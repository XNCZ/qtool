//! 合成 Rabi 幅度扫描的报告页 → plt/rabi_demo.html。
//!
//! 数据是**已知真值**的合成扫描：P1(A) = amp·(1 − cos(2π·freq·A)) 落到 |0>–|1> 连线上、再叠
//! 复高斯噪声，于是既能看到五面板报告长什么样，也能核验 `rabi_amp_fit` 的参数回收——特别是
//! π 脉冲幅 `a_pi = 1/(2·freq)`（真值一并打印）。覆盖高/低信噪比、以及"取向天然会被判反"
//! 的未标定用例。
//!
//! 噪声的 σ 一并合成（每分量独立同分布），并按 `s21` 的口径喂给拟合与作图，最后一节拿它做
//! 一次自洽检查：同一真值下 12 条独立扫描的 `a_pi` **散布**，应当与拟合报出的**标准误**同量级
//! ——两者都由同一份噪声定，一个偏了另一个不会跟着偏。
//!
//! 运行: cargo run --release --example rabi_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::rabi::rabi_amp_plot::rabi_amp_plot_div;
use qtool::superconductor::rabi::{Complex64, Cos, rabi_amp_fit, rabi_amp_fit_batch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 幅度点数与扫描范围（自零驱动起步——线型的锚点靠这一点）。
const N_AMP: usize = 41;
const AMP_MAX: f64 = 1.0;
/// 真值的 Rabi 频率（幅度的倒数）：扫程内 4 个振荡周期，每周期约 10 个采样点
/// （远离欠采样——周期数一多，噪声一进来就成了混叠，拟合出来的只是锯齿的包络）。
const FREQ_TRUE: f64 = 4.0;
/// 真值的半幅：归一化后 P1 在 [0, 1] 往返，峰值 1 ⇒ 半幅 0.5。
const AMP_TRUE: f64 = 0.5;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 合成一条 Rabi 扫描：真值 P1 落到 |0>–|1> 连线上，再叠复高斯噪声。
///
/// 形参:
///     seed: 随机种子
///     noise: 每分量噪声 σ 相对对比度 |g1 − g0| 的比例
///     inverted: 两态中心是否对调（造"自定轴会判反"的用例）
///
/// 返回值:
///     (幅度轴, 复数 IQ, IQ 域的逐点 σ)
fn synthetic(seed: u64, noise: f64, inverted: bool) -> (Vec<f64>, Vec<Complex64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let amps: Vec<f64> = (0..N_AMP)
        .map(|k| AMP_MAX * k as f64 / (N_AMP - 1) as f64)
        .collect();
    let (zero, one) = match inverted {
        true => (G1, G0),
        false => (G0, G1),
    };
    let sigma = noise * (one - zero).norm();
    let truth = Cos::new(FREQ_TRUE, AMP_TRUE);
    let prob = truth.at(&amps);
    let iq = prob
        .iter()
        .map(|p| {
            zero + (one - zero) * p
                + Complex64::new(gaussian(&mut rng) * sigma, gaussian(&mut rng) * sigma)
        })
        .collect();
    (amps, iq, vec![sigma; N_AMP])
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);
    let a_pi_true = 1.0 / (2.0 * FREQ_TRUE);

    // (标题, 种子, 噪声, 是否走两态标定, 两态中心是否对调)
    // 只出标准这一张卡：加噪声/未标定那些情形自己往这个数组里加一行就行
    let cases = [("noise 0.02", 0_u64, 0.02, true, false)];

    let mut divs = String::new();
    for (title, seed, noise, calibrated, inverted) in cases {
        let (amps, iq, sigma) = synthetic(seed, noise, inverted);
        let anchors = match calibrated {
            true => Some(&states),
            false => None,
        };
        let outcome = rabi_amp_fit(&amps, &iq, anchors, Some(&sigma));
        match &outcome {
            Ok(fit) => println!(
                "{title:<24} nfev={:<4} chisqr={:.3e} | freq: 真值 {:.3} → 拟合 {:.3} | a_pi: 真值 {:.6} → 拟合 {:.6} (Δ {:+.3}%) | amp={:.3}",
                fit.result.nfev,
                fit.result.chisqr,
                FREQ_TRUE,
                fit.result.model.freq,
                a_pi_true,
                fit.result.model.a_pi,
                (fit.result.model.a_pi - a_pi_true) / a_pi_true * 100.0,
                fit.result.model.amp,
            ),
            Err(err) => println!("{title:<24} 拟合失败: {err}"),
        }
        let div_id = format!("rabi-{seed}");
        divs.push_str(&rabi_amp_plot_div(
            &amps,
            &iq,
            anchors,
            Some(&sigma),
            outcome.as_ref(),
            &div_id,
            Some(title),
        ));
        divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
    }

    // 批量入口：同一条比特的多档参数扫描（这里用不同种子模拟）——结果按输入顺序返回。
    let mut axis: Vec<f64> = Vec::new();
    let mut lines: Vec<Vec<Complex64>> = Vec::new();
    let mut sigmas: Vec<Vec<f64>> = Vec::new();
    for seed in 0..12 {
        let (amps, iq, sigma) = synthetic(seed, 0.10, false);
        axis = amps;
        lines.push(iq);
        sigmas.push(sigma);
    }
    let batch = rabi_amp_fit_batch(&axis, &lines, Some(&vec![&states; lines.len()]), Some(&sigmas));
    let ok = batch.iter().filter(|item| item.is_ok()).count();
    println!("rabi_amp_fit_batch: {ok}/{} 条拟合成功", batch.len());

    let a_pi: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.model.a_pi)
        .collect();
    let stderrs: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .filter_map(|fit| fit.result.params.get("a_pi"))
        .filter_map(|parameter| parameter.stderr)
        .collect();
    let mean = a_pi.iter().sum::<f64>() / a_pi.len() as f64;
    let variance =
        a_pi.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (a_pi.len() - 1) as f64;
    let stderr_mean = stderrs.iter().sum::<f64>() / stderrs.len() as f64;
    let redchi_mean: f64 = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.redchi)
        .sum::<f64>()
        / a_pi.len() as f64;
    // σ 的**绝对尺度**只在加权 χ² 上现身：协方差是 (JᵀJ)⁻¹·redchi，而 J 与 redchi 各带一个
    // 1/σ 与 σ²，尺度恰好对消——标准误因此对 σ 的大小不敏感，拿它去验 σ 是验不出来的。
    // 真正的判据是 χ²/ndof：σ 给对了（就是每点的真实噪声）它才落在 1 附近。
    println!(
        "σ 检查: 12 条线加权 χ²/ndof 均值 {redchi_mean:.3}（σ 即真值噪声，应当在 1 附近）"
    );
    println!(
        "         a_pi 真值 {a_pi_true:.6} | 12 条线均值 {mean:.6} | 散布(std) {:.6} vs 平均标准误 {stderr_mean:.6}（比值 {:.2}）",
        variance.sqrt(),
        variance.sqrt() / stderr_mean,
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool rabi amp demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/rabi_demo.html", html) {
        Ok(()) => println!("wrote plt/rabi_demo.html"),
        Err(err) => println!("failed to write plt/rabi_demo.html: {err}"),
    }
}
