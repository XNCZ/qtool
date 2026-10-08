//! 合成 Ramsey（T2\*）延时扫描的报告页 → plt/ramsey_demo.html。
//!
//! 数据是**已知真值**的合成扫描：P1(τ) = offset + amplitude·exp(−τ/decay)·cos(2π·f·τ + phase)
//! 落到 |0>–|1> 连线上、再叠复高斯噪声，于是既能看到五面板报告长什么样，也能核验 `ramsey_fit`
//! 的参数回收——特别是 T2\*（`decay`，真值一并打印）。覆盖高/低信噪比、深阻尼（窗口内衰减殆尽）、
//! 以及"未标定"的用例；噪声最大的那档会滑进退化解（T2\* 欠约束），报告里由标准误与表下提示
//! 自曝，不是漏网。
//!
//! 噪声的 σ 一并合成（每分量独立同分布），并按 `s21` 的口径喂给拟合与作图，最后一节拿它做一次
//! 自洽检查：同一真值下 12 条独立扫描的 T2\* **散布**，应当与拟合报出的**标准误**同量级；而
//! σ 的绝对尺度只在**加权 χ²/ndof** 上现身（标准误对它不敏感），所以两个检查各看一个。
//!
//! T2\* 这个量在低信噪比下**本身是正偏且散布大的**（包络要在两三个时间常数里定形状，噪声总
//! 能把"更长的时间常数 + 略大的幅度"凑出同样好的残差），所以每条都带标准误一起打印——噪声最
//! 大的那档正是用来看这一点的：数字开始不可信时，标准误会先变大。
//!
//! 运行: cargo run --release --example ramsey_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::ramsey::ramsey_plot::ramsey_plot_div;
use qtool::superconductor::ramsey::{Complex64, CosDamp, ramsey_fit, ramsey_fit_batch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 延时点数与扫描范围。
const N_TAU: usize = 101;
const TAU_MAX: f64 = 60.0e-6;
/// 真值的 T2\*：扫程内衰减三个时间常数，包络看得清。
const T2_TRUE: f64 = 20.0e-6;
/// 真值的条纹频率：扫程内约 9 个周期；采样 101 点即每周期约 11 点，既远离欠采样，
/// 图上也还数得清条纹。
const FREQ_TRUE: f64 = 0.15e6;
/// 真值的常数项与幅度：P1 在 [0.1, 0.9] 之间往返，落在物理区间内。
const OFFSET_TRUE: f64 = 0.5;
const AMPLITUDE_TRUE: f64 = 0.4;
/// 真值的条纹相位 (rad)。
const PHASE_TRUE: f64 = 0.7;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 合成一条 Ramsey 扫描：真值 P1 落到 |0>–|1> 连线上，再叠复高斯噪声。
///
/// 形参:
///     seed: 随机种子
///     noise: 每分量噪声 σ 相对对比度 |g1 − g0| 的比例
///     decay_scale: 真值 T2\* 乘以这个系数（<1 就是更陡的阻尼）
///     inverted: 两态中心是否对调（造"自定轴会判反"的用例）
///
/// 返回值:
///     (延时轴, 复数 IQ, IQ 域的逐点 σ)
fn synthetic(
    seed: u64,
    noise: f64,
    decay_scale: f64,
    inverted: bool,
) -> (Vec<f64>, Vec<Complex64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let taus: Vec<f64> = (0..N_TAU)
        .map(|k| TAU_MAX * k as f64 / (N_TAU - 1) as f64)
        .collect();
    let (zero, one) = match inverted {
        true => (G1, G0),
        false => (G0, G1),
    };
    let sigma = noise * (one - zero).norm();
    // 原样构造真值线型：没有派生量，五个字段就是全部
    let truth = CosDamp {
        offset: OFFSET_TRUE,
        amplitude: AMPLITUDE_TRUE,
        frequency: FREQ_TRUE,
        phase: PHASE_TRUE,
        decay: T2_TRUE * decay_scale,
    };
    let prob = truth.at(&taus);
    let iq = prob
        .iter()
        .map(|p| {
            zero + (one - zero) * p
                + Complex64::new(gaussian(&mut rng) * sigma, gaussian(&mut rng) * sigma)
        })
        .collect();
    (taus, iq, vec![sigma; N_TAU])
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);

    // (标题, 种子, 噪声, T2* 系数, 是否走两态标定, 两态中心是否对调)
    // 只出标准这一张卡：加噪声/深阻尼/未标定那些情形自己往这个数组里加一行就行
    let cases = [("noise 0.02", 0_u64, 0.02, 1.0, true, false)];

    let mut divs = String::new();
    for (title, seed, noise, decay_scale, calibrated, inverted) in cases {
        let (taus, iq, sigma) = synthetic(seed, noise, decay_scale, inverted);
        let anchors = match calibrated {
            true => Some(&states),
            false => None,
        };
        let outcome = ramsey_fit(&taus, &iq, anchors, Some(&sigma));
        match &outcome {
            Ok(fit) => {
                // T2* 一律带标准误一起打印：低信噪比下包络本身欠约束（正偏、散布大），
                // 那个标准误就是"这个数可不可信"的自述，不带它读不出好坏
                let t2_err = fit
                    .result
                    .params
                    .get("decay")
                    .and_then(|parameter| parameter.stderr)
                    .unwrap_or(f64::NAN);
                println!(
                    "{title:<24} nfev={:<5} redchi={:.3} | T2*: 真值 {:.3e} → 拟合 {:.3e} ± {:.1e} (Δ {:+.2}%) | freq: 真值 {:.4e} → 拟合 {:.4e} | offset={:.3} amp={:.3} phase={:.3}",
                    fit.result.nfev,
                    fit.result.redchi,
                    T2_TRUE * decay_scale,
                    fit.result.model.decay,
                    t2_err,
                    (fit.result.model.decay - T2_TRUE * decay_scale) / (T2_TRUE * decay_scale) * 100.0,
                    FREQ_TRUE,
                    fit.result.model.frequency,
                    fit.result.model.offset,
                    fit.result.model.amplitude,
                    fit.result.model.phase,
                )
            }
            Err(err) => println!("{title:<24} 拟合失败: {err}"),
        }
        let div_id = format!("ramsey-{seed}");
        divs.push_str(&ramsey_plot_div(
            &taus,
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
        let (taus, iq, sigma) = synthetic(seed, 0.10, 1.0, false);
        axis = taus;
        lines.push(iq);
        sigmas.push(sigma);
    }
    let batch = ramsey_fit_batch(&axis, &lines, Some(&vec![&states; lines.len()]), Some(&sigmas));
    let ok = batch.iter().filter(|item| item.is_ok()).count();
    println!("ramsey_fit_batch: {ok}/{} 条拟合成功", batch.len());

    let t2: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.model.decay)
        .collect();
    let stderrs: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .filter_map(|fit| fit.result.params.get("decay"))
        .filter_map(|parameter| parameter.stderr)
        .collect();
    let mean = t2.iter().sum::<f64>() / t2.len() as f64;
    let variance = t2.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (t2.len() - 1) as f64;
    let stderr_mean = stderrs.iter().sum::<f64>() / stderrs.len() as f64;
    let redchi_mean: f64 = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.redchi)
        .sum::<f64>()
        / t2.len() as f64;
    println!(
        "σ 检查: 12 条线加权 χ²/ndof 均值 {redchi_mean:.3}（σ 即真值噪声，应当在 1 附近）"
    );
    println!(
        "         T2* 真值 {T2_TRUE:.6e} | 12 条线均值 {mean:.6e} | 散布(std) {:.3e} vs 平均标准误 {stderr_mean:.3e}（比值 {:.2}）",
        variance.sqrt(),
        variance.sqrt() / stderr_mean,
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool ramsey t2 demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/ramsey_demo.html", html) {
        Ok(()) => println!("wrote plt/ramsey_demo.html"),
        Err(err) => println!("failed to write plt/ramsey_demo.html: {err}"),
    }
}
