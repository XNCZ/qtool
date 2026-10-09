//! 合成 T1（能量弛豫）延时扫描的报告页 → plt/t1_demo.html。
//!
//! 数据是**已知真值**的合成扫描：P1(τ) = offset + amplitude·exp(−τ/t1) 落到 |0>–|1> 连线上、
//! 再叠复高斯噪声，于是既能看到四面板报告长什么样，也能核验 `t1_fit` 的参数回收——特别是 T1
//! （真值一并打印）。覆盖高/低信噪比、**T1 远大于窗口**（包络在窗口内几乎看不出来，T1 欠约束）、
//! 以及"未标定"和"未标定且朝向判反"的用例：后者的原始投影是朝上升的，`t1_fit` 不翻正——
//! 页面上的曲线就停在反向（`amplitude` 为负），T1 的值不受朝向影响。
//!
//! 噪声的 σ 一并合成（每分量独立同分布），并按 `s21` 的口径喂给拟合与作图，最后一节拿它做一次
//! 自洽检查：同一真值下 12 条独立扫描的 T1 **散布**，应当与拟合报出的**标准误**同量级；而 σ
//! 的绝对尺度只在**加权 χ²/ndof** 上现身（标准误对它不敏感），所以两个检查各看一个。
//!
//! T1 这个量在"窗口 ≪ T1"时**本身是正偏且散布大的**（包络在一个时间常数都不到的范围里定不出
//! 形状，噪声总能把"更长的时间常数 + 略小的幅度"凑出同样好的残差），所以每条都带标准误一起
//! 打印——数字开始不可信时，标准误会先变大。
//!
//! 运行: cargo run --release --example t1_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::t1::t1_plot::t1_plot_div;
use qtool::superconductor::t1::{Complex64, Decay, t1_fit, t1_fit_batch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 延时点数与扫描范围。
const N_TAU: usize = 101;
const TAU_MAX: f64 = 200.0e-6;
/// 真值的 T1：扫程内衰减五个时间常数，包络看得清。
const T1_TRUE: f64 = 40.0e-6;
/// 真值的本底与幅度：起点 P1 = 0.95，长延时落到 0.03——两个数都带 SPAM 的含义（读出误判、
/// π 脉冲激发不干净），正是 `offset` 与 `amplitude` 该吸收的那部分。
const OFFSET_TRUE: f64 = 0.03;
const AMPLITUDE_TRUE: f64 = 0.92;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 合成一条 T1 扫描：真值 P1 落到 |0>–|1> 连线上，再叠复高斯噪声。
///
/// 形参:
///     seed: 随机种子
///     truth: 真值线型
///     noise: 每分量噪声 σ 相对对比度 |g1 − g0| 的比例
///     inverted: 两态中心是否对调（造"自定轴会判反、曲线朝上升"的用例）
///
/// 返回值:
///     (延时轴, 复数 IQ, IQ 域的逐点 σ)
fn synthetic(seed: u64, truth: Decay, noise: f64, inverted: bool) -> (Vec<f64>, Vec<Complex64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let taus: Vec<f64> = (0..N_TAU)
        .map(|k| TAU_MAX * k as f64 / (N_TAU - 1) as f64)
        .collect();
    let (zero, one) = match inverted {
        true => (G1, G0),
        false => (G0, G1),
    };
    let sigma = noise * (one - zero).norm();
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

    // (标题, 种子, 噪声, T1 相对真值的倍数, 是否走两态标定, 两态中心是否对调)
    // 只出标准这一张卡：加噪声/长 T1/未标定那些情形自己往这个数组里加一行就行
    let cases = [("noise 0.02", 0_u64, 0.02, 1.0, true, false)];

    let mut divs = String::new();
    for (title, seed, noise, t1_scale, calibrated, inverted) in cases {
        let truth = Decay {
            offset: OFFSET_TRUE,
            amplitude: AMPLITUDE_TRUE,
            t1: T1_TRUE * t1_scale,
        };
        let (taus, iq, sigma) = synthetic(seed, truth, noise, inverted);
        let anchors = match calibrated {
            true => Some(&states),
            false => None,
        };
        let outcome = t1_fit(&taus, &iq, anchors, Some(&sigma));
        match &outcome {
            Ok(fit) => {
                // T1 一律带标准误一起打印：窗口比它短时包络本身欠约束（正偏、散布大），那个
                // 标准误就是"这个数可不可信"的自述，不带它读不出好坏
                let t1_err = fit
                    .result
                    .params
                    .get("t1")
                    .and_then(|parameter| parameter.stderr)
                    .unwrap_or(f64::NAN);
                println!(
                    "{title:<26} nfev={:<5} redchi={:.3} | T1: 真值 {:.3e} → 拟合 {:.3e} ± {:.1e} (Δ {:+.2}%) | offset={:.3} amp={:+.3}",
                    fit.result.nfev,
                    fit.result.redchi,
                    truth.t1,
                    fit.result.model.t1,
                    t1_err,
                    (fit.result.model.t1 - truth.t1) / truth.t1 * 100.0,
                    fit.result.model.offset,
                    fit.result.model.amplitude,
                )
            }
            Err(err) => println!("{title:<26} 拟合失败: {err}"),
        }
        let div_id = format!("t1-{seed}");
        divs.push_str(&t1_plot_div(
            &taus,
            &iq,
            anchors,
            Some(&sigma),
            outcome.as_ref(),
            &div_id,
            Some(title),
        ));
    }

    // 批量入口：同一条比特的多档参数扫描（这里用不同种子模拟）——结果按输入顺序返回。
    let mut axis: Vec<f64> = Vec::new();
    let mut lines: Vec<Vec<Complex64>> = Vec::new();
    let mut sigmas: Vec<Vec<f64>> = Vec::new();
    for seed in 0..12 {
        let truth = Decay {
            offset: OFFSET_TRUE,
            amplitude: AMPLITUDE_TRUE,
            t1: T1_TRUE,
        };
        let (taus, iq, sigma) = synthetic(seed, truth, 0.10, false);
        axis = taus;
        lines.push(iq);
        sigmas.push(sigma);
    }
    let batch = t1_fit_batch(&axis, &lines, Some(&vec![&states; lines.len()]), Some(&sigmas));
    let ok = batch.iter().filter(|item| item.is_ok()).count();
    println!("t1_fit_batch: {ok}/{} 条拟合成功", batch.len());

    let fitted: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.model.t1)
        .collect();
    let stderrs: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .filter_map(|fit| fit.result.params.get("t1"))
        .filter_map(|parameter| parameter.stderr)
        .collect();
    let mean = fitted.iter().sum::<f64>() / fitted.len() as f64;
    let variance =
        fitted.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (fitted.len() - 1) as f64;
    let stderr_mean = stderrs.iter().sum::<f64>() / stderrs.len() as f64;
    let redchi_mean: f64 = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.redchi)
        .sum::<f64>()
        / fitted.len() as f64;
    println!("σ 检查: 12 条线加权 χ²/ndof 均值 {redchi_mean:.3}（σ 即真值噪声，应当在 1 附近）");
    println!(
        "         T1 真值 {T1_TRUE:.6e} | 12 条线均值 {mean:.6e} | 散布(std) {:.3e} vs 平均标准误 {stderr_mean:.3e}（比值 {:.2}）",
        variance.sqrt(),
        variance.sqrt() / stderr_mean,
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool t1 demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/t1_demo.html", html) {
        Ok(()) => println!("wrote plt/t1_demo.html"),
        Err(err) => println!("failed to write plt/t1_demo.html: {err}"),
    }
}
