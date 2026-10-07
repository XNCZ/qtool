//! 合成 qspec 谱线的报告页 → plt/qspec_demo.html。
//!
//! 数据是**已知真值**的合成谱线：洛伦兹线型的 P1 落到 |0>–|1> 连线上、再叠复高斯噪声，
//! 于是既能看到四面板报告长什么样，也能核验 `qspec_fit` 的参数回收（真值一并打印）。
//! 覆盖高/中/低信噪比、窄线、以及共振落在扫描边界这几种形态。
//!
//! 运行: cargo run --release --example qspec_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::qspec_plot::{PLOTLY_JS_CDN, qspec_fit_plot_div};
use qtool::superconductor::qspec::{Complex64, Lorentz, qspec_fit, qspec_fit_batch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 频点数（与真实 qspec 扫描一致）。
const N_FREQ: usize = 51;
/// 扫描半宽：中心两侧各 3 MHz。
const SCAN_HZ: f64 = 3.0e6;
/// 扫描窗中心（比特频率标称值）。
const FQ_HZ: f64 = 5.0e9;
/// 真值的基线与峰高。
const AMP_TRUE: f64 = 1.0;
const OFFSET_TRUE: f64 = 0.05;
/// 两态中心：**任意夹角**的一对（间距 0.233，与 I 轴约 31°）。
///
/// 特意不取轴对齐的连线——竖直或水平的连线会让"主轴方向"退化成一个坐标轴，绕开
/// [`qtool::superconductor::qspec::direction`] 真正要做的事。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 合成一条谱线：真值 P1 落到 |0>–|1> 连线上，再叠复高斯噪声。
///
/// 形参:
///     seed: 随机种子
///     fwhm_true: 真值半高全宽，Hz
///     fq_true: 真值峰位，Hz
///     noise: 每分量噪声 σ 相对对比度 |g1 − g0| 的比例
///     inverted: 真值两态中心是否对调（用来造"自定轴会判反"的用例）
///
/// 返回值:
///     (频率轴, 复数 IQ)
fn synthetic(
    seed: u64,
    fwhm_true: f64,
    fq_true: f64,
    noise: f64,
    inverted: bool,
) -> (Vec<f64>, Vec<Complex64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let freqs: Vec<f64> = (0..N_FREQ)
        .map(|k| FQ_HZ + SCAN_HZ * (2.0 * k as f64 / (N_FREQ - 1) as f64 - 1.0))
        .collect();
    let truth = Lorentz {
        fq: fq_true,
        fwhm: fwhm_true,
        amp: AMP_TRUE,
        offset: OFFSET_TRUE,
    };
    // 两态中心对调：|0>→|1> 的连线 I 分量为负，自定轴的"I 分量为正"约定会把谱线判反
    let (zero, one) = match inverted {
        true => (G1, G0),
        false => (G0, G1),
    };
    let sigma = noise * (one - zero).norm();
    let prob = truth.at(&freqs);
    let iq = prob
        .iter()
        .map(|p| {
            zero + (one - zero) * p
                + Complex64::new(gaussian(&mut rng) * sigma, gaussian(&mut rng) * sigma)
        })
        .collect();
    (freqs, iq)
}

fn main() {
    // 各态标定中心：索引即态编号，P1 只用前两态
    let states = StateCenters::new(vec![G0, G1]);

    // (标题, 种子, 真值半高全宽, 真值峰位, 噪声, 是否走两态标定, 两态中心是否对调)
    // 只出标准这一张卡：加噪声/加宽/未标定那些情形自己往这个数组里加一行就行
    let cases = [("noise 0.02", 0_u64, 2.4e6, FQ_HZ, 0.02, true, false)];

    let mut divs = String::new();
    for (title, seed, fwhm_true, fq_true, noise, calibrated, inverted) in cases {
        let (freqs, iq) = synthetic(seed, fwhm_true, fq_true, noise, inverted);
        let anchors = match calibrated {
            true => Some(&states),
            false => None,
        };
        let outcome = qspec_fit(&freqs, &iq, anchors, None);
        match &outcome {
            Ok(fit) => println!(
                "{title:<24} nfev={:<5} chisqr={:.3e} | fq: 真值 {:.6} GHz → 拟合 {:.6} GHz (Δ {:+.1} kHz) | fwhm: 真值 {:.3} MHz → 拟合 {:.3} MHz | amp={:.3}",
                fit.result.nfev,
                fit.result.chisqr,
                fq_true / 1e9,
                fit.result.model.fq / 1e9,
                (fit.result.model.fq - fq_true) / 1e3,
                fwhm_true / 1e6,
                fit.result.model.fwhm / 1e6,
                fit.result.model.amp,
            ),
            Err(err) => println!("{title:<24} 拟合失败: {err}"),
        }
        let div_id = format!("qspec-{seed}");
        divs.push_str(&qspec_fit_plot_div(
            &freqs,
            &iq,
            anchors,
            None,
            outcome.as_ref(),
            &div_id,
            Some(title),
        ));
        divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
    }

    // 批量入口：同一条比特的多档功率（这里用不同种子模拟）——结果按输入顺序返回
    let mut axis: Vec<f64> = Vec::new();
    let mut lines: Vec<Vec<Complex64>> = Vec::new();
    for seed in 0..12 {
        let (freqs, iq) = synthetic(seed, 2.4e6, FQ_HZ, 0.10, false);
        axis = freqs;
        lines.push(iq);
    }
    let batch = qspec_fit_batch(&axis, &lines, Some(&states), None);
    let ok = batch.iter().filter(|item| item.is_ok()).count();
    println!("qspec_fit_batch: {}/{} 条拟合成功", ok, batch.len());

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool qspec demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/qspec_demo.html", html) {
        Ok(()) => println!("wrote plt/qspec_demo.html"),
        Err(err) => println!("failed to write plt/qspec_demo.html: {err}"),
    }
}
