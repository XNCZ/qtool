//! 合成 qspec vs Z 扫描的报告页 → plt/qspec_z_demo.html。
//!
//! 逐 Z 偏置合成一条谱线，峰位按 tunable transmon 的通量调谐线型走（这里只是**造数据**；
//! 该线型的拟合在尚未移植的 `qspec_z.rs`）。每条线各自跑 `qspec_fit`，再交给
//! `qspec_z_plot_div` 出报告：上方 P1 热图（hover 弹该 Z 的四面板）、下方参数 vs Z 线图。
//!
//! 运行: cargo run --release --example qspec_z_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::qspec::qspec_z_plot::{QspecZLine, qspec_z_plot_div};
use qtool::superconductor::qspec::{Complex64, Lorentz, qspec_fit};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 频点数（比单点 qspec 密：峰要在窗口里挪出可见的一段）。
const N_FREQ: usize = 121;
/// 扫描窗中心与半宽：覆盖整段调谐范围。
const F_CENTER_HZ: f64 = 5.04e9;
const SCAN_HZ: f64 = 20.0e6;
/// Z 扫描范围与点数。
const Z_LOW: f64 = -0.02;
const Z_HIGH: f64 = 0.02;
const N_Z: usize = 21;
/// 真值的线宽与噪声。
const FWHM_TRUE: f64 = 2.4e6;
const NOISE: f64 = 0.10;
/// 通量调谐线型的结构参数（造数据用，与将来的 `qspec_z.rs` 拟合参数同名）。
const F_MAX_HZ: f64 = 5.05e9;
const ETA_HZ: f64 = -0.22e9;
const Z_OFFSET: f64 = 0.0;
const Z_PERIOD: f64 = 0.6;
const ASYMMETRY: f64 = 0.1;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// tunable transmon 的通量调谐线型：
/// `f01(z) = (f_max + η)·[cos²θ + d²·sin²θ]^{1/4} − η`，`θ = π(z − z_offset)/z_period`。
fn flux_tunable(z: f64) -> f64 {
    let theta = std::f64::consts::PI * (z - Z_OFFSET) / Z_PERIOD;
    let bracket = theta.cos().powi(2) + ASYMMETRY.powi(2) * theta.sin().powi(2);
    (F_MAX_HZ + ETA_HZ) * bracket.powf(0.25) - ETA_HZ
}

/// 合成一条谱线：真值 P1 落到 |0>–|1> 连线上，再叠复高斯噪声。
fn synthetic(seed: u64, fq_true: f64) -> (Vec<f64>, Vec<Complex64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let freqs: Vec<f64> = (0..N_FREQ)
        .map(|k| F_CENTER_HZ + SCAN_HZ * (2.0 * k as f64 / (N_FREQ - 1) as f64 - 1.0))
        .collect();
    let truth = Lorentz {
        fq: fq_true,
        fwhm: FWHM_TRUE,
        amp: 1.0,
        offset: 0.05,
    };
    let sigma = NOISE * (G1 - G0).norm();
    let prob = truth.at(&freqs);
    let iq = prob
        .iter()
        .map(|p| {
            G0 + (G1 - G0) * p
                + Complex64::new(gaussian(&mut rng) * sigma, gaussian(&mut rng) * sigma)
        })
        .collect();
    (freqs, iq)
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);

    let mut axis: Vec<f64> = Vec::new();
    let mut iq_lines: Vec<Vec<Complex64>> = Vec::new();
    let mut fits = Vec::with_capacity(N_Z);
    for row in 0..N_Z {
        let z = Z_LOW + (Z_HIGH - Z_LOW) * row as f64 / (N_Z - 1) as f64;
        let fq_true = flux_tunable(z);
        let (freqs, iq) = synthetic(row as u64, fq_true);
        axis = freqs;
        let outcome = qspec_fit(&axis, &iq, Some(&states), None);
        match &outcome {
            Ok(fit) => println!(
                "Z {z:+.4} V: 真值 {:.4} GHz → 拟合 {:.4} GHz (Δ {:+.1} kHz), fwhm {:.3} MHz",
                fq_true / 1e9,
                fit.result.model.fq / 1e9,
                (fit.result.model.fq - fq_true) / 1e3,
                fit.result.model.fwhm / 1e6,
            ),
            Err(err) => println!("Z {z:+.4} V: 拟合失败: {err}"),
        }
        fits.push(outcome);
        iq_lines.push(iq);
    }

    let lines: Vec<QspecZLine<'_>> = (0..N_Z)
        .map(|row| QspecZLine {
            z: Z_LOW + (Z_HIGH - Z_LOW) * row as f64 / (N_Z - 1) as f64,
            iq: &iq_lines[row],
            fit: fits[row].as_ref(),
        })
        .collect();

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool qspec vs Z demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{}</body>\n</html>\n",
        qspec_z_plot_div(&axis, &lines, Some(&states), "qspec-z", Some("qspec vs Z"))
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/qspec_z_demo.html", html) {
        Ok(()) => println!("wrote plt/qspec_z_demo.html"),
        Err(err) => println!("failed to write plt/qspec_z_demo.html: {err}"),
    }
}
