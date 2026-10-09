//! 合成 qspec vs Z 扫描的报告页 → plt/qspec_z_demo.html。
//!
//! 逐 Z 偏置合成一条谱线，峰位按 tunable transmon 的通量调谐线型走（这里只是**造数据**；
//! 该线型的拟合在尚未移植的 `qspec_z.rs`）。每条线各自跑 `qspec_fit`，再交给
//! `qspec_z_plot_div` 出报告：上方 P1 热图（hover 弹该 Z 的四面板）、下方参数 vs Z 线图。
//!
//! 两张卡片：上面一张固定宽窗（整段调谐范围共用一条公共轴，`QspecZLine::freqs` 留 `None`），
//! 下面一张动窗扫描（窗心逐行外推到该行的真值峰位，每行自带频率轴，热图按各轴的并集铺开、
//! 没覆盖到的格子留空）。两张都把逐行的峰位拟成通量调谐线型（`flux_fit`），各自在报告最下方
//! 出一张 f01 vs Z 的面板（峰位点 + 拟合曲线 + 参数表）。
//!
//! 运行: cargo run --release --example qspec_z_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::qspec::flux::{FluxError, FluxFit, flux_fit};
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::qspec::qspec_z_plot::{QspecZLine, qspec_z_plot_div};
use qtool::superconductor::qspec::{Complex64, Lorentz, QspecError, QspecFit, qspec_fit};
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
/// 动窗扫描的半宽与点数：窗心逐行外推，每行只扫自己这一段。
const WINDOW_HZ: f64 = 15.0e6;
const WINDOW_N: usize = 31;
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
///
/// 形参:
///     seed: 随机种子
///     fq_true: 该行的真值峰位，Hz
///     center_hz: 该行扫描窗的中心，Hz
///     scan_hz: 该行扫描窗的半宽，Hz
///     n_freq: 该行频点数
///
/// 返回值:
///     (该行的频率轴, 该行的平均复数 IQ)
fn synthetic(
    seed: u64,
    fq_true: f64,
    center_hz: f64,
    scan_hz: f64,
    n_freq: usize,
) -> (Vec<f64>, Vec<Complex64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let freqs: Vec<f64> = (0..n_freq)
        .map(|k| center_hz + scan_hz * (2.0 * k as f64 / (n_freq - 1) as f64 - 1.0))
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

/// 逐行的拟合峰位：拟合失败的那行给 NaN（plotly 按断点跳过，也不进通量拟合）。
///
/// 形参:
///     fits: 逐行的洛伦兹拟合结果
///
/// 返回值:
///     逐行峰位 (n,)，Hz
fn fit_peaks(fits: &[Result<QspecFit, QspecError>]) -> Vec<f64> {
    fits.iter()
        .map(|outcome| match outcome {
            Ok(fit) => fit.result.model.fq,
            Err(_not_fitted) => f64::NAN,
        })
        .collect()
}

/// 通量拟合结果转成报告要的形参：失败也给 `Some(Err(..))`，报告照画峰位点并在表下说明原因。
///
/// 形参:
///     flux: 通量调谐拟合结果
///
/// 返回值:
///     [`qspec_z_plot_div`] 的通量面板形参
fn flux_arg(flux: &Result<FluxFit, FluxError>) -> Option<Result<&FluxFit, &FluxError>> {
    match flux {
        Ok(fit) => Some(Ok(fit)),
        Err(error) => Some(Err(error)),
    }
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);
    // 逐行偏置轴：两张卡片与通量拟合共用同一条
    let zs: Vec<f64> = (0..N_Z)
        .map(|row| Z_LOW + (Z_HIGH - Z_LOW) * row as f64 / (N_Z - 1) as f64)
        .collect();

    let mut axis: Vec<f64> = Vec::new();
    let mut iq_lines: Vec<Vec<Complex64>> = Vec::new();
    let mut fits = Vec::with_capacity(N_Z);
    for row in 0..N_Z {
        let z = zs[row];
        let fq_true = flux_tunable(z);
        let (freqs, iq) = synthetic(row as u64, fq_true, F_CENTER_HZ, SCAN_HZ, N_FREQ);
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
            z: zs[row],
            iq: &iq_lines[row],
            fit: fits[row].as_ref(),
            freqs: None,
        })
        .collect();

    // 动窗扫描：窗心逐行外推到该行的真值峰位，每行只扫 WINDOW_HZ 的半宽，行轴因此各不相同。
    let mut row_freqs: Vec<Vec<f64>> = Vec::with_capacity(N_Z);
    let mut window_iq: Vec<Vec<Complex64>> = Vec::with_capacity(N_Z);
    let mut window_fits = Vec::with_capacity(N_Z);
    let (mut worst_khz, mut failures) = (0.0_f64, 0_usize);
    for row in 0..N_Z {
        let z = Z_LOW + (Z_HIGH - Z_LOW) * row as f64 / (N_Z - 1) as f64;
        let fq_true = flux_tunable(z);
        let (freqs, iq) = synthetic(1000 + row as u64, fq_true, fq_true, WINDOW_HZ, WINDOW_N);
        let outcome = qspec_fit(&freqs, &iq, Some(&states), None);
        match &outcome {
            Ok(fit) => worst_khz = worst_khz.max((fit.result.model.fq - fq_true).abs() / 1e3),
            Err(_failure) => failures += 1,
        }
        row_freqs.push(freqs);
        window_iq.push(iq);
        window_fits.push(outcome);
    }
    println!(
        "动窗卡片: {N_Z} 行 x {WINDOW_N} 点（半宽 {:.1} MHz），失败 {failures} 行，最大峰位偏差 {worst_khz:.1} kHz",
        WINDOW_HZ / 1e6
    );
    let window_lines: Vec<QspecZLine<'_>> = (0..N_Z)
        .map(|row| QspecZLine {
            z: zs[row],
            iq: &window_iq[row],
            fit: window_fits[row].as_ref(),
            freqs: Some(&row_freqs[row]),
        })
        .collect();

    // 通量调谐：把动窗卡片逐偏置的峰位拟成 SQUID 线型（真值在文件头，一并对拍）
    let flux = flux_fit(&zs, &fit_peaks(&window_fits), ETA_HZ, ASYMMETRY);
    match &flux {
        Ok(fit) => {
            let model = fit.result.model;
            println!(
                "通量拟合: f_max {:.4} → {:.4} GHz | z_offset {:.4} → {:.4} V | z_period {:.3} → {:.3} V | redchi {:.2e}",
                F_MAX_HZ / 1e9,
                model.f_max / 1e9,
                Z_OFFSET,
                model.z_offset,
                Z_PERIOD,
                model.z_period,
                fit.result.redchi,
            );
            // 反解往返：把甜点频率调回去，应当落回甜点偏置
            match fit.tune_to(model.f_max, None) {
                Some(z) => println!("tune_to(f_max) → {z:.4} V（甜点 {:.4} V）", model.z_offset),
                None => println!("tune_to(f_max) → 不可达（不合预期）"),
            }
        }
        Err(err) => println!("通量拟合失败: {err}"),
    }

    // 连续扫描卡片同样出通量调谐面板：用它自己那批（固定宽窗）拟合的峰位
    let fixed_flux = flux_fit(&zs, &fit_peaks(&fits), ETA_HZ, ASYMMETRY);
    match &fixed_flux {
        Ok(fit) => println!(
            "连续扫描卡片的通量拟合: f_max {:.4} → {:.4} GHz | z_offset {:.4} V | z_period {:.3} V | redchi {:.2e}",
            F_MAX_HZ / 1e9,
            fit.result.model.f_max / 1e9,
            fit.result.model.z_offset,
            fit.result.model.z_period,
            fit.result.redchi,
        ),
        Err(err) => println!("连续扫描卡片的通量拟合失败: {err}"),
    }

    let fixed_card = qspec_z_plot_div(
        &axis,
        &lines,
        Some(&states),
        "qspec-z",
        Some("qspec vs Z"),
        flux_arg(&fixed_flux),
    );
    // 公共轴对动窗卡片用不上（每行自带轴），传空切片
    let window_card = qspec_z_plot_div(
        &[],
        &window_lines,
        Some(&states),
        "qspec-z-window",
        Some("qspec vs Z - 动窗扫描（逐行频率轴）"),
        flux_arg(&flux),
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool qspec vs Z demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{fixed_card}\
         <hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n{window_card}</body>\n</html>\n"
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
