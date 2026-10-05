//! S21 拟合的噪声抵抗能力扫描。
//!
//! 两套噪声模型，逐档升高强度，每档多次重复（不同噪声种子）：
//! - **高斯**：加性复高斯，逐点 σ 恒定（与信号无关）；
//! - **散粒（泊松型）**：把 |S21| 视作强度 ∝ 计数，σ_i ∝ √|S21_i|——
//!   即计数服从泊松分布的复数据体现：峰值处相对噪声 1/√N，notch 底部最脏。
//!
//! 统计口径：成功率（s12_fit 返回 Ok 的比例）与各参数的相对误差中位数；
//! θ 按模 π、φ 按模 2π 的等价分支折算。
//!
//! 运行: cargo run --release --example s21_noise_scan

use qtool::superconductor::s21::{Complex64, S21Model, model_at, s12_fit};
use std::f64::consts::PI;

/// 确定性 LCG + Box-Muller，保证每次运行结果可复现。
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as f64) / ((1_u64 << 31) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn truth() -> S21Model {
    S21Model::from_fit_array(&[
        6.8982e9, 8.0e3, -1.4e4, 6.02, 0.1, -2.4e-7, 4.4e9, 4.4e9, 10521.0, -3.0e6, 8.2e6,
    ])
}

/// 相对误差，θ 模 π、φ 模 2π 折算等价分支。
fn relative_error(index: usize, fitted: f64, target: f64) -> f64 {
    let delta = match index {
        3 => {
            let half = PI;
            let raw = (fitted - target).rem_euclid(half);
            raw.min(half - raw)
        }
        8 => {
            let period = 2.0 * PI;
            let raw = (fitted - target).rem_euclid(period);
            raw.min(period - raw)
        }
        _ => (fitted - target).abs(),
    };
    delta / target.abs().max(f64::MIN_POSITIVE)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| match a.partial_cmp(b) {
        Some(ordering) => ordering,
        None => std::cmp::Ordering::Equal,
    });
    values[values.len() / 2]
}

fn main() {
    let truth = truth();
    let targets = truth.to_array();
    let freqs: Vec<f64> = (0..51)
        .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
        .collect();
    let clean: Vec<Complex64> = freqs.iter().map(|f| model_at(*f, &truth)).collect();
    let peak = clean.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
    let reps = 50;

    println!("峰值 |S21| = {peak:.3e}，重复 {reps} 次/档\n");
    println!("=== 高斯噪声（逐点 σ 恒定，按峰值相对幅度给出） ===");
    println!(
        "{:>8} {:>9} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "σ/峰值", "SNR峰值", "成功", "fr误差/线宽", "稳定率", "ql 误差", "qc 误差", "ap 误差", "tau 误差"
    );
    let gaussian_levels = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0];
    scan(&freqs, &clean, &targets, peak, reps, &gaussian_levels, false);

    println!("\n=== 散粒噪声（σ ∝ √|S21|，按峰值计数 N 给出） ===");
    println!(
        "{:>8} {:>9} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "σ/峰值", "SNR峰值", "成功", "fr误差/线宽", "稳定率", "ql 误差", "qc 误差", "ap 误差", "tau 误差"
    );
    let shot_levels = [1e1, 3e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
    scan(&freqs, &clean, &targets, peak, reps, &shot_levels, true);
}

/// 一档噪声：生成 reps 条带噪数据，逐条拟合，统计成功与误差中位数。
fn scan(
    freqs: &[f64],
    clean: &[Complex64],
    targets: &[f64; 11],
    peak: f64,
    reps: usize,
    levels: &[f64],
    shot: bool,
) {
    let linewidth = targets[0] / targets[1];
    for level in levels.iter() {
        let mut errors: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); 11];
        let mut fr_lw: Vec<f64> = Vec::with_capacity(reps);
        let mut successes = 0_usize;
        for rep in 0..reps {
            let mut rng = Rng::new(0x5eed_0000 ^ level.to_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ rep as u64);
            let noisy: Vec<Complex64> = clean
                .iter()
                .map(|z| {
                    let sigma = match shot {
                        // 计数 n = N·|z|/峰值，σ_count = √n → σ_amp = |z|/√n
                        true => z.norm() / (level * z.norm() / peak).sqrt(),
                        false => level * peak,
                    };
                    z + Complex64::new(sigma * rng.next_normal(), sigma * rng.next_normal())
                })
                .collect();
            match s12_fit(freqs, &noisy, None) {
                Ok(result) => {
                    successes += 1;
                    let fitted = result.model.to_array();
                    for k in 0..11 {
                        errors[k].push(relative_error(k, fitted[k], targets[k]));
                    }
                    fr_lw.push(relative_error(0, fitted[0], targets[0]) * targets[0] / linewidth);
                }
                Err(_err) => {}
            }
        }
        let mut stat = |k: usize| -> f64 {
            match errors[k].is_empty() {
                true => f64::NAN,
                false => median(&mut errors[k]),
            }
        };
        let fr_err_lw = stat(0) * targets[0] / linewidth;
        let ql_err = stat(1);
        let qc_err = stat(2);
        let ap_err = stat(4);
        let tau_err = stat(5);
        let stable = fr_lw.iter().filter(|e| **e < 0.2).count() as f64 / reps as f64;
        let snr = match shot {
            true => level.sqrt(),
            false => 1.0 / level,
        };
        let label = format!("{level:.0e}");
        println!(
            "{:>8} {:>9.0} {:>5}/{} {:>9.2} {:>9.0}% {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}",
            label, snr, successes, reps, fr_err_lw, stable * 100.0, ql_err, qc_err, ap_err, tau_err
        );
    }
}
