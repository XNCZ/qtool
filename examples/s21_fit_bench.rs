//! S21 拟合的两侧对照基准（Rust 侧）。
//!
//! 与 Python 侧脚本使用同一份确定性数据（同参数的 LCG 噪声）；初值估计与
//! 全流程为有意改进（θ 由圆几何确定，baseline 的采样式 θ 会全部落入错误盆地），
//! 两侧输出不再相同。
//! 计时取多次重复的最小值与中位数，抑制变频抖动。
//!
//! 运行: cargo run --release --example s21_fit_bench

use qtool::superconductor::s21::{
    Complex64, JAC_NAMES, S21Model, jacobian, model_at, s12_fit,
};
use std::hint::black_box;
use std::time::Instant;

/// 与 baseline 测试夹具一致的 11 个真值参数。
fn truth_params() -> S21Model {
    S21Model::from_fit_array(&[
        6.8982e9, 8.0e3, -1.4e4, 6.02, 0.1, -2.4e-7, 4.4e9, 4.4e9, 10521.0, -3.0e6, 8.2e6,
    ])
}

/// 与 Python 侧逐位一致的确定性均匀噪声源（LCG，取值 [-0.5, 0.5)）。
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new() -> Self {
        Self { state: 0 }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5
    }
}

/// 重复计时：返回 (最小值 ms, 中位数 ms)。
fn repeat_ms<T>(reps: usize, mut body: impl FnMut() -> T) -> (f64, f64) {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        black_box(body());
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| match a.partial_cmp(b) {
        Some(ordering) => ordering,
        None => std::cmp::Ordering::Equal,
    });
    (samples[0], samples[samples.len() / 2])
}

fn main() {
    let truth = truth_params();
    let freqs: Vec<f64> = (0..51)
        .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
        .collect();
    let noiseless: Vec<Complex64> = freqs.iter().map(|f| model_at(*f, &truth)).collect();

    // 噪声按 ±2e-3 注入，两侧序列一致
    let mut lcg = Lcg::new();
    let noisy: Vec<Complex64> = noiseless
        .iter()
        .map(|z| z + Complex64::new(4e-3 * lcg.next_unit(), 4e-3 * lcg.next_unit()))
        .collect();

    println!("=== 微基准（51 点一条线） ===");
    let (model_min, model_med) = repeat_ms(200, || {
        for f in freqs.iter() {
            black_box(model_at(*f, &truth));
        }
    });
    let (jac_min, jac_med) = repeat_ms(200, || {
        for f in freqs.iter() {
            black_box(jacobian(*f, &truth));
        }
    });
    println!(
        "一条线 model_at: min {:.2} us / median {:.2} us | jacobian: min {:.2} us / median {:.2} us ({:.2}x)",
        model_min * 1e3,
        model_med * 1e3,
        jac_min * 1e3,
        jac_med * 1e3,
        jac_med / model_med
    );

    println!("\n=== 全流程 s12_fit（无噪声，3 次重复） ===");
    let (full_min, full_med) = repeat_ms(3, || s12_fit(&freqs, &noiseless, None));
    match s12_fit(&freqs, &noiseless, None) {
        Ok(result) => {
            println!(
                "time: min {full_min:.3} ms / median {full_med:.3} ms | nfev(最优候选)={} chisqr={:.6e} residual(=chisqr/n)={:.6e}",
                result.nfev,
                result.chisqr,
                result.chisqr / (result.ndata as f64 / 2.0)
            );
            print_params(&result.model);
        }
        Err(err) => println!("s12_fit 失败：{err}"),
    }

    println!("\n=== 全流程 s12_fit（带噪声：无权重 vs 常数 sigma，各 3 次重复） ===");
    let (nw_min, nw_med) = repeat_ms(3, || s12_fit(&freqs, &noisy, None));
    match s12_fit(&freqs, &noisy, None) {
        Ok(result) => {
            println!(
                "无权重: time min {nw_min:.3} ms / median {nw_med:.3} ms | chisqr={:.6e}",
                result.chisqr
            );
            print_params(&result.model);
        }
        Err(err) => println!("无权重拟合失败：{err}"),
    }
    let sigma = vec![2e-3_f64; freqs.len()];
    let (w_min, w_med) = repeat_ms(3, || s12_fit(&freqs, &noisy, Some(&sigma)));
    match s12_fit(&freqs, &noisy, Some(&sigma)) {
        Ok(result) => {
            println!(
                "常数 sigma=2e-3: time min {w_min:.3} ms / median {w_med:.3} ms | 加权 chisqr={:.6e}",
                result.chisqr
            );
            print_params(&result.model);
        }
        Err(err) => println!("加权拟合失败：{err}"),
    }
}

fn print_params(model: &S21Model) {
    let values = model.to_array();
    let text: Vec<String> = JAC_NAMES
        .iter()
        .zip(values.iter())
        .map(|(name, value)| format!("{name}={value:.10e}"))
        .collect();
    println!("{}", text.join(" "));
}

