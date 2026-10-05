//! 批量 S12 拟合基准：3000 条不同频率线的全流程拟合（初值估计 + 候选扫描），
//! 串行与 rayon 并行对照。Python 侧脚本用同一 LCG 生成同样的 3000 个问题
//! （见 /tmp 的 bench_batch.py），但其 lmfit 受 GIL 约束只能串行。
//!
//! 运行: cargo run --release --example s12_batch_bench

use qtool::superconductor::s21::{Complex64, S21Model, model_at, s12_fit, s12_fit_batch};
use std::hint::black_box;
use std::time::Instant;

/// 与 Python 侧逐位一致的确定性均匀噪声源（LCG，取值 [-0.5, 0.5)）。
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5
    }
}

/// 第 index 条线的参数：在夹具真值附近确定性抖动，构成 3000 个不同问题。
fn line_params(index: usize) -> S21Model {
    let mut lcg = Lcg::new(index as u64 + 1);
    S21Model::from_fit_array(&[
        6.8982e9 + lcg.next_unit() * 6e6,        // fr ±3 MHz
        8.0e3 * (1.0 + lcg.next_unit() * 0.5),   // ql ±25%
        -1.4e4 * (1.0 + lcg.next_unit() * 0.6),  // qc ±30%（保持负号）
        6.02 + lcg.next_unit() * 0.8,            // theta ±0.4 rad
        0.1,
        -2.4e-7,
        4.4e9,
        4.4e9,
        10521.0,
        -3.0e6,
        8.2e6,
    ])
}

fn main() {
    let n_lines = 3000;
    let freqs: Vec<f64> = (0..51)
        .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
        .collect();

    // 逐线生成数据：不同参数 + 独立噪声
    let lines: Vec<Vec<Complex64>> = (0..n_lines)
        .map(|index| {
            let params = line_params(index);
            let mut lcg = Lcg::new(0x9e37_79b9_7f4a_7c15 ^ (index as u64) << 17);
            freqs
                .iter()
                .map(|f| {
                    model_at(*f, &params)
                        + Complex64::new(4e-3 * lcg.next_unit(), 4e-3 * lcg.next_unit())
                })
                .collect()
        })
        .collect();

    println!(
        "问题数 = {n_lines} 条 × {} 频点 | rayon 线程数 = {}",
        freqs.len(),
        rayon::current_num_threads()
    );

    // 串行
    let start = Instant::now();
    let mut serial_failures = 0_usize;
    for line in lines.iter() {
        match s12_fit(&freqs, line, None) {
            Ok(result) => {
                black_box(result.chisqr);
            }
            Err(_err) => serial_failures += 1,
        }
    }
    let serial_ms = start.elapsed().as_secs_f64() * 1e3;

    // rayon 并行
    let start = Instant::now();
    let results = s12_fit_batch(&freqs, &lines, None);
    let parallel_ms = start.elapsed().as_secs_f64() * 1e3;
    let parallel_failures = results
        .iter()
        .filter(|outcome| outcome.is_err())
        .count();

    println!(
        "串行  : {serial_ms:9.1} ms（{:.3} ms/条，失败 {serial_failures}）",
        serial_ms / n_lines as f64
    );
    println!(
        "rayon : {parallel_ms:9.1} ms（{:.3} ms/条，失败 {parallel_failures}）→ 加速 {:.1}×",
        parallel_ms / n_lines as f64,
        serial_ms / parallel_ms
    );
}
