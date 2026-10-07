//! 批量 S21 拟合基准：3000 条不同频率线的全流程拟合（初值估计 + 候选扫描），
//! 串行与 rayon 并行对照；另附**同任务对照**——同一起点、同一份数据、只拟合一次，
//! 与 Python 侧 scipy 的两行逐项可比（同一个 LCG 生成同样的 3000 个问题，
//! 见 qtool-py/bench/bench_s21_batch.py）。
//!
//! 运行: cargo run --release --example s21_batch_bench

use lmfit::ComplexCurve;
use qtool::superconductor::s21::{Complex64, S21Model, model_at, s21_fit, s21_fit_batch};
use rayon::prelude::*;
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

    // 每条线的真值参数（同时是"单次拟合"对照的公共起点）
    let truth: Vec<S21Model> = (0..n_lines).map(line_params).collect();

    // 逐线生成数据：不同参数 + 独立噪声
    let lines: Vec<Vec<Complex64>> = (0..n_lines)
        .map(|index| {
            let params = truth[index];
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
        match s21_fit(&freqs, line, None) {
            Ok(result) => {
                black_box(result.chisqr);
            }
            Err(_err) => serial_failures += 1,
        }
    }
    let serial_ms = start.elapsed().as_secs_f64() * 1e3;

    // rayon 并行
    let start = Instant::now();
    let results = s21_fit_batch(&freqs, &lines, None);
    let parallel_ms = start.elapsed().as_secs_f64() * 1e3;
    let parallel_failures = results
        .iter()
        .filter(|outcome| outcome.is_err())
        .count();

    println!(
        "全流程（初值估计 + 候选扫描，每条线最多 8 个候选）"
    );
    println!(
        "串行  : {serial_ms:9.1} ms（{:.3} ms/条，失败 {serial_failures}）",
        serial_ms / n_lines as f64
    );
    println!(
        "rayon : {parallel_ms:9.1} ms（{:.3} ms/条，失败 {parallel_failures}）→ 加速 {:.1}×",
        parallel_ms / n_lines as f64,
        serial_ms / parallel_ms
    );

    // =====================================================================
    // 同任务对照：同一起点、同一份数据、只拟合一次
    // 这两行与 Python 侧 scipy 的两行逐项对应（同样的 3000 个问题、同样的 p0）。
    // =====================================================================
    let mut eval_count = 0_usize;
    let eval_start = Instant::now();
    for params in truth.iter() {
        for f in freqs.iter() {
            black_box(model_at(*f, params));
            eval_count += 1;
        }
    }
    let eval_ns = eval_start.elapsed().as_secs_f64() * 1e9 / eval_count as f64;

    // 起点 = 真值参数
    let start = Instant::now();
    let mut nfev_sum = 0_usize;
    let mut single_failures = 0_usize;
    for (params, line) in truth.iter().zip(lines.iter()) {
        match params.fit(line, &freqs) {
            Ok(result) => {
                black_box(result.chisqr);
                nfev_sum += result.nfev;
            }
            Err(_err) => single_failures += 1,
        }
    }
    let single_ms = start.elapsed().as_secs_f64() * 1e3;

    let start = Instant::now();
    let single_results: Vec<_> = truth
        .par_iter()
        .zip(lines.par_iter())
        .map(|(params, line)| params.fit(line, &freqs))
        .collect();
    let single_parallel_ms = start.elapsed().as_secs_f64() * 1e3;
    let single_parallel_failures = single_results.iter().filter(|outcome| outcome.is_err()).count();

    // 起点 = 真值 + 估计器量级的抖动（与 Python 侧控制组同一组偏移）
    let control_n = 300.min(n_lines);
    let start = Instant::now();
    let mut control_nfev = 0_usize;
    let mut control_failures = 0_usize;
    for index in 0..control_n {
        let mut values = truth[index].to_array();
        values[0] += 1.0e6; // fr
        values[1] *= 0.75; // ql
        values[2] *= 1.3; // qc
        values[3] += 0.3; // theta
        match S21Model::from_fit_array(&values).fit(&lines[index], &freqs) {
            Ok(result) => {
                black_box(result.chisqr);
                control_nfev += result.nfev;
            }
            Err(_err) => control_failures += 1,
        }
    }
    let control_ms = start.elapsed().as_secs_f64() * 1e3;

    println!("\n模型求值: {eval_ns:.1} ns/点（{eval_count} 点，单点 S21 内核）");
    println!("同任务对照（同一起点、同一份数据、只拟合一次）");
    println!(
        "真值起点 串行 : {single_ms:9.1} ms（{:.3} ms/条，平均 nfev {}，失败 {single_failures}）",
        single_ms / n_lines as f64,
        nfev_sum / n_lines.max(1)
    );
    println!(
        "真值起点 rayon: {single_parallel_ms:9.1} ms（{:.3} ms/条，失败 {single_parallel_failures}）→ 加速 {:.1}×",
        single_parallel_ms / n_lines as f64,
        single_ms / single_parallel_ms
    );
    println!(
        "抖动起点 串行 : {control_ms:9.1} ms（{:.3} ms/条，平均 nfev {}，失败 {control_failures}，前 {control_n} 条）",
        control_ms / control_n.max(1) as f64,
        control_nfev / control_n.max(1)
    );
}
