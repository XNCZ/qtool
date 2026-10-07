//! 合成 DRAG 载波失谐扫描的报告页 → plt/drag_detuning_demo.html。
//!
//! 一次画**整条升阶链**：最低阶绕零失谐铺满 ±`drag_detuning_span_hz`（驱动那一层的常数，这里
//! 是 1.5 MHz），之后每一阶围着上一阶的谷位收窄；**每一阶都拟谷**（失谐这一档的曲线不是 Rabi
//! 余弦，没有周期可给窗口定尺度），升阶半宽按经验指数 `span = 1/pairs^0.8` 缩——与 notebook 里
//! 那一档同一套政策。数据是**已知真值**的合成扫描，落到 |0>–|1> 连线上、再叠复高斯噪声。
//!
//! 真值取一个**非零的载波残差**（+0.25 MHz）：真实情形就是 `f01_working` 还差一点，谷不压在零
//! 失谐线上——页面上那条零失谐点线正是拿来一眼看出差多少的。形状直接用洛伦兹：这一档的真实形状
//! 由脉冲细节定，不是能从第一性原理写出来的式子，所以这里要验的是**机器**（单组解析初值、加权、
//! 出窗判废、逐阶收窗），不是形状——谷宽按 `1/pairs` 缩，与幅度那一档同一个道理（误差随脉冲数
//! 线性累积，对参数的分辨率就线性变尖）。
//!
//! 噪声的 σ 一并合成，最后一节拿它做一次自洽检查：同一真值、同一窗口下 12 条独立扫描的谷心
//! **散布**，应当与拟合报出的**标准误**同量级；而 σ 的绝对尺度只在**加权 χ²/ndof** 上现身。
//!
//! 运行: cargo run --release --example drag_detuning_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::drag::detuning::plot::{OrderScan, drag_detuning_plot_div};
use qtool::superconductor::drag::detuning::{
    Complex64, Valley, ValleyError, ValleyFit, valley_fit, valley_fit_batch,
};
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 真值的谷心：这一轮要修掉的载波残差 (Hz)，取个非零值。
const CENTRE_TRUE: f64 = 0.25e6;
/// 真值的谷宽：取该阶窗口半宽的这个比例。
///
/// 窗口政策的用意就是"逐阶收窄、把谷框住"——那前提是谷宽随窗口一起缩（否则收窄没有意义）。
/// 取一个定比例，各阶在各自窗口里的形状就一致：页面上是一条套一条的圆润谷，而不是越到高阶
/// 越像一根针。
const FWHM_FRACTION: f64 = 0.75;
/// 最低阶的失谐半宽 (Hz)（notebook 里 `drag_detuning_span_hz`）。
const DETUNING_SPAN_HZ: f64 = 1.5e6;
/// 升阶窗口的相对半宽：`span = DRAG_SPAN / pairs^DRAG_SPAN_DECAY`（notebook 那两个常数）。
const DRAG_SPAN: f64 = 1.0;
const DRAG_SPAN_DECAY: f64 = 0.8;
/// 每阶的点数。
const POINTS: usize = 31;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 一阶扫描轴的半宽。
fn half_span(axis: &[f64]) -> f64 {
    match (axis.first(), axis.last()) {
        (Some(first), Some(last)) => (last - first).abs() / 2.0,
        _ => 0.0,
    }
}

/// 一阶的真值 P1：以 `CENTRE_TRUE` 为谷心的倒置洛伦兹，谷宽取该阶窗口半宽的
/// [`FWHM_FRACTION`]。
fn truth(detunings_hz: &[f64]) -> Vec<f64> {
    let model = Valley {
        centre: CENTRE_TRUE,
        fwhm: FWHM_FRACTION * half_span(detunings_hz),
        amp: -1.0,
        offset: 1.0,
    };
    model.at(detunings_hz)
}

/// 一阶的扫描轴：最低阶绕 0 铺 ±`DETUNING_SPAN_HZ`，升阶围着 `centre` 取相对半宽。
fn axis(pairs: usize, centre: f64) -> Vec<f64> {
    let (low, high) = match pairs {
        1 => (-DETUNING_SPAN_HZ, DETUNING_SPAN_HZ),
        _ => {
            let span = DRAG_SPAN / (pairs as f64).powf(DRAG_SPAN_DECAY);
            (
                centre - DETUNING_SPAN_HZ * span,
                centre + DETUNING_SPAN_HZ * span,
            )
        }
    };
    (0..POINTS)
        .map(|k| low + (high - low) * k as f64 / (POINTS - 1) as f64)
        .collect()
}

/// 把真值 P1 落到 |0>–|1> 连线上，叠复高斯噪声。
fn synthetic(prob: &[f64], sigma: f64, rng: &mut StdRng) -> Vec<Complex64> {
    prob.iter()
        .map(|p| G0 + (G1 - G0) * p + Complex64::new(gaussian(rng) * sigma, gaussian(rng) * sigma))
        .collect()
}

/// 一条升阶链：各阶的轴、IQ 与拟合结果都自己持有，画图时借出去。
struct Chain {
    pairs: Vec<usize>,
    axes: Vec<Vec<f64>>,
    lines: Vec<Vec<Complex64>>,
    fits: Vec<Result<ValleyFit, ValleyError>>,
}

impl Chain {
    /// 跑一条升阶链：逐阶造数据、逐阶拟谷，窗心取上一阶的读数。
    ///
    /// 形参:
    ///     pairs_list: 各阶的脉冲对数，从低到高
    ///     noise: 每分量噪声 σ 相对对比度 |g1 − g0| 的比例
    ///     seed: 随机种子
    ///
    /// 返回值:
    ///     自持有的整条链；逐阶的回收情况同时打印出来
    fn run(pairs_list: &[usize], noise: f64, seed: u64) -> Self {
        let rng = &mut StdRng::seed_from_u64(seed);
        let sigma = noise * (G1 - G0).norm();
        let states = StateCenters::new(vec![G0, G1]);
        let mut chain = Self {
            pairs: Vec::new(),
            axes: Vec::new(),
            lines: Vec::new(),
            fits: Vec::new(),
        };
        let mut centre = 0.0;

        for pairs in pairs_list {
            let detunings_hz = axis(*pairs, centre);
            let iq = synthetic(&truth(&detunings_hz), sigma, rng);
            let weights = vec![sigma; detunings_hz.len()];
            let fit = valley_fit(&detunings_hz, &iq, &states, Some(&weights));
            match &fit {
                Ok(result) => {
                    println!(
                        "  N={pairs}：centre={:+.4e} ± {:.1e} (Δ {:+.3}%) fwhm={:.4e} redchi={:.3}",
                        result.result.model.centre,
                        result
                            .result
                            .params
                            .get("centre")
                            .and_then(|p| p.stderr)
                            .unwrap_or(f64::NAN),
                        (result.result.model.centre - CENTRE_TRUE) / CENTRE_TRUE * 100.0,
                        result.result.model.fwhm,
                        result.result.redchi,
                    );
                    centre = result.result.model.centre;
                }
                Err(err) => println!("  N={pairs}：拟合失败 {err}"),
            }
            chain.pairs.push(*pairs);
            chain.axes.push(detunings_hz);
            chain.lines.push(iq);
            chain.fits.push(fit);
        }
        chain
    }

    /// 交给作图的各阶扫描（借用本结构里的轴、IQ 与拟合）。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     各阶的扫描与拟合，次序与建链时一致
    fn scans(&self) -> Vec<OrderScan<'_>> {
        (0..self.pairs.len())
            .map(|index| OrderScan {
                pairs: self.pairs[index],
                detunings_hz: self.axes[index].as_slice(),
                iq: self.lines[index].as_slice(),
                fit: self.fits[index].as_ref(),
            })
            .collect()
    }

    /// 标定值：最高阶取到的载波残差 (Hz)。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     最高阶的读数；那一阶没拟合出来时是 NaN
    fn chosen(&self) -> f64 {
        match self.fits.last() {
            Some(Ok(fit)) => fit.result.model.centre,
            Some(Err(_)) => f64::NAN,
            None => f64::NAN,
        }
    }
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);
    // 只出标准三阶这一张卡（理由同 drag_amplitude_demo）
    let cases: [(&str, &[usize], f64, u64); 1] = [("标准三阶", &[1, 2, 4], 0.02, 3)];

    let mut divs = String::new();
    for (title, pairs_list, noise, seed) in cases {
        println!("{title}（噪声 {noise}，各阶 {pairs_list:?}）：");
        let chain = Chain::run(pairs_list, noise, seed);
        println!(
            "  载波残差 {:.1} kHz（真值 {:.1} kHz）",
            chain.chosen() / 1e3,
            CENTRE_TRUE / 1e3
        );
        let div_id = format!("drag-detuning-{seed}");
        divs.push_str(&drag_detuning_plot_div(
            &chain.scans(),
            chain.chosen(),
            &states,
            &div_id,
            Some(title),
        ));
        divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
    }

    // 批量入口：同一批比特共用一条轴（这里用不同种子模拟）
    let detunings_hz = axis(2, CENTRE_TRUE);
    let sigma = 0.10 * (G1 - G0).norm();
    let mut lines: Vec<Vec<Complex64>> = Vec::new();
    let mut sigmas: Vec<Vec<f64>> = Vec::new();
    for seed in 0..12 {
        let rng = &mut StdRng::seed_from_u64(seed);
        lines.push(synthetic(&truth(&detunings_hz), sigma, rng));
        sigmas.push(vec![sigma; detunings_hz.len()]);
    }
    let batch = valley_fit_batch(&detunings_hz, &lines, &states, Some(&sigmas));
    let ok = batch.iter().filter(|item| item.is_ok()).count();
    let centres: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.model.centre)
        .collect();
    let stderrs: Vec<f64> = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .filter_map(|fit| fit.result.params.get("centre"))
        .filter_map(|parameter| parameter.stderr)
        .collect();
    let mean = centres.iter().sum::<f64>() / centres.len() as f64;
    let std = (centres.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (centres.len() - 1) as f64)
        .sqrt();
    let stderr_mean = stderrs.iter().sum::<f64>() / stderrs.len() as f64;
    let redchi = batch
        .iter()
        .filter_map(|item| item.as_ref().ok())
        .map(|fit| fit.result.redchi)
        .sum::<f64>()
        / centres.len() as f64;
    println!("valley_fit_batch: {ok}/{} 条拟合成功", batch.len());
    println!(
        "        centre 真值 {:.4e} | 12 条线均值 {mean:.4e} | 散布 {std:.2e} vs 平均标准误 {stderr_mean:.2e}（比值 {:.2}）| χ²/ndof {redchi:.3}",
        CENTRE_TRUE,
        std / stderr_mean,
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool drag detuning demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/drag_detuning_demo.html", html) {
        Ok(()) => println!("wrote plt/drag_detuning_demo.html"),
        Err(err) => println!("failed to write plt/drag_detuning_demo.html: {err}"),
    }
}
