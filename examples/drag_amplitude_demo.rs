//! 合成 DRAG 幅度扫描的报告页 → plt/drag_amplitude_demo.html。
//!
//! 一次画**整条升阶链**：最低阶（`pairs = 1`，两发同号脉冲）从零驱动铺开、扫**两个周期**、
//! 拟余弦问出周期，之后每一阶围着上一阶的谷位收窄到半个周期（`A_π/(2·pairs)`）、拟谷，谷心
//! 既是这一阶的读数、也是下一阶的窗心。数据是**已知真值**的合成扫描：`P1 = sin²(pairs·π·A/A_π)` 落到 |0>–|1>
//! 连线上、再叠复高斯噪声，于是既能看到多阶报告长什么样，也能核验每一阶的回收（真值一并打印）。
//!
//! 覆盖三档：标准三阶、低信噪比，以及**五阶**——颜色按阶序由浅到深、任多少阶都不绕回浅色，
//! 那一档就是给这条看的。
//!
//! 噪声的 σ 一并合成（每分量独立同分布），最后一节拿它做一次自洽检查：同一真值、同一窗口下
//! 12 条独立扫描的谷心**散布**，应当与拟合报出的**标准误**同量级；而 σ 的绝对尺度只在**加权
//! χ²/ndof** 上现身（标准误对它不敏感），所以两个检查各看一个。
//!
//! 谷这一档的 χ²/ndof 会明显大于 1，且是**预期的**：洛伦兹只是顶替 sin² 的形状，残差里有一份
//! 系统性的形状错配。偏的是宽度与深度，谷心仍然回得来——读数只看谷心。
//!
//! 运行: cargo run --release --example drag_amplitude_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::drag::amplitude::plot::{OrderFit, OrderScan, drag_amplitude_plot_div};
use qtool::superconductor::drag::amplitude::{
    Complex64, FactorNError, FactorNFit, FactorOneError, FactorOneFit, factor_n_fit,
    factor_n_fit_batch, factor_one_fit, window_half_width,
};
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 真值的 π 幅度：最低阶扫到它（首个极小压在右端，是真实数据里常见的取法）。
const A_PI: f64 = 0.05;
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

/// 一阶的真值 P1：`2·pairs` 发同号脉冲下 `sin²(pairs·π·A/A_π)`。
fn truth(pairs: usize, amps: &[f64]) -> Vec<f64> {
    amps.iter()
        .map(|a| (pairs as f64 * std::f64::consts::PI * a / A_PI).sin().powi(2))
        .collect()
}

/// 一阶的扫描轴：最低阶自零驱动铺开、**扫两个周期**（`0 → 2·A_π`，极小在 `A_π`、`2·A_π` 上
/// 深度相同——这正是"取首个极小"这条判据存在的理由，扫得宽一点看得见），升阶围着 `centre`
/// 取半个周期。
fn axis(pairs: usize, centre: f64) -> Vec<f64> {
    let (low, high) = match pairs {
        1 => (0.0, 2.0 * A_PI),
        _ => {
            let half = window_half_width(A_PI, pairs);
            (centre - half, centre + half)
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

/// 一阶的拟合结果：两种线型分开装。
enum ChainFit {
    Cosine(Result<FactorOneFit, FactorOneError>),
    Valley(Result<FactorNFit, FactorNError>),
}

/// 一条升阶链：各阶的轴、IQ 与拟合结果都自己持有，画图时借出去。
struct Chain {
    pairs: Vec<usize>,
    axes: Vec<Vec<f64>>,
    lines: Vec<Vec<Complex64>>,
    fits: Vec<ChainFit>,
}

impl Chain {
    /// 跑一条升阶链：逐阶造数据、逐阶拟合（最低阶余弦、其余谷），窗心取上一阶的读数。
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
            let amps = axis(*pairs, centre);
            let prob = truth(*pairs, &amps);
            let iq = synthetic(&prob, sigma, rng);
            let weights = vec![sigma; amps.len()];
            let fit = match *pairs {
                1 => {
                    let fit = factor_one_fit(&amps, &iq, &states, Some(&weights));
                    match &fit {
                        Ok(result) => {
                            println!(
                                "  N={pairs}：a_pi={:.5} ± {:.1e} (Δ {:+.3}%) freq={:.3} redchi={:.3}",
                                result.result.model.a_pi,
                                result
                                    .result
                                    .params
                                    .get("a_pi")
                                    .and_then(|p| p.stderr)
                                    .unwrap_or(f64::NAN),
                                (result.result.model.a_pi - A_PI) / A_PI * 100.0,
                                result.result.model.freq,
                                result.result.redchi,
                            );
                            centre = result.result.model.a_pi;
                        }
                        Err(err) => println!("  N={pairs}：拟合失败 {err}"),
                    }
                    ChainFit::Cosine(fit)
                }
                _ => {
                    let fit = factor_n_fit(&amps, &iq, &states, Some(&weights));
                    match &fit {
                        Ok(result) => {
                            println!(
                                "  N={pairs}：centre={:.5} ± {:.1e} (Δ {:+.3}%) fwhm={:.5} redchi={:.3}",
                                result.result.model.centre,
                                result
                                    .result
                                    .params
                                    .get("centre")
                                    .and_then(|p| p.stderr)
                                    .unwrap_or(f64::NAN),
                                (result.result.model.centre - A_PI) / A_PI * 100.0,
                                result.result.model.fwhm,
                                result.result.redchi,
                            );
                            centre = result.result.model.centre;
                        }
                        Err(err) => println!("  N={pairs}：拟合失败 {err}"),
                    }
                    ChainFit::Valley(fit)
                }
            };
            chain.pairs.push(*pairs);
            chain.axes.push(amps);
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
                amps: self.axes[index].as_slice(),
                iq: self.lines[index].as_slice(),
                fit: match &self.fits[index] {
                    ChainFit::Cosine(result) => OrderFit::Cosine(result.as_ref()),
                    ChainFit::Valley(result) => OrderFit::Valley(result.as_ref()),
                },
            })
            .collect()
    }

    /// 标定值：最高阶取到的那个谷位（这一档就是 π 幅度）。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     最高阶的读数；那一阶没拟合出来时是 NaN
    fn chosen(&self) -> f64 {
        match self.fits.last() {
            Some(ChainFit::Cosine(result)) => match result {
                Ok(fit) => fit.result.model.a_pi,
                Err(_) => f64::NAN,
            },
            Some(ChainFit::Valley(result)) => match result {
                Ok(fit) => fit.result.model.centre,
                Err(_) => f64::NAN,
            },
            None => f64::NAN,
        }
    }
}

fn main() {
    let states = StateCenters::new(vec![G0, G1]);
    // 只出标准三阶这一张卡：demo 是给人看"这一档长什么样"的，噪声档与更高阶留给需要的人自己加
    let cases: [(&str, &[usize], f64, u64); 1] = [("标准三阶", &[1, 2, 4], 0.02, 3)];

    let mut divs = String::new();
    for (title, pairs_list, noise, seed) in cases {
        println!("{title}（噪声 {noise}，各阶 {pairs_list:?}）：");
        let chain = Chain::run(pairs_list, noise, seed);
        println!("  标定值（最高阶的谷位）{:.5}", chain.chosen());
        let div_id = format!("drag-{seed}");
        divs.push_str(&drag_amplitude_plot_div(
            &chain.scans(),
            chain.chosen(),
            &states,
            &div_id,
            Some(title),
        ));
        divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
    }

    // 批量入口：同一批比特共用一条轴（这里用不同种子模拟）
    let amps = axis(2, A_PI);
    let sigma = 0.10 * (G1 - G0).norm();
    let mut lines: Vec<Vec<Complex64>> = Vec::new();
    let mut sigmas: Vec<Vec<f64>> = Vec::new();
    for seed in 0..12 {
        let rng = &mut StdRng::seed_from_u64(seed);
        lines.push(synthetic(&truth(2, &amps), sigma, rng));
        sigmas.push(vec![sigma; amps.len()]);
    }
    let batch = factor_n_fit_batch(&amps, &lines, &vec![&states; lines.len()], Some(&sigmas));
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
    println!("factor_n_fit_batch: {ok}/{} 条拟合成功", batch.len());
    println!(
        "        centre 真值 {A_PI:.5} | 12 条线均值 {mean:.5} | 散布 {std:.2e} vs 平均标准误 {stderr_mean:.2e}（比值 {:.2}）| χ²/ndof {redchi:.3}",
        std / stderr_mean,
    );

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool drag amp demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/drag_amplitude_demo.html", html) {
        Ok(()) => println!("wrote plt/drag_amplitude_demo.html"),
        Err(err) => println!("failed to write plt/drag_amplitude_demo.html: {err}"),
    }
}
