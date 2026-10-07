//! 合成 IQ 概率实验的报告页 → plt/iq_demo.html。
//!
//! 真值已知，且**故意用非高斯形状**——香蕉（弯的）、双峰、月牙（空缺）、长条（强各向异性）、
//! 均匀盘（方块状），外加一团混了逃逸点的。目的是把"轮廓与面积算法"逼到形状假设失效的地方：
//! 协方差椭圆在这些云上会给出误导性的 68%/95%/99%，而密度网格不假设任何形状。
//!
//! 另有一组**高斯对照**：面积应当落在解析值 π·(−2 ln(1−p))·(sx·sy) 上，用来验证算法本身
//! 没有系统性偏差（打印实测/解析之比）。
//!
//! 运行: cargo run --release --example iq_demo

use qtool::superconductor::iq::iq_plot::iq_plot_div;
use qtool::superconductor::iq::{Complex64, IqStats, LEVELS, Region, iq_stats};
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use qtool::superconductor::{StateCenters, p1};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 每团的单发数。
const SHOTS: usize = 20_000;

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 高斯云：给定中心与两个轴的展宽。
fn gaussian_cloud(center: Complex64, sx: f64, sy: f64, shots: usize, rng: &mut StdRng) -> Vec<Complex64> {
    (0..shots)
        .map(|_| center + Complex64::new(gaussian(rng) * sx, gaussian(rng) * sy))
        .collect()
}

/// 跑一组云，打印关键数字并出报告 div。
fn report(
    title: &str,
    divs: &mut String,
    iqs: &[Vec<Complex64>],
    seed: u64,
) {
    match iq_stats(iqs) {
        Ok(stats) => {
            print_stats(title, &stats);
            for index in 0..iqs.len() {
                check_containment(title, &iqs[index], &stats, index);
            }
        }
        Err(err) => println!("{title:<22} 分析失败: {err}"),
    }
    let div_id = format!("iq-{seed}");
    divs.push_str(&iq_plot_div(iqs, &div_id, Some(title)));
    divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
}

/// 射线法：点是否在某个环内。
fn inside(ring: &[(f64, f64)], point: (f64, f64)) -> bool {
    let (px, py) = point;
    let mut hit = false;
    let n = ring.len();
    for index in 0..n {
        let (x0, y0) = ring[index];
        let (x1, y1) = ring[(index + 1) % n];
        if (y0 > py) != (y1 > py) {
            let x = x0 + (py - y0) / (y1 - y0) * (x1 - x0);
            if px < x {
                hit = !hit;
            }
        }
    }
    hit
}

/// 点是否落在区域内（外环内、且不在任何洞里）。
fn in_region(region: &Region, point: (f64, f64)) -> bool {
    region
        .polygons()
        .iter()
        .any(|(exterior, holes)| inside(exterior, point) && !holes.iter().any(|h| inside(h, point)))
}

/// 该态的最远单发到中心的距离：100% 区域的等效半径不应当比它小太多——小太多说明
/// 逃逸点被网格采样区间截掉了，那个面积是"网格里的足迹"，不是云的真实足迹。
fn far_shot(cloud: &[Complex64], center: Complex64) -> f64 {
    cloud
        .iter()
        .map(|z| (*z - center).norm())
        .fold(0.0_f64, f64::max)
}

/// 核验：落在每条区域内的单发比例，应当就是 68 / 95 / 99 / 100%。
fn check_containment(title: &str, cloud: &[Complex64], stats: &IqStats, index: usize) {
    let state = &stats.states()[index];
    let fractions: Vec<String> = (0..4)
        .map(|slot| {
            let inside = cloud
                .iter()
                .filter(|z| in_region(&state.regions[slot], (z.re, z.im)))
                .count();
            format!("{:.3}", inside as f64 / cloud.len() as f64)
        })
        .collect();
    let far = far_shot(cloud, state.center);
    println!(
        "{:<22}   区域内的单发占比（应 0.680/0.950/0.990/1.000）: {} | r100={:.4} 最远单发={:.4}",
        title,
        fractions.join(" / "),
        state.radii[3],
        far,
    );
}

/// 打印一组云的每态面积、等效半径与信噪比。
fn print_stats(title: &str, stats: &IqStats) {
    for (index, state) in stats.states().iter().enumerate() {
        println!(
            "{title:<22} |{index}> center={:.4} r68={:.4} r95={:.4} r99={:.4} r100={:.4} A100/A99={:.2}",
            state.center,
            state.radii[0],
            state.radii[1],
            state.radii[2],
            state.radii[3],
            state.areas[3] / state.areas[2],
        );
    }
    for pair in stats.pairs() {
        println!(
            "{:<22}   |{}>–|{}>: d={:.4} cut={:.4} err={:.4} auc={:.5} snr={:.3}",
            "",
            pair.state_p,
            pair.state_q,
            pair.separation,
            pair.threshold,
            pair.error_rate,
            pair.auc,
            pair.snr,
        );
    }
}

fn main() {
    let mut divs = String::new();

    // ---- 对照：两团标准高斯。面积应当落在解析值上 ----
    let mut rng = StdRng::seed_from_u64(7);
    let g0 = Complex64::new(-0.5, 0.0);
    let g1 = Complex64::new(0.5, 0.0);
    let gaussian_pair = vec![
        gaussian_cloud(g0, 0.05, 0.05, SHOTS, &mut rng),
        gaussian_cloud(g1, 0.05, 0.05, SHOTS, &mut rng),
    ];
    // 解析面积：2D 高斯最深 p 区域 = π·(−2 ln(1−p))·σx·σy
    println!("== 高斯对照（σx = σy = 0.05）==");
    for level in LEVELS {
        let analytic = std::f64::consts::PI * (-2.0 * (1.0 - level).ln()) * 0.05 * 0.05;
        println!("   p={level:.2} 解析面积 = {analytic:.4e}");
    }
    // 这一组只做数字自检（不上卡片）：出卡的只有下面那张三态图
    match iq_stats(&gaussian_pair) {
        Ok(stats) => {
            let analytic: Vec<f64> = LEVELS
                .iter()
                .map(|p| std::f64::consts::PI * (-2.0 * (1.0 - p).ln()) * 0.05 * 0.05)
                .collect();
            let state = &stats.states()[0];
            let ratios: Vec<String> = (0..3)
                .map(|slot| format!("{:.3}", state.areas[slot] / analytic[slot]))
                .collect();
            println!("   实测/解析: {}（1.00 附近说明无系统偏差）", ratios.join(" / "));
        }
        Err(err) => println!("   对照分析失败: {err}"),
    }

    // ---- 三态：|0>、|1>、|2> 各一团，两两可分性都要报 ----
    let mut rng = StdRng::seed_from_u64(23);
    let three = vec![
        gaussian_cloud(Complex64::new(-0.5, -0.2), 0.06, 0.05, 6000, &mut rng),
        gaussian_cloud(Complex64::new(0.0, 0.25), 0.07, 0.05, 6000, &mut rng),
        gaussian_cloud(Complex64::new(0.5, -0.15), 0.05, 0.06, 6000, &mut rng),
    ];
    report("three states", &mut divs, &three, 99);

    // ---- 标定中心能直接喂给 p1：拿三态中心投影一次，看看是否 0/1/- ----
    match iq_stats(&three) {
        Ok(stats) => {
            let centers: StateCenters = stats.centers();
            let probe = vec![
                centers.as_slice()[0],
                centers.as_slice()[1],
                centers.as_slice()[2],
            ];
            println!("三态中心喂给 p1（前两态应落在 0 与 1）：{:?}", p1(&probe, Some(&centers)));
        }
        Err(err) => println!("三态分析失败: {err}"),
    }

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool iq demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/iq_demo.html", html) {
        Ok(()) => println!("wrote plt/iq_demo.html"),
        Err(err) => println!("failed to write plt/iq_demo.html: {err}"),
    }
}
