//! 合成 Rabi 层析轨迹的报告页 → plt/bloch_demo.html。
//!
//! 扫 π 脉冲幅度 A，每个 A 上做三基层析（X 基前置 H、Y 基前置 Rx(−π/2)、Z 基直测），得到一条
//! 从 |0> 出发、绕着驱动轴的布洛赫轨迹；再叠一层退相干，让轨迹是一条往球心收的螺线——纯态在
//! 球面上，退相干把它拉向圆心，`R = √(X²+Y²+Z²)` 就是这么读的。
//!
//! 真值的转轴取在 XY 平面里、方位角 0.5 rad（既不是 0 也不是 π/2），三个投影面因此都有东西
//! 可看：扫过 1.5 圈的同时半径一路收，三条投影各是一条往原点收的螺线。
//!
//! 数据是**已知真值**的合成扫描：三基各自的 P1 = (1 − ⟨P⟩)/2 落到 |0>–|1> 连线上，再叠复高斯
//! 噪声。同一份数据跑两种投影——给出两态中心（标定路径，与真值逐个分量对照）和不给（自定轴
//! 路径，只用来看那一支能不能画：它定得了直线、定不了哪一端是 |1>，整体朝向是约定）。
//!
//! 运行: cargo run --release --example bloch_demo

use qtool::superconductor::StateCenters;
use qtool::superconductor::bloch::bloch_plot::bloch_plot_div;
use qtool::superconductor::bloch::{BlochVector, Complex64, Trajectory, bloch_vector};
use qtool::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// 扫的幅度点数与上限：A = 1 是一个 π 脉冲，扫到 3 正好一圈半。
const POINTS: usize = 61;
const AMP_MAX: f64 = 3.0;
/// 转轴在 XY 平面里的方位角 (rad)。
const AXIS_AZIMUTH: f64 = 0.5;
/// 退相干尺度：径向长度 `R = exp(−A/DECAY)`。
const DECAY: f64 = 1.6;
/// 每分量的 IQ 噪声 σ，相对对比度 |g1 − g0| 的比例。
const NOISE: f64 = 0.02;
/// 随机种子。
const SEED: u64 = 3;
/// 两态标定中心（任意夹角）。
const G0: Complex64 = Complex64::new(0.30, 0.10);
const G1: Complex64 = Complex64::new(0.42, 0.30);

/// 标准正态样本（Box–Muller）。
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 真值的布洛赫向量：绕 `n̂ = (cos φ, sin φ, 0)` 转 θ = π·A，再乘退相干 `exp(−A/DECAY)`。
///
/// 形参:
///     amps: 幅度轴
///
/// 返回值:
///     `[⟨X⟩, ⟨Y⟩, ⟨Z⟩]`，各与 `amps` 等长
fn truth(amps: &[f64]) -> [Vec<f64>; 3] {
    let (sin_phi, cos_phi) = AXIS_AZIMUTH.sin_cos();
    let mut parts = [Vec::new(), Vec::new(), Vec::new()];
    for amp in amps {
        let (sin_theta, cos_theta) = (std::f64::consts::PI * amp).sin_cos();
        let radius = (-amp / DECAY).exp();
        parts[0].push(radius * sin_phi * sin_theta);
        parts[1].push(radius * -cos_phi * sin_theta);
        parts[2].push(radius * cos_theta);
    }
    parts
}

/// 某个基的读出 IQ：把该基的泡利期望值搬成 P1 = (1 − ⟨P⟩)/2，落到两态连线上，再叠复高斯噪声。
///
/// 形参:
///     component: 该基的泡利期望值
///     sigma: 每分量的噪声 σ（IQ 域）
///     rng: 随机源
///
/// 返回值:
///     该基各点的复数 IQ，与 `component` 等长
fn synthetic(component: &[f64], sigma: f64, rng: &mut StdRng) -> Vec<Complex64> {
    component
        .iter()
        .map(|value| {
            let prob = (1.0 - value) / 2.0;
            G0 + (G1 - G0) * prob + Complex64::new(gaussian(rng) * sigma, gaussian(rng) * sigma)
        })
        .collect()
}

/// 合成一次三基扫描。
///
/// 形参:
///     seed: 随机种子
///
/// 返回值:
///     (幅度轴, `[X 基 IQ, Y 基 IQ, Z 基 IQ]`)
fn scan(seed: u64) -> (Vec<f64>, [Vec<Complex64>; 3]) {
    let rng = &mut StdRng::seed_from_u64(seed);
    let sigma = NOISE * (G1 - G0).norm();
    let amps: Vec<f64> = (0..POINTS)
        .map(|index| AMP_MAX * index as f64 / (POINTS - 1) as f64)
        .collect();
    let parts = truth(&amps);
    let iq = [
        synthetic(&parts[0], sigma, rng),
        synthetic(&parts[1], sigma, rng),
        synthetic(&parts[2], sigma, rng),
    ];
    (amps, iq)
}

/// 反解，失败即报错退出（长度是自己给的，真出问题就是 demo 本身写错了）。
///
/// 形参:
///     trajectory: 三基扫描
///     states: 各态标定中心；None 走自定轴
///
/// 返回值:
///     布洛赫向量
fn solve(trajectory: &Trajectory<'_>, states: Option<&StateCenters>) -> BlochVector {
    match bloch_vector(trajectory, states) {
        Ok(vector) => vector,
        Err(error) => panic!("bloch_vector failed: {error}"),
    }
}

/// 三个分量与真值的最大偏差。
///
/// 形参:
///     vector: 反解出来的布洛赫向量
///     expected: 真值的三分量
///
/// 返回值:
///     `[⟨X⟩, ⟨Y⟩, ⟨Z⟩]` 各自的最大偏差
fn deviation(vector: &BlochVector, expected: &[Vec<f64>; 3]) -> [f64; 3] {
    let found = [&vector.x, &vector.y, &vector.z];
    let mut worst = [0.0; 3];
    for (index, (component, truth)) in found.iter().zip(expected).enumerate() {
        worst[index] = component
            .iter()
            .zip(truth)
            .map(|(value, truth)| (value - truth).abs())
            .fold(0.0, f64::max);
    }
    worst
}

fn main() {
    let (amps, iq) = scan(SEED);
    let trajectory = Trajectory {
        axis: &amps,
        x: &iq[0],
        y: &iq[1],
        z: &iq[2],
    };
    let states = StateCenters::new(vec![G0, G1]);
    let calibrated = solve(&trajectory, Some(&states));

    let worst = deviation(&calibrated, &truth(&amps));
    println!("{} 点，噪声 {NOISE}，三基各自成扫描：", amps.len());
    println!(
        "  标定投影与真值的最大偏差：⟨X⟩ {:.4}、⟨Y⟩ {:.4}、⟨Z⟩ {:.4}（每分量 σ_P1 = {:.4}）",
        worst[0],
        worst[1],
        worst[2],
        NOISE,
    );
    let radius = calibrated.radius();
    match (radius.first(), radius.last()) {
        (Some(first), Some(last)) => println!("  径向长度 R：起点 {first:.3}（|0>，纯态）→ 终点 {last:.3}（已被退相干拉进球里）"),
        _ => println!("  径向长度 R：无从取值"),
    }

    // 只出标定那一条：自定轴投影（`solve(&trajectory, None)`）画出来的形状一样，只是整体朝向
    // 由约定定，想看得自己把 centers 换成 None
    let divs = bloch_plot_div(&calibrated, "π amplitude", "bloch", Some("标定三基"));

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool bloch demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/bloch_demo.html", html) {
        Ok(()) => println!("wrote plt/bloch_demo.html"),
        Err(err) => println!("failed to write plt/bloch_demo.html: {err}"),
    }
}
