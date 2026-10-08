//! # IQ 云的统计：各态中心、弥散与可分性
//!
//! baseline `qana` 的 Rust 移植与推广：`exps/iq.py`（单团云的矩）与
//! `exps/iq_prob_analysis.py`（两态云的中心、展宽与信噪比）。
//!
//! 一次调用收进**各态的单发 IQ 云**（索引即态编号，0 → |0>、1 → |1>、…），产出三样东西：
//!
//! 1. **各态中心**——所有单发点的均值，可直接构造 [`StateCenters`]，此后 qspec / rabi 的
//!    P1 投影就不再依赖调用方手给的中心。
//! 2. **各态的弥散**——按"最深 68% / 95% / 99% 的点占多大地方"度量（[`DensityGrid`]）。
//!    baseline 用协方差椭圆，而椭圆的等值线就是高斯的等值线：香蕉、月牙、双峰这些形状会被
//!    压成一个椭圆，68% 的那条线也可能只圈住四成的点。这里换成密度网格——**不假设任何
//!    分布形状**，非凸、多峰都如实呈现，而且面积与画出来的轮廓出自同一个网格。
//! 3. **可分性**——两两之间沿中心连线投到一维，扫出最优判别阈值与错分率，另给一个无阈值
//!    的 AUC。云的全部用途就是判别它属于哪个态，所以"弥散有多大"最终要被读成"判别错多少"。
//!
//! **与 baseline 的口径差异**（有意为之，不是疏漏）：
//!
//! - 展宽：baseline 报 `√(uᵀCu)`（二阶矩），这里报 `(16–84 分位差)/2`（分位展宽）。二阶矩
//!   对逃逸点不稳健——单发 IQ 里一两个逃逸点就能把它抬上去，分位数不受影响。两者都留着：
//!   `spread` 是分位口径，`sigma` 是二阶矩口径，便于与 baseline 逐位对拍。
//! - 展宽只按**态对**给（[`PairStats::snr`]）：`separation / √(σp∥² + σq∥²)`，与 baseline
//!   同式。两个 σ 是**各团沿中心连线方向**的二阶矩展宽（∥ 即"沿连线"，不是各向同性的总宽度，
//!   也不是垂直于连线的方向）。单团自身的展宽不单独报——判别永远是两团之间的事，脱离了对手的
//!   "自身宽度"没有用处。

use crate::superconductor::{StateCenters, direction};
use crate::utils::density::{DensityGrid, Region};
use crate::utils::{auc, best_threshold, quantile};
// 与 s21 / qspec / rabi 一致：对外直接用 `iq::Complex64`
pub use lmfit::Complex64;

/// 每个态单发云所需的最少点数：少于这个数，分位展宽与密度网格都没有意义。
const MIN_SHOTS: usize = 8;

/// 密度网格每轴的格子数。
///
/// 定 64 是量与解析值比出来的：阈值二分的判据是"整格里的单发数"，而区域边界是**穿格**的，
/// 于是面积带一个随格数变化的偏差——实测高斯云上 68/95/99% 的面积与解析值之比，64 格是
/// 0.989 / 0.980 / 0.966，96 格反而变成 0.988 / 0.979 / 0.950（越外层越偏小）。64 格的一致性
/// 更好，薄环（环带约 3 格宽）与细长条在这个格数下也已够分辨。
pub(crate) const BINS: usize = 64;

/// 四条等密度线的质量比例：1σ / 2σ / 3σ 的样本版（高斯下正是 68.27 / 95.45 / 99.73%），
/// 再加一条 100%。
///
/// **100% 与前三条性质不同**：它要求把每一个单发都圈进去，阈值因此被压到最稀的那个**有点
/// 的格子**上——面积由尾巴最远的那几个点决定，样本越多它越大，不是稳定的形状参数。它的用处
/// 是跟 99% 比出尾巴有多重（`A100/A99`），而不是当作"云有多大"。另外，落在网格采样区间之外
/// 的逃逸点会被并进边界格，于是这类云的 100% 区域会一直贴到网格边缘。
pub const LEVELS: [f64; 4] = [0.68, 0.95, 0.99, 1.0];

/// IQ 统计的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum IqError {
    /// 一个态都没给。
    NoStates,
    /// 某个态的单发点数不足以估计展宽。
    LackShots { state: usize, shots: usize, min: usize },
    /// 某个态的有限点太少（其余全是 NaN/inf），连中心都定不下来。
    Degenerate { state: usize },
    /// 两态中心重合，判别轴无定义。
    CoincidentCenters { state_p: usize, state_q: usize },
}

impl std::fmt::Display for IqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStates => write!(f, "no state cloud was given"),
            Self::LackShots { state, shots, min } => write!(
                f,
                "state {state} has only {shots} shots; at least {min} are needed to estimate a spread"
            ),
            Self::Degenerate { state } => write!(
                f,
                "state {state} has too few finite shots; no center can be estimated"
            ),
            Self::CoincidentCenters { state_p, state_q } => write!(
                f,
                "centers of state {state_p} and {state_q} coincide; the discriminant axis is undefined"
            ),
        }
    }
}

impl std::error::Error for IqError {}

/// 一个态的统计：中心与弥散（三条最深密度区域的面积）。
#[derive(Debug, Clone)]
pub struct StateStats {
    /// 云中心（所有单发点的均值）
    pub center: Complex64,
    /// 最深 68 / 95 / 99 / 100% 区域的面积，数据单位²
    pub areas: [f64; 4],
    /// 各条等密度线的"等效半径" √(A/π)，数据单位
    pub radii: [f64; 4],
    /// 68 / 95 / 99% 的区域多边形——**与 `areas` 是同一个对象**：表里的面积就是对它积出来的，
    /// 图上的轮廓就是照它画的
    pub regions: [Region; 4],
}

/// 把测得的 IQ 旋转 + 平移到 P1 坐标系的相似变换：`iq ↦ iq × rotation_scale + translation`。
///
/// 由一对态的两个中心定出，性质是：**取实部即自动归一化到 0–1**——p 态中心落在 0、q 态中心
/// 落在 1（`Re(apply(c_p)) = 0`、`Re(apply(c_q)) = 1`），虚部则是垂直方向坐标，以两中心间距
/// 为单位。所以 `Re(apply(iq))` 与"投到两中心连线上再按 |c_q − c_p| 归一"是同一件事。
#[derive(Debug, Clone, Copy)]
pub struct IqTransform {
    /// 旋转 + 缩放：`1 / (c_q − c_p)`（模长是间距的倒数，幅角是连线的负角）
    pub rotation_scale: Complex64,
    /// 平移：`−rotation_scale × c_p`，把 p 态中心搬到原点
    pub translation: Complex64,
}

impl IqTransform {
    /// 由两个态的中心构造。两中心重合时 `rotation_scale` 会发散——调用方（[`pair_stats`]）
    /// 在此之前就用 [`IqError::CoincidentCenters`] 挡住了。
    ///
    /// 形参:
    ///     state_p: p 态的中心
    ///     state_q: q 态的中心
    ///
    /// 返回值:
    ///     把 p 态中心映到 0、q 态中心映到 1 的相似变换
    pub fn new(state_p: Complex64, state_q: Complex64) -> Self {
        let rotation_scale = Complex64::new(1.0, 0.0) / (state_q - state_p);
        Self {
            rotation_scale,
            translation: -rotation_scale * state_p,
        }
    }

    /// 直接应用这个变换：`iq × rotation_scale + translation`。
    ///
    /// 形参:
    ///     iq: 测得的 IQ 点
    ///
    /// 返回值:
    ///     P1 坐标系下的复数；**实部就是 P1（0 = p 态中心，1 = q 态中心）**
    pub fn apply(&self, iq: Complex64) -> Complex64 {
        iq * self.rotation_scale + self.translation
    }
}

/// 一对态之间的可分性。
#[derive(Debug, Clone, Copy)]
pub struct PairStats {
    /// 两个态的编号（`state_p < state_q`）
    pub state_p: usize,
    pub state_q: usize,
    /// 中心间距，数据单位
    pub separation: f64,
    /// 最优判别阈值：**判别边界与两中心连线的交点**（IQ 平面上的复数）
    pub threshold: Complex64,
    /// 该阈值处的错分率
    pub error_rate: f64,
    /// 无阈值的可分性度量（曼-惠特尼统计量 `P(q 态的 P1 > p 态的 P1)`），0.5 完全重叠、1 完全分开
    pub auc: f64,
    /// baseline 口径的信噪比 `separation / √(σp∥² + σq∥²)`：两个 σ 是各团**沿中心连线方向**的
    /// 二阶矩展宽（`∥` 即"沿连线"），所以这个数是"两团相距几个连线方向的 σ"，无量纲
    pub snr: f64,
    /// 把 IQ 归一化到 P1 的旋转 + 平移（取实部即得 0–1 的 P1）
    pub transform: IqTransform,
}

/// 一次 IQ 概率实验的全部统计结果。
#[derive(Debug, Clone)]
pub struct IqStats {
    /// 各态统计，索引即态编号
    states: Vec<StateStats>,
    /// 两两可分性，按 (0,1)、(0,2)、…、(n−2,n−1) 排列
    pairs: Vec<PairStats>,
}

impl IqStats {
    /// 各态统计。
    pub fn states(&self) -> &[StateStats] {
        &self.states
    }

    /// 两两可分性。
    pub fn pairs(&self) -> &[PairStats] {
        &self.pairs
    }

    /// 各态中心，可直接作为 [`StateCenters`] 喂给 qspec / rabi 的 P1 投影。
    ///
    /// 返回值:
    ///     索引即态编号的标定中心
    pub fn centers(&self) -> StateCenters {
        StateCenters::new(self.states.iter().map(|stats| stats.center).collect())
    }
}

/// 由各态的单发 IQ 云计算中心、弥散与可分性。
///
/// 形参:
///     iqs: 各态的单发 IQ 云；`iqs[i]` 是第 i 个态，索引即态编号（与 [`StateCenters`] 同一
///          约定），不要求恰好两个态
///
/// 返回值:
///     各态统计与两两可分性；有限点不足或两态中心重合时返回错误
pub fn iq_stats(iqs: &[Vec<Complex64>]) -> Result<IqStats, IqError> {
    if iqs.is_empty() {
        return Err(IqError::NoStates);
    }
    let mut states: Vec<StateStats> = Vec::with_capacity(iqs.len());
    for (index, cloud) in iqs.iter().enumerate() {
        states.push(single_state(index, cloud)?);
    }
    // 可分性是**态对**的事：两两沿中心连线投到一维，各自扫最优阈值
    let centers: Vec<Complex64> = states.iter().map(|stats| stats.center).collect();
    let mut pairs: Vec<PairStats> = Vec::new();
    for state_p in 0..iqs.len() {
        for state_q in (state_p + 1)..iqs.len() {
            pairs.push(pair_stats(
                state_p,
                state_q,
                &centers,
                &iqs[state_p],
                &iqs[state_q],
            )?);
        }
    }
    Ok(IqStats { states, pairs })
}

/// 单个态的统计：中心、密度网格与三条等密度线的面积。
///
/// 零展布的云（所有单发重合、整个云落在一条直线上）不是坏数据：没有密度网格可建，但中心
/// 明明白白，等密度区域退化成一个点（面积 0），图上于是只剩点云、没有轮廓。
fn single_state(index: usize, cloud: &[Complex64]) -> Result<StateStats, IqError> {
    if cloud.len() < MIN_SHOTS {
        return Err(IqError::LackShots {
            state: index,
            shots: cloud.len(),
            min: MIN_SHOTS,
        });
    }
    let points: Vec<(f64, f64)> = cloud
        .iter()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
        .map(|z| (z.re, z.im))
        .collect();
    if points.len() < MIN_SHOTS {
        return Err(IqError::Degenerate { state: index });
    }
    let density = DensityGrid::from_samples(&points, BINS);
    let center = Complex64::new(
        points.iter().map(|(x, _)| x).sum::<f64>() / points.len() as f64,
        points.iter().map(|(_, y)| y).sum::<f64>() / points.len() as f64,
    );
    // 各级等密度线的区域：有网格就照它取，没有（零展布）就是一个点。两条路给的是同一类对象，
    // 往下算面积、半径、画轮廓都不必再分叉
    let regions = LEVELS.map(|level| match &density {
        Some(grid) => grid.deepest(level).1,
        None => Region::point(),
    });
    let mut areas = [0.0; LEVELS.len()];
    let mut radii = [0.0; LEVELS.len()];
    for (slot, region) in regions.iter().enumerate() {
        areas[slot] = region.area;
        radii[slot] = (region.area / std::f64::consts::PI).sqrt();
    }
    Ok(StateStats {
        center,
        areas,
        radii,
        regions,
    })
}

/// 一团点沿"中心 → 目标点"方向的展宽，两个口径一起给。
///
/// 方向退化（两点重合）时改用 +I 方向：展宽仍然是该团自己的宽度，只是没有物理含义，仅供
/// 单态情形兜底。
///
/// 形参:
///     cloud: 该态的单发点
///     center: 该态中心
///     target: 目标点（另一个态的中心）
///
/// 返回值:
///     (分位展宽, 二阶矩展宽)。两者都是"沿该方向的一维标准差量纲"，高斯下分位展宽 = σ
fn along_axis(cloud: &[Complex64], center: Complex64, target: Complex64) -> (f64, f64) {
    let delta = target - center;
    let unit = match delta.norm() > 0.0 {
        true => delta / delta.norm(),
        false => Complex64::new(1.0, 0.0),
    };
    let mut projected: Vec<f64> = cloud
        .iter()
        .filter(|z| z.re.is_finite() && z.im.is_finite())
        .map(|z| (z - center).re * unit.re + (z - center).im * unit.im)
        .collect();
    if projected.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    projected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 16–84 分位差的一半：高斯下正是 1σ，对任意形状都有限
    let spread = 0.5 * (quantile(&projected, 0.84) - quantile(&projected, 0.16));
    let mean = projected.iter().sum::<f64>() / projected.len() as f64;
    let variance = projected
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (projected.len() - 1) as f64;
    (spread, variance.sqrt())
}

/// 一对态的可分性：沿中心连线投到一维，扫最优阈值。
fn pair_stats(
    state_p: usize,
    state_q: usize,
    centers: &[Complex64],
    cloud_p: &[Complex64],
    cloud_q: &[Complex64],
) -> Result<PairStats, IqError> {
    let delta = centers[state_q] - centers[state_p];
    let separation = delta.norm();
    if !(separation > 0.0) {
        return Err(IqError::CoincidentCenters { state_p, state_q });
    }
    // 投影只有一处实现：归一化变换取实部，就是"投到中心连线、再按 |c_q − c_p| 归一"
    // （两中心因此正好落在 0 与 1，阈值于是有直接的含义）
    let transform = IqTransform::new(centers[state_p], centers[state_q]);
    let project = |cloud: &[Complex64]| -> Vec<f64> {
        cloud
            .iter()
            .filter(|z| z.re.is_finite() && z.im.is_finite())
            .map(|z| transform.apply(*z).re)
            .collect()
    };
    let (p_scores, q_scores) = (project(cloud_p), project(cloud_q));
    let (cut, error_rate) = best_threshold(&p_scores, &q_scores);
    let auc = auc(&p_scores, &q_scores);
    // baseline 口径：两团各自沿连线展宽，按平方和合并
    let (_, sigma_p) = along_axis(cloud_p, centers[state_p], centers[state_q]);
    let (_, sigma_q) = along_axis(cloud_q, centers[state_q], centers[state_p]);
    let noise = (sigma_p * sigma_p + sigma_q * sigma_q).sqrt();
    let snr = match noise > 0.0 {
        true => separation / noise,
        false => f64::NAN,
    };
    // 阈值对外是 IQ 平面上的复数：判别边界与中心连线的交点（P1 = cut 处）。中心连线的方向是
    // `delta / separation`、判别边界垂直于它——两条线都不必存：它们由中心与间距完全决定
    let threshold = centers[state_p] + cut * delta;
    Ok(PairStats {
        state_p,
        state_q,
        separation,
        threshold,
        error_rate,
        auc,
        snr,
        transform,
    })
}

/// 供绘图使用：一对态沿其判别轴投影后的一维样本，以及该对的判别阈值。
///
/// 形参:
///     iqs: 与 [`iq_stats`] 同一批云
///     stats: 统计结果
///     state_p: 态编号（`state_p < state_q`）
///     state_q: 态编号
///
/// 返回值:
///     (p 团的投影, q 团的投影, 判别边界与中心连线的交点)；编号越界时投影为空、交点为 NaN
pub fn pair_projection(
    iqs: &[Vec<Complex64>],
    stats: &IqStats,
    state_p: usize,
    state_q: usize,
) -> (Vec<f64>, Vec<f64>, Complex64) {
    let pair = stats
        .pairs
        .iter()
        .find(|item| item.state_p == state_p && item.state_q == state_q);
    let missing = (
        Vec::new(),
        Vec::new(),
        Complex64::new(f64::NAN, f64::NAN),
    );
    match (pair, iqs.get(state_p), iqs.get(state_q)) {
        (Some(pair), Some(cloud_p), Some(cloud_q)) => {
            // 投影用统计层已经算好的那个变换，不再各写一遍
            let project = |cloud: &[Complex64]| -> Vec<f64> {
                cloud.iter().map(|z| pair.transform.apply(*z).re).collect()
            };
            (project(cloud_p), project(cloud_q), pair.threshold)
        }
        (Some(_), Some(_), None)
        | (Some(_), None, Some(_))
        | (Some(_), None, None)
        | (None, Some(_), Some(_))
        | (None, Some(_), None)
        | (None, None, Some(_))
        | (None, None, None) => missing,
    }
}

/// 供绘图使用：一团点的 (I, Q) 坐标。
pub fn scatter(iq: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    (
        iq.iter().map(|z| z.re).collect(),
        iq.iter().map(|z| z.im).collect(),
    )
}

/// 供绘图使用：由一团点定出的主轴（虚线画在云图上，标明 P1 定轴用的方向）。
pub fn cloud_axis(iq: &[Complex64]) -> (Complex64, Complex64) {
    direction(iq)
}
