//! # qspec 通量调谐：f01 vs Z 的 SQUID 模型与拟合
//!
//! baseline `qana` 的 Rust 移植：`exps/qspec_analysis.py` 的 `flux_tunable` / `flux_of_f` /
//! `_flux_p0_candidates` / `fit_flux_tunable`。输入是逐偏置拟出来的峰位——谱线本身的拟合在
//! [`qspec`](crate::superconductor::qspec::qspec) 那边做过，这里只把这一串 `(z, f01)` 拟成
//! 通量调谐线型，并给出反解 [`FluxFit::tune_to`]（"把频率推到某个值，需要多大的 Z 偏置"）。
//!
//! 线型（`θ` 是归一化磁通 `π·Φ`）：
//!
//! ```text
//! f01(z) = (f_max + η)·[cos²θ + d²·sin²θ]^{1/4} − η,   θ = π(z − z_offset) / z_period
//! ```
//!
//! 括号里两项都非负，四次方根恒有定义；甜点处 `θ = 0`、频率回到 `f_max`，越过甜点越远压得
//! 越低，最低到 `(f_max + η)·√d − η`。写成对 z 而不是对磁通，是为了让拟合直接吐出**通量
//! 响应**：`z_period` 就是这台机器上"一个磁通量子对应多少伏"。
//!
//! `η`（充电能 `Ec/h`）与 `d`（结不对称度）都是**结构常数**：版图设计时定死，用设计值固定住
//! 即可（`#[param(vary = false)]`，随结果原样带回）。放开它们只会让一条小半周期的曲线去猜
//! 两个它压根约束不住的量——白多两份标准误，还会把 `z_period` 的标准误一起撑大。
//!
//! 全部物理量使用 SI 单位：频率 Hz、偏置 V。

use crate::utils::{argmax, max_value, min_value};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;
use std::f64::consts::PI;

/// 三个拟合参数的名字，顺序与 [`Flux`] 的字段声明序一致。
pub const FLUX_NAMES: [&str; 3] = ["f_max", "z_offset", "z_period"];

/// 通量调谐线型所需的最少偏置点数：2 × 参数数 = 6——曲线是周期的，点数再少约束不住三个参数。
const MIN_POINTS: usize = 6;

/// 一个磁通量子对应偏置跨度的初值候选，按扫描窗宽的倍数铺开（无量纲）。
///
/// 跨度本身读不出来，但量级有线索：窗内若只扫过小半个周期，曲线就近乎单调，跨度必然比窗宽
/// 大得多。与 baseline 的 `PERIOD_SPAN_RATIOS` 同表。
const PERIOD_SPAN_RATIOS: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];

/// 反解里判定"括号恰为 1"的容差（`np.isclose` 的默认口径：`atol = 1e-8`、`rtol = 1e-5`）。
const BRACKET_UNITY_TOLERANCE: f64 = 1e-5;

// =========================================================================
// 线型
// =========================================================================

/// 通量调谐线型的参数（SI 单位）。
///
/// 前三个字段参与拟合；`eta` 与 `asymmetry` 是版图给定的结构常数，`vary = false` 钉住不动，
/// 随结果原样带回（好让结果自成一份可复算的记录）。`#[param(value = 0.0)]` 中的 0.0 只是
/// lmfit 宏要求的占位起始值，真正的初值由调用方构造时给出。
#[derive(Model, Clone, Copy, Debug)]
pub struct Flux {
    /// 甜点处的频率，Hz
    #[param(value = 0.0)]
    pub f_max: f64,
    /// 甜点对应的偏置，V
    #[param(value = 0.0)]
    pub z_offset: f64,
    /// 一个磁通量子对应的偏置跨度，V（可正可负）
    #[param(value = 0.0)]
    pub z_period: f64,
    /// 充电能 `Ec/h`，Hz（结构常数，不参与拟合）
    #[param(value = 0.0, vary = false)]
    pub eta: f64,
    /// 两结不对称度 `d`，无量纲（结构常数，不参与拟合）
    #[param(value = 0.0, vary = false)]
    pub asymmetry: f64,
}

impl Flux {
    /// 在偏置数组上求线型值（画拟合曲线用的密集网格走这里）。
    ///
    /// 形参:
    ///     zs: 偏置数组 (n,)，V
    ///
    /// 返回值:
    ///     线型在 `zs` 上的取值 (n,)，Hz
    pub fn at(&self, zs: &[f64]) -> Vec<f64> {
        zs.iter().map(|z| self.eval(*z)).collect()
    }
}

impl Curve for Flux {
    /// 在单点 `z` 处求线型值。
    ///
    /// 形参:
    ///     z: Z 偏置，V
    ///
    /// 返回值:
    ///     该偏置处的 f01，Hz
    fn eval(&self, z: f64) -> f64 {
        let theta = PI * (z - self.z_offset) / self.z_period;
        let bracket = theta.cos().powi(2) + self.asymmetry.powi(2) * theta.sin().powi(2);
        (self.f_max + self.eta) * bracket.powf(0.25) - self.eta
    }
}

// =========================================================================
// 初值候选
// =========================================================================

/// 通量调谐线型的初值候选（baseline `_flux_p0_candidates`）。
///
/// `f_max` 取观测到的最高峰位：甜点落在扫描窗内时它就是 `f_max`，没扫到时只是个下界，交给
/// 拟合往上抬；`z_offset` 取最高峰位所在的那个偏置，甜点就在它附近；`z_period` 读不出来，
/// 按窗宽的倍数铺一列、正负各铺一遍（扫过去时 `θ` 往哪边走也是未知的）。
///
/// 形参:
///     zs: 偏置轴 (n,)，V，须非空（调用方先校验过点数）
///     peaks: 峰位轴 (n,)，Hz
///     eta: 充电能 `Ec/h`，Hz
///     asymmetry: 两结不对称度
///
/// 返回值:
///     初值列表，共 `PERIOD_SPAN_RATIOS.len() × 2` 组
fn p0_candidates(zs: &[f64], peaks: &[f64], eta: f64, asymmetry: f64) -> Vec<Flux> {
    let best = argmax(peaks);
    let f_max = peaks[best];
    let z_offset = zs[best];
    let span = max_value(zs) - min_value(zs);
    let mut out = Vec::with_capacity(PERIOD_SPAN_RATIOS.len() * 2);
    for ratio in PERIOD_SPAN_RATIOS {
        for sign in [1.0, -1.0] {
            out.push(Flux {
                f_max,
                z_offset,
                z_period: sign * ratio * span,
                eta,
                asymmetry,
            });
        }
    }
    out
}

// =========================================================================
// 拟合入口
// =========================================================================

/// 通量调谐拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum FluxError {
    /// 偏置或峰位数据为空。
    EmptyData,
    /// 偏置点数与峰位点数不一致。
    LengthMismatch { zs: usize, peaks: usize },
    /// 偏置点数不足以约束三参数线型。
    LackPoints { points: usize, min: usize },
    /// 无初值候选，或全部候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for FluxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the bias or peak data is empty; cannot run the fit"),
            Self::LengthMismatch { zs, peaks } => {
                write!(f, "bias points ({zs}) and peak points ({peaks}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} bias points cannot constrain a periodic 3-parameter model; at least {min} are needed"
            ),
            Self::AllFitsUnsuccess => write!(f, "no initial-value candidate produced a fit"),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for FluxError {}

/// 通量调谐的拟合结果。
#[derive(Debug, Clone)]
pub struct FluxFit {
    /// lmfit 的拟合结果：`model` 即五参数（前三个拟合、后两个结构常数），`params` 带标准误。
    pub result: ModelResult<Flux>,
}

impl FluxFit {
    /// 反解：把频率调到 `f` 需要多大的 Z 偏置（baseline `flux_of_f`）。
    ///
    /// **这个反解不是单值的**：`f(z)` 对 z 周期（周期 `2·z_period`，即一个磁通量子），且关于
    /// 甜点 `z_offset` 左右对称，同一个目标频率一般对应好几处偏置。本函数只回**离参照点
    /// `near` 最近的那一处**——调频率总是从当前工作点挪最小的距离过去；`near` 取 None 表示
    /// 以甜点为准。解析求解、不用迭代：由线型反解出 `cos²θ`，`θ` 有 `±φ` 与 `±(π − φ)` 四支，
    /// 各自折回 z 再整体平移整数个周期，挑最近的一个。
    ///
    /// 形参:
    ///     f: 目标频率，Hz
    ///     near: 参照偏置，V；None 表示以甜点为准
    ///
    /// 返回值:
    ///     所需的偏置 (V)；目标频率超出这条曲线可达的范围时返回 None
    pub fn tune_to(&self, f: f64, near: Option<f64>) -> Option<f64> {
        let model = self.result.model;
        let ratio = (f + model.eta) / (model.f_max + model.eta);
        let reference = match near {
            Some(z) => z,
            None => model.z_offset,
        };
        match ratio > 0.0 {
            true => {}
            false => return None,
        }
        let bracket = ratio.powi(4);
        match model.asymmetry >= 1.0 {
            // d = 1 时括号恒为 1：两结完全不对称，频率压根不随磁通变，除了甜点都调不到
            true => {
                return match (bracket - 1.0).abs() <= BRACKET_UNITY_TOLERANCE {
                    true => Some(model.z_offset),
                    false => None,
                };
            }
            false => {}
        }
        let cos_sq = (bracket - model.asymmetry.powi(2)) / (1.0 - model.asymmetry.powi(2));
        match (0.0..=1.0).contains(&cos_sq) {
            true => {}
            false => return None,
        }
        let phi = cos_sq.sqrt().acos();
        let scale = model.z_period / PI;
        let period = 2.0 * model.z_period;
        let mut best = f64::NAN;
        let mut best_distance = f64::INFINITY;
        for branch in [phi, -phi, PI - phi, -(PI - phi)] {
            let base = model.z_offset + branch * scale;
            let folded = base + ((reference - base) / period).round() * period;
            let distance = (folded - reference).abs();
            match distance < best_distance {
                true => {
                    best_distance = distance;
                    best = folded;
                }
                false => {}
            }
        }
        Some(best)
    }
}

/// 以给定初值执行一次通量调谐拟合。
///
/// 形参:
///     zs: 偏置轴 (n,)，V
///     peaks: 峰位轴 (n,)，Hz
///     p0: 初值
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
fn fit_once(zs: &[f64], peaks: &[f64], p0: &Flux) -> Result<ModelResult<Flux>, FluxError> {
    match p0.fit(peaks, zs) {
        Ok(result) => Ok(result),
        Err(err) => Err(FluxError::Lmfit(err)),
    }
}

/// 通量调谐拟合：把逐偏置的峰位 `(z, f01)` 拟成 SQUID 线型。
///
/// 复刻 baseline 的 `fit_flux_tunable`：由 [`p0_candidates`] 生成初值，逐候选执行 [`fit_once`]，
/// 失败者跳过，返回 `chisqr` 最小的一次拟合结果。候选只喂求解器初值，谁胜出仍由残差定。
///
/// 形参:
///     zs: Z 偏置轴 (n,)，V
///     peaks: 逐偏置的峰位 (n,)，Hz
///     eta: 充电能 `Ec/h`，Hz（版图给定，固定不拟合）
///     asymmetry: 两结不对称度 `d`（版图给定，固定不拟合）
///
/// 返回值:
///     残差最小的拟合结果，`model.f_max` 即甜点频率、`model.z_period` 即一个磁通量子对应的
///     偏置跨度；数据非法或全部候选未收敛时返回错误
pub fn flux_fit(zs: &[f64], peaks: &[f64], eta: f64, asymmetry: f64) -> Result<FluxFit, FluxError> {
    if zs.is_empty() || peaks.is_empty() {
        return Err(FluxError::EmptyData);
    }
    if zs.len() != peaks.len() {
        return Err(FluxError::LengthMismatch {
            zs: zs.len(),
            peaks: peaks.len(),
        });
    }
    if zs.len() < MIN_POINTS {
        return Err(FluxError::LackPoints {
            points: zs.len(),
            min: MIN_POINTS,
        });
    }
    let mut best: Option<ModelResult<Flux>> = None;
    for p0 in p0_candidates(zs, peaks, eta, asymmetry) {
        match fit_once(zs, peaks, &p0) {
            Ok(result) => {
                let take = match &best {
                    Some(current) => result.chisqr < current.chisqr,
                    None => true,
                };
                match take {
                    true => best = Some(result),
                    false => {}
                }
            }
            // 单候选失败是预期情形（周期猜得离谱），继续扫描其余候选
            Err(FluxError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(FluxFit { result }),
        None => Err(FluxError::AllFitsUnsuccess),
    }
}

/// 一条通量调谐拟合的输入：偏置轴 + 逐偏置峰位 + 两个结构常数。
///
/// 结构常数（`eta` 与 `asymmetry`）逐比特取自版图，不随扫描走，故随行携带而不是全局一套。
pub struct FluxLine<'a> {
    /// Z 偏置轴 (n,)，V
    pub zs: &'a [f64],
    /// 逐偏置的峰位 (n,)，Hz
    pub peaks: &'a [f64],
    /// 充电能 `Ec/h`，Hz（版图给定，固定不拟合）
    pub eta: f64,
    /// 两结不对称度 `d`（版图给定，固定不拟合）
    pub asymmetry: f64,
}

/// 批量通量调谐拟合：逐条独立执行 [`flux_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。
///
/// 形参:
///     lines: 逐条输入（各条自带偏置轴、峰位与两个结构常数）
///
/// 返回值:
///     与 `lines` 等长的结果列表，逐条对应
pub fn flux_fit_batch(lines: &[FluxLine<'_>]) -> Vec<Result<FluxFit, FluxError>> {
    lines
        .par_iter()
        .map(|line| flux_fit(line.zs, line.peaks, line.eta, line.asymmetry))
        .collect()
}
