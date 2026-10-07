//! # DRAG 幅度扫描的升阶（`pairs ≥ 2`）：谷的洛伦兹拟合
//!
//! 最低阶（`pairs = 1`，[`super::factor_one`]）问的是**周期**，用来定升阶窗口的尺度；从二阶
//! 起，每一阶都围着上一阶取到的谷位收窄到 `centre ± A_π/(2·pairs)`（半个周期，恰好框住一个
//! 谷），再拟一条**谷**——谷心既是这一阶的读数，也是下一阶窗口的中心，最高阶那个还要当标定值。
//!
//! 阶越高谷越尖，形状不再是 Rabi 余弦，用倒置的洛伦兹顶替：
//!
//! ```text
//! P1(A) = offset + amp·(fwhm/2)² / ((A − centre)² + (fwhm/2)²)        amp < 0 即谷
//! ```
//!
//! **初值只有一组解析解，不铺候选**（baseline `valley_p0`）。四个参数里只有谷心是要用的量，
//! 而基线、深度、半高全宽三者耦合：一条近乎平直的曲线，用"半宽很大 + 深度很大 + 基线很大"
//! 能用小半宽凑出同样的残差，最优化在这条平坦方向上走到哪里全凭初值（半高全宽因此报出过窗口
//! 的几倍到上百倍）。铺候选等于往那条平坦方向上多撒几个点，正是要躲的东西；给一组尺度合适的
//! 起点，最优化不必自己去找尺度。
//!
//! 两个锚点都由物理给，且**只是起点、不是约束**：
//!
//! - **谷心初值取数据的最低点**。`2·pairs` 发脉冲扫到 `A_π` 时比特整圈转回 |0>，P1 在谷底取
//!   极小，最低点就是谷位的第一估计。
//! - **谷底电位取 0**：`(offset0, amp0) = (+1, −1)` ⇒ `floor = 0`，就是上面那句话的电位。
//!   半高全宽那条方程里的 `0` 正是它（baseline `valley_p0` 的 "谷底落在 0、远处趋于 1"）。
//!   但**谷底的残余正是这一档要量的东西**（相位误差/泄露的最小点），把它钉死就把要看的量
//!   抹掉了，所以 `offset` 与 `amp` 拟合时全自由。
//!
//! 半高全宽由曲线自读：最高那一点必须落在洛伦兹上，
//! `w = reach·√((offset0 − y_max)/(y_max − floor))`；最高点越出 `(0, 1)` 时方程没有正解
//! （实测投影会过冲到 1.03–1.05），退到位置差 `reach`。
//!
//! **谷心必须落在扫过的区间里**，出窗一律按"没拟合出来"处理：平线上 (谷心, 宽度) 几乎完全
//! 简并，一条任意宽的洛伦兹在窗口内就是平线、谷心能滑到窗外任何值；而它既当下一阶的窗心、
//! 最高阶还要写进标定表，一阶跑飞会把整条链带跑。
//!
//! P1 一律由**标定过的**两态中心投影（理由与 [`super::factor_one`] 同）：谷底的残余正是要看
//! 的量，按数据自身极值归一化会把它抹掉。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{argmax, argmin, max_value, min_value};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;

pub use lmfit::Complex64;

/// 四参数洛伦兹所需的**最少**幅度点数：6 点 − 4 参数 = 2 个残余自由度（与 qspec 那条同值）。
const MIN_POINTS: usize = 6;

/// 深度的初值。与 [`OFFSET_P0`] 一起给出"谷底落在 0、远处趋于 1"这个有物理含义的起点——
/// 两个量本身仍然自由，最优化从这里起步（baseline `VALLEY_P0_AMP`）。
const AMP_P0: f64 = -1.0;

/// 基线的初值，含义与理由同 [`AMP_P0`]（baseline `VALLEY_P0_OFFSET`）。
const OFFSET_P0: f64 = 1.0;

// =========================================================================
// 线型
// =========================================================================

/// 谷的线型：倒置的洛伦兹，谷心、半高全宽、深度、基线四个参数全自由。
///
/// 字段次序与 baseline 的 `lorentz(f, fq, fwhm, amp, offset)` 一致，只有第一个换了名字——
/// `fq` 是频率那一路留下的，这里的轴可能是幅度、系数或失谐（baseline 自己的注释）。
#[derive(Model, Clone, Copy, Debug)]
pub struct Valley {
    /// 谷心：扫描轴上 P1 取极小的位置，与扫描量同单位
    #[param(value = 0.0)]
    pub centre: f64,
    /// 半高全宽
    ///
    /// 线型里只用它的半值平方 `(fwhm/2)²`，负宽度因此是写法而不是取值——下界钉在 0 把这条
    /// 规范一次定死，不必拟合完再取绝对值。
    #[param(value = 1.0, min = 0.0)]
    pub fwhm: f64,
    /// 深度（带符号）：**谷取负**，谷底相对基线低 `|amp|`
    ///
    /// 上界钉在 0：谷的深度朝下是物理，不是约定。不钉的话，一条**峰形**曲线（窗落错瓣、
    /// 标定次序给反这类真会发生的故障）会被 `amp > 0` 拟得好好的，而它的谷心落在窗内、出窗
    /// 那条判据拦不住，最后写进标定表；钉上之后那种曲线只能退化成平线，谷心随之滑走，判据
    /// 拦得下。
    #[param(value = -1.0, max = 0.0)]
    pub amp: f64,
    /// 远处（谷外）的基线电平
    ///
    /// 自由：谷底的残余——`offset + amp`——正是这一档要量的量，钉死它就把要看的量抹掉了。
    #[param(value = 1.0)]
    pub offset: f64,
}

impl Valley {
    /// 在幅度数组上求线型值（画拟合曲线用的密集网格走这里）。
    ///
    /// 形参:
    ///     amps: 幅度数组 (n,)
    ///
    /// 返回值:
    ///     线型在 `amps` 上的取值 (n,)
    pub fn at(&self, amps: &[f64]) -> Vec<f64> {
        amps.iter().map(|a| self.eval(*a)).collect()
    }
}

impl Curve for Valley {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 驱动幅度（或该实验扫描的别的量）
    ///
    /// 返回值:
    ///     该点处的激发概率
    fn eval(&self, x: f64) -> f64 {
        let half = 0.5 * self.fwhm;
        let delta = x - self.centre;
        self.offset + self.amp * half * half / (delta * delta + half * half)
    }
}

// =========================================================================
// 初值
// =========================================================================

/// 谷的初值：谷心取逐点最低点，半高全宽由"最高点落在线上"那条方程解出（baseline `valley_p0`）。
///
/// 方程要有正解，`y_max` 必须严格落在谷底与基线之间（`floor < y_max < offset0`）；最高点若不
/// 严格低于基线，就没有有限的半宽能让曲线升到那里去，此时退到**位置差**（谷心到最高点的距离
/// 就是半高半宽的量级）。
///
/// 形参:
///     amps: 扫描轴 (n,)
///     prob: 该轴上的 P1 曲线 (n,)
///
/// 返回值:
///     初值；轴或曲线为空、或曲线是平的（最高点与谷底同在一处，半高全宽无从谈起）时返回 None
fn valley_p0(amps: &[f64], prob: &[f64]) -> Option<Valley> {
    if amps.is_empty() || prob.is_empty() || amps.len() != prob.len() {
        return None;
    }
    let floor = OFFSET_P0 + AMP_P0;
    let top = max_value(prob);
    let centre = amps[argmin(prob)];
    let reach = (amps[argmax(prob)] - centre).abs();
    match top.is_finite() && reach > 0.0 {
        false => return None,
        true => {}
    }
    let half = match floor < top && top < OFFSET_P0 {
        true => reach * ((OFFSET_P0 - top) / (top - floor)).sqrt(),
        // 最高点落到基线之上（或整条贴在谷底）：方程没有正解，退到位置差
        false => reach,
    };
    Some(Valley {
        centre,
        // 线型里的 w 是半高半宽，模型参数是半高全宽，故乘二
        fwhm: 2.0 * half,
        amp: AMP_P0,
        offset: OFFSET_P0,
    })
}

/// 拟合不可用时这一阶的谷位：逐点取最低点（baseline `first_minimum` 的兜底那条路）。
///
/// 最低阶的轴从零驱动起步，那里脉冲面积为零、末态原样回到 |0>，P1 与谷底同高——不排掉的话
/// `argmin` 会指到起点，标出来的 π 幅度整整差一个周期。升阶的窗不含零点（`centre ± 半宽`），
/// 这条自然空转；保留它是为了让这个读数规则对**任何**阶都是对的。
///
/// 形参:
///     amps: 扫描轴 (n,)
///     prob: 该轴上的 P1 曲线 (n,)
///
/// 返回值:
///     最低点所在的轴坐标；轴或曲线为空时返回 NaN
pub fn lowest_point(amps: &[f64], prob: &[f64]) -> f64 {
    let skip = match amps.first() {
        Some(first) => usize::from(*first == 0.0),
        None => 0,
    };
    let points: Vec<(f64, f64)> = amps
        .iter()
        .zip(prob.iter())
        .skip(skip)
        .map(|(amp, value)| (*amp, *value))
        .collect();
    let values: Vec<f64> = points.iter().map(|(_, value)| *value).collect();
    match values.is_empty() {
        true => f64::NAN,
        false => points[argmin(&values)].0,
    }
}

/// 第 `pairs` 阶该开的窗口半宽（baseline `window_half_width`）。
///
/// `2·pairs` 发脉冲下 P1 的周期是 `A_π/pairs`，窗口要恰好框住一个谷、又不把隔壁那两个框进来，
/// 半宽就取**半个周期**。整条升阶链的窗口尺度都从最低阶那一次余弦拟合（`a_pi = 1/freq`）长
/// 出来，不靠经验指数。
///
/// 形参:
///     a_pi: 最低阶拟合出的 π 幅度（[`super::factor_one`]），与幅度轴同单位
///     pairs: 脉冲对数，`≥ 1`（0 阶没有意义）
///
/// 返回值:
///     窗口半宽，与幅度轴同单位
pub fn window_half_width(a_pi: f64, pairs: usize) -> f64 {
    a_pi / (2.0 * pairs as f64)
}

// =========================================================================
// 拟合入口
// =========================================================================

/// 升阶谷拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum FactorNError {
    /// 幅度或 IQ 数据为空。
    EmptyData,
    /// 幅度点数与 IQ 点数不一致。
    LengthMismatch { amps: usize, iq: usize },
    /// 幅度点数不足以约束四参数洛伦兹。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与幅度扫描数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 曲线是平的：最高点与谷底同在一处，初值里的半高全宽无从谈起。
    FlatCurve,
    /// 谷心落在扫描区间之外（含端点，出窗一律按没拟合出来处理）。
    CentreOutOfWindow { centre: f64, low: f64, high: f64 },
    /// 数据无有效投影值，或拟合未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for FactorNError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the amplitude or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { amps, iq } => {
                write!(f, "amplitude points ({amps}) and IQ points ({iq}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} amplitude points cannot constrain a 4-parameter Lorentzian; at least {min} are needed"
            ),
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of amplitude scans ({lines})"
            ),
            Self::FlatCurve => write!(
                f,
                "the curve is flat: the highest and lowest samples coincide, so no finite width can reach the top"
            ),
            Self::CentreOutOfWindow { centre, low, high } => write!(
                f,
                "the fitted centre ({centre}) falls outside the scanned window [{low}, {high}]: on a flat curve the centre and the width are degenerate, so such a solution only repeats the noise"
            ),
            Self::AllFitsUnsuccess => write!(
                f,
                "the projection carries no finite value, or the fit did not converge"
            ),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for FactorNError {}

/// 单条升阶扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct FactorNFit {
    /// lmfit 的拟合结果：`model` 即四参数，`model.centre` 即谷心；`params` 带标准误。
    pub result: ModelResult<Valley>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次谷拟合。
///
/// 形参:
///     amps: 扫描轴 (n,)
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    amps: &[f64],
    y: &[f64],
    p0: &Valley,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<Valley>, FactorNError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, amps, weights),
        None => p0.fit(y, amps),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(FactorNError::Lmfit(err)),
    }
}

/// 升阶谷拟合：IQ 投影为 P1，再拟一条倒置洛伦兹，谷心即这一阶的读数。
///
/// 这条线型带自由 `offset`，但它并不对 `1 − P1` 封闭——翻转会把谷变成峰，而 `amp ≤ 0` 把那条
/// 路堵死，所以没有 rabi 那样的两遍取向拟合。谷心出窗、曲线是平的、求解器没收敛，都按"没拟合
/// 出来"报错，由调用方决定退回 [`lowest_point`]（错误类型把三种情形分开，便于分辨是哪一种）。
///
/// 形参:
///     amps: XY 驱动幅度 (n,)，与最低阶同一单位；这一阶的窗由上一阶的谷心与
///           [`window_half_width`] 给出
///     iq: 单条幅度扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；**必传**（理由见模块文档）
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径）；
///            给定时按 1/σ² 加权，经 [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     拟合结果，`model.centre` 即谷心、`model.fwhm` 即半高全宽（与扫描量同单位）；
///     数据非法、初值解不出、谷心出窗或未收敛时返回错误
pub fn factor_n_fit(
    amps: &[f64],
    iq: &[Complex64],
    states: &StateCenters,
    sigma: Option<&[f64]>,
) -> Result<FactorNFit, FactorNError> {
    if amps.is_empty() || iq.is_empty() {
        return Err(FactorNError::EmptyData);
    }
    if amps.len() != iq.len() {
        return Err(FactorNError::LengthMismatch {
            amps: amps.len(),
            iq: iq.len(),
        });
    }
    if amps.len() < MIN_POINTS {
        return Err(FactorNError::LackPoints {
            points: amps.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, Some(states));
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(FactorNError::AllFitsUnsuccess);
    }
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, Some(states), values)),
        None => None,
    };

    let p0 = match valley_p0(amps, &prob) {
        Some(p0) => p0,
        None => return Err(FactorNError::FlatCurve),
    };
    let result = fit_once(amps, &prob, &p0, weights.as_deref())?;

    // 谷心必须落在扫过的范围里（含端点）。平线上谷心与宽度几乎完全简并，一条任意宽的洛伦兹在
    // 窗口内就是一条平线，谷心可以滑到窗外任何值上；而它既当下一阶的窗心、最高阶还要当标定值，
    // 放它过去的代价不止这一步。出窗的解一律按"没拟合出来"处理。
    let (low, high) = (min_value(amps), max_value(amps));
    let centre = result.model.centre;
    match low <= centre && centre <= high {
        true => {}
        false => {
            return Err(FactorNError::CentreOutOfWindow {
                centre,
                low,
                high,
            });
        }
    }

    Ok(FactorNFit { result, p1: prob })
}

/// 批量升阶谷拟合：对多条幅度扫描并行执行 [`factor_n_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心由各线
/// **共用**——同一批比特共用一套中心，换标定就再调一次本函数。
///
/// 形参:
///     amps: 公共幅度轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 各态标定中心；必传（同 [`factor_n_fit`]）
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`FactorNError::BatchSigmaMismatch`]
pub fn factor_n_fit_batch(
    amps: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: &StateCenters,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<FactorNFit, FactorNError>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => factor_n_fit(amps, line, states, Some(sigma)),
                None => Err(FactorNError::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| factor_n_fit(amps, line, states, None))
            .collect(),
    }
}
