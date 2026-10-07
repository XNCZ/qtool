//! # T1（能量弛豫）：指数线型与拟合
//!
//! baseline `qana` 的 Rust 移植：`exps/t1_analysis.py` 的线型 `decay` 与拟合 `t1_fit`。
//!
//! 物理：把比特激发到 |1> 之后让它自由演化 τ，再读一次——末态落在 |1> 的概率随 τ 指数衰减
//!
//! ```text
//! P1(τ) = offset + amplitude·exp(−τ / t1)
//! ```
//!
//! `amplitude` 吸收 π 脉冲本身的不完美（激发不干净时起点不到 1），`offset` 吸收读出把 |1>
//! 误判成 |0> 的那一部分——两者都是 SPAM，不该算进 T1（baseline 的文档原话）。
//!
//! **朝向**：未传标定中心时曲线的朝向是约定（自定轴的正负号本身任意），这里把它钉成 T1 的
//! 物理朝向——**τ = 0 处 P1 = 1**：比特先被 π 脉冲摆到 |1>，然后才往下衰减。拟合出来的朝向
//! 与它不符时按 `1 − P1` 翻过来重拟一次（做法照 qspec 的二次处理：先拟一次、按 `amplitude`
//! 的符号判、必要时翻过来再拟）；传了标定中心时朝向由 g0 → 0、g1 → 1 定死，不再动它。
//!
//! 这里只有"给一条曲线、还一组参数"的纯计算，不含实验语义：延时怎么排、读出怎么定轴、
//! 标定值往哪写，都由调用方决定。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{max_value, min_value, orient};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;

pub use lmfit::Complex64;

/// 三参数指数所需的最少延时点数：4 点 − 3 参数 = 1 个残余自由度（与 baseline 的
/// `times.size < 4` 同值）。
const MIN_POINTS: usize = 4;

/// T1 的初值候选，单位是**窗口长度的倍数**。
///
/// 下限保证窗口内看得到明显衰减，上限允许"窗口比 T1 还短"的情形——那时衰减在窗口内几乎看不
/// 出来，真值可能远在窗口之外。baseline 的单点初值 `span/3` 正夹在头两档之间。
const T1_RATIOS: [f64; 4] = [0.2, 0.5, 1.0, 2.0];

/// 本底电平的初值：T1 的渐近态是 |0>，P1 落到 0。
const OFFSET_P0: f64 = 0.0;

/// 衰减幅度的初值：π 脉冲把布居整个摆到 |1>，τ = 0 处 P1 就是 1。
///
/// 取常数而不取 `prob[0]`：起点是一个**单点**，噪声全落在它身上，而它本该是个已知的物理量
/// （激发不干净、读出误判那些正是拟合要吸收的，初值不必替它们先打折）。自定轴路径上曲线按
/// 投影极值归一化，起点同样是 1，同一个常数两条路都够用。
const AMPLITUDE_P0: f64 = 1.0;

// =========================================================================
// 线型
// =========================================================================

/// 带本底的指数衰减线型。
///
/// 三个参数全自由（baseline 的 `bounds` 也只钉了 T1 的下界），曲线朝上还是朝下由 `amplitude`
/// 的符号承担——不另设取向开关。
#[derive(Model, Clone, Copy, Debug)]
pub struct Decay {
    /// 长延时下的本底：读出把 |1> 误判成 |0> 的那一部分落在这里
    #[param(value = 0.0)]
    pub offset: f64,
    /// 衰减项的幅度（带符号）：π 脉冲激发不干净时绝对值小于 1
    ///
    /// 符号即曲线朝向——取正随 τ 下降、取负随 τ 上升。它不设界也不规范化：`1 − P1` 那种反向
    /// 读数正是靠它与 `offset` 吸收成同一族曲线，两种写法对残差没有区别。
    #[param(value = 1.0)]
    pub amplitude: f64,
    /// 能量弛豫时间 (s)
    ///
    /// 下界钉在 0：负的时间常数会让幅度随 τ 发散，不是本模型要表达的东西（baseline 的
    /// `bounds` 也只钉了这一条）。
    #[param(value = 1.0e-6, min = 0.0)]
    pub t1: f64,
}

impl Decay {
    /// 在延时数组上求线型值（画拟合曲线用的密集网格走这里）。
    ///
    /// 形参:
    ///     taus: 延时轴 (n,)，单位 s
    ///
    /// 返回值:
    ///     线型在 `taus` 上的取值 (n,)
    pub fn at(&self, taus: &[f64]) -> Vec<f64> {
        taus.iter().map(|tau| self.eval(*tau)).collect()
    }
}

impl Curve for Decay {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 延时，单位 s
    ///
    /// 返回值:
    ///     该延时处的 P1
    fn eval(&self, x: f64) -> f64 {
        self.offset + self.amplitude * (-x / self.t1).exp()
    }
}

// =========================================================================
// 初值
// =========================================================================

/// T1 的初值候选：本底与幅度由物理给定，唯一铺开的是时间常数。
///
/// - `offset` 取 [`OFFSET_P0`] = 0：τ 大了以后比特已经弛豫回 |0>，P1 的渐近值就是 0——这是
///   物理，不必从数据里猜（读出误判那一部分正是拟合要吸收的，初值不必替它先占个位）。自定轴
///   路径上曲线按投影极值归一化，渐近端同样是 0；
/// - `amplitude` 取 [`AMPLITUDE_P0`] = 1：起点在 |1> 上（朝向另说，见 [`t1_fit`] 的二次处理）；
/// - `t1` 取窗口长度（`max(taus) − min(taus)`）的 [`T1_RATIOS`] 倍——这一维读不出来，只能铺开。
///
/// 形参:
///     taus: 延时轴 (n,)
///
/// 返回值:
///     初值列表，共 4 组
fn p0_candidates(taus: &[f64]) -> Vec<Decay> {
    let span = max_value(taus) - min_value(taus);
    T1_RATIOS
        .iter()
        .map(|ratio| Decay {
            offset: OFFSET_P0,
            amplitude: AMPLITUDE_P0,
            t1: ratio * span,
        })
        .collect()
}

// =========================================================================
// 拟合入口
// =========================================================================

/// T1 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum T1Error {
    /// 延时或 IQ 数据为空。
    EmptyData,
    /// 延时点数与 IQ 点数不一致。
    LengthMismatch { taus: usize, iq: usize },
    /// 延时点数不足以约束三参数指数。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与延时扫描数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 数据无有效投影值，或全部初值候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for T1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the delay or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { taus, iq } => {
                write!(f, "delay points ({taus}) and IQ points ({iq}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} delay points cannot constrain a 3-parameter exponential; at least {min} are needed"
            ),
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of delay scans ({lines})"
            ),
            Self::AllFitsUnsuccess => write!(
                f,
                "the projection carries no finite value, or no initial-value candidate produced a fit"
            ),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for T1Error {}

/// 单条 T1 弛豫扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct T1Fit {
    /// lmfit 的拟合结果：`model` 即三参数（`t1` 即能量弛豫时间），`params` 带标准误。
    pub result: ModelResult<Decay>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条，也是报告里画的那条。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次指数拟合。
///
/// 形参:
///     taus: 延时轴 (n,)，单位 s
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///     sigma: P1 域的逐点不确定度；None 表示不加权（与 baseline 的 `curve_fit` 一致）
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    taus: &[f64],
    y: &[f64],
    p0: &Decay,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<Decay>, T1Error> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, taus, weights),
        None => p0.fit(y, taus),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(T1Error::Lmfit(err)),
    }
}

/// 初值候选逐个试，取 `chisqr` 最小者（baseline 的单点初值扩成四档）。
///
/// 形参:
///     taus: 延时轴 (n,)
///     prob: 待拟合的 P1 曲线 (n,)
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果；全部候选失败时返回错误
fn fit_candidates(
    taus: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
) -> Result<ModelResult<Decay>, T1Error> {
    let mut best: Option<ModelResult<Decay>> = None;
    for p0 in p0_candidates(taus) {
        match fit_once(taus, prob, &p0, sigma) {
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
            // 单候选失败是预期情形（初值把时间常数放在窗口外），继续扫描其余候选
            Err(T1Error::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(T1Error::AllFitsUnsuccess),
    }
}

/// T1 拟合：IQ 投影为 P1，再拟一条带本底的指数衰减。
///
/// 复刻 baseline 的 `t1_fit`，并补上它的一个缺口：baseline 的初值假定曲线是下降的，自定轴
/// 判反（数据朝上升）时会从 `amplitude = 0` 这种退化初值起步。这里初值改成与朝向无关的写法
/// （见 [`p0_candidates`]），朝向则按下面的默认钉住。
///
/// 形参:
///     taus: 延时轴 (n,)，单位 s
///     iq: 单条延时扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；None 表示未标定，P1 改由数据自身定轴
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.t1` 即能量弛豫时间（单位 s）；数据非法或全部候选未收敛时
///     返回错误
pub fn t1_fit(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
) -> Result<T1Fit, T1Error> {
    if taus.is_empty() || iq.is_empty() {
        return Err(T1Error::EmptyData);
    }
    if taus.len() != iq.len() {
        return Err(T1Error::LengthMismatch {
            taus: taus.len(),
            iq: iq.len(),
        });
    }
    if taus.len() < MIN_POINTS {
        return Err(T1Error::LackPoints {
            points: taus.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, states);
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(T1Error::AllFitsUnsuccess);
    }
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };

    let trial = match fit_candidates(taus, &prob, weights.as_deref()) {
        Ok(result) => result,
        Err(error) => return Err(error),
    };
    // 自定轴路径下朝向该是 T1 的物理朝向：τ = 0 处 P1 在**高**的一头（先摆到 |1> 再衰减）。
    // 判据取拟合出来的 `amplitude` 符号而不是端点比较——拟合把噪声平均掉了，符号要稳得多。
    // 标定路径下朝向由 g0 → 0、g1 → 1 定死，负的 `amplitude` 是"标定与数据对不上"的信号，
    // 翻过去只会把标定错误盖住，所以只对自定轴路径翻。
    let inversion = states.is_none() && trial.model.amplitude < 0.0;
    match inversion {
        false => Ok(T1Fit {
            result: trial,
            p1: prob,
        }),
        true => {
            // 翻转是 1 − P1，带负号的仿射变换，满量程不变，故两个朝向共用同一份 σ_P1
            let orientation = orient(&prob, true);
            match fit_candidates(taus, &orientation, weights.as_deref()) {
                Ok(result) => Ok(T1Fit {
                    result,
                    p1: orientation,
                }),
                Err(error) => Err(error),
            }
        }
    }
}

/// 批量 T1 拟合：对多条弛豫扫描并行执行 [`t1_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心由各线
/// **共用**——同一条比特的多档参数扫描共用一套中心，换比特/换标定就再调一次本函数。
///
/// 形参:
///     taus: 公共延时轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 各态标定中心；None 表示未标定
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`T1Error::BatchSigmaMismatch`]
pub fn t1_fit_batch(
    taus: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: Option<&StateCenters>,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<T1Fit, T1Error>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => t1_fit(taus, line, states, Some(sigma)),
                None => Err(T1Error::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| t1_fit(taus, line, states, None))
            .collect(),
    }
}
