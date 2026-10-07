//! # T2 echo（回波退相位）：指数线型与拟合
//!
//! 物理：`π/2 — τ/2 — π — τ/2 — π/2`。中间的 π 脉冲把准静态失谐积累起来的相位折回去，
//! Ramsey 那条会振荡的条纹因此被压成一条单调的指数衰减
//!
//! ```text
//! P1(τ) = offset + amplitude·exp(−τ / t2_echo)
//! ```
//!
//! **τ 是总自由演化时间**（两个 τ/2 的和，也就是扫描轴本身）：拟合出的时间常数就是 T2 echo，
//! 不做任何 2× 换算。若实验只记了半延时 τ/2，调用方要自己把轴翻成总延时再送进来。
//!
//! 单个 π 脉冲只折得回准静态的那部分失相，快涨落仍会退相位，所以 `T2* ≤ T2echo ≤ 2·T1` 是
//! 这条曲线的物理次序——那是判断拟合结果合不合理时的参照，不是拟合本身的约束。
//!
//! `amplitude` 吸收末态制备的不完美，`offset` 吸收读出把 |1> 误判成 |0> 的那一部分——两者
//! 都是 SPAM，不该算进时间常数（与 T1 那边同一口径）。
//!
//! **曲线的形状与朝向**：零延时处三个脉冲直接相接（π/2 · π · π/2 合成一个 2π），比特回到
//! |0>，所以 **P1(0) = 0**；长延时那边相干性耗光、退相干到完全混合态，最后一次 π/2 把它映到
//! **P1 = 0.5**——曲线整体是**从 0 升到 0.5**，`amplitude` 因此取负，扫程内也不该看到它跑到
//! 0.5 之上（那是拟合或读出出问题的信号，不是这条曲线的行为）。
//!
//! 未传标定中心时朝向就按上面这个默认钉住；拟合出来的朝向与它不符时按 `1 − P1` 翻过来重拟
//! 一次（做法照 qspec 的二次处理：先拟一次、按 `amplitude` 的符号判、必要时翻过来再拟）。
//! 传了标定中心时朝向由 g0 → 0、g1 → 1 定死，不再动它。
//!
//! baseline `qana` 里没有回波实验，本模块没有可对照的 Python 实现：线型、初值与拟合照
//! `exps/t1_analysis.py` 的 `decay` / `t1_fit` 那套写（同一条曲线、同一个三参数模型），只是
//! 时间常数的名字与横轴的约定按回波实验来。
//!
//! 这里只有"给一条曲线、还一组参数"的纯计算，不含实验语义：延时怎么排、读出怎么定轴、
//! 标定值往哪写，都由调用方决定。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{max_value, min_value, orient};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;

pub use lmfit::Complex64;

/// 三参数指数所需的最少延时点数：4 点 − 3 参数 = 1 个残余自由度。
const MIN_POINTS: usize = 4;

/// T2 echo 的初值候选，单位是**窗口长度的倍数**。
///
/// 下限保证窗口内看得到明显衰减，上限允许"窗口比时间常数还短"的情形——那时衰减在窗口内几乎
/// 看不出来，真值可能远在窗口之外。
const T2_ECHO_RATIOS: [f64; 4] = [0.2, 0.5, 1.0, 2.0];

/// 本底电平的初值：长延时处相干性耗光、退相干到完全混合态，最后一次 π/2 把它映到 P1 = 0.5。
const OFFSET_P0: f64 = 0.5;

/// 衰减幅度的初值：起点 P1 = 0 比本底低一整个电平，所以是 **−0.5**——取负是它的物理符号
/// （曲线朝上升），不是量值上的写法。填正号等于把初值放到镜像那条支上（从 1 降到 0.5，与数据
/// 反着走），迭代得先穿过 `A ≈ 0`，而 `∂/∂t2_echo = A·(τ/t2_echo²)·e^{−τ/t2_echo}` 恰好在
/// 那里整列为零；何况这个符号后面还要当翻转判据用，初值不该先站在错的一侧。
const AMPLITUDE_P0: f64 = -0.5;

// =========================================================================
// 线型
// =========================================================================

/// 带本底的指数衰减线型。
///
/// 三个参数全自由（只钉时间常数的下界），曲线朝上还是朝下由 `amplitude` 的符号承担——不另设
/// 取向开关。
#[derive(Model, Clone, Copy, Debug)]
pub struct EchoDecay {
    /// 长延时下的本底：读出把 |1> 误判成 |0> 的那一部分落在这里
    #[param(value = 0.0)]
    pub offset: f64,
    /// 衰减项的幅度（带符号）：末态制备不干净时绝对值小于 1
    ///
    /// 符号即曲线朝向——取正随 τ 下降、取负随 τ 上升。它不设界也不规范化：`1 − P1` 那种反向
    /// 读数正是靠它与 `offset` 吸收成同一族曲线，两种写法对残差没有区别。
    #[param(value = 1.0)]
    pub amplitude: f64,
    /// 回波退相位时间 (s)：**τ 取总自由演化时间**，拟合出的它直接就是 T2 echo
    ///
    /// 下界钉在 0：负的时间常数会让幅度随 τ 发散，不是本模型要表达的东西。
    #[param(value = 1.0e-6, min = 0.0)]
    pub t2_echo: f64,
}

impl EchoDecay {
    /// 在延时数组上求线型值（画拟合曲线用的密集网格走这里）。
    ///
    /// 形参:
    ///     taus: 总自由演化时间轴 (n,)，单位 s
    ///
    /// 返回值:
    ///     线型在 `taus` 上的取值 (n,)
    pub fn at(&self, taus: &[f64]) -> Vec<f64> {
        taus.iter().map(|tau| self.eval(*tau)).collect()
    }
}

impl Curve for EchoDecay {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 总自由演化时间，单位 s
    ///
    /// 返回值:
    ///     该延时处的 P1
    fn eval(&self, x: f64) -> f64 {
        self.offset + self.amplitude * (-x / self.t2_echo).exp()
    }
}

// =========================================================================
// 初值
// =========================================================================

/// T2 echo 的初值候选：本底与幅度由物理给定，唯一铺开的是时间常数。
///
/// - `offset` 取 [`OFFSET_P0`] = 0.5：长延时处退相干到完全混合态，最后一次 π/2 把它映到 0.5
///   ——这是物理，不必从数据里猜（读出误判那一部分正是拟合要吸收的，初值不必替它先占个位）。
///   自定轴路径上曲线按投影极值归一化、绝对电平不在里面，0.5 那里只是个中位的起点；
/// - `amplitude` 取 [`AMPLITUDE_P0`] = −0.5：起点 P1 = 0 比本底低一整个电平；
/// - `t2_echo` 取窗口长度（`max(taus) − min(taus)`）的 [`T2_ECHO_RATIOS`] 倍——这一维读不
///   出来，只能铺开。
///
/// 形参:
///     taus: 总自由演化时间轴 (n,)
///
/// 返回值:
///     初值列表，共 4 组
fn p0_candidates(taus: &[f64]) -> Vec<EchoDecay> {
    let span = max_value(taus) - min_value(taus);
    T2_ECHO_RATIOS
        .iter()
        .map(|ratio| EchoDecay {
            offset: OFFSET_P0,
            amplitude: AMPLITUDE_P0,
            t2_echo: ratio * span,
        })
        .collect()
}

// =========================================================================
// 拟合入口
// =========================================================================

/// T2 echo 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum T2EchoError {
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

impl std::fmt::Display for T2EchoError {
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

impl std::error::Error for T2EchoError {}

/// 单条回波扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct T2EchoFit {
    /// lmfit 的拟合结果：`model` 即三参数（`t2_echo` 即回波退相位时间），`params` 带标准误。
    pub result: ModelResult<EchoDecay>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条，也是报告里画的那条。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次指数拟合。
///
/// 形参:
///     taus: 总自由演化时间轴 (n,)，单位 s
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    taus: &[f64],
    y: &[f64],
    p0: &EchoDecay,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<EchoDecay>, T2EchoError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, taus, weights),
        None => p0.fit(y, taus),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(T2EchoError::Lmfit(err)),
    }
}

/// 初值候选逐个试，取 `chisqr` 最小者。
///
/// 形参:
///     taus: 总自由演化时间轴 (n,)
///     prob: 待拟合的 P1 曲线 (n,)
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果；全部候选失败时返回错误
fn fit_candidates(
    taus: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
) -> Result<ModelResult<EchoDecay>, T2EchoError> {
    let mut best: Option<ModelResult<EchoDecay>> = None;
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
            Err(T2EchoError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(T2EchoError::AllFitsUnsuccess),
    }
}

/// T2 echo 拟合：IQ 投影为 P1，再拟一条带本底的指数衰减。
///
/// 线型对 `1 − P1` 封闭（自由 `offset` + 自由 `amplitude`），两种朝向的残差恒等；未标定时
/// 朝向按回波的物理默认钉住（P1(0) = 0，见模块文档），标定时由两个中心定死。
///
/// 形参:
///     taus: **总自由演化时间**轴 (n,)，单位 s。若手上只有半延时 τ/2，先乘 2 再送进来——
///           这条约定不在函数里替调用方决定（见模块文档）
///     iq: 单条延时扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；None 表示未标定，P1 改由数据自身定轴
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.t2_echo` 即回波退相位时间（单位 s）；数据非法或全部候选未
///     收敛时返回错误
pub fn t2_echo_fit(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
) -> Result<T2EchoFit, T2EchoError> {
    if taus.is_empty() || iq.is_empty() {
        return Err(T2EchoError::EmptyData);
    }
    if taus.len() != iq.len() {
        return Err(T2EchoError::LengthMismatch {
            taus: taus.len(),
            iq: iq.len(),
        });
    }
    if taus.len() < MIN_POINTS {
        return Err(T2EchoError::LackPoints {
            points: taus.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, states);
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(T2EchoError::AllFitsUnsuccess);
    }
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };

    let trial = match fit_candidates(taus, &prob, weights.as_deref()) {
        Ok(result) => result,
        Err(error) => return Err(error),
    };
    // 自定轴路径下朝向该是回波的物理朝向：P1(0) = 0、往上升到 0.5，即 `amplitude` 取负那一支。
    // 判据取拟合出来的符号而不是端点比较——拟合把噪声平均掉了，符号要稳得多。标定路径下朝向
    // 由 g0 → 0、g1 → 1 定死，正的 `amplitude` 是"标定与数据对不上"的信号，翻过去只会把标定
    // 错误盖住，所以只对自定轴路径翻。
    let inversion = states.is_none() && trial.model.amplitude > 0.0;
    match inversion {
        false => Ok(T2EchoFit {
            result: trial,
            p1: prob,
        }),
        true => {
            // 翻转是 1 − P1，带负号的仿射变换，满量程不变，故两个朝向共用同一份 σ_P1
            let orientation = orient(&prob, true);
            match fit_candidates(taus, &orientation, weights.as_deref()) {
                Ok(result) => Ok(T2EchoFit {
                    result,
                    p1: orientation,
                }),
                Err(error) => Err(error),
            }
        }
    }
}

/// 批量 T2 echo 拟合：对多条回波扫描并行执行 [`t2_echo_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心由各线
/// **共用**——同一条比特的多档参数扫描共用一套中心，换比特/换标定就再调一次本函数。
///
/// 形参:
///     taus: 公共总自由演化时间轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 各态标定中心；None 表示未标定
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`T2EchoError::BatchSigmaMismatch`]
pub fn t2_echo_fit_batch(
    taus: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: Option<&StateCenters>,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<T2EchoFit, T2EchoError>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => t2_echo_fit(taus, line, states, Some(sigma)),
                None => Err(T2EchoError::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| t2_echo_fit(taus, line, states, None))
            .collect(),
    }
}
