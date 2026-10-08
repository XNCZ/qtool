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
//! **朝向**：曲线的物理朝向是 **τ = 0 处 P1 = 1**——比特先被 π 脉冲摆到 |1>，然后才往下
//! 衰减——但本模块不做任何翻转，拟合出什么朝向就交付什么朝向：自定轴路径下朝向是投影方向
//! 自身的约定（`amplitude` 为正即 τ = 0 落在高的一头），标定路径下由 g0 → 0、g1 → 1 定死。
//! 自定轴路径的初值候选把镜像（朝上升）一支一并铺上，两条朝向都能直接收敛（见
//! [`p0_candidates`]）；标定路径不铺——那里拟出上升是"标定与数据对不上"的信号。
//! 线型对 `y ↦ 1 − P1` 封闭（自由 `offset` 吸收镜像），两种朝向残差恒等，`t1` 的值不受
//! 朝向影响，`amplitude` 的符号本身留给调用方当判据。
//!
//! 这里只有"给一条曲线、还一组参数"的纯计算，不含实验语义：延时怎么排、读出怎么定轴、
//! 标定值往哪写，都由调用方决定。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{max_value, min_value};
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
/// 投影极值归一化，起点同样是 1，同一个常数两条路都够用；反过来的那一支由 [`p0_candidates`]
/// 的 `mirror` 另铺一份。
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
/// - `amplitude` 取 [`AMPLITUDE_P0`] = 1：起点在 |1> 上；
/// - `mirror` 为真时把镜像支 `(offset, amplitude) → (1 − offset, −amplitude)` 一并铺上：
///   自定轴路径的朝向本是约定，数据可能朝上升，镜像支让那条曲线不必从 `amplitude = 0`
///   这个退化起点穿过去（`∂/∂t1 = amplitude·(τ/t1²)·e^{−τ/t1}` 在 `amplitude = 0` 处整列
///   为零）。只对自定轴路径开（见 [`t1_fit`]）：标定路径拟出上升曲线是"标定与数据对不上"
///   的信号，不替它补好起点；
/// - `t1` 取窗口长度（`max(taus) − min(taus)`）的 [`T1_RATIOS`] 倍——这一维读不出来，只能铺开。
///
/// 形参:
///     taus: 延时轴 (n,)
///     mirror: 是否把镜像（朝上升）一支的初值一并铺上
///
/// 返回值:
///     初值列表，共 4 组（`mirror` 为真时 8 组）
fn p0_candidates(taus: &[f64], mirror: bool) -> Vec<Decay> {
    let span = max_value(taus) - min_value(taus);
    let mut branches = vec![(OFFSET_P0, AMPLITUDE_P0)];
    match mirror {
        true => branches.push((1.0 - OFFSET_P0, -AMPLITUDE_P0)),
        false => {}
    }
    let mut out = Vec::with_capacity(branches.len() * T1_RATIOS.len());
    for (offset, amplitude) in branches {
        for ratio in T1_RATIOS {
            out.push(Decay {
                offset,
                amplitude,
                t1: ratio * span,
            });
        }
    }
    out
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
///     mirror: 初值候选是否铺上镜像（朝上升）一支，透传 [`p0_candidates`]
///
/// 返回值:
///     残差最小的拟合结果；全部候选失败时返回错误
fn fit_candidates(
    taus: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
    mirror: bool,
) -> Result<ModelResult<Decay>, T1Error> {
    let mut best: Option<ModelResult<Decay>> = None;
    for p0 in p0_candidates(taus, mirror) {
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
/// 判反（数据朝上升）时会从 `amplitude = 0` 这种退化初值起步。这里初值改成与朝向无关的
/// 写法（见 [`p0_candidates`]），朝向原样交付、不做翻转（见模块文档）。
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

    // 镜像（朝上升）一支只对自定轴路径铺：那里的朝向本是约定，上升曲线是合法形态；
    // 标定路径的朝向由两个中心定死，拟出上升是"标定与数据对不上"的信号，不替它补好起点。
    let trial = match fit_candidates(taus, &prob, weights.as_deref(), states.is_none()) {
        Ok(result) => result,
        Err(error) => return Err(error),
    };
    Ok(T1Fit {
        result: trial,
        p1: prob,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 延时轴：41 点、100 µs 窗；真值 T1 = 30 µs。
    const TAU_MAX: f64 = 100e-6;
    const T1_TRUE: f64 = 30e-6;

    /// 两态中心（任意夹角的一对）。
    const G0: Complex64 = Complex64::new(0.30, 0.10);
    const G1: Complex64 = Complex64::new(0.42, 0.30);

    /// 无噪声合成：P1 = exp(−τ/T1) 落到 `(zero, one)` 连线上；`inverted` 为真时把两个
    /// 中心对调，自定轴随之反了（投影出的是上升曲线）。
    fn clean_scan(inverted: bool) -> (Vec<f64>, Vec<Complex64>) {
        let (zero, one) = match inverted {
            true => (G1, G0),
            false => (G0, G1),
        };
        let taus: Vec<f64> = (0..41).map(|k| TAU_MAX * k as f64 / 40.0).collect();
        let iq = taus
            .iter()
            .map(|tau| zero + (one - zero) * (-tau / T1_TRUE).exp())
            .collect();
        (taus, iq)
    }

    /// 标定路径回收 T1；朝向原样交付——中心对调（投影反了）时不翻正，`amplitude` 为负、
    /// 曲线留在反向，T1 的值不受影响。
    #[test]
    fn calibrated_fit_recovers_truth_without_flipping() {
        let (taus, iq) = clean_scan(false);
        let straight = StateCenters::new(vec![G0, G1]);
        let swapped = StateCenters::new(vec![G1, G0]);

        let forward = match t1_fit(&taus, &iq, Some(&straight), None) {
            Ok(fit) => fit,
            Err(err) => panic!("不该拟合失败：{err}"),
        };
        assert!((forward.result.model.t1 - T1_TRUE).abs() < 1e-7, "{}", forward.result.model.t1);
        assert!(forward.result.model.amplitude > 0.0, "{}", forward.result.model.amplitude);

        let reversed = match t1_fit(&taus, &iq, Some(&swapped), None) {
            Ok(fit) => fit,
            Err(err) => panic!("不该拟合失败：{err}"),
        };
        assert!((reversed.result.model.t1 - T1_TRUE).abs() < 1e-7, "{}", reversed.result.model.t1);
        assert!(reversed.result.model.amplitude < 0.0, "{}", reversed.result.model.amplitude);
        assert!(reversed.p1[0] < 0.5, "{}", reversed.p1[0]);
    }

    /// 镜像一支只对自定轴路径铺：`mirror = true` 的候选含上升支（`offset = 1`、
    /// `amplitude = −1`），下降支也保留；`mirror = false`（标定路径）只有下降支。
    #[test]
    fn mirror_candidates_are_opt_in() {
        let (taus, _) = clean_scan(false);

        let mirrored = p0_candidates(&taus, true);
        assert!(
            mirrored
                .iter()
                .any(|p| p.amplitude < 0.0 && (p.offset - 1.0).abs() < 1e-12),
            "{mirrored:?}"
        );
        assert!(mirrored.iter().any(|p| p.amplitude > 0.0), "{mirrored:?}");

        let descending_only = p0_candidates(&taus, false);
        assert!(descending_only.iter().all(|p| p.amplitude > 0.0), "{descending_only:?}");
    }

    /// 自定轴路径同理：朝向由数据自身的投影决定，判反了也不翻。
    #[test]
    fn auto_axis_keeps_the_data_orientation() {
        let (taus, iq) = clean_scan(true);
        let fit = match t1_fit(&taus, &iq, None, None) {
            Ok(fit) => fit,
            Err(err) => panic!("不该拟合失败：{err}"),
        };
        assert!((fit.result.model.t1 - T1_TRUE).abs() < 1e-6, "{}", fit.result.model.t1);
        assert!(fit.result.model.amplitude < 0.0, "{}", fit.result.model.amplitude);
    }
}
