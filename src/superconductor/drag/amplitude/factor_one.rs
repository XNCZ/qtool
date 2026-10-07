//! # DRAG 幅度扫描的最低阶（`pairs = 1`）：余弦线型与拟合
//!
//! 一连打 `2·pairs` 发**同号**脉冲，脉冲本身不带 DRAG 修正（`coeff = 0`），峰值幅度逐点扫。
//! 同号的两发不抵消任何东西——它只是把单发的旋转误差累加 `2·pairs` 倍，P1 对幅度的偏离因此比
//! 单发敏感得多。最低阶（`pairs = 1`，两发）的曲线形状恰好是 Rabi 余弦
//!
//! ```text
//! P1(A) = amp·(1 − cos(2π·freq·A))
//! ```
//!
//! 与 rabi 幅度扫描是逐字同一条线型（baseline 里就是同一个 `cos_wave`），但**两处口径不同，
//! 弄混会静默差一倍**：
//!
//! - 这里扫的是**两发**的累计转角，`P1 = sin²(π·A/A_π)`，首个**极小**落在 `A_π = 1/freq`
//!   上；单发那条 `P1 = sin²(π·A/(2·A_π))` 的首个**极大**才落在 `1/(2·freq)`。同一个物理量、
//!   两条曲线的 `freq` 差一倍，所以本模块的 [`CosWave::a_pi`] 取 `1/freq`，与
//!   [`crate::superconductor::rabi::rabi_amp::Cos::a_pi`] 的 `1/(2·freq)` 不是同一个式子。
//! - 最低阶这档**不是**用来定 π 幅度的：它担的是**升阶窗口的尺度**——周期既然问得出来，
//!   `pairs` 阶的半宽取半个周期（`A_π/(2·pairs)`）就恰好框住一个谷、又不把隔壁框进来，整条
//!   升阶链不必靠经验指数。标定值取最高阶的谷心，不取这里。
//!
//! 线型锚定在零驱动处的 |0> 上：零驱动时脉冲面积为零、末态原样回到 |0>，故 `P1(0) = 0`、没有
//! 自由常数项（与 rabi 同）。这也意味着**幅度轴须自零驱动起步**，否则锚点失效。
//!
//! P1 一律由**标定过的**两态中心投影：谷底的残余正是这一档要看的量，按数据自身的极值归一化
//! 会把它强行拉到 0、抹掉那点残余（这也正是本模块只收 [`StateCenters`] 而不收 `Option` 的
//! 理由）。取向随之由 g0 → 0、g1 → 1 定死——两个取向里选残差小的那个，只是替"标定次序给反了"
//! 兜底，不是让数据自己定轴。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{argmax, local_maxima, orient, spectrum};
use lmfit::{Curve, Model, ModelParams, ModelResult};
use rayon::prelude::*;
use std::f64::consts::PI;

pub use lmfit::Complex64;

/// 两参数余弦所需的最少幅度点数：4 点 − 2 参数 = 2 个残余自由度（与 baseline `fit_cos` 的
/// `n < 4` 同值）。
const MIN_POINTS: usize = 4;

/// 半幅的初值：这条曲线在 [0, 1] 之间往返，峰值 1 ⇒ 半幅 0.5。
const AMP_P0: f64 = 0.5;

// =========================================================================
// 线型
// =========================================================================

/// `2·pairs` 发同号脉冲下 P1 随幅度的线型：零驱动处锚在 |0> 上的余弦。
///
/// 与 rabi 的 `Cos` 是同一个形状，差别只在派生量 `a_pi` 的式子（见模块文档）。
#[derive(Model, Clone, Copy, Debug)]
pub struct CosWave {
    /// 振荡频率，单位是幅度的倒数
    ///
    /// 线型对 `freq` 是偶函数（`cos(2π·(−f)·A) = cos(2π·f·A)`），符号纯属标记差异——把下界
    /// 钉在 0 就把这条规范一次定死，`a_pi = 1/freq` 也随之恒正，不必事后取绝对值。
    #[param(value = 1.0, min = 0.0)]
    pub freq: f64,
    /// 半幅（峰值的一半）
    ///
    /// 符号区分线型落在 0 的哪一侧：取正时自 0 上升，取负时整条不超过 0——它正是取向判据，
    /// 不能规范化，也不设界。
    #[param(value = 0.5)]
    pub amp: f64,
    /// π 幅度（派生量：`1/freq`）
    ///
    /// **本曲线**的首个极小：`2·pairs` 发同号脉冲下 `P1 = sin²(pairs·π·A/A_π)`，极小落在
    /// `A_π`、`2·A_π`、`3·A_π`… 深度相同，要的是第一个。标准误由 lmfit 按 delta 方法从协方差
    /// 传播：`Var(a_pi) = (a_pi/freq)²·Var(freq)`。
    #[param(derive)]
    pub a_pi: f64,
}

impl CosWave {
    /// 由初值构造，派生量随后按公式重算。
    ///
    /// 形参:
    ///     freq: 振荡频率（幅度的倒数）
    ///     amp: 半幅
    ///
    /// 返回值:
    ///     派生字段与公式一致的模型
    pub fn new(freq: f64, amp: f64) -> Self {
        let mut model = Self {
            freq,
            amp,
            a_pi: f64::NAN,
        };
        ModelParams::refresh_derive(&mut model);
        model
    }

    /// 派生量：π 幅度 `1/freq`——本节曲线的首个极小处。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     π 幅度，与 `freq` 同单位（幅度的倒数）
    fn a_pi(&self) -> f64 {
        1.0 / self.freq
    }

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

impl Curve for CosWave {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 驱动幅度
    ///
    /// 返回值:
    ///     该幅度处的激发概率
    fn eval(&self, x: f64) -> f64 {
        self.amp * (1.0 - (2.0 * PI * self.freq * x).cos())
    }
}

// =========================================================================
// 初值候选
// =========================================================================

/// 振荡频率的初值候选（与 rabi 的同名函数同源，baseline `_freq_candidates`）。
///
/// 四路各自独立估计同一个量，去重后逐个尝试（见 [`fit_once`] 的调用方），取残差最小者：
///
/// - **傅里叶**：对去均值后的曲线取实频谱，主导谱峰即振荡频率；谱峰位置用三点抛物线插值细化，
///   否则分辨率只有 `1/(n·ΔA)`，三十来点的扫描上误差可到一成，作初值偏粗。
/// - **首个极值**：起点必为 |0> 态（锚点），故第一个极值必是 |1> 处的极大，二者相隔半个周期。
///   这一路单列而非并入下一路：`amps[0]` 这个已知的极值 [`local_maxima`] 报不出来（它不报数组
///   端点），极值间距那一路看不见它。
/// - **极值间距**：相邻的两个极值（必是一极大配一极小）相隔半个周期。
/// - **峰间距**：相邻极大相隔一个完整周期。
///
/// 后三路都对每一对相邻点各出一个候选：阻尼会让间距疏密不均，各段估计本就不同，全列出来交给
/// 残差比较，好过先算个平均或只取两端。
///
/// 四路全空当且仅当曲线无振荡——非恒定的曲线必有非零谱 bin，傅里叶那一路不会缺席。此时返回
/// 空集，拟合会报"无候选"而不是拿一个凑数的初值硬拟合出看似成功的结果。
///
/// 形参:
///     amps: 幅度轴 (n,)，须等间距、单调递增
///     prob: 该取向下的 P1 曲线 (n,)
///
/// 返回值:
///     频率候选，升序去重；曲线无振荡时为空
fn freq_candidates(amps: &[f64], prob: &[f64]) -> Vec<f64> {
    let mut candidates: Vec<f64> = Vec::new();

    // 傅里叶：取主导谱峰（谱轴与面板共用 [`crate::utils::spectrum`] 同一个定义）
    let (spectrum_freqs, spectrum_amps) = spectrum(amps, prob);
    match spectrum_freqs.len() >= 2 {
        true => {
            let peak = argmax(&spectrum_amps);
            // 三点抛物线插值细化，横坐标单位是 bin；谱峰贴边时凑不满以峰为中心的三点窗，
            // 退化为 bin 中心
            let offset = match spectrum_amps
                .get(peak.wrapping_sub(1)..)
                .and_then(|rest| rest.first_chunk::<3>())
            {
                Some([left, mid, right]) => {
                    let denominator = left - 2.0 * mid + right;
                    match denominator != 0.0 {
                        true => 0.5 * (left - right) / denominator,
                        false => 0.0,
                    }
                }
                None => 0.0,
            };
            candidates.push(spectrum_freqs[peak] + offset * (spectrum_freqs[1] - spectrum_freqs[0]));
        }
        false => {}
    }

    // 极值：极大与极小各找一遍，合并后升序
    let peaks = local_maxima(prob);
    let troughs: Vec<usize> = local_maxima(&prob.iter().map(|value| -value).collect::<Vec<f64>>());
    let mut extrema: Vec<usize> = peaks.iter().chain(troughs.iter()).copied().collect();
    extrema.sort_unstable();

    // 首个极值：起点必是 |0>，故第一个极值必是极大，二者相隔半个周期
    match extrema.first() {
        Some(index) => {
            let gap = amps[*index] - amps[0];
            match gap > 0.0 {
                true => candidates.push(1.0 / (2.0 * gap)),
                false => {}
            }
        }
        None => {}
    }

    // 极值间距：相邻极值（一极大配一极小）相隔半个周期
    for pair in extrema.windows(2) {
        let gap = amps[pair[1]] - amps[pair[0]];
        match gap > 0.0 {
            true => candidates.push(1.0 / (2.0 * gap)),
            false => {}
        }
    }

    // 峰间距：相邻极大相隔一个完整周期
    for pair in peaks.windows(2) {
        let gap = amps[pair[1]] - amps[pair[0]];
        match gap > 0.0 {
            true => candidates.push(1.0 / gap),
            false => {}
        }
    }

    candidates.retain(|value| value.is_finite() && *value > 0.0);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup();
    candidates
}

/// 余弦初值候选：频率维见 [`freq_candidates`]，半幅固定为 [`AMP_P0`]。
///
/// 形参:
///     amps: 幅度轴 (n,)
///     prob: 该取向下的 P1 曲线 (n,)
///
/// 返回值:
///     初值列表；无候选时为空
fn p0_candidates(amps: &[f64], prob: &[f64]) -> Vec<CosWave> {
    freq_candidates(amps, prob)
        .into_iter()
        .map(|freq| CosWave::new(freq, AMP_P0))
        .collect()
}

// =========================================================================
// 拟合入口
// =========================================================================

/// 最低阶拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum FactorOneError {
    /// 幅度或 IQ 数据为空。
    EmptyData,
    /// 幅度点数与 IQ 点数不一致。
    LengthMismatch { amps: usize, iq: usize },
    /// 幅度点数不足以约束两参数余弦。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与幅度扫描数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 数据无有效投影值，或无初值候选、全部候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for FactorOneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the amplitude or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { amps, iq } => {
                write!(f, "amplitude points ({amps}) and IQ points ({iq}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} amplitude points cannot constrain a 2-parameter cosine; at least {min} are needed"
            ),
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of amplitude scans ({lines})"
            ),
            Self::AllFitsUnsuccess => write!(
                f,
                "the projection carries no finite value, or no initial-value candidate produced a fit"
            ),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for FactorOneError {}

/// 单条最低阶幅度扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct FactorOneFit {
    /// lmfit 的拟合结果：`model` 即三参数（含派生量 `a_pi`），`params` 带标准误。
    pub result: ModelResult<CosWave>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条。
    ///
    /// 它总是**朝对的那条**（0 端为基态）：胜出的取向若把谱线倒了过来，这里存的就是 `1 − P1`。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次余弦拟合。
///
/// 形参:
///     amps: 幅度轴 (n,)
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    amps: &[f64],
    y: &[f64],
    p0: &CosWave,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<CosWave>, FactorOneError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, amps, weights),
        None => p0.fit(y, amps),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(FactorOneError::Lmfit(err)),
    }
}

/// 单取向拟合：初值候选逐个试，取 `chisqr` 最小者。
///
/// 形参:
///     amps: 幅度轴 (n,)
///     prob: 待拟合的 P1 曲线 (n,)
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果；无候选或全部候选失败时返回错误
fn fit_orient(
    amps: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
) -> Result<ModelResult<CosWave>, FactorOneError> {
    let mut best: Option<ModelResult<CosWave>> = None;
    for p0 in p0_candidates(amps, prob) {
        match fit_once(amps, prob, &p0, sigma) {
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
            // 单候选失败是预期情形，继续扫描其余候选
            Err(FactorOneError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(FactorOneError::AllFitsUnsuccess),
    }
}

/// 最低阶（`pairs = 1`）拟合：IQ 投影为 P1，拟一条零驱动处锚在 |0> 的余弦。
///
/// 取向由**残差**择优——线型没有自由常数项，锚点钉在零驱动处的 |0> 上，标定次序给反了（两个
/// 中心对调）时数据是倒过来的，同一族线型够不着它（最好的结果退化成接近平线），残差比正确
/// 取向大一到两个量级。
///
/// 形参:
///     amps: XY 驱动幅度 (n,)，取与 XY 通道 DAC 满幅的比值（无量纲）；**须自零驱动起步**，
///           否则 `amps[0]` 处不再是 |0> 态，线型的锚点失效
///     iq: 单条幅度扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；**必传**——这一档量的是脉冲打歪多少，谷底的残余正是要看的量，
///             按数据自身极值归一化会把它抹掉（见模块文档）
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.a_pi` 即 π 幅度（本曲线的首个极小 `1/freq`）；
///     数据非法或全部候选未收敛时返回错误
pub fn factor_one_fit(
    amps: &[f64],
    iq: &[Complex64],
    states: &StateCenters,
    sigma: Option<&[f64]>,
) -> Result<FactorOneFit, FactorOneError> {
    if amps.is_empty() || iq.is_empty() {
        return Err(FactorOneError::EmptyData);
    }
    if amps.len() != iq.len() {
        return Err(FactorOneError::LengthMismatch {
            amps: amps.len(),
            iq: iq.len(),
        });
    }
    if amps.len() < MIN_POINTS {
        return Err(FactorOneError::LackPoints {
            points: amps.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, Some(states));
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(FactorOneError::AllFitsUnsuccess);
    }
    // 翻转是 1 − P1，带负号的仿射变换，满量程不变，故两个取向共用同一份 σ_P1
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, Some(states), values)),
        None => None,
    };

    let mut best: Option<(ModelResult<CosWave>, bool)> = None;
    for flipped in [false, true] {
        // 投影反向使归一化变成 1 − P1，这才是倒置的真实形式
        let oriented = orient(&prob, flipped);
        match fit_orient(amps, &oriented, weights.as_deref()) {
            Ok(result) => {
                let take = match &best {
                    Some((current, _)) => result.chisqr < current.chisqr,
                    None => true,
                };
                match take {
                    true => best = Some((result, flipped)),
                    false => {}
                }
            }
            // 某一取向整条拟合不出来是预期情形（例如翻转后的曲线够不着），换另一个取向
            Err(FactorOneError::Lmfit(..)) => {}
            Err(FactorOneError::AllFitsUnsuccess) => {}
            Err(other) => return Err(other),
        }
    }

    match best {
        Some((result, flipped)) => Ok(FactorOneFit {
            result,
            p1: orient(&prob, flipped),
        }),
        None => Err(FactorOneError::AllFitsUnsuccess),
    }
}

/// 批量最低阶拟合：对多条幅度扫描并行执行 [`factor_one_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心由各线
/// **共用**——同一批比特共用一套中心，换标定就再调一次本函数。
///
/// 形参:
///     amps: 公共幅度轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 各态标定中心；必传（同 [`factor_one_fit`]）
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`FactorOneError::BatchSigmaMismatch`]
pub fn factor_one_fit_batch(
    amps: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: &StateCenters,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<FactorOneFit, FactorOneError>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => factor_one_fit(amps, line, states, Some(sigma)),
                None => Err(FactorOneError::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| factor_one_fit(amps, line, states, None))
            .collect(),
    }
}
