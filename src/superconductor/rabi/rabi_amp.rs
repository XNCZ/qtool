//! # Rabi 幅度扫描：余弦线型与拟合
//!
//! baseline `qana` 的 Rust 移植：`exps/rabi_amp_analysis.py` 的余弦拟合流程合并于此
//! （`cos_wave` / `_freq_candidates` / `_p0_candidates` / `fit_cos` / `rabi_amp_fit`）。
//!
//! 物理：固定 XY 脉冲长度、扫驱动幅度，激发概率随脉冲面积（正比于幅度）振荡
//!
//! ```text
//! P1(A) = amp·(1 − cos(2π·freq·A))
//! ```
//!
//! 相邻极大值相隔一个完整 Rabi 周期，故峰间距的倒数即 Rabi 频率；首个极大值处脉冲面积达到
//! π，即 π 脉冲幅 `a_pi = 1/(2·freq)`。这里先把 IQ 投影为激发概率 P1（[`crate::superconductor::p1`]，
//! 两态中心未标定时由数据自身定轴），再对 P1 vs 幅度拟合余弦。
//!
//! **线型锚定在零驱动处的 |0> 态上**：零驱动时比特必处于 |0>、归一化后 P1(0) = 0，故起点钉在
//! 0、半幅为 `amp`（峰值落在 2·amp 处），**没有自由常数项**。这条锚点直接写进线型而不交给常数
//! 项，一是 min-max 归一化本就把最低电平钉在 0、再放一个自由常数只是冗余自由度，二是它同时是
//! **取向判据**的来源：`amp` 取负时线型整条落在 0 以下，贴合不了翻转过来的数据（那条起点是 1），
//! 「是否翻转」因此可由残差分辨。常数项若自由，整体平移一下便与翻转等价，两种取向对同一条
//! 数据给出同样的残差，残差就挑不出高下。
//!
//! 线型不含衰减包络，隐含"各阶 Rabi 极小值都回到同一电平"；扫描区间内振荡次数多、阻尼明显时
//! 残差会出现系统性形状，此时需要补一个包络因子。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{argmax, local_maxima, orient, spectrum};
use lmfit::{Curve, Model, ModelParams, ModelResult};
use rayon::prelude::*;
use std::f64::consts::PI;

pub use lmfit::Complex64;

/// 两参数余弦所需的最少幅度点数：4 点 − 2 参数 = 2 个残余自由度。
const MIN_POINTS: usize = 4;

/// 半幅的初值：数据经 min-max 归一化后 P1 在 [0, 1] 之间往返，峰值 1.0 ⇒ 半幅 0.5。
const AMP_P0: f64 = 0.5;

// =========================================================================
// 线型
// =========================================================================

/// Rabi 余弦线型的参数。
///
/// 线型没有自由常数项：零驱动处的 |0> 态把起点钉在 0（见模块文档），这既是物理锚点，也是
/// 取向判据的来源。
#[derive(Model, Clone, Copy, Debug)]
pub struct Cos {
    /// 振荡频率，单位是幅度的倒数（Hz 每单位幅度）
    ///
    /// 线型对 `freq` 是偶函数（`cos(2π·(−f)·A) = cos(2π·f·A)`），符号纯属标记差异——把下界
    /// 钉在 0 就把这条规范一次定死，`a_pi = 1/(2·freq)` 也随之恒正，不必事后取绝对值。
    #[param(value = 1.0, min = 0.0)]
    pub freq: f64,
    /// 半幅（峰值的一半）
    ///
    /// 符号区分线型落在 0 的哪一侧：取正时自 0 上升，取负时整条不超过 0——它正是取向判据，
    /// 不能规范化，也不设界。
    #[param(value = 0.5)]
    pub amp: f64,
    /// π 脉冲幅（派生量：`1/(2·freq)`）
    ///
    /// 标准误由 lmfit 按 delta 方法从协方差传播：`Var(a_pi) = (a_pi/freq)²·Var(freq)`，
    /// 与 baseline 手写的 `_pi_amp_sem` 逐字等价。
    #[param(derive)]
    pub a_pi: f64,
}

impl Cos {
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

    /// 派生量：π 脉冲幅 `1/(2·freq)`——首个极大值处脉冲面积达到 π。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     π 脉冲幅，与 `freq` 同单位（幅度的倒数）
    fn a_pi(&self) -> f64 {
        0.5 / self.freq
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

impl Curve for Cos {
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

/// 振荡频率的初值候选（baseline `_freq_candidates`）。
///
/// 四路各自独立估计同一个量，去重后逐个尝试（见 [`fit_once`] 的调用方），取残差最小者：
///
/// - **傅里叶**：对去均值后的曲线取实频谱，主导谱峰即振荡频率；谱峰位置用三点抛物线插值细化，
///   否则分辨率只有 `1/(n·ΔA)`，二十来点的扫描上误差可到一成，作初值偏粗。
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
///     初值列表，每项 `[freq, amp]`；无候选时为空
fn p0_candidates(amps: &[f64], prob: &[f64]) -> Vec<Cos> {
    freq_candidates(amps, prob)
        .into_iter()
        .map(|freq| Cos::new(freq, AMP_P0))
        .collect()
}

// =========================================================================
// 拟合入口
// =========================================================================

/// Rabi 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum RabiError {
    /// 幅度或 IQ 数据为空。
    EmptyData,
    /// 幅度点数与 IQ 点数不一致。
    LengthMismatch { amps: usize, iq: usize },
    /// 幅度点数不足以约束两参数余弦。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `states` 的份数与幅度扫描数量不一致。
    BatchStatesMismatch { lines: usize, states: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与幅度扫描数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 无初值候选，或全部候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for RabiError {
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
            Self::BatchStatesMismatch { lines, states } => write!(
                f,
                "per-line state-center count ({states}) does not match the number of amplitude scans ({lines})"
            ),
            Self::AllFitsUnsuccess => write!(f, "no initial-value candidate produced a fit"),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for RabiError {}

/// 单条幅度扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct RabiFit {
    /// lmfit 的拟合结果：`model` 即三参数（含派生量 `a_pi`），`params` 带标准误。
    pub result: ModelResult<Cos>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条，也是报告里画的那条。
    ///
    /// 它总是**朝对的那条**（0 端为基态）：自定轴的朝向本是约定，胜出的取向若把谱线倒了过来，
    /// 这里存的就是 `1 − P1`。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次余弦拟合。
///
/// 形参:
///     amps: 幅度轴 (n,)
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///     sigma: P1 域的逐点不确定度；None 表示不加权（与 baseline 的 `curve_fit` 一致）
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    amps: &[f64],
    y: &[f64],
    p0: &Cos,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<Cos>, RabiError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, amps, weights),
        None => p0.fit(y, amps),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(RabiError::Lmfit(err)),
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
) -> Result<ModelResult<Cos>, RabiError> {
    let mut best: Option<ModelResult<Cos>> = None;
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
            Err(RabiError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(RabiError::AllFitsUnsuccess),
    }
}

/// Rabi 幅度扫描拟合：IQ 投影为 P1，两个取向各拟合一遍，取残差最小者。
///
/// 复刻 baseline 的 `rabi_amp_fit`。取向由**残差**择优——线型锚定在零驱动处的 |0> 态上、
/// 没有自由常数项，翻转过来的数据起点是 1，同一族线型够不着它（最好的结果退化成接近平线），
/// 残差比正确取向大一到两个量级。这与 qspec 那边的判据不同是有原因的：qspec 的线型带自由
/// 基线，对 `y → 1 − y` 封闭，两种取向残差恒等，才必须改用 `amp` 的符号来判。
///
/// 形参:
///     amps: XY 驱动幅度 (n,)，取与 XY 通道 DAC 满幅的比值（无量纲）；**须自零驱动起步**，
///           否则 `amps[0]` 处不再是 |0> 态，线型的锚点失效
///     iq: 单条幅度扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；None 表示未标定，P1 改由数据自身定轴
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.a_pi` 即 π 脉冲幅、`model.freq` 即 Rabi 频率；
///     数据非法或全部候选未收敛时返回错误
pub fn rabi_amp_fit(
    amps: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
) -> Result<RabiFit, RabiError> {
    if amps.is_empty() || iq.is_empty() {
        return Err(RabiError::EmptyData);
    }
    if amps.len() != iq.len() {
        return Err(RabiError::LengthMismatch {
            amps: amps.len(),
            iq: iq.len(),
        });
    }
    if amps.len() < MIN_POINTS {
        return Err(RabiError::LackPoints {
            points: amps.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, states);
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(RabiError::AllFitsUnsuccess);
    }
    // 翻转是 1 − P1，带负号的仿射变换，满量程不变，故两个取向共用同一份 σ_P1
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };

    let mut best: Option<(ModelResult<Cos>, bool)> = None;
    for flip in [false, true] {
        // 投影反向使归一化变成 1 − P1，这才是倒置的真实形式
        let oriented = orient(&prob, flip);
        match fit_orient(amps, &oriented, weights.as_deref()) {
            Ok(result) => {
                let take = match &best {
                    Some((current, _)) => result.chisqr < current.chisqr,
                    None => true,
                };
                match take {
                    true => best = Some((result, flip)),
                    false => {}
                }
            }
            // 某一取向整条拟合不出来是预期情形（例如翻转后的曲线够不着），换另一个取向
            Err(RabiError::Lmfit(..)) => {}
            Err(RabiError::AllFitsUnsuccess) => {}
            Err(other) => return Err(other),
        }
    }

    match best {
        Some((result, flipped)) => Ok(RabiFit {
            result,
            p1: orient(&prob, flipped),
        }),
        None => Err(RabiError::AllFitsUnsuccess),
    }
}

/// 批量 Rabi 拟合：对多条幅度扫描并行执行 [`rabi_amp_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心**逐线给**：
/// 二十颗比特同时测时每颗有自己的读出中心；同一条比特的多档扫描则把同一个中心重复
/// `iq_lines.len()` 遍（逐线一份引用，不做广播约定）。
///
/// 形参:
///     amps: 公共幅度轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 每条线各自的各态标定中心，逐线对应；给定时份数须与 `iq_lines` 相同；
///             None 表示全部未标定
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`states` / `sigmas` 缺少对应份的线分别返回
///     [`RabiError::BatchStatesMismatch`] / [`RabiError::BatchSigmaMismatch`]
pub fn rabi_amp_fit_batch(
    amps: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: Option<&[&StateCenters]>,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<RabiFit, RabiError>> {
    iq_lines
        .par_iter()
        .enumerate()
        .map(|(index, line)| {
            let centers = match states {
                Some(list) => match list.get(index) {
                    Some(centers) => Some(*centers),
                    None => {
                        return Err(RabiError::BatchStatesMismatch {
                            lines: iq_lines.len(),
                            states: list.len(),
                        });
                    }
                },
                None => None,
            };
            match sigmas {
                Some(list) => match list.get(index) {
                    Some(sigma) => rabi_amp_fit(amps, line, centers, Some(sigma)),
                    None => Err(RabiError::BatchSigmaMismatch {
                        lines: iq_lines.len(),
                        sigmas: list.len(),
                    }),
                },
                None => rabi_amp_fit(amps, line, centers, None),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 无噪声的 Rabi 扫描：真值线型直接落到 |0>–|1> 连线上。
    ///
    /// 幅度取满量程 0→1；调用方自己保证频率别把采样打到欠采样区（41 点时
    /// `freq = 4` ⇒ 4 个周期、每周期约 10 点）。
    ///
    /// 形参:
    ///     freq: 真值振荡频率（幅度的倒数）
    ///     points: 幅度点数
    ///
    /// 返回值:
    ///     (幅度轴, 复数 IQ)
    fn clean_scan(freq: f64, points: usize) -> (Vec<f64>, Vec<Complex64>) {
        let amps: Vec<f64> = (0..points).map(|k| k as f64 / (points - 1) as f64).collect();
        let g0 = Complex64::new(0.30, 0.10);
        let g1 = Complex64::new(0.42, 0.30);
        let truth = Cos::new(freq, 0.5);
        let iq = truth
            .at(&amps)
            .iter()
            .map(|p| g0 + (g1 - g0) * p)
            .collect();
        (amps, iq)
    }

    /// 频率候选里应当含有真值（傅里叶那一路的谱峰就落在它附近）。
    #[test]
    fn freq_candidates_contain_truth() {
        let (amps, iq) = clean_scan(4.0, 41);
        let prob = crate::superconductor::p1(&iq, None);
        let candidates = freq_candidates(&amps, &prob);
        assert!(!candidates.is_empty(), "有振荡就该有候选");
        let closest = candidates
            .iter()
            .map(|value| (value - 4.0).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(closest < 0.3, "候选中离真值最近的一个应当足够近：{closest}");
    }

    /// 无噪声拟合回收真值：Rabi 频率与 π 脉冲幅都由 `freq` 派生。
    #[test]
    fn fit_recovers_truth() {
        let (amps, iq) = clean_scan(4.0, 41);
        let outcome = rabi_amp_fit(&amps, &iq, None, None);
        let fit = match outcome {
            Ok(fit) => fit,
            Err(err) => panic!("无噪声数据不该拟合失败：{err}"),
        };
        assert!((fit.result.model.freq - 4.0).abs() < 0.01, "{}", fit.result.model.freq);
        assert!((fit.result.model.a_pi - 1.0 / 8.0).abs() < 1e-4, "{}", fit.result.model.a_pi);
        // 归一化后 P1 在 [0, 1] 往返，峰值 1 ⇒ 半幅 0.5
        assert!((fit.result.model.amp - 0.5).abs() < 0.02, "{}", fit.result.model.amp);
    }

    /// 派生量的标准误按 delta 方法传播：`σ(a_pi) = (a_pi/freq)·σ(freq)`。
    #[test]
    fn a_pi_stderr_matches_delta_method() {
        let (amps, iq) = clean_scan(4.0, 41);
        let fit = match rabi_amp_fit(&amps, &iq, None, None) {
            Ok(fit) => fit,
            Err(err) => panic!("不该拟合失败：{err}"),
        };
        let model = fit.result.model;
        let (sigma_freq, sigma_a_pi) = (
            fit.result.params.get("freq").and_then(|p| p.stderr),
            fit.result.params.get("a_pi").and_then(|p| p.stderr),
        );
        match (sigma_freq, sigma_a_pi) {
            (Some(sf), Some(sa)) => {
                let expected = (model.a_pi / model.freq) * sf;
                assert!((sa - expected).abs() < 1e-12, "{sa} vs {expected}");
            }
            _ => panic!("两个参数都该有标准误"),
        }
    }

    /// 取向判据：把两态中心对调（自定轴的朝向会判反），锚定线型仍应收敛到同一组参数。
    #[test]
    fn inverted_axis_still_recovers() {
        let (amps, iq) = clean_scan(4.0, 41);
        let g0 = Complex64::new(0.30, 0.10);
        let g1 = Complex64::new(0.42, 0.30);
        // 把数据沿投影轴翻过来：相当于把两态中心对调
        let flipped: Vec<Complex64> = iq.iter().map(|z| g0 + g1 - z).collect();
        let fit = match rabi_amp_fit(&amps, &flipped, None, None) {
            Ok(fit) => fit,
            Err(err) => panic!("翻转后的数据也该拟合出来：{err}"),
        };
        assert!((fit.result.model.freq - 4.0).abs() < 0.01, "{}", fit.result.model.freq);
        // 胜出的取向必须让半幅为正——负半幅的线型整条落在 0 以下，够不着非负的数据
        assert!(fit.result.model.amp > 0.0, "{}", fit.result.model.amp);
    }

    /// 点数不足以约束两参数余弦时拒绝拟合，而不是硬凑一个结果。
    #[test]
    fn too_few_points_is_rejected() {
        let (amps, iq) = clean_scan(4.0, 41);
        let outcome = rabi_amp_fit(&amps[..3], &iq[..3], None, None);
        assert_eq!(
            outcome.err(),
            Some(RabiError::LackPoints {
                points: 3,
                min: MIN_POINTS
            })
        );
    }

    /// 长度对不上时拒绝拟合。
    #[test]
    fn length_mismatch_is_rejected() {
        let (amps, iq) = clean_scan(4.0, 41);
        let outcome = rabi_amp_fit(&amps[..10], &iq, None, None);
        assert_eq!(
            outcome.err(),
            Some(RabiError::LengthMismatch {
                amps: 10,
                iq: 41
            })
        );
    }
}
