//! # T2\*（Ramsey）延时扫描：阻尼余弦线型与拟合
//!
//! baseline `qana` 的 Rust 移植：`strategy/fitting/cosine_damp.py` 的阻尼余弦拟合与
//! `exps/t2_analysis.py` 的 T2 分析合并于此。
//!
//! 物理：两个 π/2 之间让比特自由演化 τ，第二个 π/2 带相位 `2π·Δf·τ`——相位斜坡就是一个频移，
//! 把退相干摊成一条能看得见的条纹：
//!
//! ```text
//! P1(τ) = offset + amplitude·exp(−τ/decay)·cos(2π·frequency·τ + phase)
//! ```
//!
//! 包络的时间常数 `decay` 即 **T2\***。条纹频率本身不是要标的量，它由人为加的 Δf 与比特自己的
//! 残余失谐一起决定，所以在拟合里当自由参数放掉。
//!
//! 与 rabi 的两个差别值得记一笔：
//!
//! - **没有取向翻转循环**。rabi 的线型锚定在零驱动处的 |0>（无自由常数项），`y → 1 − y` 够不着，
//!   残差因此能分辨两种取向；这里的线型带自由 `offset` 与自由 `phase`，而
//!   `1 − y = (1 − offset) + (−amplitude)·exp(−τ/decay)·cos(2π·frequency·τ + phase)` 仍是同一族
//!   （把 `amplitude` 取负即可），翻转被线型自己吸收，两种取向对同一条数据给出同样的残差。
//! - **初值只有一路**（傅里叶），照 baseline：rabi 的极值间距那几路依赖"零驱动处是 |0>、首个极值
//!   必是极大"的锚点，这里起点相位自由，`taus[0]` 不一定是极值，那几路的前提不成立。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{argmax, max_value, mean, min_value, spectrum};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;
use std::f64::consts::PI;

pub use lmfit::Complex64;

/// 五参数阻尼余弦所需的延时时长点数：10 点 − 5 参数 = 5 个残余自由度（与 baseline 的
/// `MIN_SAMPLES` 同值）。
const MIN_POINTS: usize = 10;

/// 相位的初值候选 (rad)。
///
/// 五个维度里**只有相位没法由数据直接读出来**，也不是仿射变换能吸收的量（相位错了，任何常数项
/// 与振幅都补不回来），所以它单独铺开一圈：一圈四点加一个端点，任意真实相位离其中最近的一个不
/// 超过 π/4。与 baseline 的 `PHASE_CANDIDATES` 同表。
const PHASE_CANDIDATES: [f64; 5] = [-PI, -0.5 * PI, 0.0, 0.5 * PI, PI];

/// 衰减时间常数的初值候选，单位是**窗口长度的倍数**。
///
/// 下限保证窗口内看得到明显衰减，上限允许"窗口比衰减还短"的情形——那时阻尼在窗口内几乎看不出来，
/// 真值可能远在窗口之外。与 baseline 的 `DECAY_RATIOS` 同表。
const DECAY_RATIOS: [f64; 4] = [0.2, 0.5, 1.0, 2.0];

/// 条纹本底电平的另一档初值：退相干到头就是完全混合态，读出把它映到 P1 = 0.5。
///
/// 与 `mean(prob)` 并列成两档候选：窗口里装得下几个周期时两者给出的数差不多，装不下时
/// `mean(prob)` 会明显偏离渐近电平，而读出误差让真实渐近值偏离 0.5 时反过来由它兜住。
const OFFSET_P0: f64 = 0.5;

/// 条纹幅度的另一档初值：满对比度的条纹在 [0, 1] 之间往返，半幅就是 0.5。
///
/// 与 `(max(prob) − min(prob))/2` 并列成两档候选：后者取的是数据的极值，窗口里振荡不满
/// 几个周期、或者阻尼明显时极值没被采到，它会偏大；0.5 是那条理想振幅。两档都留着，交给
/// 残差挑。
const AMPLITUDE_P0: f64 = 0.5;

// =========================================================================
// 线型
// =========================================================================

/// 阻尼余弦线型的参数：常数项 + 指数包络的余弦。
///
/// 与 rabi 的 `Cos` 不同，这里**五个参数全自由**：起点电平由 `offset` 承担（Ramsey 的条纹可以
/// 停在任意电平上，不像 rabi 被零驱动处的 |0> 钉在 0），相位与衰减也都要拟。
#[derive(Model, Clone, Copy, Debug)]
pub struct CosDamp {
    /// 常数项：条纹衰减完之后剩下的电平
    #[param(value = 0.5)]
    pub offset: f64,
    /// 振荡项的峰值幅度（带符号）
    ///
    /// 符号与相位是同一件事：`(A, φ)` 与 `(−A, φ + π)` 给出同一条曲线。初值取正、相位候选盖满
    /// 一圈，所以拟合正常会落在正幅度那一支；给负号解也是对的，只是同一族里的另一种写法。
    #[param(value = 0.5)]
    pub amplitude: f64,
    /// 条纹频率 (Hz)
    ///
    /// 它含**人为加的 Δf 与比特自身的残余失谐**两部分，不是本实验要标的量，只当自由参数。
    /// 下界钉在 0：`(f, φ)` 与 `(−f, −φ)` 是同一条曲线，这条规范把两重简并砍掉一重。
    #[param(value = 1.0e6, min = 0.0)]
    pub frequency: f64,
    /// 条纹相位 (rad)
    #[param(value = 0.0)]
    pub phase: f64,
    /// 包络的衰减时间常数 (s)，**即 T2\***
    ///
    /// 下界钉在 0：负的时间常数会让包络随 τ 发散，不是本模型要表达的东西。
    #[param(value = 1.0e-6, min = 0.0)]
    pub decay: f64,
}

impl CosDamp {
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

impl Curve for CosDamp {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 延时，单位 s
    ///
    /// 返回值:
    ///     该延时处的 P1
    fn eval(&self, x: f64) -> f64 {
        self.offset
            + self.amplitude * (-x / self.decay).exp() * (2.0 * PI * self.frequency * x + self.phase).cos()
    }
}

// =========================================================================
// 初值候选
// =========================================================================

/// 条纹频率的初值：FFT 主峰，三点抛物线插值细化（baseline `_freq_candidate`）。
///
/// 去均值后取单边谱（[`crate::utils::spectrum`] 已丢直流，与 baseline 把直流 bin 清零同一口径）。
/// 阻尼余弦的频谱是中心在 ±frequency 的洛伦兹线、关于中心对称，所以衰减只把它展宽、不把峰位
/// 挪走——谱峰仍是无偏的频率初值。不细化的话分辨率只有 `1/(n·Δτ)`，窗口几十微秒时就是几十 kHz，
/// 作初值太粗。
///
/// 形参:
///     taus: 延时轴 (n,)，**须等间距、单调递增**
///     prob: 待拟合的 P1 曲线 (n,)
///
/// 返回值:
///     频率初值 (Hz)；延时轴不等间距或点数不足时是 NaN（谱为空），调用方会因此报"无候选"
fn frequency_candidate(taus: &[f64], prob: &[f64]) -> f64 {
    let (spectrum_freqs, spectrum_amps) = spectrum(taus, prob);
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
            spectrum_freqs[peak] + offset * (spectrum_freqs[1] - spectrum_freqs[0])
        }
        false => f64::NAN,
    }
}

/// 阻尼余弦的初值候选：本底两档 × 幅度两档 × 相位一圈 × 衰减四档，只有频率一路由数据读出
/// （baseline `_p0_candidates` 再加一档本底、一档幅度）。
///
/// - `offset` 取 `mean(prob)`（振荡项在窗口内平均掉大半，剩下的电平主要由常数项决定）与
///   [`OFFSET_P0`]（退相干到头是完全混合态，读出把它映到 0.5）两档；
/// - `amplitude` 取 `(max − min)/2`（曲线能取到的最大摆幅，真值不会超过它）与
///   [`AMPLITUDE_P0`]（满对比度的半幅）两档；
/// - `frequency` 见 [`frequency_candidate`]；
/// - `phase` 铺一圈 [`PHASE_CANDIDATES`]；
/// - `decay` 取窗口长度（`max(taus)`）的 [`DECAY_RATIOS`] 倍。
///
/// 四维各铺各的、取笛卡尔积（不是只把两档配成两对）：一维读偏了不该把另外几维的候选也一起
/// 废掉，多出来的组合由 `chisqr` 挑。
///
/// 形参:
///     taus: 延时轴 (n,)，须等间距、单调递增
///     prob: 待拟合的 P1 曲线 (n,)
///
/// 返回值:
///     初值列表，共 80 组；谱为空（延时轴不合规）时频率维是 NaN，逐组拟合必然失败
fn p0_candidates(taus: &[f64], prob: &[f64]) -> Vec<CosDamp> {
    let window = max_value(taus);
    let offsets = [mean(prob), OFFSET_P0];
    let amplitudes = [0.5 * (max_value(prob) - min_value(prob)), AMPLITUDE_P0];
    let frequency = frequency_candidate(taus, prob);
    let mut candidates = Vec::with_capacity(
        offsets.len() * amplitudes.len() * PHASE_CANDIDATES.len() * DECAY_RATIOS.len(),
    );
    for offset in offsets {
        for amplitude in amplitudes {
            for phase in PHASE_CANDIDATES {
                for ratio in DECAY_RATIOS {
                    candidates.push(CosDamp {
                        offset,
                        amplitude,
                        frequency,
                        phase,
                        decay: ratio * window,
                    });
                }
            }
        }
    }
    candidates
}

// =========================================================================
// 拟合入口
// =========================================================================

/// T2\* 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum RamseyError {
    /// 延时或 IQ 数据为空。
    EmptyData,
    /// 延时点数与 IQ 点数不一致。
    LengthMismatch { taus: usize, iq: usize },
    /// 延时点数不足以约束五参数线型。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与延时扫描数量不一致。
    /// 批量拟合时逐线 `states` 的份数与延时扫描数量不一致。
    BatchStatesMismatch { lines: usize, states: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与延时扫描数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 无初值候选，或全部候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for RamseyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the delay or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { taus, iq } => {
                write!(f, "delay points ({taus}) and IQ points ({iq}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} delay points cannot constrain a 5-parameter damped cosine; at least {min} are needed"
            ),
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of delay scans ({lines})"
            ),
            Self::BatchStatesMismatch { lines, states } => write!(
                f,
                "per-line state-center count ({states}) does not match the number of delay scans ({lines})"
            ),
            Self::AllFitsUnsuccess => write!(f, "no initial-value candidate produced a fit"),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for RamseyError {}

/// 单条延时扫描的拟合结果。
#[derive(Debug, Clone)]
pub struct RamseyFit {
    /// lmfit 的拟合结果：`model` 即五参数，`params` 带标准误。
    pub result: ModelResult<CosDamp>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条，也是报告里画的那条。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次阻尼余弦拟合。
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
    p0: &CosDamp,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<CosDamp>, RamseyError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, taus, weights),
        None => p0.fit(y, taus),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(RamseyError::Lmfit(err)),
    }
}

/// 初值候选逐个试，取 `chisqr` 最小者（baseline `cosine_damp` 的循环）。
///
/// 形参:
///     taus: 延时轴 (n,)
///     prob: 待拟合的 P1 曲线 (n,)
///     sigma: P1 域的逐点不确定度；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果；无候选或全部候选失败时返回错误
fn fit_candidates(
    taus: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
) -> Result<ModelResult<CosDamp>, RamseyError> {
    let mut best: Option<ModelResult<CosDamp>> = None;
    for p0 in p0_candidates(taus, prob) {
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
            // 单候选失败是预期情形（相位/衰减的初值不对），继续扫描其余候选
            Err(RamseyError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(RamseyError::AllFitsUnsuccess),
    }
}

/// T2\* 拟合：IQ 投影为 P1，再拟一条阻尼余弦。
///
/// 复刻 baseline 的 `t2_fit` → `cosine_damp`。取向不必挑：线型带自由常数项与自由相位，`1 − P1`
/// 只是把 `amplitude` 翻个号，仍在同一族里（见模块文档）。
///
/// 形参:
///     taus: 延时轴 (n,)，单位 s；**须等间距**——频率初值由 FFT 给出，它要求等间距
///     iq: 单条延时扫描的平均 IQ (n,)，复数
///     states: 各态标定中心；None 表示未标定，P1 改由数据自身定轴
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.decay` 即 T2\*（单位 s）、`model.frequency` 即条纹频率 (Hz)；
///     数据非法或全部候选未收敛时返回错误
pub fn ramsey_fit(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
) -> Result<RamseyFit, RamseyError> {
    if taus.is_empty() || iq.is_empty() {
        return Err(RamseyError::EmptyData);
    }
    if taus.len() != iq.len() {
        return Err(RamseyError::LengthMismatch {
            taus: taus.len(),
            iq: iq.len(),
        });
    }
    if taus.len() < MIN_POINTS {
        return Err(RamseyError::LackPoints {
            points: taus.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, states);
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(RamseyError::AllFitsUnsuccess);
    }
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };

    match fit_candidates(taus, &prob, weights.as_deref()) {
        Ok(result) => Ok(RamseyFit { result, p1: prob }),
        Err(error) => Err(error),
    }
}

/// 批量 T2\* 拟合：对多条延时扫描并行执行 [`ramsey_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心**逐线给**：
/// 二十颗比特同时测时每颗有自己的读出中心；同一条比特的多档扫描则把同一个中心重复
/// `iq_lines.len()` 遍（逐线一份引用，不做广播约定）。
///
/// 形参:
///     taus: 公共延时轴 (n,)
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 每条线各自的各态标定中心，逐线对应；给定时份数须与 `iq_lines` 相同；
///             None 表示全部未标定
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`states` / `sigmas` 缺少对应份的线分别返回
///     [`RamseyError::BatchStatesMismatch`] / [`RamseyError::BatchSigmaMismatch`]
pub fn ramsey_fit_batch(
    taus: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: Option<&[&StateCenters]>,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<RamseyFit, RamseyError>> {
    iq_lines
        .par_iter()
        .enumerate()
        .map(|(index, line)| {
            let centers = match states {
                Some(list) => match list.get(index) {
                    Some(centers) => Some(*centers),
                    None => {
                        return Err(RamseyError::BatchStatesMismatch {
                            lines: iq_lines.len(),
                            states: list.len(),
                        });
                    }
                },
                None => None,
            };
            match sigmas {
                Some(list) => match list.get(index) {
                    Some(sigma) => ramsey_fit(taus, line, centers, Some(sigma)),
                    None => Err(RamseyError::BatchSigmaMismatch {
                        lines: iq_lines.len(),
                        sigmas: list.len(),
                    }),
                },
                None => ramsey_fit(taus, line, centers, None),
            }
        })
        .collect()
}
