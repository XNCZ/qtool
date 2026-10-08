//! # qspec 比特谱：洛伦兹线型与 P1 投影拟合
//!
//! baseline `qana` 的 Rust 移植：`exps/qspec_analysis.py` 的单谱线拟合（`lorentz` /
//! `_half_max_width` / `_p0_candidates` / `fit_lorentz` / `qspec_fit`）与它所依赖的
//! `exps/iq_norm.py` 投影（`direction` / `p1` / `projection_refs`）合并于此。通量调谐
//! （`flux_tunable` / `qspec_vs_z_*`）不在本模块内。
//!
//! 物理：qspec 扫 XY 驱动频率，共振处比特被激发、读出腔响应随之改变，平均 IQ 因而在
//! 复平面上沿 |0>–|1> 连线移动。这里先把 IQ 投影为激发概率 P1（[`p1`]，两态中心未标定时
//! 由数据自身拟合直线定轴，见 [`crate::superconductor::direction`]），再对 P1 vs 驱动频率
//! 拟合洛伦兹峰：峰中心即比特频率，半高全宽即谱线线宽。旋转波近似下稳态激发概率为
//!
//! ```text
//! P1(Δ) = (Ω²/2) / (Δ² + Γ₂²/4 + Ω²/2)
//! ```
//!
//! 即洛伦兹线型，峰中心落在驱动失谐为零处，观测线宽含驱动功率展宽；P1 的投影相对真实
//! 布居数只差一次仿射变换，而洛伦兹在仿射变换下仍是洛伦兹，故峰中心与线宽不受该变换影响。
//!
//! 线型：`offset + amp·(fwhm/2)² / ((f − fq)² + (fwhm/2)²)`。基线 `offset` 保留为自由
//! 参数、不按"远离共振处为 |0>"锚定到 0：扫描窗有限，端点处的真实值离渐近基线还差
//! `amp·(fwhm/2)²/Δ²`，模型够不到那个电平，错配会被线宽吸收。
//!
//! 线型对镜像 `y ↦ 1 − y` 封闭——`amp` 取负、`offset` 取 `1 − offset` 即得同一条曲线的
//! 镜像——谷形数据（读出轴与常规相反，或谱线本身是吸收谷）只能靠负 `amp` 表达：自定轴
//! 路径下初值候选把镜像一组一并铺上，让谷直接收敛、原样交付（见 [`p0_candidates`]）；
//! 标定路径不铺——那里拟出谷是"标定与数据对不上"的信号，不替它兜底（见 [`qspec_fit`]）。
//!
//! 全部频率量使用 SI 单位 Hz。

use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::{argmax, local_maxima, orient};
use lmfit::{Curve, Model, ModelResult};
use rayon::prelude::*;

pub use lmfit::Complex64;

/// 洛伦兹四个参数的名字，顺序与 [`Lorentz`] 的字段声明序一致。
pub const LORENTZ_NAMES: [&str; 4] = ["fq", "fwhm", "amp", "offset"];

/// 四参数洛伦兹所需的**最少**频点数：6 点 − 4 参数 = 2 个残余自由度。
const MIN_POINTS: usize = 6;

// =========================================================================
// 线型
// =========================================================================

/// 洛伦兹线型的参数（SI 单位）。
///
/// 线型只以 `fwhm²` 出现，正负给出同一条曲线；初值候选恒取正宽度，故拟合结果通常为正。
/// `#[param(value = 0.0)]` 中的 0.0 只是 lmfit 宏要求的占位起始值，真正的初值由调用方
/// 构造时给出。
#[derive(Model, Clone, Copy, Debug)]
pub struct Lorentz {
    /// 峰中心（比特频率），Hz
    #[param(value = 0.0)]
    pub fq: f64,
    /// 半高全宽，Hz（≥ 0，见下方说明）
    ///
    /// 线型只以 `fwhm²` 出现，正负给出同一条曲线——符号是规范（gauge）而非物理。把下界钉在
    /// 0 就把这条规范一次定死：求解器只在正半轴上游走，结果不会出现"负线宽"，也不必像
    /// baseline 那样事后取 `abs`。真正的最优曲线不受影响：负解是正解的镜像，取正后
    /// `chisqr` 逐位相同。
    #[param(value = 1.0, min = 0.0)]
    pub fwhm: f64,
    /// 峰高（峰值相对基线；谷形数据为负）
    ///
    /// 正负是**判据**而不是噪声：谷只能靠负 `amp` 表达，符号因此直接说明投影的哪一端
    /// 映射到了 0（见 [`qspec_fit`]）。不设下界，免得求解器把符号信息压在边界上。
    #[param(value = 1.0)]
    pub amp: f64,
    /// 远离共振处的基线电平
    #[param(value = 0.0)]
    pub offset: f64,
}

impl Lorentz {
    /// 在频率数组上求线型值（画拟合曲线用的密集网格走这里）。
    ///
    /// 形参:
    ///     freqs_hz: 频率数组 (n,)，Hz
    ///
    /// 返回值:
    ///     线型在 `freqs_hz` 上的取值 (n,)
    pub fn at(&self, freqs_hz: &[f64]) -> Vec<f64> {
        freqs_hz.iter().map(|f| self.eval(*f)).collect()
    }
}

impl Curve for Lorentz {
    /// 在单点 `x` 处求线型值。
    ///
    /// 形参:
    ///     x: 频率，Hz
    ///
    /// 返回值:
    ///     该频率处的线型值
    fn eval(&self, x: f64) -> f64 {
        let half = 0.5 * self.fwhm;
        let delta = x - self.fq;
        self.offset + self.amp * half * half / (delta * delta + half * half)
    }
}

// =========================================================================
// IQ 投影
// =========================================================================

// =========================================================================
// 初值候选
// =========================================================================

/// 半高宽度：从下标 `k` 处的极大值向两侧走到半高处的宽度（baseline `_half_max_width`）。
///
/// 一侧走不到半高（峰贴着扫描边界）时该侧取端点，给出的是从峰到该端点的跨度，在边界
/// 情形下偏小。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz，须单调
///     y: 曲线 (n,)
///     k: 极大值所在下标
///
/// 返回值:
///     半高宽度 Hz；峰顶电平非正时无法定义半高，返回 0
pub(crate) fn half_max_width(freqs_hz: &[f64], y: &[f64], k: usize) -> f64 {
    let half = 0.5 * y[k];
    let mut lo = freqs_hz[0];
    for i in (0..k).rev() {
        match y[i] <= half {
            true => {
                lo = freqs_hz[i];
                break;
            }
            false => {}
        }
    }
    let mut hi = freqs_hz[freqs_hz.len() - 1];
    for i in (k + 1)..y.len() {
        match y[i] <= half {
            true => {
                hi = freqs_hz[i];
                break;
            }
            false => {}
        }
    }
    hi - lo
}

/// 宽度候选：各峰位自己的半高宽，钳到 `[一个采样间距, 全扫描范围]`。
///
/// 宽度须落在扫描范围内，否则初值不可行：窄过一个采样间距的峰在数据上无从分辨，宽过
/// 整个扫描窗的线型则与常数基线不可区分。一个都取不到时退回扫描范围的 1/5。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz，须单调
///     prob: 待取候选的曲线 (n,)，共振处为峰
///     positions: 峰位下标
///
/// 返回值:
///     宽度候选，Hz
fn width_candidates(freqs_hz: &[f64], prob: &[f64], positions: &[usize]) -> Vec<f64> {
    let n = freqs_hz.len();
    let span = freqs_hz[n - 1] - freqs_hz[0];
    let df = span / (n - 1) as f64;
    let mut widths = Vec::new();
    for k in positions {
        let width = half_max_width(freqs_hz, prob, *k);
        match width > 0.0 {
            true => widths.push(width.max(df).min(span)),
            false => {}
        }
    }
    match widths.is_empty() {
        true => vec![span / 5.0],
        false => widths,
    }
}

/// 峰位 × 宽度候选的笛卡尔积；幅值与基线固定为 1 与 0。
///
/// 数据经 min-max 归一化，基线钉在 0、峰顶钉在 1，任何别的初值都比它离真值更远；基线
/// 在拟合中是自由的（见 [`Lorentz`]），初值取 0 只是起点。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz
///     positions: 峰位下标
///     widths: 宽度候选，Hz
///
/// 返回值:
///     初值列表，`positions.len() × widths.len()` 个
fn candidates_from(freqs_hz: &[f64], positions: &[usize], widths: &[f64]) -> Vec<Lorentz> {
    let mut out = Vec::with_capacity(positions.len() * widths.len());
    for k in positions {
        for width in widths {
            out.push(Lorentz {
                fq: freqs_hz[*k],
                fwhm: *width,
                amp: 1.0,
                offset: 0.0,
            });
        }
    }
    out
}

/// 初值候选：峰形一组，`mirror` 为真时把镜像（谷形）一组一并铺上，谁胜出由残差定。
///
/// 峰形初值（峰位钉在极大值、`amp = 1`）只够得到共振处为峰的曲线；数据是谷时谷只能靠
/// 负的 `amp` 表达，峰形起点会收敛到"又宽又浅的正洛伦兹"这类残差巨大的局部极小并报
/// 成功。`mirror` 就是给谷补的那一组起点——谷底朝哪边、起点就取哪边的符号，不必事先判定。
///
/// 谷的候选不另起一套几何：曲线取 `1 − y` 后峰、谷互换，[`peak_candidates`] 的峰位/
/// 宽度机制原样复用，候选再经 [`reflect`] 翻回原曲线。
///
/// **只对自定轴路径开**（调用处见 [`qspec_fit`]）：自定轴的朝向本是约定，谷是合法形态；
/// 标定路径的朝向由 g0 → 0、g1 → 1 定死，数据该是峰，拟出谷是"标定与数据对不上"的信号，
/// 铺上镜像候选只会把它盖住。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz
///     prob: 该取向下的 P1 曲线 (n,)
///     mirror: 是否把镜像（谷形）候选一并铺上
///
/// 返回值:
///     初值列表；频点过少或长度不符时为空
fn p0_candidates(freqs_hz: &[f64], prob: &[f64], mirror: bool) -> Vec<Lorentz> {
    match freqs_hz.len() < 2 || prob.len() != freqs_hz.len() {
        true => Vec::new(),
        false => {
            let mut candidates = peak_candidates(freqs_hz, prob);
            match mirror {
                true => candidates.extend(
                    peak_candidates(freqs_hz, &orient(prob, true))
                        .into_iter()
                        .map(reflect),
                ),
                false => {}
            }
            candidates
        }
    }
}

/// 峰形初值：局部极大 ∪ 全局极大，宽度只取全局极大那一处的半高宽。
///
/// 局部极大的判定不含数组端点，而共振落在扫描边界时端点就是峰，故无条件并入全局极大；
/// 噪声底上报出的假峰不必事先甄别，交给拟合后的残差比较剔除。
///
/// 宽度只取一处：baseline 对每个峰各取一个半突起宽度，对拍（四套候选集 × 30 个格子）
/// 表明宽度候选的多寡对结果没有影响——胜者由残差定，多出来的噪声峰候选赢不了——只多算
/// 候选，故收成一处。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz
///     y: 待取候选的曲线 (n,)，共振处为峰
///
/// 返回值:
///     初值列表；无极大值时为空
fn peak_candidates(freqs_hz: &[f64], y: &[f64]) -> Vec<Lorentz> {
    let peak_of_global_max = argmax(y);
    let mut positions = local_maxima(y);
    positions.push(peak_of_global_max);
    let widths = width_candidates(freqs_hz, y, &[peak_of_global_max]);
    candidates_from(freqs_hz, &positions, &widths)
}

/// 峰形初值 → 谷形初值：曲线换成镜像 `1 − y` 时 `(amp, offset) → (−amp, 1 − offset)`，
/// 峰位与宽度不变——峰顶（`L(fq) = 1`）翻成谷底，远离共振的基线翻成峰形基线的镜面。
///
/// 形参:
///     peak: 峰形初值
///
/// 返回值:
///     镜像曲线上的同一组初值
fn reflect(peak: Lorentz) -> Lorentz {
    Lorentz {
        fq: peak.fq,
        fwhm: peak.fwhm,
        amp: -peak.amp,
        offset: 1.0 - peak.offset,
    }
}

// =========================================================================
// 拟合入口
// =========================================================================

/// qspec 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum QspecError {
    /// 频率或 IQ 数据为空。
    EmptyData,
    /// 频率点数与 IQ 点数不一致。
    LengthMismatch { freqs: usize, iq: usize },
    /// 频点数不足以约束四参数线型。
    LackPoints { points: usize, min: usize },
    /// 批量拟合时逐线 `sigmas` 的份数与频率线数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 无初值候选，或全部候选均未收敛。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for QspecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the frequency or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { freqs, iq } => {
                write!(f, "frequency points ({freqs}) and IQ points ({iq}) have different lengths")
            }
            Self::LackPoints { points, min } => write!(
                f,
                "{points} frequency points cannot constrain a 4-parameter Lorentzian; at least {min} are needed"
            ),
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of frequency lines ({lines})"
            ),
            Self::AllFitsUnsuccess => {
                write!(f, "no initial-value candidate produced a fit")
            }
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for QspecError {}

/// qspec 单条谱线的拟合结果。
#[derive(Debug, Clone)]
pub struct QspecFit {
    /// lmfit 的拟合结果：`model` 即四参数，`params` 带标准误，`chisqr` 为 Σ(残差)²。
    pub result: ModelResult<Lorentz>,
    /// 实际参与拟合的 P1 曲线——即 `result` 所拟合的那条曲线，也是报告里画的那条。
    ///
    /// 朝向原样交付、不翻正：自定轴路径下它可能是谷（`amp` 为负），符号就是投影方向
    /// 落点的读数；标定路径下按 g0 → 0、g1 → 1 的朝向（见 [`qspec_fit`]）。
    pub p1: Vec<f64>,
}

/// 以给定初值执行一次洛伦兹拟合（不加权，与 baseline 的 `curve_fit` 一致）。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz
///     y: 待拟合曲线 (n,)，实值
///     p0: 初值
///
/// 返回值:
///     lmfit 的拟合结果；求解器报错时透传
pub(crate) fn fit_once(
    freqs_hz: &[f64],
    y: &[f64],
    p0: &Lorentz,
    sigma: Option<&[f64]>,
) -> Result<ModelResult<Lorentz>, QspecError> {
    let outcome = match sigma {
        Some(weights) => p0.fit_sigma(y, freqs_hz, weights),
        None => p0.fit(y, freqs_hz),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(QspecError::Lmfit(err)),
    }
}

/// 单取向拟合：初值候选逐个试，取 `chisqr` 最小者。
///
/// 形参:
///     freqs_hz: 频率轴 (n,)，Hz
///     prob: 待拟合的 P1 曲线 (n,)，已定向
///     sigma: P1 域的逐点不确定度；None 表示不加权
///     mirror: 初值候选是否铺上镜像（谷形）一组，透传 [`p0_candidates`]
///
/// 返回值:
///     残差最小的拟合结果；无候选或全部候选失败的返回错误
fn fit_orient(
    freqs_hz: &[f64],
    prob: &[f64],
    sigma: Option<&[f64]>,
    mirror: bool,
) -> Result<ModelResult<Lorentz>, QspecError> {
    let mut best: Option<ModelResult<Lorentz>> = None;
    for p0 in p0_candidates(freqs_hz, prob, mirror) {
        match fit_once(freqs_hz, prob, &p0, sigma) {
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
            Err(QspecError::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(QspecError::AllFitsUnsuccess),
    }
}

/// qspec 全流程拟合：IQ 投影为 P1，扫描初值候选，取残差最小者。
///
/// 复刻 baseline 的 `qspec_fit`：由 [`p0_candidates`] 生成初值，逐候选执行 [`fit_once`]，
/// 失败者跳过，返回 `chisqr` 最小的一次拟合结果。候选只喂求解器初值，谁胜出仍由残差定。
///
/// **朝向**：不做翻转，拟合出什么朝向就交付什么朝向。线型带自由基线、对 `y ↦ 1 − p`
/// 封闭，两种朝向残差恒等——"哪边是 |1>"由投影方向决定，不是拟合能判的（baseline 正反
/// 各拟一遍再比残差，胜负也只由浮点噪声定）。自定轴路径下投影方向的正负号本是约定
/// （[`crate::superconductor::direction`] 统一取 I 分量为正），归一化可能把 |0> 一侧映射
/// 到 1，谱线于是倒过来；此时初值里的镜像一组（见 [`p0_candidates`]）让谷直接收敛，结果
/// 就按这个朝向交付——`amp` 为负即谷，符号本身就是"哪一端映射到了 0"的读数，要相反朝向
/// 的调用方自行取 `1 − p1`。标定路径不铺镜像：朝向由 g0 → 0、g1 → 1 定死，数据该是峰，
/// 拟出谷是"标定与数据对不上"的信号，替它兜底只会把问题盖住。
///
/// 形参:
///     freqs_hz: XY 驱动频率数组 (n,)，Hz，取绝对频率（比特频率加扫描失谐）
///     iq: 单条谱线的平均 IQ (n,)，复数
///     states: 各态标定中心；None 表示未标定，P1 改由数据自身定轴
///     sigma: **IQ 域**的逐点测量不确定度（每个实/虚分量的 σ，与 `s21` 同一口径；
///            点是 n 次单发平均时传 `std(shots)/√n`）。给定时按 1/σ² 加权，经
///            [`p1_sigma`] 折算到 P1 空间；None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果，`model.fq` 即比特频率、`model.fwhm` 即谱线线宽；
///     数据非法或全部候选未收敛时返回错误
pub fn qspec_fit(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
) -> Result<QspecFit, QspecError> {
    if freqs_hz.is_empty() || iq.is_empty() {
        return Err(QspecError::EmptyData);
    }
    if freqs_hz.len() != iq.len() {
        return Err(QspecError::LengthMismatch {
            freqs: freqs_hz.len(),
            iq: iq.len(),
        });
    }
    if freqs_hz.len() < MIN_POINTS {
        return Err(QspecError::LackPoints {
            points: freqs_hz.len(),
            min: MIN_POINTS,
        });
    }

    let prob = p1(iq, states);
    if !prob.iter().any(|value| value.is_finite()) {
        return Err(QspecError::AllFitsUnsuccess);
    }
    let weights = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };

    // 镜像（谷形）候选只对自定轴路径铺：那里的朝向本是约定，谷是合法形态；标定路径的
    // 朝向由两个中心定死，拟出谷是"标定与数据对不上"的信号，不该被能拟合谷的初值盖住。
    let mirror = states.is_none();
    let trial = match fit_orient(freqs_hz, &prob, weights.as_deref(), mirror) {
        Ok(result) => result,
        Err(err) => return Err(err),
    };
    Ok(QspecFit {
        result: trial,
        p1: prob,
    })
}

/// 批量 qspec 拟合：对多条谱线并行执行 [`qspec_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。各态标定中心由各线
/// **共用**——同一条比特的多档功率共用一套中心，换比特/换标定就再调一次本函数。
///
/// 形参:
///     freqs_hz: 公共 XY 驱动频率轴 (n,)，Hz
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     states: 各态标定中心；None 表示未标定，P1 由各线自身定轴
///     sigmas: 每条线各自的 IQ 域逐点不确定度，逐线对应；给定时份数须与 `iq_lines`
///             相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`QspecError::BatchSigmaMismatch`]
pub fn qspec_fit_batch(
    freqs_hz: &[f64],
    iq_lines: &[Vec<Complex64>],
    states: Option<&StateCenters>,
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<QspecFit, QspecError>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => qspec_fit(freqs_hz, line, states, Some(sigma)),
                None => Err(QspecError::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| qspec_fit(freqs_hz, line, states, None))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::argmin;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// 比特频率真值，Hz。
    const FQ_HZ: f64 = 4.42e9;
    /// 谱线线宽真值，Hz。
    const FWHM_HZ: f64 = 20e6;

    /// 频率轴：200 MHz 窗、101 点（与使用反馈里的最小复现同一网格）。
    fn freqs() -> Vec<f64> {
        (0..101).map(|k| 4.32e9 + 2e6 * k as f64).collect()
    }

    /// 两态中心（任意夹角的一对）。
    fn centers() -> StateCenters {
        StateCenters::new(vec![Complex64::new(0.20, 0.10), Complex64::new(0.50, 0.40)])
    }

    /// 单位峰高的洛伦兹：中心 [`FQ_HZ`]、线宽 [`FWHM_HZ`]，远端的基线为 0。
    fn lorentz(freqs_hz: &[f64]) -> Vec<f64> {
        let half = 0.5 * FWHM_HZ;
        freqs_hz
            .iter()
            .map(|f| half * half / ((f - FQ_HZ).powi(2) + half * half))
            .collect()
    }

    /// 由 P1 真值合成 IQ：各点落在 g0 → g1 连线上。
    fn synth(prob: &[f64], states: &StateCenters) -> Vec<Complex64> {
        let (g0, g1) = (states.as_slice()[0], states.as_slice()[1]);
        prob.iter().map(|p| g0 + (g1 - g0) * *p).collect()
    }

    /// 标准正态样本（Box–Muller）。
    fn gaussian(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// 谷形数据（吸收谷：共振处 P1 下凹）走自定轴路径：峰形初值够不到谷底，曾被拟成
    /// "又宽又浅的正洛伦兹"且 `success = True`；镜像候选让谷直接收敛，交付的也是这条
    /// 谷——朝向不翻正，`amp` 为负。
    #[test]
    fn valley_on_auto_axis_recovers_truth() {
        let freqs_hz = freqs();
        let states = centers();
        let prob: Vec<f64> = lorentz(&freqs_hz).iter().map(|value| 1.0 - 0.7 * value).collect();
        let iq = synth(&prob, &states);

        let fit = match qspec_fit(&freqs_hz, &iq, None, None) {
            Ok(fit) => fit,
            Err(err) => panic!("谷形数据不该拟合失败：{err}"),
        };
        assert!((fit.result.model.fq - FQ_HZ).abs() < 1e6, "{}", fit.result.model.fq);
        assert!((fit.result.model.fwhm - FWHM_HZ).abs() < 1e6, "{}", fit.result.model.fwhm);
        // 朝向原样交付：amp 为负，p1 的极小值落在共振处
        assert!(fit.result.model.amp < 0.0, "{}", fit.result.model.amp);
        let bottom = argmin(&fit.p1);
        assert!((freqs_hz[bottom] - FQ_HZ).abs() < 2e6, "{}", freqs_hz[bottom]);
    }

    /// 镜像候选只对自定轴路径铺：同一份谷形曲线，`mirror = true` 的候选集含谷底
    /// （`amp` 为负）、峰形一组也保留；`mirror = false`（标定路径）只有峰形一组。
    #[test]
    fn mirror_candidates_are_opt_in() {
        let freqs_hz = freqs();
        let prob: Vec<f64> = lorentz(&freqs_hz).iter().map(|value| 1.0 - 0.7 * value).collect();

        let mirrored = p0_candidates(&freqs_hz, &prob, true);
        assert!(
            mirrored
                .iter()
                .any(|p| p.amp < 0.0 && (p.fq - FQ_HZ).abs() < 2e6),
            "{mirrored:?}"
        );
        assert!(mirrored.iter().any(|p| p.amp > 0.0), "{mirrored:?}");

        let peaks_only = p0_candidates(&freqs_hz, &prob, false);
        assert!(peaks_only.iter().all(|p| p.amp > 0.0), "{peaks_only:?}");
    }

    /// 峰形数据回归：两种路径都回收真值、`amp` 为正（谷候选不抢胜）。
    #[test]
    fn peak_still_recovers_on_both_paths() {
        let freqs_hz = freqs();
        let states = centers();
        let iq = synth(&lorentz(&freqs_hz), &states);

        for outcome in [
            qspec_fit(&freqs_hz, &iq, None, None),
            qspec_fit(&freqs_hz, &iq, Some(&states), None),
        ] {
            let fit = match outcome {
                Ok(fit) => fit,
                Err(err) => panic!("峰形数据不该拟合失败：{err}"),
            };
            assert!((fit.result.model.fq - FQ_HZ).abs() < 1e6, "{}", fit.result.model.fq);
            assert!((fit.result.model.fwhm - FWHM_HZ).abs() < 1e6, "{}", fit.result.model.fwhm);
            assert!(fit.result.model.amp > 0.0, "{}", fit.result.model.amp);
        }
    }

    /// 带噪声的谷形数据走自定轴路径同样回收：20 MHz 的线宽在复高斯噪声（每分量 σ = 0.01）
    /// 下仍应落在 3 MHz 内；朝向原样交付，`amp` 为负。
    #[test]
    fn noisy_valley_recovers_truth() {
        let freqs_hz = freqs();
        let states = centers();
        let prob: Vec<f64> = lorentz(&freqs_hz).iter().map(|value| 1.0 - 0.7 * value).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let iq: Vec<Complex64> = synth(&prob, &states)
            .iter()
            .map(|z| z + Complex64::new(0.01 * gaussian(&mut rng), 0.01 * gaussian(&mut rng)))
            .collect();

        let fit = match qspec_fit(&freqs_hz, &iq, None, None) {
            Ok(fit) => fit,
            Err(err) => panic!("带噪声的谷形数据不该拟合失败：{err}"),
        };
        assert!((fit.result.model.fq - FQ_HZ).abs() < 1e6, "{}", fit.result.model.fq);
        assert!((fit.result.model.fwhm - FWHM_HZ).abs() < 3e6, "{}", fit.result.model.fwhm);
        assert!(fit.result.model.amp < 0.0, "{}", fit.result.model.amp);
    }
}
