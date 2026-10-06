//! # S21 谐振腔模型与拟合
//!
//! baseline `qana` 的 Rust 移植：`src/cardano.rs` 的解析求根（改由 [`roots`]
//! crate 承担，物理分支选择留在本模块）、`src/s21.rs` 的模型与解析雅可比、
//! `exps/s12_analysis.py` 的初值策略与拟合流程合并于此，求解器用 [`lmfit`] crate。
//!
//! 模型形式（与 baseline 逐项对应）：
//!
//! ```text
//! S21 = Zc + (A·cos(2πfτ) − j·B·sin(2πfτ))·e^{−jφ}·res
//! res = 1 − (Ql/Qc·e^{jθ})/((1 + 2j·y)·cosθ)
//! ```
//!
//! 有效失谐 y 由 Duffing 隐式方程 `y = y0 + ap/(1+4y²)`、`y0 = Ql·(f−fr)/fr`
//! 确定，解析求根见 solve_detuning_y。偏导经隐函数定理穿过 y 传导：
//! `∂y/∂p = −(∂F/∂p)/(∂F/∂y)`，其中 `∂F/∂y = 1 + 8·ap·y/(1+4y²)²`；y 复用
//! 求根结果，导数不需要再次求根。
//!
//! 拟合入口 [`s21_fit`] 复刻 baseline 的 `s12_fit`：由 estimate 从数据读出
//! 初值候选，逐候选执行 fit_once，取残差最小者。`estimate` 与 baseline 有一处
//! 有意差异：notch 的 θ 改由代数圆拟合的几何唯一确定（见 estimate 的文档），
//! 其余估计量与 baseline 一致。`qi` 与 `kappa_ex` 是派生量，经 `#[param(derive)]`
//! 声明，值由同名方法重算，标准误由 lmfit 按 delta 方法从协方差传播。
//!
//! 全部物理量使用 SI 单位：频率 Hz、时间 s、相位 rad；`JAC_NAMES` 的列序与
//! 结构体字段声明序一致，前 11 个字段即拟合参数。

use crate::utils::{argmax, argmin, linear_fit, max_value, mean, median, min_value, unwrap_phase};
use lmfit::{ComplexCurve, ComplexResult, Model, ModelParams, PartialValues};
use rayon::prelude::*;
use roots::find_roots_cubic;
use std::f64::consts::PI;

pub use lmfit::Complex64;

/// 雅可比矩阵的列顺序，与 baseline 的 `JAC_NAMES` 一致（baseline 中记作
/// `A`/`B` 的两列在此对应字段 `a`/`b`）。
pub const JAC_NAMES: [&str; 11] = [
    "fr", "ql", "qc", "theta", "ap", "tau", "a", "b", "phi", "zc_re", "zc_im",
];

/// S21 模型的全部参数（SI 单位）。
///
/// 前 11 个字段是按 [`JAC_NAMES`] 序排列的拟合参数；末尾两个是派生量，
/// 值由同名方法给出、不参与拟合。`#[param(value = 0.0)]` 中的 0.0 只是
/// lmfit 宏要求的占位起始值，真正的初值由调用方构造时给出。
#[derive(Model, Clone, Copy, Debug)]
pub struct S21Model {
    /// 谐振频率，Hz
    #[param(value = 0.0)]
    pub fr: f64,
    /// 有载品质因数
    #[param(value = 0.0)]
    pub ql: f64,
    /// 耦合品质因数
    #[param(value = 0.0)]
    pub qc: f64,
    /// notch 相位，rad
    #[param(value = 0.0)]
    pub theta: f64,
    /// Duffing 非线性参数
    #[param(value = 0.0)]
    pub ap: f64,
    /// 背景延迟，s
    #[param(value = 0.0)]
    pub tau: f64,
    /// cos 支背景幅度
    #[param(value = 0.0)]
    pub a: f64,
    /// sin 支背景幅度
    #[param(value = 0.0)]
    pub b: f64,
    /// 背景相位，rad
    #[param(value = 0.0)]
    pub phi: f64,
    /// 常数复偏置实部
    #[param(value = 0.0)]
    pub zc_re: f64,
    /// 常数复偏置虚部
    #[param(value = 0.0)]
    pub zc_im: f64,
    /// 内部品质因数（派生量：`1 / (1/ql − cosθ/qc)`）
    #[param(derive)]
    pub qi: f64,
    /// 外部耦合率（派生量：`fr / qc`），Hz
    #[param(derive)]
    pub kappa_ex: f64,
}

impl S21Model {
    /// 按 [`JAC_NAMES`] 的顺序收集 11 个拟合参数。
    ///
    /// 返回值:
    ///     长度 11 的参数数组
    pub fn to_array(self) -> [f64; 11] {
        [
            self.fr, self.ql, self.qc, self.theta, self.ap, self.tau, self.a, self.b, self.phi,
            self.zc_re, self.zc_im,
        ]
    }

    /// 由 11 元数组构造参数，派生量随后按公式重算。
    ///
    /// 形参:
    ///     values: 按 [`JAC_NAMES`] 序排列的 11 个拟合参数
    ///
    /// 返回值:
    ///     参数结构；未知长度的输入返回 None
    pub fn from_array(values: &[f64]) -> Option<Self> {
        if values.len() != 11 {
            return None;
        }
        let mut fit_values = [0.0_f64; 11];
        for (slot, value) in fit_values.iter_mut().zip(values.iter()) {
            *slot = *value;
        }
        Some(Self::from_fit_array(&fit_values))
    }

    /// 由定长数组构造参数，派生量随后按公式重算。
    ///
    /// 形参:
    ///     values: 按 [`JAC_NAMES`] 序排列的 11 个拟合参数
    ///
    /// 返回值:
    ///     派生字段与公式一致的参数结构
    pub fn from_fit_array(values: &[f64; 11]) -> Self {
        let mut params = Self {
            fr: values[0],
            ql: values[1],
            qc: values[2],
            theta: values[3],
            ap: values[4],
            tau: values[5],
            a: values[6],
            b: values[7],
            phi: values[8],
            zc_re: values[9],
            zc_im: values[10],
            qi: f64::NAN,
            kappa_ex: f64::NAN,
        };
        ModelParams::refresh_derive(&mut params);
        params
    }

    /// 派生量：内部品质因数。
    ///
    /// 返回值:
    ///     由 `ql`、`qc`、`theta` 按 baseline 表达式算出的 Qi
    fn qi(&self) -> f64 {
        1.0 / (1.0 / self.ql - self.theta.cos() / self.qc)
    }

    /// 派生量：外部耦合率。
    ///
    /// 返回值:
    ///     `fr / qc`，Hz
    fn kappa_ex(&self) -> f64 {
        self.fr / self.qc
    }
}

// =========================================================================
// Duffing 隐式方程的解析求根
// =========================================================================

/// 求 Duffing 隐式方程 `y = y0 + ap/(1+4y²)` 的有效失谐解。
///
/// 隐式方程等价于三次方程 `4y³ − 4y0·y² + y − (y0+ap) = 0`，实根由
/// [`roots::find_roots_cubic`] 解析给出（该 crate 在全部网格上与 baseline 的
/// Cardano/Viète 实现一致到 3e-15，退化重根处更精确）；物理分支取最接近 y0 的
/// 实根——双稳态区存在三个实根，只有最接近线性失谐的那支是实际响应的分支。
///
/// 形参:
///     y0: 线性失谐 ql·(f−fr)/fr
///     ap: Duffing 非线性参数
///
/// 返回值:
///     有效失谐 y；ap=0 时退化为 y0（此时三次方程仅有一个实根）
pub(crate) fn solve_detuning_y(y0: f64, ap: f64) -> f64 {
    // 4y³ − 4y0·y² + y − (y0+ap) = 0
    let polynomial_roots = find_roots_cubic(4.0, -4.0 * y0, 1.0, -(y0 + ap));
    let mut best = f64::NAN;
    let mut best_distance = f64::INFINITY;
    for root in polynomial_roots.as_ref() {
        let distance = (*root - y0).abs();
        if distance < best_distance {
            best_distance = distance;
            best = *root;
        }
    }
    best
}

// =========================================================================
// 模型与解析雅可比
// =========================================================================

/// 在单个频点上求值复数 S21 模型。
///
/// 形参:
///     f: 读出频率，Hz
///     p: 模型参数
///
/// 返回值:
///     该频率处的复数 S21
pub fn model_at(f: f64, p: &S21Model) -> Complex64 {
    Complex64::new(p.zc_re, p.zc_im) + background_at(f, p) * notch_at(f, p)
}

/// notch 因子 `1 − (Ql/Qc·e^{jθ})/((1+2j·y)·cosθ)`，不含背景与常数偏置。
///
/// 形参:
///     f: 读出频率，Hz
///     p: 模型参数
///
/// 返回值:
///     该频率处的 notch 因子
pub(crate) fn notch_at(f: f64, p: &S21Model) -> Complex64 {
    let y = solve_detuning_y(p.ql * (f - p.fr) / p.fr, p.ap);
    let cos_t = p.theta.cos();
    let kappa = Complex64::from_polar(p.ql / p.qc / cos_t, p.theta);
    Complex64::new(1.0, 0.0) - kappa / Complex64::new(1.0, 2.0 * y)
}

/// 驻波背景因子 `(A·cos(2πfτ) − j·B·sin(2πfτ))·e^{−jφ}`，不含常数偏置。
///
/// 形参:
///     f: 读出频率，Hz
///     p: 模型参数
///
/// 返回值:
///     该频率处的背景因子
pub(crate) fn background_at(f: f64, p: &S21Model) -> Complex64 {
    let arg = 2.0 * PI * f * p.tau;
    (p.a * arg.cos() - Complex64::i() * p.b * arg.sin()) * Complex64::from_polar(1.0, -p.phi)
}

/// 在单个频点上求模型对 11 个参数的解析偏导。
///
/// 形参:
///     f: 读出频率，Hz
///     p: 模型参数
///
/// 返回值:
///     长度 11 的复数偏导数组，顺序与 [`JAC_NAMES`] 一致
pub fn jacobian(f: f64, p: &S21Model) -> [Complex64; 11] {
    // 隐式方程的解与隐函数定理所需的中间量
    let y = solve_detuning_y(p.ql * (f - p.fr) / p.fr, p.ap);
    let quad = 1.0 + 4.0 * y * y;
    let d_impl = 1.0 + 8.0 * p.ap * y / (quad * quad);
    let u = Complex64::new(1.0, 2.0 * y);
    let cos_t = p.theta.cos();
    let kappa = Complex64::from_polar(p.ql / p.qc / cos_t, p.theta);
    let res = Complex64::new(1.0, 0.0) - kappa / u;

    // 背景项与相位因子
    let arg = 2.0 * PI * f * p.tau;
    let (cos_tau, sin_tau) = (arg.cos(), arg.sin());
    let e = Complex64::from_polar(1.0, -p.phi);
    let bg = (p.a * cos_tau - Complex64::i() * p.b * sin_tau) * e;

    // ∂y/∂p（隐函数定理）
    let dy_dfr = -p.ql * f / (p.fr * p.fr * d_impl);
    let dy_dql = (f - p.fr) / (p.fr * d_impl);
    let dy_dap = 1.0 / (quad * d_impl);

    // ∂res/∂p = −(∂K/∂p)/u + 2j·K·(∂y/∂p)/u²
    let u2 = u * u;
    let dres_fr = Complex64::i() * 2.0 * kappa * dy_dfr / u2;
    let dres_ql = -(kappa / p.ql) / u + Complex64::i() * 2.0 * kappa * dy_dql / u2;
    let dres_qc = (kappa / p.qc) / u;
    let dres_theta = -(Complex64::i() * p.ql / (p.qc * cos_t * cos_t)) / u;
    let dres_ap = Complex64::i() * 2.0 * kappa * dy_dap / u2;

    let w = 2.0 * PI * f;
    let one = Complex64::new(1.0, 0.0);
    [
        bg * dres_fr,
        bg * dres_ql,
        bg * dres_qc,
        bg * dres_theta,
        bg * dres_ap,
        (p.a * (-w * sin_tau) - Complex64::i() * p.b * (w * cos_tau)) * e * res,
        cos_tau * e * res,
        -Complex64::i() * sin_tau * e * res,
        -Complex64::i() * bg * res,
        one,
        Complex64::i() * one,
    ]
}

impl ComplexCurve for S21Model {
    /// 在单点 `x` 处求复数 S21；自变量为实数频率，以实部承载。
    ///
    /// 形参:
    ///     x: 复数自变量，实部为读出频率（Hz）
    ///
    /// 返回值:
    ///     该频率处的复数 S21
    fn eval(&self, x: Complex64) -> Complex64 {
        model_at(x.re, self)
    }

    /// 解析偏导包：第 11 列之前与 [`JAC_NAMES`] 一一对应，派生量没有偏导。
    ///
    /// 形参:
    ///     x: 复数自变量，实部为读出频率（Hz）
    ///
    /// 返回值:
    ///     复数偏导包；本模型恒为 Some
    fn partials_at(&self, x: Complex64) -> Option<impl PartialValues<Scalar = Complex64>> {
        let d = jacobian(x.re, self);
        Some(S21ModelPartials {
            fr: d[0],
            ql: d[1],
            qc: d[2],
            theta: d[3],
            ap: d[4],
            tau: d[5],
            a: d[6],
            b: d[7],
            phi: d[8],
            zc_re: d[9],
            zc_im: d[10],
        })
    }
}

// =========================================================================
// 初值候选（baseline exps/s12_analysis.py 的 estimate）
// =========================================================================

/// 从数据读出的初值候选。
///
/// 与 baseline `estimate` 的差异（有意为之）：Notch 的 θ 不再从 `angle(circle_data)`
/// 的 5 个启发式采样里取，而由代数圆拟合的几何唯一确定——θ 取 `arg K`，`K` 为
/// 圆轨迹的直径向量（`|K| = 2r`），`Qc = Ql / Re K`，符号随几何自然给出。
/// baseline 的采样法在背景除法不完美（圆不过原点）时会把唯一正确的分支以
/// `mask < 3` 丢弃，32 个候选全部落到错误盆地；fr、AB、τ、φ 的估计与 baseline 相同。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 平均后的复数 IQ 数组 (n,)
///
/// 返回值:
///     11 元初值数组的列表，数量为 2（AB 候选）× 4（fr 候选）；
///     数据为空、圆拟合失败或掩码不足以拟合时可能为空
pub(crate) fn estimate(freqs_hz: &[f64], iq: &[Complex64]) -> Vec<[f64; 11]> {
    let n = freqs_hz.len();
    let mut candidates = Vec::new();
    if n == 0 || iq.len() != n {
        return candidates;
    }

    let abs_data: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
    let phase: Vec<f64> = iq.iter().map(|z| z.arg()).collect();
    let phase_unwrapped = unwrap_phase(&phase);

    // baseline 的 detrend 以索引为自变量（scipy 默认），第二步 polyfit 才按频率；
    // 两步各自成立，此处按原样保留以保持数值一致。
    let index_axis: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let (index_slope, index_intercept) = linear_fit(&index_axis, &phase_unwrapped);
    let angle_data: Vec<f64> = phase_unwrapped
        .iter()
        .enumerate()
        .map(|(i, v)| v - (index_slope * i as f64 + index_intercept))
        .collect();
    let line: Vec<f64> = phase_unwrapped
        .iter()
        .zip(angle_data.iter())
        .map(|(raw, detrended)| raw - detrended)
        .collect();
    let (slope, intercept) = linear_fit(freqs_hz, &line);

    let i_abs_min = argmin(&abs_data);
    let i_angle_min = argmin(&angle_data);
    let i_angle_max = argmax(&angle_data);
    let fr_space = [
        freqs_hz[i_abs_min],
        freqs_hz[i_angle_min],
        freqs_hz[i_angle_max],
        (freqs_hz[i_angle_min] + freqs_hz[i_angle_max]) / 2.0,
    ];

    let ab_space = [mean(&abs_data), median(&abs_data)];
    let tau = -slope / (2.0 * PI);
    let phi = -intercept;

    for ab in ab_space {
        let phase_factor: Vec<Complex64> = freqs_hz
            .iter()
            .map(|f| Complex64::from_polar(1.0, intercept - 2.0 * PI * f * (-slope / (2.0 * PI))))
            .collect();
        let norm_data: Vec<Complex64> = iq
            .iter()
            .zip(phase_factor.iter())
            .map(|(z, factor)| z / ab / factor)
            .collect();
        let circle_data: Vec<Complex64> = norm_data
            .iter()
            .map(|z| Complex64::new(1.0, 0.0) - z)
            .collect();
        let circle_amp: Vec<f64> = circle_data.iter().map(|z| z.norm()).collect();

        // 几何定 θ：circle_data 在复平面上是一条圆（背景残留使其不过原点），
        // 对圆心取角后 arg(z−c) = arg K − 2·atan(2y)，与残留无关。
        let (center, radius) = match circle_fit(&circle_data) {
            Some(fit) => fit,
            None => continue,
        };
        let centered_phase: Vec<f64> = circle_data.iter().map(|z| (z - center).arg()).collect();
        let centered_angle = unwrap_phase(&centered_phase);
        // α 的极值出现在 y→∓∞，其中点即 arg K；|K| = 2r
        let theta_alpha = 0.5 * (max_value(&centered_angle) + min_value(&centered_angle));
        let kappa_abs = 2.0 * radius;
        // 参数 θ 与 K 同向（(θ, Qc) 与 (θ+π, Qc) 给出同一模型，取此分支）
        let theta = (theta_alpha + PI).rem_euclid(2.0 * PI) - PI;
        let max_circle_amp = max_value(&circle_amp);

        let mask: Vec<bool> = (0..n)
            .map(|i| {
                circle_amp[i] > 0.25 * max_circle_amp
                    && (centered_angle[i] - theta_alpha).abs() < 2.8
            })
            .collect();
        let kept: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        if kept.len() < 3 {
            continue;
        }
        // −tan((α−θ_α)/2) = 2y 对频率线性，斜率 2·Ql/fr
        let xs: Vec<f64> = kept.iter().map(|&i| freqs_hz[i]).collect();
        let ys: Vec<f64> = kept
            .iter()
            .map(|&i| -((centered_angle[i] - theta_alpha) / 2.0).tan())
            .collect();
        let (slp, _intercept) = linear_fit(&xs, &ys);

        for fr in fr_space {
            let ql = slp / 2.0 * fr;
            // Qc = Ql / Re K，符号随圆几何自然给出
            let qc = ql / (kappa_abs * theta_alpha.cos());
            candidates.push([fr, ql, qc, theta, 0.0, tau, ab, ab, phi, 0.0, 0.0]);
        }
    }
    candidates
}

/// 代数圆拟合（Kåsa 最小二乘，先减去质心以改善条件数）。
///
/// 形参:
///     points: 复平面上的样本点
///
/// 返回值:
///     (圆心, 半径)；点数不足 3 或方程奇异时返回 None
fn circle_fit(points: &[Complex64]) -> Option<(Complex64, f64)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let x_mean = points.iter().map(|z| z.re).sum::<f64>() / n as f64;
    let y_mean = points.iter().map(|z| z.im).sum::<f64>() / n as f64;
    // 圆方程 (x−a)² + (y−b)² = r² 展开为 u² + v² = 2a·u + 2b·v + c（u、v 为去心坐标）
    let mut matrix = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for z in points {
        let u = z.re - x_mean;
        let v = z.im - y_mean;
        let row = [2.0 * u, 2.0 * v, 1.0];
        let value = u * u + v * v;
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] += row[i] * row[j];
            }
            rhs[i] += row[i] * value;
        }
    }
    let solution = match solve3(matrix, rhs) {
        Some(values) => values,
        None => return None,
    };
    let (a, b, c) = (solution[0], solution[1], solution[2]);
    let center = Complex64::new(a + x_mean, b + y_mean);
    let squared_radius = a * a + b * b + c;
    if squared_radius <= 0.0 {
        return None;
    }
    Some((center, squared_radius.sqrt()))
}

/// 三阶线性方程组，Cramer 法则求解。
///
/// 形参:
///     matrix: 3×3 系数矩阵
///     rhs: 右端向量
///
/// 返回值:
///    解向量；行列式为零时返回 None
fn solve3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if det == 0.0 {
        return None;
    }
    let det_of = |column: usize| -> f64 {
        let mut m = matrix;
        for row in 0..3 {
            m[row][column] = rhs[row];
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    Some([det_of(0) / det, det_of(1) / det, det_of(2) / det])
}

// =========================================================================
// 拟合入口
// =========================================================================

/// S21 拟合的错误类型。
#[derive(Debug, Clone, PartialEq)]
pub enum S21Error {
    /// 频率或 IQ 数据为空。
    EmptyData,
    /// 频率点数与 IQ 点数不一致。
    LengthMismatch { freqs: usize, iq: usize },
    /// 初值候选为空，无从拟合。
    NoCandidates,
    /// 批量拟合时逐线 `sigmas` 的份数与频率线数量不一致。
    BatchSigmaMismatch { lines: usize, sigmas: usize },
    /// 全部初值候选的拟合均未成功。
    AllFitsUnsuccess,
    /// 单次拟合失败，透传 lmfit 的错误。
    Lmfit(lmfit::Error),
}

impl std::fmt::Display for S21Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the frequency or IQ data is empty; cannot run the fit"),
            Self::LengthMismatch { freqs, iq } => {
                write!(f, "frequency points ({freqs}) and IQ points ({iq}) have different lengths")
            }
            Self::NoCandidates => {
                write!(f, "initial-value estimation produced no candidate; nothing to fit")
            }
            Self::BatchSigmaMismatch { lines, sigmas } => write!(
                f,
                "per-line sigma count ({sigmas}) does not match the number of frequency lines ({lines})"
            ),
            Self::AllFitsUnsuccess => write!(f, "all initial-value candidates failed to fit"),
            Self::Lmfit(err) => write!(f, "lmfit failed: {err}"),
        }
    }
}

impl std::error::Error for S21Error {}

/// 以给定初值执行一次 S21 拟合。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 平均后的复数 IQ 数组 (n,)
///     p0: 按 [`JAC_NAMES`] 序排列的 11 个初值
///     sigma: 每个频点的测量不确定度，None 表示不加权
///
/// 返回值:
///     lmfit 的复数拟合结果；`sigma` 为 Some 时按 1/σ² 加权（实虚部同权）。
///     baseline 的 `residual`（Σ|fit−data|²/n）等于结果的 `chisqr / (ndata / 2)`
pub(crate) fn fit_once(
    freqs_hz: &[f64],
    iq: &[Complex64],
    p0: &[f64; 11],
    sigma: Option<&[f64]>,
) -> Result<ComplexResult<S21Model>, S21Error> {
    if freqs_hz.is_empty() || iq.is_empty() {
        return Err(S21Error::EmptyData);
    }
    if freqs_hz.len() != iq.len() {
        return Err(S21Error::LengthMismatch {
            freqs: freqs_hz.len(),
            iq: iq.len(),
        });
    }
    let model = S21Model::from_fit_array(p0);
    let outcome = match sigma {
        Some(weights) => model.fit_sigma(iq, freqs_hz, weights),
        None => model.fit(iq, freqs_hz),
    };
    match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(S21Error::Lmfit(err)),
    }
}

/// S21 全流程拟合：初值候选扫描，取残差最小者。
///
/// 复刻 baseline 的 `s12_fit`：由 estimate 生成候选，逐候选执行 fit_once，
/// 失败者跳过，返回 `chisqr` 最小的一次拟合结果（baseline 对胜者再做一次重拟合
/// 只为绘图，本移植无绘图，直接返回扫描结果）。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 平均后的复数 IQ 数组 (n,)
///     sigma: 每个频点的测量不确定度，None 表示不加权
///
/// 返回值:
///     残差最小的拟合结果；无候选或全部失败时返回错误
pub fn s21_fit(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
) -> Result<ComplexResult<S21Model>, S21Error> {
    if freqs_hz.is_empty() || iq.is_empty() {
        return Err(S21Error::EmptyData);
    }
    if freqs_hz.len() != iq.len() {
        return Err(S21Error::LengthMismatch {
            freqs: freqs_hz.len(),
            iq: iq.len(),
        });
    }
    let candidates = estimate(freqs_hz, iq);
    if candidates.is_empty() {
        return Err(S21Error::NoCandidates);
    }
    let mut best: Option<ComplexResult<S21Model>> = None;
    for p0 in candidates {
        match fit_once(freqs_hz, iq, &p0, sigma) {
            Ok(result) => {
                let take = match &best {
                    Some(current) => result.chisqr < current.chisqr,
                    None => true,
                };
                if take {
                    best = Some(result);
                }
            }
            // 单候选失败是预期情形，继续扫描其余候选
            Err(S21Error::Lmfit(..)) => {}
            Err(other) => return Err(other),
        }
    }
    match best {
        Some(result) => Ok(result),
        None => Err(S21Error::AllFitsUnsuccess),
    }
}

/// 批量 S21 拟合：对多条频率线并行执行 [`s21_fit`]，结果按输入顺序返回。
///
/// 每条线互不依赖，rayon 按当前线程池并行；单线失败不影响其余线。
///
/// 形参:
///     freqs_hz: 公共读出频率轴 (n,)，Hz
///     iq_lines: 每条线的平均复数 IQ，长度均为 n
///     sigmas: 每条线各自的逐点测量不确定度，逐线对应；给定时数量须与
///             `iq_lines` 相同，每份长度须为 n；None 表示全部不加权
///
/// 返回值:
///     与 `iq_lines` 等长的结果列表，逐线对应；`sigmas` 缺少对应份的线返回
///     [`S21Error::BatchSigmaMismatch`]
pub fn s21_fit_batch(
    freqs_hz: &[f64],
    iq_lines: &[Vec<Complex64>],
    sigmas: Option<&[Vec<f64>]>,
) -> Vec<Result<ComplexResult<S21Model>, S21Error>> {
    match sigmas {
        Some(list) => iq_lines
            .par_iter()
            .enumerate()
            .map(|(index, line)| match list.get(index) {
                Some(sigma) => s21_fit(freqs_hz, line, Some(sigma)),
                None => Err(S21Error::BatchSigmaMismatch {
                    lines: iq_lines.len(),
                    sigmas: list.len(),
                }),
            })
            .collect(),
        None => iq_lines
            .par_iter()
            .map(|line| s21_fit(freqs_hz, line, None))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一组量级接近真实数据的参数（与 baseline 测试夹具一致）。
    fn sample_params() -> S21Model {
        S21Model::from_fit_array(&[
            6.8982e9, 8.0e3, -1.4e4, 6.02, 0.1, -2.4e-7, 4.4e9, 4.4e9, 10521.0, -3.0e6, 8.2e6,
        ])
    }

    /// 隐式方程残差 `y − y0 − ap/(1+4y²)`，用于验证解满足原方程。
    fn implicit_residual(y: f64, y0: f64, ap: f64) -> f64 {
        y - y0 - ap / (1.0 + 4.0 * y * y)
    }

    /// 三次方程 `4y³ − 4y0·y² + y − (y0+ap) = 0` 的左侧取值，供二分基准使用。
    fn cubic_value(y: f64, y0: f64, ap: f64) -> f64 {
        4.0 * y * y * y - 4.0 * y0 * y * y + y - (y0 + ap)
    }

    #[test]
    fn ap_zero_reduces_to_linear_detuning() {
        for y0 in [-3.0, -1.0, -0.5, 0.0, 0.25, 0.9, 2.5] {
            let y = solve_detuning_y(y0, 0.0);
            assert!(
                (y - y0).abs() < 1e-14,
                "ap=0 时应退化为 y0，y0={y0} 得到 y={y}"
            );
        }
    }

    #[test]
    fn implicit_equation_is_satisfied() {
        let mut worst = 0.0_f64;
        for i in -40..=40 {
            let y0 = i as f64 * 0.05;
            for j in -20..=20 {
                let ap = j as f64 * 0.05;
                let y = solve_detuning_y(y0, ap);
                let r = implicit_residual(y, y0, ap);
                worst = worst.max(r.abs());
            }
        }
        assert!(worst < 1e-12, "隐式方程最大残差 {worst} 超出容差");
    }

    /// 退化情形：判别式恰为零时存在重根，取根不能只返回那个单根。
    #[test]
    fn degenerate_double_root_is_not_missed() {
        let y = solve_detuning_y(-1.0, 1.0);
        assert!(
            (y + 0.5).abs() < 1e-12,
            "退化重根情形应取 -0.5，得到 {y}"
        );
    }

    /// 扫描 + 二分构成的独立基准：返回三次方程全部实根中最接近 y0 者。
    fn bisection_reference(y0: f64, ap: f64) -> f64 {
        let lo = y0.min(0.0) - 4.0;
        let hi = y0.max(0.0) + 4.0;
        let steps = 40000;
        let dx = (hi - lo) / (steps as f64);
        let mut best = f64::NAN;
        let mut best_dist = f64::INFINITY;
        let mut prev_x = lo;
        let mut prev_f = cubic_value(lo, y0, ap);
        for i in 1..=steps {
            let x = lo + dx * (i as f64);
            let fx = cubic_value(x, y0, ap);
            if prev_f == 0.0 {
                let d = (prev_x - y0).abs();
                if d < best_dist {
                    best_dist = d;
                    best = prev_x;
                }
            } else if prev_f * fx < 0.0 {
                let mut a = prev_x;
                let mut b = x;
                for _ in 0..80 {
                    let m = 0.5 * (a + b);
                    if cubic_value(a, y0, ap) * cubic_value(m, y0, ap) <= 0.0 {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                let root = 0.5 * (a + b);
                let d = (root - y0).abs();
                if d < best_dist {
                    best_dist = d;
                    best = root;
                }
            }
            prev_x = x;
            prev_f = fx;
        }
        best
    }

    #[test]
    fn matches_bisection_reference_on_grid() {
        let mut worst = 0.0_f64;
        let mut worst_at = (0.0_f64, 0.0_f64);
        for i in -60..=60 {
            let y0 = i as f64 * 0.1;
            for j in -30..=30 {
                let ap = j as f64 * 0.1;
                let got = solve_detuning_y(y0, ap);
                let want = bisection_reference(y0, ap);
                let d = (got - want).abs();
                if d > worst {
                    worst = d;
                    worst_at = (y0, ap);
                }
            }
        }
        assert!(
            worst < 1e-9,
            "与二分基准最大偏差 {worst}（y0={}, ap={}）",
            worst_at.0,
            worst_at.1
        );
    }

    #[test]
    fn solution_satisfies_equation_to_machine_precision() {
        let mut worst = 0.0_f64;
        let mut worst_at = (0.0_f64, 0.0_f64);
        for i in -60..=60 {
            let y0 = i as f64 * 0.1;
            for j in -30..=30 {
                let ap = j as f64 * 0.1;
                let y = solve_detuning_y(y0, ap);
                let scale = y.abs().max(y0.abs()).max(1.0);
                // 重根处导数退化为零，残差只能到 sqrt(eps) 量级
                let r = implicit_residual(y, y0, ap).abs() / scale;
                if r > worst {
                    worst = r;
                    worst_at = (y0, ap);
                }
            }
        }
        assert!(
            worst < 1e-13,
            "隐式方程相对残差 {worst}（y0={}, ap={}）",
            worst_at.0,
            worst_at.1
        );
    }

    #[test]
    fn root_is_closest_to_y0() {
        // 三实根区（ap<0 且 y0 较大）下解应当是三个实根中最接近 y0 者
        let y0 = 1.2;
        let ap = -1.5;
        let y = solve_detuning_y(y0, ap);
        let a = -y0;
        let b = 0.25;
        let c = -(y0 + ap) / 4.0;
        let mut roots: Vec<f64> = Vec::new();
        for k in -2000..=2000 {
            let x = k as f64 * 0.005;
            let f = x * x * x + a * x * x + b * x + c;
            let x2 = x + 0.005;
            let f2 = x2 * x2 * x2 + a * x2 * x2 + b * x2 + c;
            if f * f2 < 0.0 {
                roots.push(x);
            }
        }
        let mut expected = f64::NAN;
        let mut best = f64::INFINITY;
        for r in roots {
            let d = (r - y0).abs();
            if d < best {
                best = d;
                expected = r;
            }
        }
        assert!(
            (y - expected).abs() < 0.02,
            "取根未落在最接近 y0 的分支：得到 {y}，期望约 {expected}"
        );
    }

    /// 解析偏导与中心差分逐列比对。
    ///
    /// 步长按参数量级定基：`zc_*` 一类参数可能取零值，若按自身取值定步长，
    /// 模型变化量会低于浮点分辨率而被舍入吞掉。
    #[test]
    fn jacobian_matches_central_difference() {
        let p = sample_params();
        let freqs = [
            6.8982e9 - 5e6,
            6.8982e9 - 3.7e6,
            6.8982e9,
            6.8982e9 + 1.9e6,
            6.8982e9 + 5e6,
        ];
        let scale = [1e9, 1e4, 1e4, 1.0, 1.0, 1e-7, 1e9, 1e9, 1.0, 1e9, 1e9];
        let base = p.to_array();
        let h_scale = 1e-7;

        let mut worst = 0.0_f64;
        let mut worst_name = "";
        for f in freqs {
            let ana = jacobian(f, &p);
            for k in 0..JAC_NAMES.len() {
                let h = base[k].abs().max(scale[k]) * h_scale;
                let mut vp = base;
                vp[k] += h;
                let mut vm = base;
                vm[k] -= h;
                let pp = match S21Model::from_array(&vp) {
                    Some(params) => params,
                    None => panic!("长度应为 11"),
                };
                let pm = match S21Model::from_array(&vm) {
                    Some(params) => params,
                    None => panic!("长度应为 11"),
                };
                let num = (model_at(f, &pp) - model_at(f, &pm)) / (2.0 * h);
                let denom = ana[k].norm().max(num.norm()).max(1e-30);
                let err = (ana[k] - num).norm() / denom;
                if err > worst {
                    worst = err;
                    worst_name = JAC_NAMES[k];
                }
            }
        }
        assert!(
            worst < 1e-5,
            "解析偏导与差分不符，最大相对误差 {worst}（列 {worst_name}）"
        );
    }

    /// 常数列：`zc_re` 的偏导恒为 1，`zc_im` 恒为 j。
    #[test]
    fn constant_columns_are_exact() {
        let p = sample_params();
        let row = jacobian(6.8982e9, &p);
        assert!((row[9] - Complex64::new(1.0, 0.0)).norm() < 1e-15);
        assert!((row[10] - Complex64::new(0.0, 1.0)).norm() < 1e-15);
    }

    /// 参数往返：数组与结构体互转保持数值不变，派生量与公式一致。
    #[test]
    fn params_roundtrip() {
        let p = sample_params();
        let back = match S21Model::from_array(&p.to_array()) {
            Some(params) => params,
            None => panic!("长度应为 11"),
        };
        for (l, r) in p.to_array().iter().zip(back.to_array().iter()) {
            assert!((l - r).abs() < 1e-30);
        }
        assert!(S21Model::from_array(&[0.0; 10]).is_none());
        assert!((back.qi - (1.0 / (1.0 / back.ql - back.theta.cos() / back.qc))).abs() < 1e-30);
        assert!((back.kappa_ex - back.fr / back.qc).abs() < 1e-30);
    }

    /// 夹具参数上的无噪声数据：良好初值的单次拟合应回到真值。
    #[test]
    fn fit_once_recovers_noiseless_model() {
        let truth = sample_params();
        let freqs: Vec<f64> = (0..51)
            .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
            .collect();
        let iq: Vec<Complex64> = freqs.iter().map(|f| model_at(*f, &truth)).collect();

        let mut p0 = truth.to_array();
        p0[0] += 2e5;
        p0[1] *= 0.7;
        p0[2] *= 1.3;
        p0[3] += 0.3;

        let result = match fit_once(&freqs, &iq, &p0, None) {
            Ok(result) => result,
            Err(err) => panic!("单次拟合失败：{err}"),
        };
        assert!(result.success, "拟合未收敛：{}", result.message);
        for k in 0..JAC_NAMES.len() {
            let truth_value = truth.to_array()[k];
            let fitted = result.model.to_array()[k];
            let rel = (fitted - truth_value).abs() / truth_value.abs().max(1.0);
            assert!(
                rel < 1e-6,
                "参数 {} 相对偏差 {rel}：拟合 {fitted} 真值 {truth_value}",
                JAC_NAMES[k]
            );
        }
    }

    /// 全流程（含圆几何定 θ 的初值）在夹具数据上回到真值。
    ///
    /// 这是对 baseline 的行为改进：baseline 的采样式 θ 估计使 32 个候选全部落入
    /// 错误盆地（残差 2.59e17），圆几何定 θ 后候选直达最优。θ 与 Qc 存在
    /// (θ+π, Qc) 的等价分支，比较时按模 π 处理。
    #[test]
    fn s21_fit_recovers_noiseless_model() {
        let truth = sample_params();
        let freqs: Vec<f64> = (0..51)
            .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
            .collect();
        let iq: Vec<Complex64> = freqs.iter().map(|f| model_at(*f, &truth)).collect();

        let result = match s21_fit(&freqs, &iq, None) {
            Ok(result) => result,
            Err(err) => panic!("全流程拟合失败：{err}"),
        };
        assert!(
            result.chisqr < 1e-2,
            "残差未到浮点底：chisqr = {:e}",
            result.chisqr
        );
        let fitted = result.model.to_array();
        let target = truth.to_array();
        for k in 0..JAC_NAMES.len() {
            if k == 3 {
                // θ 的等价分支：差 π 视为同一点
                let delta = (fitted[k] - target[k]).rem_euclid(PI);
                let distance = delta.min(PI - delta);
                assert!(distance < 1e-6, "theta 分支差 {distance}");
                continue;
            }
            if k == 8 {
                // φ 以 2π 为周期
                let delta = (fitted[k] - target[k]).rem_euclid(2.0 * PI);
                let distance = delta.min(2.0 * PI - delta);
                assert!(distance < 1e-6, "phi 周期差 {distance}");
                continue;
            }
            let rel = (fitted[k] - target[k]).abs() / target[k].abs().max(1.0);
            assert!(
                rel < 1e-6,
                "参数 {} 相对偏差 {rel}：拟合 {} 真值 {}",
                JAC_NAMES[k],
                fitted[k],
                target[k]
            );
        }
        // 派生量与公式一致，并出现在参数表中
        assert!(
            (result.model.qi - 1.0 / (1.0 / result.model.ql - result.model.theta.cos() / result.model.qc))
                .abs()
                < 1e-9
        );
        assert!((result.model.kappa_ex - result.model.fr / result.model.qc).abs() < 1e-9);
        match result.params.get("qi") {
            Some(parameter) => assert!(parameter.derive, "qi 应标记为派生参数"),
            None => panic!("派生量 qi 应出现在参数表中"),
        }
    }
}
