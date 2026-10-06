//! 超导量子器件的分析与拟合。
//!
//! 目前包含 S21 谐振腔模型及其拟合流程（[`s21`]）与比特谱（qspec）的洛伦兹拟合（[`qspec`]）。
//!
//! 读出 IQ 平面的公共词汇放在本模块：各态标定中心 [`StateCenters`]，以及把 IQ 沿
//! |0>–|1> 连线投影归一化成激发概率的 [`p1`]（自定轴时用到的主轴拟合见 [`direction`]）。
//! baseline 里这几个量在 `exps/iq_norm.py`，被 qspec / rabi / T1 / jazz 共用。

use crate::utils::{max_value, min_value};
use lmfit::Complex64;

pub mod qspec;

pub mod s21;

/// 读出 IQ 平面上的各态标定中心，**索引即态编号**（0 → |0>，1 → |1>，…）。
///
/// 态编号天然就是下标，故内部只存一段数组：|0> 中心与 |2> 中心是同一个量的不同取值，
/// 拆成具名字段等于把同一件事写两遍，多出来的态还得另开一个容器收。P1 投影只用前两态，
/// 更高激发态一并存着备查。
#[derive(Debug, Clone, PartialEq)]
pub struct StateCenters {
    /// 各态中心，按态序排列
    centers: Vec<Complex64>,
}

impl StateCenters {
    /// 由按态序排列的中心构造。
    ///
    /// 形参:
    ///     centers: 各态中心，`centers[0]` 为 |0>、`centers[1]` 为 |1>，其后依次为 |2>、|3>…
    ///
    /// 返回值:
    ///     标定中心。不足两态也能构造——"标定够不够投影"是使用处的事，由 [`p1`] 按退化
    ///     情形处理（各点 NaN），构造处不替调用方做决定。
    pub fn new(centers: Vec<Complex64>) -> Self {
        Self { centers }
    }

    /// 各态中心的切片，索引即态编号。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///    各态中心，按态序排列
    pub fn as_slice(&self) -> &[Complex64] {
        &self.centers
    }
}

/// 复平面点云的重心与主轴方向（baseline `iq_norm.direction`）。
///
/// 数据理想情况下落在 |0> 与 |1> 两态中心连成的直线上，故对点云拟合直线——取协方差矩阵
/// 的主成分，等价于正交回归（最小化各点到拟合直线的垂距平方和），相比把 Q 对 I 做普通
/// 最小二乘不要求直线接近水平。2×2 对称阵的主特征向量有闭式解，不需要特征分解库。
///
/// 形参:
///     iq: 复数 IQ 数组 (n,)
///
/// 返回值:
///     (重心, 单位方向)。空输入给出 NaN 重心与任意方向，调用方的投影随后整体退化为 NaN；
///     点云退化（所有点重合）时方向无意义，但同样由调用方的极值判据拦下，不在此处分支。
pub fn direction(iq: &[Complex64]) -> (Complex64, Complex64) {
    let n = iq.len();
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for z in iq {
        sum_re += z.re;
        sum_im += z.im;
    }
    let mean_re = sum_re / n as f64;
    let mean_im = sum_im / n as f64;

    let (mut cov_ii, mut cov_iq, mut cov_qq) = (0.0, 0.0, 0.0);
    for z in iq {
        let d_re = z.re - mean_re;
        let d_im = z.im - mean_im;
        cov_ii += d_re * d_re;
        cov_iq += d_re * d_im;
        cov_qq += d_im * d_im;
    }

    // 2×2 对称阵 [[a, b], [b, c]] 的主特征向量方向为 ½·atan2(2b, a − c)
    let theta = 0.5 * (2.0 * cov_iq).atan2(cov_ii - cov_qq);
    let (sin, cos) = theta.sin_cos();
    // 主成分的正负号本身任意（特征分解只定方向、不定朝向），统一取 I 分量为正、
    // I 分量恰为零时取 Q 分量为正，使同一份数据在任何后端下得到同一个轴
    let unit = Complex64::new(cos, sin);
    let unit = match cos == 0.0 {
        true => match sin < 0.0 {
            true => -unit,
            false => unit,
        },
        false => match cos < 0.0 {
            true => -unit,
            false => unit,
        },
    };
    (Complex64::new(mean_re, mean_im), unit)
}

/// 两态 IQ 投影归一化：|0> 映射 0、|1> 映射 1（baseline `iq_norm.p1`）。
///
/// 给定各态中心时沿前两态的连线 `g0 → g1` 投影：`P1 = ((z − g0)·(g1 − g0)) / |g1 − g0|²`，
/// 投影值直接作为概率——这条路径不做 min-max 归一化，噪声涨落可能越出 [0, 1]。更高激发态
/// （|2>、|3>…）与 P1 无关，带上来只是备查。
///
/// `states` 取 None 时改由数据自身定轴：对 `iq` 在复平面拟合直线、取拟合方向为投影轴
/// （见 [`direction`]），再按投影的极值归一化——最大映射 1、最小映射 0。直线的哪一端是
/// |1> 无法由数据判定，取向为约定（I 分量为正），需要相反取向时在调用侧取 `1 − p1(..)`。
///
/// 形参:
///     iq: 复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定，改由数据自身定轴
///
/// 返回值:
///     与 `iq` 等长的 P1 数组。标定不足两态、两态中心重合（`|g1 − g0| = 0`）、或投影
///     退化为单点（投影极值相等）时各点为 NaN
pub fn p1(iq: &[Complex64], states: Option<&StateCenters>) -> Vec<f64> {
    match states {
        Some(centers) => match centers.as_slice() {
            // 投影只用前两态；`span` 是 `g1 − g0`，`denom` 为其模方
            [zero, one, ..] => {
                let span = *one - *zero;
                let denom = span.norm_sqr();
                iq.iter()
                    .map(|z| {
                        let d = z - *zero;
                        let proj = (d.re * span.re + d.im * span.im) / denom;
                        match denom > 0.0 {
                            true => proj,
                            false => f64::NAN,
                        }
                    })
                    .collect()
            }
            // 标定不足两态：投影轴无定义
            [] | [_] => vec![f64::NAN; iq.len()],
        },
        None => {
            let (origin, unit) = direction(iq);
            let proj: Vec<f64> = iq
                .iter()
                .map(|z| {
                    let d = z - origin;
                    d.re * unit.re + d.im * unit.im
                })
                .collect();
            let lo = min_value(&proj);
            let span = max_value(&proj) - lo;
            proj.iter()
                .map(|value| match span > 0.0 {
                    true => (value - lo) / span,
                    false => f64::NAN,
                })
                .collect()
        }
    }
}
