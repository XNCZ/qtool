//! 超导量子器件的分析与拟合。
//!
//! 目前包含 S21 谐振腔模型及其拟合流程（[`s21`]）与比特谱（qspec）的洛伦兹拟合（[`qspec`]）。
//!
//! 读出 IQ 平面的公共词汇放在本模块：各态标定中心 [`StateCenters`]，把 IQ 沿
//! |0>–|1> 连线投影归一化成激发概率的 [`p1`]（自定轴时用到的主轴拟合见 [`direction`]），
//! 以及把 IQ 域的逐点不确定度送进 P1 空间的 [`p1_sigma`]。
//! baseline 里这几个量在 `exps/iq_norm.py`，被 qspec / rabi / T1 / jazz 共用。

use crate::utils::{max_value, min_value};
use lmfit::Complex64;

pub mod qspec;

pub mod s21;
pub mod rabi;
pub mod ramsey;
pub mod t1;
pub mod t2_echo;
pub mod drag;
pub mod iq;
pub mod bloch;

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
    projection(iq, states).0
}

/// IQ 的逐点不确定度经过 P1 投影后的不确定度。
///
/// 传入的 `sigma` 是**复数 IQ 域**的逐点标准差（每个实/虚分量的 σ，与 `s21` 同一口径：
/// 噪声取圆对称复高斯，两个分量同 σ，故一个实数即可描述）。圆噪声沿任意方向的分量都是
/// 同一个 σ，于是投影只做一件事——除以该路径的 P1 满量程：
///
/// ```text
/// σ_P1 = σ_IQ / L
/// ```
///
/// 标定路径的 `L = |g1 − g0|`，自定轴路径的 `L` 是投影极值之差（见 [`projection`]），
/// 两条路同一个式子，所以本函数是"σ 进入 P1 空间"的唯一入口。
///
/// 只取一阶传播：自定轴路径的满量程本身由数据估出，其涨落不计入。
///
/// 形参:
///     iq: 复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定，同 [`p1`]
///     sigma: IQ 域的逐点标准差，长度须与 `iq` 一致
///
/// 返回值:
///     逐点的 σ_P1，长度与 `sigma` 相同；投影退化（满量程为 0）时各点为 NaN
pub fn p1_sigma(iq: &[Complex64], states: Option<&StateCenters>, sigma: &[f64]) -> Vec<f64> {
    let (_, scale) = projection(iq, states);
    sigma
        .iter()
        .map(|value| match scale > 0.0 {
            true => value / scale,
            false => f64::NAN,
        })
        .collect()
}

/// P1 投影的核心：[`p1`] 的曲线与 [`p1_sigma`] 的满量程。
///
/// 两条路径都是"IQ 沿某条轴投影、再按一个尺度归一化成 0 → 1"，`scale` 即 P1 从 0 走到 1
/// 所跨的 IQ 平面距离。把它一并交出来，"σ 进 P1 空间"才只有一处除法，不必在标定与自定轴
/// 两条路上各写一遍传播公式。
///
/// 形参: 同 [`p1`]
///
/// 返回值:
///     (P1 曲线, 满量程)。退化情形（标定不足两态、两态中心重合、自定轴投影为单点）曲线各点
///     为 NaN、满量程为 0
fn projection(iq: &[Complex64], states: Option<&StateCenters>) -> (Vec<f64>, f64) {
    match states {
        Some(centers) => match centers.as_slice() {
            // 投影只用前两态；`span` 是 `g1 − g0`，`denom` 为其模方
            [zero, one, ..] => {
                let span = *one - *zero;
                let denom = span.norm_sqr();
                let prob: Vec<f64> = iq
                    .iter()
                    .map(|z| {
                        let d = z - *zero;
                        let proj = (d.re * span.re + d.im * span.im) / denom;
                        match denom > 0.0 {
                            true => proj,
                            false => f64::NAN,
                        }
                    })
                    .collect();
                // 投影已按 |g1 − g0| 归一化，满量程即两态中心的间距
                (prob, denom.sqrt())
            }
            // 标定不足两态：投影轴无定义
            [] | [_] => (vec![f64::NAN; iq.len()], 0.0),
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
            let prob: Vec<f64> = proj
                .iter()
                .map(|value| match span > 0.0 {
                    true => (value - lo) / span,
                    false => f64::NAN,
                })
                .collect();
            // 归一化正是除以投影的极值之差，它就是这条路径的满量程
            (prob, span)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 两态中心：任意夹角的一对（连线不平行于任一坐标轴）。
    fn centers() -> StateCenters {
        StateCenters::new(vec![Complex64::new(0.30, 0.10), Complex64::new(0.42, 0.30)])
    }

    /// 标定路径：`zero → one` 连线投影，两个中心分别映射到 0 与 1。
    #[test]
    fn calibrated_projection_maps_centers_to_0_and_1() {
        let states = centers();
        let iq = vec![
            Complex64::new(0.30, 0.10),
            Complex64::new(0.42, 0.30),
            Complex64::new(0.36, 0.20),
        ];
        let prob = p1(&iq, Some(&states));
        assert!((prob[0] - 0.0).abs() < 1e-12, "{}", prob[0]);
        assert!((prob[1] - 1.0).abs() < 1e-12, "{}", prob[1]);
        assert!((prob[2] - 0.5).abs() < 1e-12, "{}", prob[2]);
    }

    /// 标定路径是线性投影，不做 min-max：噪声涨落会越出 [0, 1]。
    #[test]
    fn calibrated_projection_is_not_clipped() {
        let states = centers();
        let extended = Complex64::new(0.30, 0.10) + (centers().as_slice()[1] - centers().as_slice()[0]) * 2.0;
        let prob = p1(&[extended], Some(&states));
        assert!(prob[0] > 1.0, "{}", prob[0]);
    }

    /// 自定轴路径：投影的极值被钉在 0 与 1 上。
    #[test]
    fn auto_axis_normalizes_extremes() {
        let states = centers();
        let g0 = states.as_slice()[0];
        let g1 = states.as_slice()[1];
        // 沿两态连线均匀取点，再叠一点横向偏移（让主轴拟合有非零的垂距）
        let iq: Vec<Complex64> = (0..=10)
            .map(|k| g0 + (g1 - g0) * (k as f64 / 10.0) + Complex64::new(0.0, 0.004 * (k % 3) as f64))
            .collect();
        let prob = p1(&iq, None);
        let lo = prob.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = prob.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((lo - 0.0).abs() < 1e-12, "{lo}");
        assert!((hi - 1.0).abs() < 1e-12, "{hi}");
    }

    /// 两态中心重合成一点：投影轴无定义，各点为 NaN。
    #[test]
    fn coincident_centers_give_nan() {
        let same = Complex64::new(0.30, 0.10);
        let states = StateCenters::new(vec![same, same]);
        let prob = p1(&[same, Complex64::new(0.4, 0.2)], Some(&states));
        assert!(prob.iter().all(|value| value.is_nan()), "{prob:?}");
    }

    /// 标定不足两态：与两态重合走同一个退化出口（NaN），不 panic。
    #[test]
    fn short_calibration_gives_nan() {
        let only_zero = StateCenters::new(vec![Complex64::new(0.30, 0.10)]);
        let prob = p1(&[Complex64::new(0.30, 0.10)], Some(&only_zero));
        assert_eq!(prob.len(), 1);
        assert!(prob[0].is_nan(), "{}", prob[0]);
    }

    /// 主轴方向：沿已知直线取点，方向应当与它平行；符号约定取 I 分量为正。
    #[test]
    fn direction_follows_the_line() {
        // 斜率 2 的直线（I 分量为正的一侧）
        let iq: Vec<Complex64> = (0..=10)
            .map(|k| Complex64::new(k as f64 * 0.01, k as f64 * 0.02))
            .collect();
        let (origin, unit) = direction(&iq);
        assert!((origin.re - 0.05).abs() < 1e-12, "{}", origin.re);
        assert!((origin.im - 0.10).abs() < 1e-12, "{}", origin.im);
        // 单位方向与 (1, 2)/√5 同向（同向而非反向：I 分量为正）
        let expected = 1.0 / 5.0_f64.sqrt();
        assert!((unit.re - expected).abs() < 1e-9, "{}", unit.re);
        assert!((unit.im - 2.0 * expected).abs() < 1e-9, "{}", unit.im);
    }
}
