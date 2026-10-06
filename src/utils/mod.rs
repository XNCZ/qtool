//! 通用数值助手。
//!
//! 这些函数按 numpy 的语义实现（相位展开的分支修正、偶数取中的中位数、
//! 并列取首个的下标等），语义与 baseline 保持一致，供各分析模块复用。

use std::f64::consts::PI;

/// 按 numpy 的语义展开相位：相邻差折叠进 (−π, π]，再累加校正量。
///
/// 形参:
///     phase: 原始相位数组，rad
///
/// 返回值:
///     展开后的相位数组，长度与输入相同
pub(crate) fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    let n = phase.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(phase[0]);
    let mut correction = 0.0;
    for i in 1..n {
        let dd = phase[i] - phase[i - 1];
        let mut ddmod = (dd + PI).rem_euclid(2.0 * PI) - PI;
        if ddmod == -PI && dd > 0.0 {
            ddmod = PI;
        }
        // numpy 累加的是差值修正 `ddmod − dd`：相邻差本身在 (−π, π] 内时为零，
        // 只在跨支跳变时贡献 ±2π。
        correction += ddmod - dd;
        out.push(phase[i] + correction);
    }
    out
}

/// 算术平均。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     平均值；空数组返回 NaN
pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 中位数，偶数个元素取中间两数的均值（与 numpy 一致）。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     中位数；空数组返回 NaN
pub(crate) fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| match left.partial_cmp(right) {
        Some(ordering) => ordering,
        None => std::cmp::Ordering::Equal,
    });
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// 最小值下标；并列时取首个（与 numpy 一致）。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     最小元素的下标；空数组返回 0
pub(crate) fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] < values[best] {
            best = i;
        }
    }
    best
}

/// 最大值下标；并列时取首个（与 numpy 一致）。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     最大元素的下标；空数组返回 0
pub(crate) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

/// 最大值。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     最大元素；空数组返回 NaN
pub(crate) fn max_value(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut best = values[0];
    for &v in values.iter() {
        if v > best {
            best = v;
        }
    }
    best
}

/// 最小值。
///
/// 形参:
///     values: 输入数组
///
/// 返回值:
///     最小元素；空数组返回 NaN
pub(crate) fn min_value(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut best = values[0];
    for &v in values.iter() {
        if v < best {
            best = v;
        }
    }
    best
}

/// 一元一次最小二乘拟合 `y ≈ slope·x + intercept`。
///
/// 形参:
///     x: 自变量数组
///     y: 因变量数组，长度与 x 相同
///
/// 返回值:
///     (slope, intercept)；点数不足或自变量无展布时退回 (0, y 均值)
pub(crate) fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 {
        return (0.0, mean(y));
    }
    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        sxx += dx * dx;
        sxy += dx * (y[i] - y_mean);
    }
    if sxx == 0.0 {
        return (0.0, y_mean);
    }
    let slope = sxy / sxx;
    (slope, y_mean - slope * x_mean)
}

/// 线性去趋势（scipy `signal.detrend(type="linear")` 的口径），但自变量取**频率**而不是
/// 索引 —— 系数因此带物理单位，也能直接套用到别的频率网格上（见 [`linear_detrend`]）。
///
/// 形参:
///     x: 自变量（频率），长度须与 `y` 一致
///     y: 待去趋势的序列
///
/// 返回值:
///     (去趋势后的序列, 斜率, 截距)；后两者即频率域那条直线 `slope * x + intercept`
pub(crate) fn detrend(x: &[f64], y: &[f64]) -> (Vec<f64>, f64, f64) {
    let (slope, intercept) = linear_fit(x, y);
    let detrended = y
        .iter()
        .zip(x.iter())
        .map(|(value, at)| value - (slope * at + intercept))
        .collect();
    (detrended, slope, intercept)
}

/// 用**已知**的（频率域）直线去趋势：只套用、不重新拟合，所以可以直接用在另一条（更密
/// 或更疏的）频率网格上。相位面板上数据点与拟合曲线必须共用同一条趋势线，否则残差会
/// 整体倾斜 —— 那条线在数据网格上拟合一次（[`detrend`]），再用本函数套到拟合曲线上。
pub(crate) fn linear_detrend(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(value, at)| value - (slope * at + intercept))
        .collect()
}

#[cfg(feature = "plot")]
pub(crate) mod heatmap;
#[cfg(feature = "plot")]
pub(crate) mod params;
#[cfg(feature = "plot")]
pub(crate) mod bubble;
#[cfg(feature = "plot")]
pub(crate) mod resize;
