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

/// 分位数（线性插值，与 numpy 的 `percentile` 默认口径一致）。
///
/// 形参:
///     sorted: **已升序排好**的样本
///     q: 分位，0..=1
///
/// 返回值:
///     该分位的值；空数组返回 NaN，q 越界时按端点截断
pub(crate) fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let position = match q < 0.0 {
        true => 0.0,
        false => match q > 1.0 {
            true => 1.0,
            false => q,
        },
    } * (n - 1) as f64;
    let low = position.floor();
    let high = position.ceil();
    let (i, j) = (low as usize, high as usize);
    match i == j {
        true => sorted[i],
        false => sorted[i] + (sorted[j] - sorted[i]) * (position - low),
    }
}

/// 一维两样本的最优判别阈值与错分率。
///
/// 两团样本按值排序后，只有"相邻两值之间"这些切点值得考虑——切点落在别处，错分数只增不减。
/// 于是扫一遍即可：某个切点左侧的 `b` 与右侧的 `a` 都是错分的。
///
/// 这就是分类里那个判别阈值的经验版：**不假设两团各是什么分布**，纯计数。
///
/// 形参:
///     a: 第一组样本
///     b: 第二组样本
///
/// 返回值:
///     (阈值, 错分率)。阈值取最优切点两侧值的中点；错分率 = 最优切点处的错分数 / 样本总数。
///     任一组为空或全为 NaN 时返回 (NaN, NaN)
pub(crate) fn best_threshold(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.is_empty() || b.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    // 合并后按值排序，同时带上来源标记
    let mut merged: Vec<(f64, bool)> = a
        .iter()
        .map(|value| (*value, false))
        .chain(b.iter().map(|value| (*value, true)))
        .collect();
    merged.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(std::cmp::Ordering::Equal));
    let total = merged.len() as f64;
    // 切点在 k 之前（即前 k 个落在阈值左侧）时的错分数：左侧的 b + 右侧的 a
    let (mut errors_best, mut cut_best) = (f64::INFINITY, 0_usize);
    let (mut left_a, mut left_b) = (0.0, 0.0);
    for k in 0..=merged.len() {
        if k > 0 {
            match merged[k - 1].1 {
                true => left_b += 1.0,
                false => left_a += 1.0,
            }
        }
        let right_a = a.len() as f64 - left_a;
        let errors = left_b + right_a;
        if errors < errors_best - f64::EPSILON {
            errors_best = errors;
            cut_best = k;
        }
    }
    // 阈值取最优切点两侧值的中点；切点贴边时外推一个样本间距
    let threshold = match cut_best {
        0 => merged[0].0,
        k if k >= merged.len() => merged[merged.len() - 1].0,
        k => 0.5 * (merged[k - 1].0 + merged[k].0),
    };
    (threshold, errors_best / total)
}

/// 两样本的 AUC（= 曼-惠特尼统计量 = **P(b > a)** 的样本估计，平局各计一半；`a` 是第一组、
/// `b` 是第二组，所以两团分得越开这个数越接近 1）。
///
/// 与 [`best_threshold`] 的关键差别：**它不需要选阈值**，所以没有"在同一批样本上既选阈值
/// 又报错分率"那种乐观偏差，是纯粹的可分性度量。两个数一起看：阈值能用，AUC 不会被样本
/// 量骗。0.5 表示两团完全重叠，1 表示完全分开。
///
/// 形参:
///     a: 第一组样本
///     b: 第二组样本
///
/// 返回值:
///     AUC；任一组为空时返回 NaN
pub(crate) fn auc(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let mut merged: Vec<(f64, bool)> = a
        .iter()
        .map(|value| (*value, false))
        .chain(b.iter().map(|value| (*value, true)))
        .collect();
    merged.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(std::cmp::Ordering::Equal));
    // 逐点给秩（并列取平均秩），再把 b 的秩和转成"a 大于 b 的期望比例"
    let mut rank_sum_b = 0.0;
    let mut index = 0;
    while index < merged.len() {
        let mut end = index;
        while end + 1 < merged.len() && merged[end + 1].0 == merged[index].0 {
            end += 1;
        }
        // 并列组的平均秩（1 基）
        let rank = 0.5 * ((index + 1) as f64 + (end + 1) as f64);
        for item in &merged[index..=end] {
            match item.1 {
                true => rank_sum_b += rank,
                false => {}
            }
        }
        index = end + 1;
    }
    let (n_a, n_b) = (a.len() as f64, b.len() as f64);
    (rank_sum_b - n_b * (n_b + 1.0) / 2.0) / (n_a * n_b)
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

/// 局部极大的下标（scipy `find_peaks` 的口径）。
///
/// 严格大于左邻、且右侧下降的样点是峰；平台（连续相等的样点）取其中点，与 scipy 的
/// plateau 处理一致。数组端点不算峰——峰落在扫描边界时要靠调用方自行兜底（例如并入全局极大）。
///
/// 形参:
///     y: 曲线 (n,)
///
/// 返回值:
///    峰位下标，升序
pub(crate) fn local_maxima(y: &[f64]) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = y.len();
    let mut i = 1;
    while i + 1 < n {
        match y[i] > y[i - 1] {
            true => {
                // 平台：向右吃掉相等的一段，峰位取平台中点
                let mut j = i;
                while j + 1 < n && y[j + 1] == y[i] {
                    j += 1;
                }
                match j + 1 < n && y[j + 1] < y[i] {
                    true => peaks.push((i + j) / 2),
                    false => {}
                }
                i = j + 1;
            }
            false => i += 1,
        }
    }
    peaks
}

/// 取向候选：`flipped` 为真时取 `1 − x`（物理量的"投影反向"就是这条曲线），否则原样。
///
/// 投影方向的正负号在自定轴时只是约定，故调用方常把两个取向各拟合一遍、按残差择优。
///
/// 形参:
///     values: 曲线 (n,)
///     flipped: 是否取反向
///
/// 返回值:
///     该取向下的曲线 (n,)
pub(crate) fn orient(values: &[f64], flipped: bool) -> Vec<f64> {
    match flipped {
        true => values.iter().map(|value| 1.0 - value).collect(),
        false => values.to_vec(),
    }
}

/// 去均值后的一维幅度谱（单边）。
///
/// 频率轴 `f_k = k / (n·Δx)`，只取 `k = 1..=n/2`：直流项按定义不入结果——曲线无振荡时谱上
/// 只剩直流，把它留在里面会让"无振荡"与"振荡频率为零"混为一谈。网格取自前两点，x 须等间距
/// 单调递增。
///
/// 朴素 DFT 是 O(n²)，但扫描点数只有几十到几百，够快，也就不必为此引一个 FFT 依赖。本函数
/// 同时供初值估计（`rabi_amp` 的傅里叶候选）与报告里的频谱面板使用——两处各写一份的话，
/// 面板上看到的峰与拟合用的初值就不再是同一个定义。
///
/// 形参:
///     x: 自变量轴 (n,)，须等间距、单调递增
///     y: 曲线 (n,)
///
/// 返回值:
///     (频率轴, 幅度谱)，两者等长；点数不足、长度不一致或网格退化时皆空
pub(crate) fn spectrum(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    if n < 2 || x.len() != n {
        return (Vec::new(), Vec::new());
    }
    let step = x[1] - x[0];
    if step <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let center = mean(y);
    let mut freqs = Vec::with_capacity(n / 2);
    let mut amps = Vec::with_capacity(n / 2);
    for k in 1..=n / 2 {
        let (mut re, mut im) = (0.0, 0.0);
        for (j, value) in y.iter().enumerate() {
            let phase = 2.0 * PI * (k * j) as f64 / n as f64;
            let centered = value - center;
            re += centered * phase.cos();
            im -= centered * phase.sin();
        }
        freqs.push(k as f64 / (n as f64 * step));
        amps.push((re * re + im * im).sqrt());
    }
    (freqs, amps)
}

pub(crate) mod density;

#[cfg(feature = "plot")]
pub(crate) mod heatmap;
#[cfg(feature = "plot")]
pub(crate) mod panels;
#[cfg(feature = "plot")]
pub(crate) mod params;
#[cfg(feature = "plot")]
pub(crate) mod bubble;
#[cfg(feature = "plot")]
pub(crate) mod resize;

#[cfg(test)]
mod tests {
    use super::*;

    /// 局部极大的口径（与 scipy `find_peaks` 逐数组对拍过，这里固化成可见的样例）。
    #[test]
    fn local_maxima_matches_scipy_semantics() {
        let cases: [(&[f64], &[usize]); 7] = [
            (&[1.0, 3.0, 2.0], &[1]),
            // 端点不算峰：共振落在扫描边界时要靠调用方兜底
            (&[3.0, 2.0, 1.0], &[]),
            // 平台取中点
            (&[1.0, 3.0, 3.0, 3.0, 2.0], &[2]),
            // 平台顶到数组末端不算峰
            (&[1.0, 3.0, 3.0], &[]),
            (&[0.0, 1.0, 0.0, 2.0, 0.0], &[1, 3]),
            (&[1.0, 2.0, 3.0], &[]),
            // 噪声抖动每一个都算峰——修峰交给下游的残差比较
            (&[0.0, 1.0, 0.2, 1.1, 0.3], &[1, 3]),
        ];
        for (y, expected) in cases {
            assert_eq!(local_maxima(y), expected, "{y:?}");
        }
    }

    /// 取向候选：取反向就是 `1 − x`，不反就是原样。
    #[test]
    fn orient_is_the_complement() {
        let values = [0.0, 0.25, 1.0];
        assert_eq!(orient(&values, false), vec![0.0, 0.25, 1.0]);
        assert_eq!(orient(&values, true), vec![1.0, 0.75, 0.0]);
    }
}
