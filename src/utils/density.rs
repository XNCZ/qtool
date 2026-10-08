//! 二维点云的密度网格与"最深 p 区域"的面积（crate 内部件）。
//!
//! 读出 IQ 云的形状常常不是高斯（香蕉、月牙、双峰都见得到），用协方差椭圆去代表它等于
//! 宣称它是高斯的——椭圆的等值线本就是高斯的等值线。这里换一条不依赖任何分布假设的路：
//! 把平面切成均匀格子数点，再按格子密度从高到低压入，压满 p 比例的点即"最深 p 区域"。
//!
//! 这样得到的区域天然是非凸、天然多峰（双峰会先各取各的核心，而不是跨过中间的空洞连成
//! 一片），也不需要凸包——凸包会把凹陷填平、把不该连的地方连起来，正好毁掉"任意形状"这个
//! 目标。**面积与轮廓出自同一个网格**，报出去的那个数与画出来的那条线不可能是两回事。

use crate::utils::quantile;
use contour::{Contour, ContourBuilder, Float};
use geo::Area;

/// 采样区间取样本的这两个分位，用来定"展宽"这个尺度的基准。
///
/// 用分位而不是 min/max，是免得几个逃逸点把这把尺子本身拉歪。
const EXTENT_LOW: f64 = 0.001;
const EXTENT_HIGH: f64 = 0.999;

/// 采样区间相对分位展宽的最大外扩倍数。
///
/// 区间取"数据范围"而不是"分位区间"：**最深 100% 那一层要装下每一个单发**，框不够大它就被
/// 截掉了（实测高斯云上 r100 比最远单发小 28%）。但也不能完全放任——逃逸点偶尔跑到十几倍
/// 展宽外，框一撑大，格子就粗到分辨不出薄环这类细结构。取 2 倍展宽封顶：寻常云（最远点约
/// 4σ）落在封顶之内、框正好裹住数据，真正的野点则被挡在门外（那一层的面积因此偏小，见
/// [`DensityGrid::deepest`]）。
const EXTENT_CAP: f64 = 2.0;

/// 区间在数据范围外再留这么多（占展宽的比例），给抹平的晕留点落脚处。
const EXTENT_PAD: f64 = 0.05;

/// 抹平核的宽度，单位是格子。
///
/// **整格计数做不了这件事**：99% 那层的边界上，一个格子的期望点数还不到 1，边界那圈因此大量
/// 是空格——而"最深 99% 区域"恰恰就落在那里。按整格压，边界处那圈的小数质量无处安放，面积会
/// 系统性偏小（实测高斯云上偏 23%，且越外层越偏）。先把计数抹成连续密度场，格子值不再被整数
/// 卡住，压在边界上的那个格子才能贡献出真正属于它的那一小部分。
///
/// 取 2 格：格子的尺寸本身是"样本展布 / 格数"，所以这也是个跟着数据缩放的带宽（几万点的
/// 二维云上大致相当于 Silverman 法则给的量级），不必再让调用方调参。
const SMOOTH_SIGMA_CELLS: f64 = 2.0;

/// 高斯核截断到几倍标准差。
const SMOOTH_TRUNCATE: f64 = 3.0;

/// 二维点云的均匀网格计数，以及由它算出的"最深 p 区域"。
#[derive(Debug, Clone)]
pub(crate) struct DensityGrid {
    /// 网格原点（左下角）与格子尺寸
    x0: f64,
    y0: f64,
    cell_x: f64,
    cell_y: f64,
    /// 每轴的格子数
    nx: usize,
    ny: usize,
    /// 各格子的**原始**点数（未抹平），行优先（`y` 自下而上）；用它定"装了多少单发"
    counts: Vec<f64>,
    /// 各格子的密度（已抹平），单位是"一个格子的期望点数"；用它定**几何**（水平集在哪）
    density: Vec<f64>,
    /// 落进网格的总点数（区间外的点按边界计入）
    total: f64,
}

impl DensityGrid {
    /// 由点云建网格。
    ///
    /// 采样区间取 x、y 各自 [`EXTENT_LOW`]–[`EXTENT_HIGH`] 分位构成的矩形；落在区间外的点
    /// 按边界计入，**总点数不丢**——总数是"68% 是多少个点"的基准，丢点会让密度阈值与面积
    /// 一起偏。
    ///
    /// 形参:
    ///     points: 样本点 (x, y)
    ///     bins: 每轴的格子数
    ///
    /// 返回值:
    ///     密度网格；无点、格子数为 0、或样本在某轴上展布为零时返回 None
    pub(crate) fn from_samples(points: &[(f64, f64)], bins: usize) -> Option<Self> {
        if points.is_empty() || bins == 0 {
            return None;
        }
        let mut xs: Vec<f64> = points.iter().map(|(x, _)| *x).collect();
        let mut ys: Vec<f64> = points.iter().map(|(_, y)| *y).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // 展宽（分位半宽）用来封顶与留白，区间本身取数据范围
        let x_mid = quantile(&xs, 0.5);
        let x_half = 0.5 * (quantile(&xs, EXTENT_HIGH) - quantile(&xs, EXTENT_LOW));
        let y_mid = quantile(&ys, 0.5);
        let y_half = 0.5 * (quantile(&ys, EXTENT_HIGH) - quantile(&ys, EXTENT_LOW));
        let (x_lo, x_hi) = (
            (xs[0] - EXTENT_PAD * x_half).max(x_mid - EXTENT_CAP * x_half),
            (xs[xs.len() - 1] + EXTENT_PAD * x_half).min(x_mid + EXTENT_CAP * x_half),
        );
        let (y_lo, y_hi) = (
            (ys[0] - EXTENT_PAD * y_half).max(y_mid - EXTENT_CAP * y_half),
            (ys[ys.len() - 1] + EXTENT_PAD * y_half).min(y_mid + EXTENT_CAP * y_half),
        );
        let (span_x, span_y) = (x_hi - x_lo, y_hi - y_lo);
        if !(span_x > 0.0) || !(span_y > 0.0) {
            return None;
        }
        // 端点不落在右/上边界之外：落在最大值上的点要归进最后一格
        let cell_x = span_x / bins as f64;
        let cell_y = span_y / bins as f64;
        let mut counts = vec![0.0; bins * bins];
        for (x, y) in points {
            let ix = clamp_index((x - x_lo) / cell_x, bins);
            let iy = clamp_index((y - y_lo) / cell_y, bins);
            counts[iy * bins + ix] += 1.0;
        }
        let grid = Self {
            x0: x_lo,
            y0: y_lo,
            cell_x,
            cell_y,
            nx: bins,
            ny: bins,
            counts: counts.clone(),
            density: counts,
            total: points.len() as f64,
        };
        Some(grid.smooth())
    }

    /// 高斯抹平（KDE 的网格版）：可分离核，先按行再按列。
    ///
    /// 抹平**不改总质量**——最后按"抹平前 / 抹平后"的总和把它们对齐。边界用零填充，落在网格
    /// 外的那一点点质量因此会漏出去，对齐这一下正好把它补回来，`deepest` 里"压满 68% 的点"
    /// 才有意义。
    ///
    /// 返回值:
    ///     抹平后的网格
    fn smooth(&self) -> Self {
        let radius = (SMOOTH_TRUNCATE * SMOOTH_SIGMA_CELLS).ceil() as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|d| (-0.5 * (d as f64 / SMOOTH_SIGMA_CELLS).powi(2)).exp())
            .collect();
        let (nx, ny) = (self.nx, self.ny);
        let at = |x: isize, y: isize| -> f64 {
            match x >= 0 && x < nx as isize && y >= 0 && y < ny as isize {
                true => self.density[y as usize * nx + x as usize],
                false => 0.0,
            }
        };
        // 先按 x 方向
        let mut horizontal = vec![0.0; nx * ny];
        for j in 0..ny as isize {
            for i in 0..nx as isize {
                let mut acc = 0.0;
                for (offset, weight) in kernel.iter().enumerate() {
                    acc += weight * at(i + offset as isize - radius, j);
                }
                horizontal[j as usize * nx + i as usize] = acc;
            }
        }
        // 再按 y 方向
        let mut blurred = vec![0.0; nx * ny];
        for j in 0..ny as isize {
            for i in 0..nx as isize {
                let mut acc = 0.0;
                for (offset, weight) in kernel.iter().enumerate() {
                    let y = j + offset as isize - radius;
                    acc += weight
                        * match y >= 0 && y < ny as isize {
                            true => horizontal[y as usize * nx + i as usize],
                            false => 0.0,
                        };
                }
                blurred[j as usize * nx + i as usize] = acc;
            }
        }
        // 核未归一，且边缘漏了质量：按总和对齐回来
        let before: f64 = self.density.iter().sum();
        let after: f64 = blurred.iter().sum();
        let scale = match after > 0.0 {
            true => before / after,
            false => 1.0,
        };
        Self {
            x0: self.x0,
            y0: self.y0,
            cell_x: self.cell_x,
            cell_y: self.cell_y,
            nx,
            ny,
            counts: self.counts.clone(),
            density: blurred.iter().map(|value| value * scale).collect(),
            total: self.total,
        }
    }

    /// 最深 `fraction` 比例的点所占的区域。
    ///
    /// **两条判据分工**：量"装了多少单发"用原始计数，定"边界在哪"用抹平场。若只按抹平场压
    /// 质量，硬边之外那条抹平尾巴会被算进去——实测均匀圆盘的"95% 区域"因此装进了 100% 的
    /// 单发，区域一直撑到云外。改用原始计数，这个区域就老老实实装 95%。
    ///
    /// 阈值取**按密度从高到低压进、累积质量恰好够 fraction 的那一格的密度**——Hyndman 的
    /// density-quantile 在网格上的写法（排序 + 累积，一步到位）。
    ///
    /// 这里原先是对密度阈值做 12 步二分，坏在分辨率：能探到的最低档是 峰值/2¹²，而野点云的
    /// 100% 恰恰需要比这更低的阈值——判据一次都不成立，阈值就停在初值（峰值）上，区域空掉、
    /// 面积**静默变成 0**（不是 NaN，是"半径 0"这种看起来合法的数）。排序没有这个下限。
    ///
    /// 形参:
    ///     fraction: 质量比例，(0, 1]
    ///
    /// 返回值:
    ///     (密度阈值, 区域)。区域与面积是同一个对象：面积就是对这块多边形积出来的
    pub(crate) fn deepest(&self, fraction: f64) -> (f64, Region) {
        let target = fraction * self.total;
        let mut cells: Vec<(f64, f64)> = self
            .density
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect();
        // 密度降序。同密度的格子谁先谁后都不影响结果：阈值取的是"跨越那一档"的密度值
        cells.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut mass = 0.0;
        // 网格非空（`from_samples` 挡住空样本与 0 格），且 Σcounts = total ≥ target，
        // 所以下面这趟必定赋上一个真正的密度；万一没赋上就是 NaN——显式地坏掉，不静默给 0
        let mut level = f64::NAN;
        for (density, count) in cells {
            mass += count;
            match mass >= target {
                true => {
                    level = density;
                    break;
                }
                false => {}
            }
        }
        (level, self.region(level))
    }

    /// 阈值以上的区域：marching squares 出的多边形，坐标已是数据单位。
    ///
    /// 交给 `contour` 算——它是 d3-contour 的移植，而 plotly.js 的等值线内部用的正是
    /// d3-contour，所以这里算出来的几何与图上画的是同一族算法；区别在于多边形拿在我们手里，
    /// 面积可以对它直接积分，于是**画出来的轮廓与报出去的面积是同一个对象**。
    ///
    /// 形参:
    ///     threshold: 密度阈值（"一个格子的期望点数"），区域取"值 ≥ 阈值"那一侧
    ///
    /// 返回值:
    ///     区域；坐标已按 `origin + step × 索引` 落到数据单位上
    pub(crate) fn region(&self, threshold: f64) -> Region {
        let builder = ContourBuilder::new(self.nx, self.ny, true)
            .x_origin(self.x0)
            .x_step(self.cell_x)
            .y_origin(self.y0)
            .y_step(self.cell_y);
        let contour = match builder.contours(&self.density, &[threshold as Float]) {
            Ok(mut found) => match found.pop() {
                Some(found) => found,
                None => return Region::empty(),
            },
            // 网格与阈值都由本函数自己给出，正常不会失败；真失败了给 NaN 面积，
            // 让它显式地坏掉，而不是悄悄报一个 0
            Err(_) => return Region::empty(),
        };
        Region::from_contour(contour)
    }
}

/// 阈值以上的那块区域：多边形（数据单位）与它的面积。
///
/// 环按"外环 + 洞"成组保存：绘图照着它画，报数照着它积分，两者用的是同一份坐标。
#[derive(Debug, Clone)]
pub struct Region {
    /// 每个多边形的 (外环, 洞)，环上各点是数据单位的 (x, y)
    polygons: Vec<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>,
    /// 面积（外环减洞），数据单位²
    pub area: f64,
}

impl Region {
    /// 空区域（面积 NaN：这不是"没有面积"，是"没算出来"）。
    fn empty() -> Self {
        Self {
            polygons: Vec::new(),
            area: f64::NAN,
        }
    }

    /// 退了化的云（所有单发重合、整个云落在一条直线上）用：区域就是一个点——多边形为空、
    /// 面积是 0。与上面的"没算出来"不同，这里是**算出来的 0**：云的中心明明白白，只是没有展宽。
    pub(crate) fn point() -> Self {
        Self {
            polygons: Vec::new(),
            area: 0.0,
        }
    }

    /// 由 `contour` 的结果装配，顺便把面积算出来。
    ///
    /// 面积交给 `geo` 的 [`geo::Area`]：它的 `Polygon` 实现会**归一化洞的绕向**，而 marching
    /// squares 给出的环方向不保证——自己写鞋带等于默认环方向一定是对的，靠不住。
    ///
    /// 多边形环转成我们自己的 `(f64, f64)` 存着（绘图的消费端不该认识 `geo_types` 的类型名）。
    fn from_contour(contour: Contour) -> Self {
        let (multi, _) = contour.into_inner();
        let area = multi.unsigned_area();
        let mut polygons = Vec::with_capacity(multi.0.len());
        for polygon in &multi.0 {
            let exterior: Vec<(f64, f64)> = polygon
                .exterior()
                .0
                .iter()
                .map(|point| (point.x, point.y))
                .collect();
            let holes: Vec<Vec<(f64, f64)>> = polygon
                .interiors()
                .iter()
                .map(|ring| ring.0.iter().map(|point| (point.x, point.y)).collect())
                .collect();
            polygons.push((exterior, holes));
        }
        Self { polygons, area }
    }
    /// 各多边形的环，供绘图用：每项是 (外环, 洞们)。
    ///
    /// 返回值:
    ///     数据单位的多边形环
    pub fn polygons(&self) -> &[(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)] {
        &self.polygons
    }
}

/// 把浮点格子坐标夹进 `[0, bins − 1]`：区间外的点按边界计入，不丢总数。
fn clamp_index(value: f64, bins: usize) -> usize {
    let index = match value.is_finite() {
        true => value.floor(),
        false => 0.0,
    };
    let index = match index < 0.0 {
        true => 0.0,
        false => index,
    };
    let index = match index >= bins as f64 {
        true => bins as f64 - 1.0,
        false => index,
    };
    index as usize
}
