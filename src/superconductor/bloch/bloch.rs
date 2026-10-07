//! 三基读出 IQ → 布洛赫向量。
//!
//! 三个基各是一条自己的扫描（X 基：前置 H；Y 基：前置 `Rx(−π/2)`；Z 基：直测），沿同一根扫描轴
//! 逐点对齐。每个基的点先经 [`p1`] 投影成激发概率，再按 `⟨P⟩ = 1 − 2P1` 变成该基上的泡利期望
//! 值——这就是"布居到球坐标"的全部内容（baseline `plot_Li.py` 里那段 markdown 推的三条式子：
//! Z 基直读、X 基等价于在 H 之后读、Y 基等价于在 `Rx(−π/2)` 之后读）。
//!
//! **投影只做一次，在 3n 个点的并集上做**：三段接成一条点云交给 [`p1`]，再按 n 切回来。标定路径
//! 因此共用同一条 `g0 → g1` 连线（三个分量同一个尺度）；自定轴路径也只在整条云上拟一根主轴、
//! 极值只取一次。若改成逐基各拟一根轴线、各自归一化，三个分量的尺度互不相同，
//! `R = √(⟨X⟩² + ⟨Y⟩² + ⟨Z⟩²)` 会被三段各自的缩放扭成一条没有意义的曲线。

use crate::superconductor::{StateCenters, p1};

pub use lmfit::Complex64;

/// 一条布洛赫轨迹的原始输入：扫描轴 + 三基 IQ。
///
/// 三个基各一条序列，长度须与 `axis` 一致。**哪个基是哪条由字段名定**，不靠位置：baseline 那边
/// 三个基是交错存成一维、靠 `[0::3]` 拆开的，那套约定（以及 X/Y 究竟各取第几条）是移植时最容易
/// 翻车的地方，这里摊成三个具名字段，把次序写在明面上。
pub struct Trajectory<'a> {
    /// 扫描轴（π 幅度、Z 幅度、延时…），单位随实验
    pub axis: &'a [f64],
    /// X 基（前置 H）的读出 IQ
    pub x: &'a [Complex64],
    /// Y 基（前置 `Rx(−π/2)`）的读出 IQ
    pub y: &'a [Complex64],
    /// Z 基（直测）的读出 IQ
    pub z: &'a [Complex64],
}

/// 布洛赫向量 `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)(扫描)`；扫描轴原样带回。
#[derive(Debug, Clone, PartialEq)]
pub struct BlochVector {
    /// 扫描轴，与输入一致
    pub axis: Vec<f64>,
    /// X 基上的泡利期望值 `1 − 2P1`
    pub x: Vec<f64>,
    /// Y 基上的泡利期望值 `1 − 2P1`
    pub y: Vec<f64>,
    /// Z 基上的泡利期望值 `1 − 2P1`
    pub z: Vec<f64>,
}

impl BlochVector {
    /// 径向长度 `R = √(⟨X⟩² + ⟨Y⟩² + ⟨Z⟩²)`：理想纯态为 1，缩进球内即退相干或读出误差。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     与扫描轴等长的 R 序列
    pub fn radius(&self) -> Vec<f64> {
        self.x
            .iter()
            .zip(&self.y)
            .zip(&self.z)
            .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
            .collect()
    }
}

/// 布洛赫向量的构造错误。
#[derive(Debug, Clone, PartialEq)]
pub enum BlochError {
    /// 扫描轴为空，没有轨迹可言。
    EmptyData,
    /// 某个基的点数与扫描轴不一致。
    LengthMismatch { axis: usize, series: usize },
}

impl std::fmt::Display for BlochError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "the scan axis is empty; there is no trajectory to build"),
            Self::LengthMismatch { axis, series } => write!(
                f,
                "a basis series has {series} points but the scan axis has {axis}"
            ),
        }
    }
}

impl std::error::Error for BlochError {}

/// 三基读出 IQ → 布洛赫向量。
///
/// 形参:
///     trajectory: 扫描轴与三基 IQ，三个基的长度须与扫描轴一致
///     states: 各态标定中心；None 表示未标定，改由数据自身定轴（同 [`p1`]）
///
/// 返回值:
///     布洛赫向量；扫描轴为空、或某个基的点数对不上时给出 [`BlochError`]——两者都是调用侧的形状
///     问题，不当成"退化数据"静默变 NaN
///
/// 取向：标定路径由 `g0 → g1` 定死（`|0>` 那一端是 +1）；自定轴路径只能定到直线、定不了哪一端
/// 是 `|1>`，整体可能反号——反号即球心对称，三个分量同时变号。
pub fn bloch_vector(
    trajectory: &Trajectory<'_>,
    states: Option<&StateCenters>,
) -> Result<BlochVector, BlochError> {
    let n = trajectory.axis.len();
    if n == 0 {
        return Err(BlochError::EmptyData);
    }
    for series in [trajectory.x, trajectory.y, trajectory.z] {
        if series.len() != n {
            return Err(BlochError::LengthMismatch {
                axis: n,
                series: series.len(),
            });
        }
    }

    // 三段接成一条点云，只投影一次：三个分量的尺度因此由同一条轴、同一对极值定下来
    let mut cloud = Vec::with_capacity(3 * n);
    cloud.extend_from_slice(trajectory.x);
    cloud.extend_from_slice(trajectory.y);
    cloud.extend_from_slice(trajectory.z);
    let prob = p1(&cloud, states);

    let (prob_x, rest) = prob.split_at(n);
    let (prob_y, prob_z) = rest.split_at(n);
    Ok(BlochVector {
        axis: trajectory.axis.to_vec(),
        x: pauli(prob_x),
        y: pauli(prob_y),
        z: pauli(prob_z),
    })
}

/// 投影值 → 该基上的泡利期望值：`⟨P⟩ = 1 − 2P1`（投影退化时 P1 是 NaN，这里原样传下去）。
///
/// 形参:
///     prob: 某一基各点的 P1
///
/// 返回值:
///     该基上的泡利期望值，长度与 `prob` 一致
fn pauli(prob: &[f64]) -> Vec<f64> {
    prob.iter().map(|value| 1.0 - 2.0 * value).collect()
}
