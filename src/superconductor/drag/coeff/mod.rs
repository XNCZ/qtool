//! DRAG 系数扫描：扫 DRAG 系数，定泄露最小的那个系数。
//!
//! `pairs` 对**反号**脉冲（`X_π` 接 `X_{−π}`）打上去，单发的旋转误差在一对里抵消、只剩二阶项，
//! P1 随系数画出一条谷——谷底是相位误差最小的那一点，而标定表里记的泄露最优在它的**两倍**处。
//!
//! 与幅度扫描的差别有两条：**每一阶都拟谷**（系数从零起、误差是二阶量，曲线不是 Rabi 余弦，
//! 没有周期可给升阶窗口定尺度，窗口沿用经验指数，见驱动那一层）；`coeff = 0` 处不是极小（那里
//! 没有 DRAG 修正、脉冲对漏得多），所以逐点取谷时不必排除起点。

pub mod valley;

// 模型文件是 `coeff/valley.rs`，不提一层的话对外路径是 `coeff::valley::valley_fit`；这里把它的
// 公共项收平到本模块，外部统一写 `coeff::valley_fit`（照 s21 / qspec / rabi / ramsey / t1 /
// t2_echo / drag::amplitude 的做法）。
pub use valley::*;

#[cfg(feature = "plot")]
pub mod plot;
