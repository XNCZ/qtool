//! T2 echo（回波退相位）：指数线型与拟合（`t2_echo`），以及报告渲染（`t2_echo_plot`，
//! feature = "plot"）。
//!
//! `π/2 — τ/2 — π — τ/2 — π/2`：中间的 π 脉冲把准静态失谐积累的相位折回去，P1 随总自由演化
//! 时间 τ 单调指数衰减，时间常数即 T2 echo。baseline `qana` 里没有回波实验，见 [`t2_echo`]
//! 的模块文档。

pub mod t2_echo;

// 模型文件是 `t2_echo/t2_echo.rs`，不提一层的话对外路径是 `t2_echo::t2_echo::t2_echo_fit`；
// 这里把它的公共项收平到本模块，外部统一写 `t2_echo::t2_echo_fit`（照 s21 / qspec / rabi /
// ramsey 的做法）。
pub use t2_echo::*;

#[cfg(feature = "plot")]
pub mod t2_echo_plot;
