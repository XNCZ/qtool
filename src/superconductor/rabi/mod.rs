//! Rabi 幅度扫描：余弦线型与拟合（`rabi_amp`），以及报告渲染（`rabi_amp_plot`，feature = "plot"）。
//!
//! baseline `qana` 的 Rust 移植：`exps/rabi_amp_analysis.py`。固定 XY 脉冲长度、扫驱动幅度，
//! 激发概率随脉冲面积（正比于幅度）振荡；线型 `P1(A) = amp·(1 − cos(2π·freq·A))` 锚定在零
//! 驱动处的 |0> 态上（P1(0) = 0），首个极大值处即 π 脉冲幅 `a_pi = 1/(2·freq)`。

pub mod rabi_amp;

// 模型文件是 `rabi/rabi_amp.rs`，不提一层的话对外路径是 `rabi::rabi_amp::Cos`；这里把它的
// 公共项收平到本模块，外部统一写 `rabi::Cos`（照 s21 / qspec 的做法）。
pub use rabi_amp::*;

#[cfg(feature = "plot")]
pub mod rabi_amp_plot;
