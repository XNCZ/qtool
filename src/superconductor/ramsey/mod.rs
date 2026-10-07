//! T2\*（Ramsey）延时扫描：阻尼余弦线型与拟合（`ramsey`），以及报告渲染（`ramsey_plot`，
//! feature = "plot"）。
//!
//! baseline `qana` 的 Rust 移植：`strategy/fitting/cosine_damp.py` 与 `exps/t2_analysis.py`。
//! 两个 π/2 之间让比特自由演化 τ，第二个 π/2 带相位斜坡 `2π·Δf·τ`，退相干摊成一条条纹：
//! `P1(τ) = offset + amplitude·exp(−τ/decay)·cos(2π·frequency·τ + phase)`，包络的时间常数
//! `decay` 即 T2\*。

pub mod ramsey;

// 模型文件是 `ramsey/ramsey.rs`，不提一层的话对外路径是 `ramsey::ramsey::ramsey_fit`；这里把
// 它的公共项收平到本模块，外部统一写 `ramsey::ramsey_fit`（照 s21 / qspec / rabi 的做法）。
pub use ramsey::*;

#[cfg(feature = "plot")]
pub mod ramsey_plot;
