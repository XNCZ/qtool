//! T1（能量弛豫）：指数线型与拟合（`t1`），以及报告渲染（`t1_plot`，feature = "plot"）。
//!
//! baseline `qana` 的 Rust 移植：`exps/t1_analysis.py`。把比特激发到 |1> 之后让它自由演化 τ，
//! 再读一次——末态落在 |1> 的概率随 τ 指数衰减，时间常数即 T1。

pub mod t1;

// 模型文件是 `t1/t1.rs`，不提一层的话对外路径是 `t1::t1::t1_fit`；这里把它的公共项收平到本
// 模块，外部统一写 `t1::t1_fit`（照 s21 / qspec / rabi / ramsey 的做法）。
pub use t1::*;

#[cfg(feature = "plot")]
pub mod t1_plot;
