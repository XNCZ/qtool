//! S21 谐振腔：模型与拟合（[`s21`]），以及报告渲染（[`s21_plot`] / [`s21_power_plot`]，feature = "plot"）。

pub mod s21;

// 模型文件是 `s21/s21.rs`，不提一层的话对外路径是 `s21::s21::S21Model`；这里把它的公共项
// 收平到本模块，外部统一写 `s21::S21Model`（`pub(crate)` 的那些也照原可见性一并转出）。
pub use s21::*;

#[cfg(feature = "plot")]
pub mod s21_plot;

#[cfg(feature = "plot")]
pub mod s21_power_plot;
