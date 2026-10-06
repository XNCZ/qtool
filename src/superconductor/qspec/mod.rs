//! 比特谱（qspec）：洛伦兹线型的拟合（[`qspec`]），以及报告渲染（[`qspec_plot`]，feature = "plot"）。

pub mod qspec;

// 模型文件是 `qspec/qspec.rs`，不提一层的话对外路径是 `qspec::qspec::Lorentz`；这里把它的公共项
// 收平到本模块，外部统一写 `qspec::Lorentz`（`pub(crate)` 的那些也照原可见性一并转出）。
pub use qspec::*;

#[cfg(feature = "plot")]
pub mod qspec_plot;

#[cfg(feature = "plot")]
pub mod qspec_z_plot;
