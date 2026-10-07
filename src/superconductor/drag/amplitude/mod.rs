//! DRAG 幅度扫描：扫 XY 幅度，定 π 脉冲幅。
//!
//! 逐阶各扫一遍幅度轴——最低阶从零驱动铺开，之后每一阶围着上一阶的谷位收窄到半个周期
//! （`centre ± A_π/(2·pairs)`，恰好框住一个谷）。最低阶的曲线形状恰好是 Rabi 余弦，它一次问出
//! **周期**，整条升阶链的窗口尺度都从那个周期长出来（[`factor_one`]）；从二阶起拟的是**谷**，
//! 谷心既是这一阶的读数、也是下一阶窗口的中心，最高阶的谷心即标定值（[`factor_n`]）。
//!
//! 升阶循环本身（逐阶扫、窗心传递）留在驱动里，本模块给的是它用的那几件。

pub mod factor_n;
pub mod factor_one;

#[cfg(feature = "plot")]
pub mod plot;

// 模型文件是 `amplitude/factor_*.rs`，不提一层的话对外路径是 `amplitude::factor_one::factor_one_fit`；
// 这里把它们的公共项收平到本模块，外部统一写 `amplitude::factor_one_fit` / `amplitude::factor_n_fit`
// （照 s21 / qspec / rabi / ramsey / t1 / t2_echo 的做法）。
pub use factor_n::*;
pub use factor_one::*;
