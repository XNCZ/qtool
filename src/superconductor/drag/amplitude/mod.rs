//! DRAG 幅度扫描：扫 XY 幅度，定 π 脉冲幅。
//!
//! 逐阶各扫一遍幅度轴——最低阶从零驱动铺开，之后每一阶围着上一阶的谷位收窄；最低阶的曲线形状
//! 恰好是 Rabi 余弦，所以它能一次问出**周期**，而升阶窗口的尺度（半个周期）正是从那个周期长
//! 出来的（[`factor_one`]）。最高阶拟的是谷，谷心即标定值（那一档还没落）。

pub mod factor_one;

// 模型文件是 `amplitude/factor_one.rs`，不提一层的话对外路径是 `amplitude::factor_one::factor_one_fit`；
// 这里把它的公共项收平到本模块，外部统一写 `amplitude::factor_one_fit`（照 s21 / qspec / rabi /
// ramsey / t1 / t2_echo 的做法）。
pub use factor_one::*;
