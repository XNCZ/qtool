//! DRAG 载波失谐扫描：扫载波失谐，把 `f01_working` 上剩下的那点残差量出来。
//!
//! `pairs` 对**反号**脉冲（`X_π` 接 `X_{−π}`），DRAG 系数用上一轮标出来的值，失谐逐点扫。谷若
//! 压在零失谐上，说明 `f01_working` 已经够准；偏离多少，就是这一轮要修掉的载波残差——修出来的
//! 量记进残差字段，**不回写 `f01_working`**（那是 qspec 实测出来的，两条独立证据不搅在一起）。
//!
//! 与系数扫描同形：**每一阶都拟谷**（没有周期可给升阶窗口定尺度，窗口沿用经验指数，见驱动那一
//! 层）；逐点取谷时不必排除起点（轴是绕零失谐铺的，起点不是"什么都没加"那一端）。横轴是频率、
//! 单位 SI（Hz），三个与失谐同轴的面板上画一条零失谐参考线。

pub mod valley;

// 模型文件是 `detuning/valley.rs`，不提一层的话对外路径是 `detuning::valley::valley_fit`；这里
// 把它的公共项收平到本模块，外部统一写 `detuning::valley_fit`（照 s21 / qspec / rabi / ramsey /
// t1 / t2_echo / drag::amplitude 的做法）。
pub use valley::*;

#[cfg(feature = "plot")]
pub mod plot;
