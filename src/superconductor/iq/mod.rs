//! IQ 概率实验：各态单发云的统计（`iq`），以及报告渲染（`iq_plot`，feature = "plot"）。
//!
//! baseline `qana` 的 Rust 移植与推广：`exps/iq.py` 与 `exps/iq_prob_analysis.py`。实验
//! 分别制备 |0>、|1>（以及将来的 |2>…），各采一批单发 IQ，本模块负责把这几团云变成实验
//! 真正要产出的两样东西——**读出标定的中心**与**读出的可分性**。
//!
//! 中心喂给 [`crate::superconductor::p1`]，此后 qspec / rabi 的投影轴与两个参考点就有了
//! 物理标定，不再依赖数据自身拟合直线；弥散与可分性回答另一个问题：这套读出到底能把几个
//! 态分得多开。

pub mod iq;

// 模型文件是 `iq/iq.rs`，不提一层的话对外路径是 `iq::iq::iq_stats`；这里把它的公共项
// 收平到本模块（照 s21 / qspec / rabi 的做法）。
pub use iq::*;

// 各态最深 p 区域的多边形：`StateStats::regions` 就是它，表里的面积与图上的轮廓是同一份坐标，
// 调用方也可以直接拿它去判定一个 IQ 点落在哪个态的区域里
pub use crate::utils::density::Region;

#[cfg(feature = "plot")]
pub mod iq_plot;
