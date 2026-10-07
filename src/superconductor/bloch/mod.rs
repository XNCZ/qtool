//! 布洛赫球轨迹：三基（X/Y/Z）读出 IQ → 布洛赫向量（`bloch`），以及报告渲染（`bloch_plot`，
//! feature = "plot"）。
//!
//! baseline `qana` 的 Rust 移植：`plot/plot_Li.py` 里"从布居到布洛赫球"那段换算的公共部分
//! （`⟨P⟩ = 1 − 2P1`，P1 是 IQ 沿 |0>–|1> 连线投影归一化后的激发概率）。baseline 把它逐字抄在
//! rabi trace / tomo phase / jazz pulse tomo / iSWAP tomo 等五六个函数里，本模块只留所有实验
//! 共用的那一步：**给定扫描轴与三基 IQ，还一组 `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)(扫描)`**。实验侧的东西——相位
//! 解缠与差分剥相、随扫描量走的拟合、读出按 3 交错存的拆轴约定——都不在这里。

pub mod bloch;

// 模型文件是 `bloch/bloch.rs`，不提一层的话对外路径是 `bloch::bloch::bloch_vector`；这里把它的
// 公共项收平到本模块，外部统一写 `bloch::bloch_vector`（照 s21 / qspec / rabi / ramsey 的做法）。
pub use bloch::*;

#[cfg(feature = "plot")]
pub mod bloch_plot;
