//! # qtool — Quantum Process Tomography (QPT) toolkit
//!
//! 纯 Rust（`faer` 矩阵计算 + `plotly` 绘图）实现的任意 $n$ 比特完全正保迹
//! (CPTP) 极大似然量子过程层析 (QPT-MLE)：基于 FISTA 加速投影梯度法输出
//! Pauli 转移矩阵 (PTM)。
//!
//! 关键类型定义在 [`qpt`] 模块内，外部直接通过
//! `use qtool::qpt::*;` 或 `use qtool::qpt::{QptDataset, QptSolver, ...};` 引用。

pub mod qpt;
