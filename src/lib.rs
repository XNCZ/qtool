//! # cz-qtool — Quantum Process Tomography (QPT) toolkit
//!
//! Pure-Rust (`faer` for linear algebra, `plotly` for plotting) quantum process
//! tomography (QPT) for an arbitrary number of qubits under the completely positive
//! and trace-preserving (CPTP) constraint: maximum-likelihood estimation (QPT-MLE)
//! solved with an accelerated projected-gradient method (FISTA) that produces a
//! Pauli transfer matrix (PTM).
//!
//! All public types live in the [`qpt`] module. Import them with
//! `use cz_qtool::qpt::*;` or `use cz_qtool::qpt::{QptDataset, QptSolver, ...};`.
//!
//! # Crate layout
//!
//! - [`qpt::QptDataset`] — collect experimental data points.
//! - [`qpt::QptSolver`] — run the FISTA reconstruction.
//! - [`qpt::QptResult`], [`qpt::PtmMatrix`] — the reconstructed PTM plus diagnostics.
//! - [`qpt::QuantumState`], [`qpt::PauliString`], [`qpt::Pauli`], [`qpt::SingleQubitState`] —
//!   states and operators.

pub mod qpt;
