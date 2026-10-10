//! # qtool — analysis toolbox for quantum computation
//!
//! Two halves, one crate: **superconducting-qubit calibration** ([`superconductor`]) and
//! **quantum process tomography** ([`qpt`]).
//!
//! The calibration half fits the standard device experiments —
//! [`s21`](superconductor::s21), [`qspec`](superconductor::qspec),
//! [`rabi`](superconductor::rabi), [`ramsey`](superconductor::ramsey),
//! [`t1`](superconductor::t1), [`t2_echo`](superconductor::t2_echo) and the
//! [`drag`](superconductor::drag) family — and renders each one as an interactive plotly
//! report: a self-contained HTML fragment that assumes only that the host page loads
//! `plotly.js`. It ships with the `plot` feature, on by default; without it the numerics
//! still build.
//!
//! The tomography half is pure-Rust (`faer` for linear algebra, `plotly` for plotting)
//! quantum process tomography (QPT) for an arbitrary number of qubits under the completely
//! positive and trace-preserving (CPTP) constraint: maximum-likelihood estimation (QPT-MLE)
//! solved with an accelerated projected-gradient method (FISTA) that produces a Pauli
//! transfer matrix (PTM).
//!
//! # Crate layout
//!
//! - [`superconductor`] — one module per experiment (the fit, its `_batch` twin and the
//!   report entry points), plus the readout vocabulary every experiment shares
//!   ([`StateCenters`](superconductor::StateCenters), [`p1`](superconductor::p1)) and the
//!   readout diagnostics ([`iq`](superconductor::iq)) / trajectory plots
//!   ([`bloch`](superconductor::bloch)).
//! - [`qpt::QptDataset`] — collect experimental data points.
//! - [`qpt::QptSolver`] — run the FISTA reconstruction.
//! - [`qpt::QptResult`], [`qpt::PtmMatrix`] — the reconstructed PTM plus diagnostics.
//! - [`qpt::QuantumState`], [`qpt::PauliString`], [`qpt::Pauli`], [`qpt::SingleQubitState`] —
//!   states and operators.
//!
//! The full tour — every report with screenshots, the Rust and Python quick-starts and the
//! data-export guide — is in the README:
//! [GitHub](https://github.com/xncz/qtool#readme) · [crates.io](https://crates.io/crates/qtool).

pub mod common;
pub mod superconductor;
pub use common::qpt;
mod utils;
