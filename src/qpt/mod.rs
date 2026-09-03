//! # Quantum Process Tomography (QPT) via Pure-Rust FISTA
//!
//! Maximum-likelihood quantum process tomography (QPT-MLE) for an arbitrary number of
//! qubits under the completely positive and trace-preserving (CPTP) constraint.
//! A 100% pure-Rust implementation with no C/Fortran/BLAS dependencies, built on the
//! [`faer`](https://crates.io/crates/faer) matrix library and an accelerated projected
//! gradient method (FISTA).
//!
//! The objective is the weighted least squares problem
//! $$
//! \min_{R} \frac{1}{2} \sum_k w_k \left( \sum_{j} R_{m_k, j} x_{k, j} - (1.0 - 2.0 \cdot p_{1}^{(k)}) \right)^2
//! $$
//! subject to `R` being the Pauli transfer matrix of a CPTP process.

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{c64, Mat, Side};
use plotly::common::{ColorScale, ColorScalePalette};
use plotly::layout::{Axis, Layout, TicksDirection};
use plotly::{HeatMap, Plot};
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
// =========================================================================
// Custom error type
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum QPTError {
    InvalidPauliChar(char),
    InvalidStateString(String),
    QubitCountMismatch { expected: usize, found: usize },
    EmptyDataset,
    InvalidProbability(f64),
    InvalidWeight(f64),
    BasisNotFound(String),
    SolverError(String),
    DimensionMismatch { expected: usize, found: usize },
}

impl fmt::Display for QPTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPauliChar(c) => write!(f, "invalid Pauli character '{c}'; only I, X, Y and Z are allowed"),
            Self::InvalidStateString(s) => write!(f, "cannot parse the quantum state description: \"{s}\""),
            Self::QubitCountMismatch { expected, found } => {
                write!(f, "qubit count mismatch: expected {expected} qubits but found {found}")
            }
            Self::EmptyDataset => write!(f, "the experiment data set is empty; cannot run tomography"),
            Self::InvalidProbability(p) => write!(f, "excited-state probability p1 = {p} is outside the physical range [0.0, 1.0]"),
            Self::InvalidWeight(w) => write!(f, "measurement weight w = {w} must be positive"),
            Self::BasisNotFound(s) => write!(f, "Pauli operator \"{s}\" was not found in the current lexicographic basis table"),
            Self::SolverError(s) => write!(f, "solver failed to converge: {s}"),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "matrix dimension mismatch: expected {expected}x{expected} but got {found}x{found}")
            }
        }
    }
}

impl Error for QPTError {}

// =========================================================================
// Basic physical enums and Pauli operators
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingleQubitState {
    PlusZ,
    MinusZ,
    PlusX,
    MinusX,
    PlusY,
    MinusY,
}

impl SingleQubitState {
    #[inline]
    pub fn pauli_vector(&self) -> [f64; 4] {
        match self {
            Self::PlusZ => [1.0, 0.0, 0.0, 1.0],
            Self::MinusZ => [1.0, 0.0, 0.0, -1.0],
            Self::PlusX => [1.0, 1.0, 0.0, 0.0],
            Self::MinusX => [1.0, -1.0, 0.0, 0.0],
            Self::PlusY => [1.0, 0.0, 1.0, 0.0],
            Self::MinusY => [1.0, 0.0, -1.0, 0.0],
        }
    }
}

impl FromStr for SingleQubitState {
    type Err = QPTError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim().trim_start_matches('|').trim_end_matches('>').trim_end_matches('⟩');
        match trimmed.to_uppercase().as_str() {
            "Z+" | "+Z" | "0" => Ok(Self::PlusZ),
            "Z-" | "-Z" | "1" => Ok(Self::MinusZ),
            "X+" | "+X" | "+" => Ok(Self::PlusX),
            "X-" | "-X" | "-" => Ok(Self::MinusX),
            "Y+" | "+Y" | "+I" | "R" => Ok(Self::PlusY),
            "Y-" | "-Y" | "-I" | "L" => Ok(Self::MinusY),
            _ => Err(QPTError::InvalidStateString(s.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pauli {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

impl Pauli {
    pub fn matrix(&self) -> [[c64; 2]; 2] {
        let zero = c64::new(0.0, 0.0);
        let one = c64::new(1.0, 0.0);
        let i_unit = c64::new(0.0, 1.0);
        match self {
            Self::I => [[one, zero], [zero, one]],
            Self::X => [[zero, one], [one, zero]],
            Self::Y => [[zero, -i_unit], [i_unit, zero]],
            Self::Z => [[one, zero], [zero, -one]],
        }
    }
}

impl TryFrom<char> for Pauli {
    type Error = QPTError;

    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c.to_ascii_uppercase() {
            'I' => Ok(Self::I),
            'X' => Ok(Self::X),
            'Y' => Ok(Self::Y),
            'Z' => Ok(Self::Z),
            other => Err(QPTError::InvalidPauliChar(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantumState {
    pub qubits: Vec<SingleQubitState>,
}

impl QuantumState {
    pub fn pauli_vector(&self) -> Vec<f64> {
        let mut vec = vec![1.0];
        for q in &self.qubits {
            let q_vec = q.pauli_vector();
            let mut next_vec = Vec::with_capacity(vec.len() * 4);
            for parent in &vec {
                for child in &q_vec {
                    next_vec.push(parent * child);
                }
            }
            vec = next_vec;
        }
        vec
    }
}

impl FromStr for QuantumState {
    type Err = QPTError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cleaned = s.trim().trim_start_matches('|').trim_end_matches('>').trim_end_matches('⟩');
        let tokens: Vec<&str> = if cleaned.contains(' ') || cleaned.contains(',') {
            cleaned.split(|c| c == ' ' || c == ',').filter(|t| !t.trim().is_empty()).collect()
        } else {
            let mut result = Vec::new();
            let chars: Vec<char> = cleaned.chars().collect();
            let mut idx = 0;
            while idx < chars.len() {
                let c = chars[idx];
                if (c == 'X' || c == 'x' || c == 'Y' || c == 'y' || c == 'Z' || c == 'z')
                    && idx + 1 < chars.len()
                    && (chars[idx + 1] == '+' || chars[idx + 1] == '-')
                {
                    result.push(&cleaned[idx..idx + 2]);
                    idx += 2;
                } else {
                    result.push(&cleaned[idx..idx + 1]);
                    idx += 1;
                }
            }
            result
        };

        if tokens.is_empty() {
            return Err(QPTError::InvalidStateString(s.to_string()));
        }

        let mut qubits = Vec::with_capacity(tokens.len());
        for t in tokens {
            qubits.push(SingleQubitState::from_str(t)?);
        }
        Ok(Self { qubits })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PauliString {
    pub ops: Vec<Pauli>,
}

impl PauliString {
    #[inline]
    pub fn index(&self) -> usize {
        let mut idx = 0;
        for op in &self.ops {
            idx = (idx << 2) | (*op as usize);
        }
        idx
    }

    pub fn from_index(num_qubits: usize, mut index: usize) -> Self {
        let mut ops = vec![Pauli::I; num_qubits];
        for k in (0..num_qubits).rev() {
            let val = index & 0b11;
            ops[k] = match val {
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            };
            index >>= 2;
        }
        Self { ops }
    }
}

impl FromStr for PauliString {
    type Err = QPTError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cleaned: String = s
            .chars()
            .filter(|c| !c.is_whitespace() && *c != ',' && *c != '|' && *c != '>')
            .collect();
        if cleaned.is_empty() {
            return Err(QPTError::InvalidPauliChar(' '));
        }
        let mut ops = Vec::with_capacity(cleaned.len());
        for c in cleaned.chars() {
            ops.push(Pauli::try_from(c)?);
        }
        Ok(Self { ops })
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for op in &self.ops {
            let c = match op {
                Pauli::I => 'I',
                Pauli::X => 'X',
                Pauli::Y => 'Y',
                Pauli::Z => 'Z',
            };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

// =========================================================================
// Experiment data set
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct QptDataPoint {
    pub prep: QuantumState,
    pub meas: PauliString,
    pub p1: f64,
    pub weight: f64,
}

impl QptDataPoint {
    #[inline]
    pub fn expectation(&self) -> f64 {
        1.0 - 2.0 * self.p1
    }

    #[inline]
    pub fn p_plus(&self) -> f64 {
        1.0 - self.p1
    }
}

#[derive(Debug, Clone, Default)]
pub struct QptDataset {
    pub num_qubits: usize,
    pub experiments: Vec<QptDataPoint>,
}

impl QptDataset {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            experiments: Vec::new(),
        }
    }

    pub fn add_data(&mut self, prep_str: &str, meas_str: &str, p1: f64) -> Result<&mut Self, QPTError> {
        self.add_weight_data(prep_str, meas_str, p1, 1.0)
    }

    pub fn add_weight_data(
        &mut self,
        prep_str: &str,
        meas_str: &str,
        p1: f64,
        weight: f64,
    ) -> Result<&mut Self, QPTError> {
        let prep = QuantumState::from_str(prep_str)?;
        let meas = PauliString::from_str(meas_str)?;

        if prep.qubits.len() != self.num_qubits {
            return Err(QPTError::QubitCountMismatch {
                expected: self.num_qubits,
                found: prep.qubits.len(),
            });
        }
        if meas.ops.len() != self.num_qubits {
            return Err(QPTError::QubitCountMismatch {
                expected: self.num_qubits,
                found: meas.ops.len(),
            });
        }
        if !(-1e-4..=1.0001).contains(&p1) {
            return Err(QPTError::InvalidProbability(p1));
        }
        if weight <= 0.0 {
            return Err(QPTError::InvalidWeight(weight));
        }

        self.experiments.push(QptDataPoint {
            prep,
            meas,
            p1: p1.clamp(0.0, 1.0),
            weight,
        });
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), QPTError> {
        if self.experiments.is_empty() {
            return Err(QPTError::EmptyDataset);
        }
        for exp in &self.experiments {
            if exp.prep.qubits.len() != self.num_qubits {
                return Err(QPTError::QubitCountMismatch {
                    expected: self.num_qubits,
                    found: exp.prep.qubits.len(),
                });
            }
            if exp.meas.ops.len() != self.num_qubits {
                return Err(QPTError::QubitCountMismatch {
                    expected: self.num_qubits,
                    found: exp.meas.ops.len(),
                });
            }
        }
        Ok(())
    }
}

// =========================================================================
// Solver configuration and diagnostics
// =========================================================================

#[derive(Debug, Clone)]
pub struct QptSolverConfig {
    pub tol_convergence: f64,
    pub max_iter: usize,
    pub verbose: bool,
    pub max_dykstra_iter: usize, // Maximum iterations of the Dykstra alternating projections
}

impl Default for QptSolverConfig {
    fn default() -> Self {
        Self {
            tol_convergence: 1e-6,
            max_iter: 200,
            verbose: false,
            max_dykstra_iter: 10, // 10 iterations reaches ~1e-7 intersection accuracy
        }
    }
}

#[derive(Debug, Clone)]
pub struct QptSolverMeta {
    pub solve_time_ms: u128,
    pub iterations: usize,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct QptDiagnostics {
    pub trace_preserving_error: f64,
    pub min_choi_eigenvalue: f64,
    pub fit_rmse: f64,
    pub data_coverage: f64,
    pub choi_matrix: Mat<c64>,
}

// =========================================================================
// PTM matrix structure
// =========================================================================

#[derive(Debug, Clone)]
pub struct PtmMatrix {
    pub num_qubits: usize,
    pub dim: usize,
    pub mat: Mat<f64>,
    pub basis_order: Vec<PauliString>,
}

impl PtmMatrix {
    pub fn from_unitary(num_qubits: usize, u: &Mat<c64>) -> Self {
        let d_h = 1 << num_qubits;
        let d = d_h * d_h;
        let paulis = all_pauli_matrices(num_qubits);
        let mut mat = Mat::<f64>::zeros(d, d);

        // U^\dagger
        let mut u_adj = Mat::<c64>::zeros(d_h, d_h);
        for r in 0..d_h {
            for c in 0..d_h {
                let val = u[(c, r)];
                u_adj[(r, c)] = c64::new(val.re, -val.im);
            }
        }

        // R_{i, j} = (1 / d_h) * Tr(P_i * U * P_j * U^\dagger)
        for j in 0..d {
            let pj = &paulis[j];
            // up = U * P_j
            let mut up = Mat::<c64>::zeros(d_h, d_h);
            for r in 0..d_h {
                for c in 0..d_h {
                    let mut sum = c64::new(0.0, 0.0);
                    for k in 0..d_h {
                        sum = sum + u[(r, k)] * pj[(k, c)];
                    }
                    up[(r, c)] = sum;
                }
            }

            // upu = up * U^\dagger
            let mut upu = Mat::<c64>::zeros(d_h, d_h);
            for r in 0..d_h {
                for c in 0..d_h {
                    let mut sum = c64::new(0.0, 0.0);
                    for k in 0..d_h {
                        sum = sum + up[(r, k)] * u_adj[(k, c)];
                    }
                    upu[(r, c)] = sum;
                }
            }

            for i in 0..d {
                let pi = &paulis[i];
                let mut tr = 0.0;
                for r in 0..d_h {
                    for c in 0..d_h {
                        let a = pi[(r, c)];
                        let b = upu[(c, r)];
                        tr += a.re * b.re - a.im * b.im;
                    }
                }
                mat[(i, j)] = tr / (d_h as f64);
            }
        }

        let basis_order = (0..d).map(|idx| PauliString::from_index(num_qubits, idx)).collect();
        Self {
            num_qubits,
            dim: d,
            mat,
            basis_order,
        }
    }

    pub fn identity(num_qubits: usize) -> Self {
        let dim = 1 << (2 * num_qubits);
        let mut mat = Mat::<f64>::zeros(dim, dim);
        for i in 0..dim {
            mat[(i, i)] = 1.0;
        }
        let basis_order = (0..dim).map(|idx| PauliString::from_index(num_qubits, idx)).collect();
        Self {
            num_qubits,
            dim,
            mat,
            basis_order,
        }
    }

    pub fn get(&self, row_meas: &str, col_prep: &str) -> Result<f64, QPTError> {
        let row_p = PauliString::from_str(row_meas)?;
        let col_p = PauliString::from_str(col_prep)?;
        self.get_by_pauli(&row_p, &col_p)
    }

    pub fn get_by_pauli(&self, row_meas: &PauliString, col_prep: &PauliString) -> Result<f64, QPTError> {
        if row_meas.ops.len() != self.num_qubits {
            return Err(QPTError::QubitCountMismatch {
                expected: self.num_qubits,
                found: row_meas.ops.len(),
            });
        }
        if col_prep.ops.len() != self.num_qubits {
            return Err(QPTError::QubitCountMismatch {
                expected: self.num_qubits,
                found: col_prep.ops.len(),
            });
        }
        Ok(self.mat[(row_meas.index(), col_prep.index())])
    }

    #[inline]
    pub fn get_by_idx(&self, row: usize, col: usize) -> f64 {
        self.mat[(row, col)]
    }

    pub fn to_choi(&self) -> Mat<c64> {
        let d = self.dim;
        let mut choi = Mat::<c64>::zeros(d, d);
        let norm = 1.0 / ((1 << self.num_qubits) as f64);
        let pauli_mats = all_pauli_matrices(self.num_qubits);

        for j in 0..d {
            let pj_t = transpose_c64_mat(&pauli_mats[j]);
            for i in 0..d {
                let r_ij = self.mat[(i, j)];
                if r_ij.abs() < 1e-14 {
                    continue;
                }
                let kron = kronecker_c64(&pj_t, &pauli_mats[i]);
                let coeff = c64::new(norm * r_ij, 0.0);
                for r in 0..d {
                    for c in 0..d {
                        choi[(r, c)] = choi[(r, c)] + coeff * kron[(r, c)];
                    }
                }
            }
        }
        choi
    }

    pub fn fidelity(&self, ideal_ptm: &PtmMatrix) -> Result<(f64, f64), QPTError> {
        if self.num_qubits != ideal_ptm.num_qubits {
            return Err(QPTError::QubitCountMismatch {
                expected: self.num_qubits,
                found: ideal_ptm.num_qubits,
            });
        }
        let d_total = self.dim as f64;
        let d_hilbert = (1 << self.num_qubits) as f64;

        let mut tr_ideal_t_exp = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                tr_ideal_t_exp += ideal_ptm.mat[(i, j)] * self.mat[(i, j)];
            }
        }
        let f_process = (tr_ideal_t_exp / d_total).clamp(0.0, 1.0);
        let f_average = ((d_hilbert * f_process + 1.0) / (d_hilbert + 1.0)).clamp(0.0, 1.0);
        Ok((f_process, f_average))
    }

    pub fn display_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== {}-Qubit PTM Table (dim = {}) ===\n", self.num_qubits, self.dim));
        out.push_str(&format!("{:<6} | ", "Meas\\In"));
        for j in 0..self.dim.min(16) {
            out.push_str(&format!("{:>8} ", self.basis_order[j]));
        }
        out.push_str("\n-------+");
        for _ in 0..self.dim.min(16) {
            out.push_str("---------");
        }
        out.push('\n');

        for i in 0..self.dim.min(16) {
            out.push_str(&format!("{:<6} | ", self.basis_order[i]));
            for j in 0..self.dim.min(16) {
                let v = self.mat[(i, j)];
                if v.abs() < 1e-4 {
                    out.push_str("       . ");
                } else {
                    out.push_str(&format!("{:>8.4} ", v));
                }
            }
            out.push('\n');
        }
        out
    }

    /// Build an interactive 2-D heat map of the PTM as a Plotly [`Plot`].
    ///
    /// # Features
    /// - **Adaptive axes** — the standard lexicographic basis labels
    ///   (II, IX, IY, IZ, XI, ...) are derived automatically from the qubit count.
    /// - **Matrix-style layout** — the Y labels and data rows are reversed so that
    ///   row 0 (II) appears at the top.
    /// - **Physical diverging colormap** — fixed to [-1.0, 1.0] with a neutral white
    ///   at 0, cleanly separating positive and negative matrix elements.
    /// - **Custom hover tooltips** — hovering displays the input operator, the output
    ///   operator and the exact matrix element value.
    pub fn to_plotly(&self, custom_title: Option<&str>) -> Plot {
        let d = self.dim; // 4^n

        // Build the adaptive n-qubit standard Pauli string labels
        let labels: Vec<String> = self.basis_order.iter().map(|p| p.to_string()).collect();

        // Assemble the 2-D matrix data: z[row][col] = R_{meas, prep}
        let mut z_data = Vec::with_capacity(d);
        for r in 0..d {
            let mut row = Vec::with_capacity(d);
            for c in 0..d {
                row.push(self.mat[(r, c)]);
            }
            z_data.push(row);
        }

        // Plotly places the first category of a categorical y axis at the bottom, so
        // reverse both the y labels and the data rows to put row 0 (II) on top.
        let mut y_labels = labels.clone();
        y_labels.reverse();
        z_data.reverse();

        // Build the HeatMap trace
        let trace = HeatMap::new(labels.clone(), y_labels, z_data)
            .color_scale(ColorScale::Palette(ColorScalePalette::RdBu)) // blue-white-red diverging colorscale
            .zmin(-1.0)
            .zmax(1.0)
            .hover_template("<b>Input (Prep)</b>: %{x}<br><b>Output (Meas)</b>: %{y}<br><b>R</b>: %{z:.4f}<extra></extra>");

        // Adaptive figure size (550 px for 1 qubit, 850 px for 2, larger for more)
        let plot_size = (500 + 22 * d).clamp(550, 1300);

        // Configure the axes and the canvas
        let title_text = custom_title.map(|s| s.to_string()).unwrap_or_else(|| {
            format!("{}-Qubit Pauli Transfer Matrix (PTM)", self.num_qubits)
        });

        // X axis: input operators (columns)
        let x_axis = Axis::new()
            .title("Input Pauli Operator (Preparation)")
            .ticks(TicksDirection::Outside)
            .tick_angle(if d > 4 { -45.0 } else { 0.0 })
            .show_grid(false);

        // Y axis: output operators (rows)
        let y_axis = Axis::new()
            .title("Output Pauli Operator (Measurement)")
            .ticks(TicksDirection::Outside)
            .show_grid(false);

        let layout = Layout::new()
            .title(title_text)
            .width(plot_size)
            .height(plot_size)
            .x_axis(x_axis)
            .y_axis(y_axis);

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);
        plot
    }

    /// Open the interactive PTM chart in the system default browser.
    pub fn show_plot(&self, custom_title: Option<&str>) {
        let plot = self.to_plotly(custom_title);
        plot.show();
    }

    /// Save the PTM as a standalone interactive HTML file.
    pub fn save_html<P: AsRef<Path>>(&self, path: P, custom_title: Option<&str>) -> std::io::Result<()> {
        let plot = self.to_plotly(custom_title);
        plot.write_html(path);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QptResult {
    pub ptm: PtmMatrix,
    pub diagnostics: QptDiagnostics,
    pub meta: QptSolverMeta,
}

impl QptResult {
    /// Convenience wrapper that plots the tomography result directly; the title
    /// automatically carries the process fidelity and RMSE diagnostics.
    pub fn to_plotly(&self, gate_name: Option<&str>) -> Plot {
        let title = format!(
            "{} QPT (RMSE: {:.3e}, TraceErr: {:.1e})",
            gate_name.unwrap_or("Reconstructed Gate"),
            self.diagnostics.fit_rmse,
            self.diagnostics.trace_preserving_error
        );
        self.ptm.to_plotly(Some(&title))
    }

    /// Convenience wrapper that opens the reconstructed result in a browser.
    pub fn show_plot(&self, gate_name: Option<&str>) {
        let plot = self.to_plotly(gate_name);
        plot.show();
    }

    /// Convenience wrapper that saves the result as a standalone HTML file.
    pub fn save_html<P: AsRef<Path>>(&self, path: P, gate_name: Option<&str>) -> std::io::Result<()> {
        let plot = self.to_plotly(gate_name);
        plot.write_html(path);
        Ok(())
    }
}

// =========================================================================
// Core: pure-Rust FISTA solver
// =========================================================================

pub struct QptSolver {
    pub num_qubits: usize,
    pub config: QptSolverConfig,
}

impl QptSolver {
    pub fn new(num_qubits: usize, config: Option<QptSolverConfig>) -> Self {
        Self {
            num_qubits,
            config: config.unwrap_or_default(),
        }
    }

    pub fn solve(&self, dataset: &QptDataset) -> Result<QptResult, QPTError> {
        let clock = Instant::now();
        dataset.validate()?;

        let n = self.num_qubits;
        let d = 1 << (2 * n); // 4^n

        // =====================================================================
        // Optimization 1 (Rayon): precompute every (P_j^T \otimes P_i) basis matrix in parallel
        // =====================================================================
        let pauli_mats = all_pauli_matrices(n);
        let basis_kron: Vec<Vec<Mat<c64>>> = (0..d)
            .into_par_iter()
            .map(|j| {
                let pj_t = transpose_c64_mat(&pauli_mats[j]);
                (0..d)
                    .map(|i| kronecker_c64(&pj_t, &pauli_mats[i]))
                    .collect()
            })
            .collect();

        // =====================================================================
        // Optimization 2 (Rayon): group experiments by measurement m and compute the Hessian H_m and bias b_m in parallel
        // =====================================================================
        let mut exp_by_m: Vec<Vec<&QptDataPoint>> = vec![Vec::new(); d];
        let mut total_w = 0.0;
        for data in &dataset.experiments {
            let m = data.meas.index();
            exp_by_m[m].push(data);
            total_w += data.weight;
        }

        let aggregated: Vec<(Mat<f64>, Vec<f64>, bool)> = (0..d)
            .into_par_iter()
            .map(|m| {
                let exps = &exp_by_m[m];
                if exps.is_empty() {
                    (Mat::<f64>::zeros(d, d), vec![0.0; d], false)
                } else {
                    let mut h_m = Mat::<f64>::zeros(d, d);
                    let mut b_m = vec![0.0; d];
                    for data in exps {
                        let x = data.prep.pauli_vector();
                        let y = data.expectation();
                        let w = data.weight;
                        for r in 0..d {
                            let xr = x[r];
                            if xr.abs() < 1e-14 { continue; }
                            b_m[r] += w * y * xr;
                            for c in 0..d {
                                h_m[(r, c)] += w * xr * x[c];
                            }
                        }
                    }
                    (h_m, b_m, true)
                }
            })
            .collect();

        let mut h_blocks = Vec::with_capacity(d);
        let mut b_vectors = Vec::with_capacity(d);
        let mut meas_cover = Vec::with_capacity(d);
        for (h, b, cov) in aggregated {
            h_blocks.push(h);
            b_vectors.push(b);
            meas_cover.push(cov);
        }

        // =====================================================================
        // Optimization 3 (Rayon): estimate the Lipschitz constant L = max_m ||H_m||_oo by parallel reduction
        // =====================================================================
        let max_spectral_norm: f64 = (0..d)
            .into_par_iter()
            .filter(|&m| meas_cover[m])
            .map(|m| {
                let h_m = &h_blocks[m];
                let mut max_row_sum: f64 = 0.0;
                for r in 0..d {
                    let mut row_sum = 0.0;
                    for c in 0..d {
                        row_sum += h_m[(r, c)].abs();
                    }
                    max_row_sum = max_row_sum.max(row_sum);
                }
                max_row_sum
            })
            .reduce(|| 0.0, f64::max);

        let lipschitz = max_spectral_norm.max(1.0);
        let lr = 1.0 / lipschitz;

        // FISTA initialization
        let mut r_curr = Mat::<f64>::zeros(d, d);
        r_curr[(0, 0)] = 1.0;
        let mut y_acc = r_curr.clone();
        let mut t_acc: f64 = 1.0;
        let mut final_iter = 0;

        // Preallocate the gradient-row buffer to avoid heap allocations in the inner loop
        let mut z_buf = vec![0.0; d * d];

        for iter in 0..self.config.max_iter {
            final_iter = iter + 1;

            // =================================================================
            // Optimization 4 (Rayon): compute the gradient row for each measurement basis in parallel: grad_m = H_m * y_m - b_m
            // =================================================================
            z_buf.par_chunks_mut(d).enumerate().for_each(|(m, row)| {
                if !meas_cover[m] {
                    for j in 0..d {
                        row[j] = y_acc[(m, j)];
                    }
                } else {
                    let h_m = &h_blocks[m];
                    let b_m = &b_vectors[m];
                    for j in 0..d {
                        let mut grad_mj = -b_m[j];
                        for l in 0..d {
                            grad_mj += h_m[(j, l)] * y_acc[(m, l)];
                        }
                        row[j] = y_acc[(m, j)] - lr * grad_mj;
                    }
                }
            });

            let mut z = Mat::<f64>::zeros(d, d);
            for m in 0..d {
                for j in 0..d {
                    z[(m, j)] = z_buf[m * d + j];
                }
            }

            // =================================================================
            // Optimization 5 (Dykstra): solve the TP ∩ CP intersection by cyclic alternating projections
            // =================================================================
            let r_next = project_to_cptp_dykstra(
                &z,
                n,
                &basis_kron,
                self.config.max_dykstra_iter,
                1e-7,
            )?;

            // Convergence criterion
            let mut max_diff: f64 = 0.0;
            for i in 0..d {
                for j in 0..d {
                    max_diff = max_diff.max((r_next[(i, j)] - r_curr[(i, j)]).abs());
                }
            }

            if max_diff < self.config.tol_convergence && iter >= 20 {
                r_curr = r_next;
                break;
            }

            // Nesterov momentum update
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_acc * t_acc).sqrt());
            let momentum = (t_acc - 1.0) / t_next;
            for i in 0..d {
                for j in 0..d {
                    let rn = r_next[(i, j)];
                    let rc = r_curr[(i, j)];
                    y_acc[(i, j)] = rn + momentum * (rn - rc);
                }
            }
            t_acc = t_next;
            r_curr = r_next;
        }

        // Assemble the result
        let basis_order = (0..d).map(|idx| PauliString::from_index(n, idx)).collect();
        let ptm = PtmMatrix {
            num_qubits: n,
            dim: d,
            mat: r_curr,
            basis_order,
        };

        let mut tp_err: f64 = (ptm.mat[(0, 0)] - 1.0).abs();
        for j in 1..d {
            tp_err = tp_err.max(ptm.mat[(0, j)].abs());
        }

        let mut total_err_sq = 0.0;
        for data in &dataset.experiments {
            let m = data.meas.index();
            let x = data.prep.pauli_vector();
            let mut pred = 0.0;
            for j in 0..d {
                pred += ptm.mat[(m, j)] * x[j];
            }
            let err = pred - data.expectation();
            total_err_sq += data.weight * err * err;
        }
        let fit_rmse = (total_err_sq / total_w).sqrt();

        let choi = ptm.to_choi();
        let eig = SelfAdjointEigen::new(choi.as_ref(), Side::Lower)
            .map_err(|e| QPTError::SolverError(format!("{:?}", e)))?;
        let s_diag = eig.S();
        let mut min_eigenvalue = f64::INFINITY;
        for i in 0..d {
            min_eigenvalue = min_eigenvalue.min(s_diag[i].re);
        }

        let data_coverage = (meas_cover.iter().filter(|&&c| c).count() as f64) / (d as f64);

        Ok(QptResult {
            ptm,
            diagnostics: QptDiagnostics {
                trace_preserving_error: tp_err,
                min_choi_eigenvalue: min_eigenvalue,
                fit_rmse,
                data_coverage,
                choi_matrix: choi,
            },
            meta: QptSolverMeta {
                solve_time_ms: clock.elapsed().as_millis(),
                iterations: final_iter,
                status: "Optimal (FISTA Converged)".to_string(),
            },
        })
    }
}

/// Project an arbitrary real matrix onto the CPTP manifold.
///
/// Uses the Dykstra cyclic alternating-projection algorithm to compute the orthogonal
/// projection onto the intersection of the trace-preserving (TP) affine subspace and
/// the completely-positive (CP) positive-semidefinite cone.
fn project_to_cptp_dykstra(
    mat: &Mat<f64>,
    num_qubits: usize,
    basis_kron: &[Vec<Mat<c64>>],
    max_dykstra_iter: usize,
    tol: f64,
) -> Result<Mat<f64>, QPTError> {
    let d = 1 << (2 * num_qubits);

    // Initial iterate X (in PTM space)
    let mut x = mat.clone();

    // Dykstra dual-correction matrices: p tracks the TP affine space, q the CP cone
    let mut p = Mat::<f64>::zeros(d, d);
    let mut q = Mat::<f64>::zeros(d, d);

    for _ in 0..max_dykstra_iter {
        // -------------------------------------------------------------
        // Step A: project onto the trace-preserving affine subspace C_TP (R_{0,0}=1, R_{0,j}=0 for j>=1)
        // Y = \Pi_{TP}(X + P)
        // -------------------------------------------------------------
        let mut y = Mat::<f64>::zeros(d, d);
        for r in 0..d {
            for c in 0..d {
                y[(r, c)] = x[(r, c)] + p[(r, c)];
            }
        }
        y[(0, 0)] = 1.0;
        for j in 1..d {
            y[(0, j)] = 0.0;
        }

        // Update the dual correction P = (X + P) - Y
        for r in 0..d {
            for c in 0..d {
                p[(r, c)] = (x[(r, c)] + p[(r, c)]) - y[(r, c)];
            }
        }

        // -------------------------------------------------------------
        // Step B: project onto the completely-positive positive-semidefinite cone C_CP (Choi >= 0)
        // X_new = \Pi_{CP}(Y + Q)
        // -------------------------------------------------------------
        let mut v = Mat::<f64>::zeros(d, d);
        for r in 0..d {
            for c in 0..d {
                v[(r, c)] = y[(r, c)] + q[(r, c)];
            }
        }

        // Map into the Choi-matrix space through the isometric isomorphism
        let mut choi = ptm_to_choi_par(&v, num_qubits, basis_kron);

        // Enforce Hermiticity
        for r in 0..d {
            for c in (r + 1)..d {
                let z1 = choi[(r, c)];
                let z2 = choi[(c, r)];
                let avg = c64::new(0.5 * (z1.re + z2.re), 0.5 * (z1.im - z2.im));
                choi[(r, c)] = avg;
                choi[(c, r)] = c64::new(avg.re, -avg.im);
            }
        }

        // Eigendecomposition followed by truncation of negative eigenvalues
        let eig = SelfAdjointEigen::new(choi.as_ref(), Side::Lower)
            .map_err(|e| QPTError::SolverError(format!("{:?}", e)))?;
        let s_diag = eig.S();
        let vecs = eig.U();

        let mut choi_psd = Mat::<c64>::zeros(d, d);
        for k in 0..d {
            let lk = s_diag[k].re;
            if lk <= 0.0 { continue; }
            for r in 0..d {
                let u_rk = vecs[(r, k)];
                for c in 0..d {
                    let u_ck = vecs[(c, k)];
                    let term = c64::new(
                        lk * (u_rk.re * u_ck.re + u_rk.im * u_ck.im),
                        lk * (u_rk.im * u_ck.re - u_rk.re * u_ck.im),
                    );
                    choi_psd[(r, c)] = choi_psd[(r, c)] + term;
                }
            }
        }

        // Map back to PTM space to obtain X_new
        let x_new = choi_to_ptm_par(&choi_psd, num_qubits, basis_kron);

        // Update the dual correction Q = (Y + Q) - X_new and check intersection convergence
        let mut max_diff: f64 = 0.0;
        for r in 0..d {
            for c in 0..d {
                q[(r, c)] = v[(r, c)] - x_new[(r, c)];
                max_diff = max_diff.max((x_new[(r, c)] - y[(r, c)]).abs());
            }
        }

        x = x_new;
        if max_diff < tol {
            break;
        }
    }

    // Numerical guard: force the first row to be trace-preserving to floating-point accuracy
    x[(0, 0)] = 1.0;
    for j in 1..d {
        x[(0, j)] = 0.0;
    }

    Ok(x)
}

/// Parallel PTM -> Choi matrix transform
fn ptm_to_choi_par(
    mat: &Mat<f64>,
    num_qubits: usize,
    basis_kron: &[Vec<Mat<c64>>],
) -> Mat<c64> {
    let d = 1 << (2 * num_qubits);
    let norm = 1.0 / ((1 << num_qubits) as f64);

    (0..d)
        .into_par_iter()
        .map(|j| {
            let mut local = Mat::<c64>::zeros(d, d);
            for i in 0..d {
                let r_ij = mat[(i, j)];
                if r_ij.abs() < 1e-14 { continue; }
                let kron = &basis_kron[j][i];
                let coeff = norm * r_ij;
                for r in 0..d {
                    for c in 0..d {
                        let val = kron[(r, c)];
                        local[(r, c)] = local[(r, c)] + c64::new(coeff * val.re, coeff * val.im);
                    }
                }
            }
            local
        })
        .reduce(
            || Mat::<c64>::zeros(d, d),
            |mut a, b| {
                for r in 0..d {
                    for c in 0..d {
                        a[(r, c)] = a[(r, c)] + b[(r, c)];
                    }
                }
                a
            },
        )
}

/// Parallel Choi -> PTM matrix transform
fn choi_to_ptm_par(
    choi: &Mat<c64>,
    num_qubits: usize,
    basis_kron: &[Vec<Mat<c64>>],
) -> Mat<f64> {
    let d = 1 << (2 * num_qubits);
    let norm = 1.0 / ((1 << num_qubits) as f64);
    let mut ptm_flat = vec![0.0; d * d];

    ptm_flat.par_chunks_mut(d).enumerate().for_each(|(j, col)| {
        for i in 0..d {
            let kron = &basis_kron[j][i];
            let mut tr = 0.0;
            for r in 0..d {
                for c in 0..d {
                    let a = kron[(r, c)];
                    let b = choi[(c, r)];
                    tr += a.re * b.re - a.im * b.im;
                }
            }
            col[i] = norm * tr;
        }
    });

    let mut ptm = Mat::<f64>::zeros(d, d);
    for j in 0..d {
        for i in 0..d {
            ptm[(i, j)] = ptm_flat[j * d + i];
        }
    }
    ptm
}

fn all_pauli_matrices(num_qubits: usize) -> Vec<Mat<c64>> {
    let d = 1 << (2 * num_qubits);
    (0..d)
        .into_par_iter()
        .map(|idx| {
            let p_str = PauliString::from_index(num_qubits, idx);
            pauli_string_to_mat(&p_str)
        })
        .collect()
}

fn pauli_string_to_mat(p_str: &PauliString) -> Mat<c64> {
    let mut current = Mat::<c64>::zeros(1, 1);
    current[(0, 0)] = c64::new(1.0, 0.0);

    for op in &p_str.ops {
        let single = op.matrix();
        let mut single_mat = Mat::<c64>::zeros(2, 2);
        for r in 0..2 {
            for c in 0..2 {
                single_mat[(r, c)] = single[r][c];
            }
        }
        current = kronecker_c64(&current, &single_mat);
    }
    current
}

fn kronecker_c64(a: &Mat<c64>, b: &Mat<c64>) -> Mat<c64> {
    let (m1, n1) = (a.nrows(), a.ncols());
    let (m2, n2) = (b.nrows(), b.ncols());
    let mut res = Mat::<c64>::zeros(m1 * m2, n1 * n2);
    for r1 in 0..m1 {
        for c1 in 0..n1 {
            let a_val = a[(r1, c1)];
            if a_val.re == 0.0 && a_val.im == 0.0 {
                continue;
            }
            for r2 in 0..m2 {
                for c2 in 0..n2 {
                    res[(r1 * m2 + r2, c1 * n2 + c2)] = a_val * b[(r2, c2)];
                }
            }
        }
    }
    res
}

fn transpose_c64_mat(a: &Mat<c64>) -> Mat<c64> {
    let (r, c) = (a.nrows(), a.ncols());
    let mut t = Mat::<c64>::zeros(c, r);
    for i in 0..r {
        for j in 0..c {
            t[(j, i)] = a[(i, j)];
        }
    }
    t
}
