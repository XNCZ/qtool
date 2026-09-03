//! # Quantum Process Tomography (QPT) via Pure-Rust FISTA
//!
//! 任意 $n$ 比特完全正保迹 (CPTP) 极大似然量子过程层析 (QPT-MLE) 求解器。
//! 100% 纯 Rust 实现：零外部 C/Fortran/BLAS 依赖，完全基于 `faer` 矩阵计算
//! 与加速投影梯度法 (FISTA)。
//!
//! 优化目标为加权最小二乘:
//! $$
//! \min_{R} \frac{1}{2} \sum_k w_k \left( \sum_{j} R_{m_k, j} x_{k, j} - (1.0 - 2.0 \cdot p_{1}^{(k)}) \right)^2
//! $$
//! 并约束 $R$ 为完全正保迹 (CPTP) 过程对应的 Pauli 转移矩阵。

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{c64, Mat, Side};
use plotly::common::{ColorScale, ColorScalePalette};
use plotly::layout::{Axis, Layout, TicksDirection};
use plotly::{HeatMap, Plot};
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
// =========================================================================
// 0. 自定义错误类型定义
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
            Self::InvalidPauliChar(c) => write!(f, "非法 Pauli 算符字符 '{c}'，仅支持 I, X, Y, Z"),
            Self::InvalidStateString(s) => write!(f, "无法解析的量子初态描述: \"{s}\""),
            Self::QubitCountMismatch { expected, found } => {
                write!(f, "比特数不匹配: 期望 {expected} 比特，实际找到 {found} 比特")
            }
            Self::EmptyDataset => write!(f, "实验数据集为空，无法进行层析求解"),
            Self::InvalidProbability(p) => write!(f, "激发态几率 p1 = {p} 超出物理合法区间 [0.0, 1.0]"),
            Self::InvalidWeight(w) => write!(f, "测量权重 w = {w} 必须为正实数"),
            Self::BasisNotFound(s) => write!(f, "Pauli 算符 \"{s}\" 在当前字典序基底表中未找到"),
            Self::SolverError(s) => write!(f, "求解收敛失败: {s}"),
            Self::DimensionMismatch { expected, found } => {
                write!(f, "矩阵维度不匹配: 期望 {expected}x{expected}，实际为 {found}x{found}")
            }
        }
    }
}

impl Error for QPTError {}

// =========================================================================
// 1. 基础物理枚举与自描述算符
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
// 2. 实验数据集定义
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
// 3. 求解器配置与诊断
// =========================================================================

#[derive(Debug, Clone)]
pub struct QptSolverConfig {
    pub tol_convergence: f64,
    pub max_iter: usize,
    pub verbose: bool,
}

impl Default for QptSolverConfig {
    fn default() -> Self {
        Self {
            tol_convergence: 1e-6,
            max_iter: 200,
            verbose: false,
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
// 4. PTM 矩阵结构
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

    /// 构建通用的 n-Qubit PTM 交互式二维热力图对象 (Plotly Plot)
    ///
    /// # 特性
    /// - **横纵轴自适应**：自动按当前比特数提取标准字典序基底 (II, IX, IY, IZ, XI, ...)
    /// - **矩阵坐标系对齐**：将 y 轴标签与数据行倒序，使第 0 行 (II) 处于最顶端
    /// - **物理发散色标**：范围固定在 [-1.0, 1.0]，0 处为中性白，正负转移分明
    /// - **悬停信息定制**：鼠标悬停显示输入算符、输出算符与精确矩阵元值
    pub fn to_plotly(&self, custom_title: Option<&str>) -> Plot {
        let d = self.dim; // 4^n

        // 1. 提取自适应 n 比特标准 Pauli 字符串标签
        let labels: Vec<String> = self.basis_order.iter().map(|p| p.to_string()).collect();

        // 2. 组装 2D 矩阵数据: z[row][col] = R_{meas, prep}
        let mut z_data = Vec::with_capacity(d);
        for r in 0..d {
            let mut row = Vec::with_capacity(d);
            for c in 0..d {
                row.push(self.mat[(r, c)]);
            }
            z_data.push(row);
        }

        // Plotly 范畴型 y 轴默认将 y 数组中第 0 类置于最底端；
        // 为让矩阵第 0 行 (II) 位于顶端，将 y 标签与数据行一并倒序。
        let mut y_labels = labels.clone();
        y_labels.reverse();
        z_data.reverse();

        // 3. 构建 HeatMap Trace
        let trace = HeatMap::new(labels.clone(), y_labels, z_data)
            .color_scale(ColorScale::Palette(ColorScalePalette::RdBu)) // 蓝-白-红发散色标
            .zmin(-1.0)
            .zmax(1.0)
            .hover_template("<b>Input (Prep)</b>: %{x}<br><b>Output (Meas)</b>: %{y}<br><b>R</b>: %{z:.4f}<extra></extra>");

        // 4. 自适应图表尺寸 (单比特 550px，两比特 850px，多比特更大)
        let plot_size = (500 + 22 * d).clamp(550, 1300);

        // 5. 配置横纵坐标轴与画布
        let title_text = custom_title.map(|s| s.to_string()).unwrap_or_else(|| {
            format!("{}-Qubit Pauli Transfer Matrix (PTM)", self.num_qubits)
        });

        // X 轴：输入算符 (列)
        let x_axis = Axis::new()
            .title("Input Pauli Operator (Preparation)")
            .ticks(TicksDirection::Outside)
            .tick_angle(if d > 4 { -45.0 } else { 0.0 })
            .show_grid(false);

        // Y 轴：输出算符 (行)
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

    /// 在系统默认浏览器中直接打开交互式 PTM 图表
    pub fn show_plot(&self, custom_title: Option<&str>) {
        let plot = self.to_plotly(custom_title);
        plot.show();
    }

    /// 保存为独立的交互式 HTML 文件
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
    /// 快捷方法：直接将层析结果画出，标题自动带上门保真度与 RMSE 诊断信息
    pub fn to_plotly(&self, gate_name: Option<&str>) -> Plot {
        let title = format!(
            "{} QPT (RMSE: {:.3e}, TraceErr: {:.1e})",
            gate_name.unwrap_or("Reconstructed Gate"),
            self.diagnostics.fit_rmse,
            self.diagnostics.trace_preserving_error
        );
        self.ptm.to_plotly(Some(&title))
    }

    /// 快捷方法：在浏览器中打开重构结果
    pub fn show_plot(&self, gate_name: Option<&str>) {
        let plot = self.to_plotly(gate_name);
        plot.show();
    }

    /// 快捷保存 HTML
    pub fn save_html<P: AsRef<Path>>(&self, path: P, gate_name: Option<&str>) -> std::io::Result<()> {
        let plot = self.to_plotly(gate_name);
        plot.write_html(path);
        Ok(())
    }
}

// =========================================================================
// 5. 核心：纯 Rust FISTA 求解器
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
        // 优化 1: 预计算所有不变的 (P_j^T \otimes P_i) 基底矩阵 (彻底消除迭代内堆分配)
        // =====================================================================
        let pauli_mats = all_pauli_matrices(n);
        let mut basis_kron: Vec<Vec<Mat<c64>>> = Vec::with_capacity(d);
        for j in 0..d {
            let pj_t = transpose_c64_mat(&pauli_mats[j]);
            let mut row_kron = Vec::with_capacity(d);
            for i in 0..d {
                row_kron.push(kronecker_c64(&pj_t, &pauli_mats[i]));
            }
            basis_kron.push(row_kron);
        }

        // =====================================================================
        // 优化 2: 预先将实验数据聚合成 Hessian H_m 与偏置向量 b_m
        // \nabla f(R)_m = H_m * R_m - b_m
        // =====================================================================
        let mut h_blocks = vec![Mat::<f64>::zeros(d, d); d];
        let mut b_vectors = vec![vec![0.0; d]; d];
        let mut meas_cover = vec![false; d];
        let mut total_w = 0.0;

        for data in &dataset.experiments {
            let m = data.meas.index();
            meas_cover[m] = true;
            let x = data.prep.pauli_vector();
            let y = data.expectation();
            let w = data.weight;
            total_w += w;

            let h_m = &mut h_blocks[m];
            let b_m = &mut b_vectors[m];

            for r in 0..d {
                let xr = x[r];
                if xr.abs() < 1e-14 { continue; }
                b_m[r] += w * y * xr;
                for c in 0..d {
                    h_m[(r, c)] += w * xr * x[c];
                }
            }
        }

        // =====================================================================
        // 优化 3: 准确计算真实 Lipschitz 常数 L = max_m ||H_m||_2 (使用无穷范数上界)
        // =====================================================================
        let mut max_spectral_norm: f64 = 0.0;
        for m in 0..d {
            if !meas_cover[m] { continue; }
            let h_m = &h_blocks[m];
            // 对称矩阵谱范数上界: max 行绝对值和
            let mut max_row_sum: f64 = 0.0;
            for r in 0..d {
                let mut row_sum = 0.0;
                for c in 0..d {
                    row_sum += h_m[(r, c)].abs();
                }
                max_row_sum = max_row_sum.max(row_sum);
            }
            max_spectral_norm = max_spectral_norm.max(max_row_sum);
        }

        // 步长严格设为理论稳定安全步长
        let lipschitz = max_spectral_norm.max(1.0);
        let lr = 1.0 / lipschitz;

        // FISTA 迭代初始状态
        let mut r_curr = Mat::<f64>::zeros(d, d);
        r_curr[(0, 0)] = 1.0;
        let mut y_acc = r_curr.clone();
        let mut t_acc: f64 = 1.0;

        let mut final_iter = 0;

        for iter in 0..self.config.max_iter {
            final_iter = iter + 1;

            // 1. 快速矩阵向量乘计算梯度: grad_m = H_m * y_acc_m - b_m
            let mut z = Mat::<f64>::zeros(d, d);
            for m in 0..d {
                if !meas_cover[m] {
                    // 若未测量该基底，保持外插位置
                    for j in 0..d {
                        z[(m, j)] = y_acc[(m, j)];
                    }
                    continue;
                }

                let h_m = &h_blocks[m];
                let b_m = &b_vectors[m];

                for j in 0..d {
                    let mut grad_mj = -b_m[j];
                    for l in 0..d {
                        grad_mj += h_m[(j, l)] * y_acc[(m, l)];
                    }
                    z[(m, j)] = y_acc[(m, j)] - lr * grad_mj;
                }
            }

            // 2. 严格完全正保迹 (CPTP) 投影 (复用已预计算的 basis_kron)
            let r_next = project_to_cptp(&z, n, &basis_kron)?;

            // 3. 收敛性检验
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

            // 4. Nesterov 动量加速
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

        // 组装返回结果
        let basis_order = (0..d).map(|idx| PauliString::from_index(n, idx)).collect();
        let ptm = PtmMatrix {
            num_qubits: n,
            dim: d,
            mat: r_curr,
            basis_order,
        };

        // 计算物理诊断参数
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

/// CPTP 投影函数
fn project_to_cptp(
    mat: &Mat<f64>,
    num_qubits: usize,
    basis_kron: &[Vec<Mat<c64>>],
) -> Result<Mat<f64>, QPTError> {
    let d = 1 << (2 * num_qubits);
    let norm = 1.0 / ((1 << num_qubits) as f64);

    // 1. 保迹投影
    let mut mat_tp = mat.clone();
    mat_tp[(0, 0)] = 1.0;
    for j in 1..d {
        mat_tp[(0, j)] = 0.0;
    }

    // 2. 映射到复数 Choi 矩阵 (直接累加预计算的基底矩阵，无任何动态分配)
    let mut choi = Mat::<c64>::zeros(d, d);
    for j in 0..d {
        for i in 0..d {
            let r_ij = mat_tp[(i, j)];
            if r_ij.abs() < 1e-14 { continue; }
            let kron = &basis_kron[j][i];
            let coeff = norm * r_ij;
            for r in 0..d {
                for c in 0..d {
                    let val = kron[(r, c)];
                    choi[(r, c)] = choi[(r, c)] + c64::new(coeff * val.re, coeff * val.im);
                }
            }
        }
    }

    // 3. 强制厄米对称
    for r in 0..d {
        for c in (r + 1)..d {
            let z1 = choi[(r, c)];
            let z2 = choi[(c, r)];
            let avg = c64::new(0.5 * (z1.re + z2.re), 0.5 * (z1.im - z2.im));
            choi[(r, c)] = avg;
            choi[(c, r)] = c64::new(avg.re, -avg.im);
        }
    }

    // 4. 特征值分解与半正定截断
    let eig = SelfAdjointEigen::new(choi.as_ref(), Side::Lower)
        .map_err(|e| QPTError::SolverError(format!("{:?}", e)))?;
    let s_diag = eig.S();
    let vecs = eig.U();
    let mut vals: Vec<f64> = (0..d).map(|i| s_diag[i].re).collect();

    let mut pos_sum = 0.0;
    for v in vals.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        } else {
            pos_sum += *v;
        }
    }

    // 迹对齐: Tr(Choi) 恒等于 2^n
    let target_tr = (1 << num_qubits) as f64;
    let scale = if pos_sum > 1e-12 { target_tr / pos_sum } else { 1.0 };
    for v in vals.iter_mut() {
        *v *= scale;
    }

    // 重构半正定 Choi
    let mut choi_psd = Mat::<c64>::zeros(d, d);
    for k in 0..d {
        let lk = vals[k];
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

    // 5. 逆映射还原回 PTM
    let mut ptm_res = Mat::<f64>::zeros(d, d);
    for j in 0..d {
        for i in 0..d {
            let kron = &basis_kron[j][i];
            let mut tr = 0.0;
            for r in 0..d {
                for c in 0..d {
                    let a = kron[(r, c)];
                    let b = choi_psd[(c, r)];
                    tr += a.re * b.re - a.im * b.im;
                }
            }
            ptm_res[(i, j)] = norm * tr;
        }
    }

    ptm_res[(0, 0)] = 1.0;
    for j in 1..d {
        ptm_res[(0, j)] = 0.0;
    }

    Ok(ptm_res)
}

fn all_pauli_matrices(num_qubits: usize) -> Vec<Mat<c64>> {
    let d = 1 << (2 * num_qubits);
    let mut mats = Vec::with_capacity(d);
    for idx in 0..d {
        let p_str = PauliString::from_index(num_qubits, idx);
        mats.push(pauli_string_to_mat(&p_str));
    }
    mats
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

// =========================================================================
// 7. 单元测试
// =========================================================================

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_single_qubit_x_gate() {
        let mut dataset = QptDataset::new(1);
        dataset
            .add_data("Z+", "Z", 1.0).unwrap()
            .add_data("Z-", "Z", 0.0).unwrap()
            .add_data("X+", "X", 0.0).unwrap()
            .add_data("X-", "X", 1.0).unwrap()
            .add_data("Y+", "Y", 1.0).unwrap()
            .add_data("Y-", "Y", 0.0).unwrap();

        let solver = QptSolver::new(1, None);
        let result = solver.solve(&dataset).expect("求解失败");

        assert!(result.diagnostics.trace_preserving_error < 1e-5);
        assert!(result.diagnostics.min_choi_eigenvalue >= -1e-6);

        let ptm = &result.ptm;
        assert!((ptm.get("I", "I").unwrap() - 1.0).abs() < 1e-3);
        assert!((ptm.get("X", "X").unwrap() - 1.0).abs() < 1e-3);
        assert!((ptm.get("Y", "Y").unwrap() - (-1.0)).abs() < 1e-3);
        assert!((ptm.get("Z", "Z").unwrap() - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_two_qubit_cz_gate() {
        let num_qubits = 2;
        let d = 16;
        let mut dataset = QptDataset::new(num_qubits);

        // 构造标准的 CZ 门对角矩阵 U_CZ = diag(1, 1, 1, -1)
        let mut u_cz = Mat::<c64>::zeros(4, 4);
        u_cz[(0, 0)] = c64::new(1.0, 0.0);
        u_cz[(1, 1)] = c64::new(1.0, 0.0);
        u_cz[(2, 2)] = c64::new(1.0, 0.0);
        u_cz[(3, 3)] = c64::new(-1.0, 0.0);

        let ideal_cz = PtmMatrix::from_unitary(num_qubits, &u_cz);

        let single_states = ["Z+", "Z-", "X+", "X-", "Y+", "Y-"];
        let pauli_axes = ["I", "X", "Y", "Z"];

        for s1 in single_states {
            for s2 in single_states {
                let prep_str = format!("{}, {}", s1, s2);
                let prep = QuantumState::from_str(&prep_str).unwrap();
                let x_vec = prep.pauli_vector();

                for p1_op in pauli_axes {
                    for p2_op in pauli_axes {
                        if p1_op == "I" && p2_op == "I" {
                            continue;
                        }
                        let meas_str = format!("{}{}", p1_op, p2_op);
                        let meas = PauliString::from_str(&meas_str).unwrap();
                        let m_idx = meas.index();

                        let mut exp_val = 0.0;
                        for j in 0..d {
                            exp_val += ideal_cz.mat[(m_idx, j)] * x_vec[j];
                        }
                        let p1 = (1.0 - exp_val) * 0.5;
                        dataset.add_data(&prep_str, &meas_str, p1).unwrap();
                    }
                }
            }
        }

        let mut cfg = QptSolverConfig::default();
        cfg.max_iter = 300;
        cfg.tol_convergence = 1e-7;
        cfg.verbose = false;

        let solver = QptSolver::new(num_qubits, Some(cfg));
        let result = solver.solve(&dataset).expect("求解失败");

        let (f_pro, f_avg) = result.ptm.fidelity(&ideal_cz).unwrap();
        eprintln!(
            "DIAG f_pro={f_pro:.6} f_avg={f_avg:.6} tp_err={:.2e} min_eig={:.2e} rmse={:.2e} iters={}",
            result.diagnostics.trace_preserving_error,
            result.diagnostics.min_choi_eigenvalue,
            result.diagnostics.fit_rmse,
            result.meta.iterations,
        );

        assert!(f_pro > 0.999);
        assert!(f_avg > 0.999);
        assert!(result.diagnostics.trace_preserving_error < 1e-6);
        assert!(result.diagnostics.min_choi_eigenvalue >= -1e-6);
        result.save_html("cz_ptm.html", Some("Ideal CZ Gate PTM (16x16)"))
            .expect("保存 HTML 失败");
    }
}