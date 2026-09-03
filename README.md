# qtool

Pure-Rust **quantum process tomography** (QPT) toolkit.

Given the measurement statistics of an unknown quantum process acting on `n`
qubits, `qtool` performs maximum-likelihood reconstruction of the process under
the **completely positive and trace-preserving (CPTP)** constraint and returns
its **Pauli transfer matrix (PTM)** `R`, where

```
E[ P_m | state ] = (R · x)_m ,      x_j = <P_j> of the prepared state.
```

The optimization

```
min_R  1/2 Σ_k w_k ( Σ_j R_{m_k,j} x_{k,j} − (1 − 2 p1_k) )²
s.t.   R is the PTM of a CPTP process
```

is solved with an accelerated projected-gradient method (**FISTA**). The
projection onto the CPTP set goes through the Choi representation: map to the
complex Choi matrix, project onto the positive-semidefinite cone by a Hermitian
eigendecomposition, renormalize the trace, and map back.

There are **no C / Fortran / BLAS dependencies**. Linear algebra is provided by
[`faer`](https://crates.io/crates/faer); interactive plotting is provided by
[`plotly`](https://crates.io/crates/plotly).

## Features

- Arbitrary number of qubits (the cost grows as `4^n`, as expected for full QPT).
- Pure-Rust solver core (`faer` only) — easy to cross-compile.
- Convenient string input for preparation states and measurement Pauli strings.
- Physical diagnostics: trace-preserving error, minimum Choi eigenvalue,
  fit RMSE, data coverage.
- Plotly integration: interactive PTM heat maps (`to_plotly`, `show_plot`,
  `save_html`).
- Ideal-PTM constructors for validation: `PtmMatrix::identity`,
  `PtmMatrix::from_unitary`.

## Installation

```toml
[dependencies]
qtool = "0.1"

# Add these only when you need them from your own code:
faer = "0.24"   # build unitaries / handle the returned Mat<c64>, Mat<f64>
plotly = "0.14" # drive the Plot objects returned by to_plotly yourself
```

## Using the library

All public types live in the `qtool::qpt` module:

```rust
use qtool::qpt::*;
```

### Public API quick reference

| Item | Purpose |
| --- | --- |
| `QptDataset::new(n)` | New, empty data set for `n` qubits. |
| `dataset.add_data(prep, meas, p1)` | Record one experiment (see string conventions below). |
| `dataset.add_weight_data(prep, meas, p1, w)` | Same, with a weight `w > 0`. |
| `dataset.validate()` | Check consistency and physical ranges. |
| `QptSolver::new(n, config)` | Build a solver; pass `None` for defaults. |
| `QptSolver::solve(&dataset)` | Run FISTA, return `Result<QptResult, QPTError>`. |
| `QptResult.ptm` | Reconstructed `PtmMatrix`. |
| `QptResult.diagnostics` | `trace_preserving_error`, `min_choi_eigenvalue`, `fit_rmse`, `data_coverage`, `choi_matrix`. |
| `QptResult.meta` | `solve_time_ms`, `iterations`, `status`. |
| `PtmMatrix::identity(n)` | Ideal identity-gate PTM. |
| `PtmMatrix::from_unitary(n, &u)` | Ideal PTM of the unitary `u` (`faer::Mat<c64>`). |
| `ptm.get("meas", "prep")` | Look up a single PTM element by Pauli strings. |
| `ptm.fidelity(&ideal)` | `(process_fidelity, average_fidelity)` vs. an ideal PTM. |
| `ptm.display_table()` | Plain-text table of the PTM. |
| `ptm.to_plotly(title)` / `.show_plot(..)` / `.save_html(path, ..)` | Plotly heat map. |
| `result.save_html(path, gate_name)` | Convenience wrapper on `QptResult`. |
| `SingleQubitState`, `Pauli`, `QuantumState`, `PauliString` | Building blocks (`FromStr`, `index()`, ...). |

#### String conventions

- Preparation state: one token per qubit from the Bloch extremes —
  `Z+` (|0>), `Z-` (|1>), `X+`, `X-`, `Y+`, `Y-`. Multi-qubit states are
  comma or space separated, e.g. `"Z+, X+"` or the compact `"ZX"`.
- Measurement: an `n`-character Pauli string over `I, X, Y, Z`
  (lexicographic qubit order), e.g. `"Z"`, `"XX"`, `"ZI"`.
- `p1` is the probability of detecting the excited state along the measurement
  axis; the expectation value used internally is `E = 1 − 2·p1`.

## Examples

### Example 1 — one-qubit Pauli-X gate

Reconstruct the PTM of a Pauli-X gate from six ideal experiments
(preparation × measurement along each of the `Z`, `X` and `Y` axes).

```rust
use qtool::qpt::{QptDataset, QptSolver};

fn main() {
    // |0> and |1> prepared and measured along Z:
    //   X|0> = |1>  -> p1 = 1        X|1> = |0> -> p1 = 0
    let mut dataset = QptDataset::new(1);
    dataset
        .add_data("Z+", "Z", 1.0).unwrap()
        .add_data("Z-", "Z", 0.0).unwrap()
        .add_data("X+", "X", 0.0).unwrap()
        .add_data("X-", "X", 1.0).unwrap()
        .add_data("Y+", "Y", 1.0).unwrap()
        .add_data("Y-", "Y", 0.0).unwrap();

    // Solve (QptSolver::new(num_qubits, config) with the default config).
    let solver = QptSolver::new(1, None);
    let result = solver.solve(&dataset).expect("tomography failed");

    // Physical sanity checks exposed by QptResult.diagnostics.
    assert!(result.diagnostics.trace_preserving_error < 1e-5);
    assert!(result.diagnostics.min_choi_eigenvalue >= -1e-6);

    // The X gate has PTM diag(1, 1, -1, -1) in the {I, X, Y, Z} basis.
    let ptm = &result.ptm;
    assert!((ptm.get("I", "I").unwrap() - 1.0).abs() < 1e-3);
    assert!((ptm.get("X", "X").unwrap() - 1.0).abs() < 1e-3);
    assert!((ptm.get("Y", "Y").unwrap() - (-1.0)).abs() < 1e-3);
    assert!((ptm.get("Z", "Z").unwrap() - (-1.0)).abs() < 1e-3);

    println!("{}", ptm.display_table());
}
```

### Example 2 — two-qubit CZ gate

Reconstruct a controlled-Z gate. A **complete** data set is synthesized
analytically from the ideal `U_CZ = diag(1, 1, 1, −1)` using
`PtmMatrix::from_unitary`, over all 6×6 preparation states and all 15
non-trivial two-qubit Pauli measurements.

```rust
use faer::{c64, Mat};
use qtool::qpt::{QuantumState, PauliString, PtmMatrix, QptDataset, QptSolver, QptSolverConfig};

fn main() {
    let num_qubits = 2;
    let d = 16; // 4^num_qubits
    let mut dataset = QptDataset::new(num_qubits);

    // Ideal CZ gate: U_CZ = diag(1, 1, 1, -1) in the computational basis.
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

                    // Ideal expectation value <P> = (R_ideal * x)_m,
                    // converted to the excitation probability p1 = (1 - <P>) / 2.
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

    // Tighter convergence settings for a 2-qubit problem.
    let mut cfg = QptSolverConfig::default();
    cfg.max_iter = 300;
    cfg.tol_convergence = 1e-7;

    let solver = QptSolver::new(num_qubits, Some(cfg));
    let result = solver.solve(&dataset).expect("tomography failed");

    // Process and average gate fidelity against the ideal CZ PTM.
    let (f_pro, f_avg) = result.ptm.fidelity(&ideal_cz).unwrap();
    println!("process fidelity = {f_pro:.6}, average fidelity = {f_avg:.6}");
    assert!(f_pro > 0.999);
    assert!(f_avg > 0.999);
    assert!(result.diagnostics.trace_preserving_error < 1e-6);
    assert!(result.diagnostics.min_choi_eigenvalue >= -1e-6);

    // Export an interactive 16x16 heat map to an HTML file.
    result
        .save_html("cz_ptm.html", Some("Ideal CZ Gate PTM (16x16)"))
        .expect("failed to save the HTML report");
}
```

## Diagnostics

`QptResult::diagnostics` provides:

- `trace_preserving_error` — max deviation of the first PTM row from
  `[1, 0, ..., 0]`.
- `min_choi_eigenvalue` — smallest eigenvalue of the reconstructed Choi matrix;
  near `0` (or `≥ −tol`) means the CP constraint is satisfied.
- `fit_rmse` — weighted root-mean-square error between the model predictions
  and the input expectations.
- `data_coverage` — fraction of measurement basis elements present in the data.
- `choi_matrix` — the reconstructed complex Choi matrix (`faer::Mat<c64>`).

## License

MIT — see [LICENSE](LICENSE).
