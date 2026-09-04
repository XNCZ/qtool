# cz-qtool

Pure-Rust **quantum process tomography** (QPT) toolkit.

Given the measurement statistics of an unknown quantum process acting on `n`
qubits, `cz-qtool` performs maximum-likelihood reconstruction of the process
under the **completely positive and trace-preserving (CPTP)** constraint and
returns its **Pauli transfer matrix (PTM)** `R`, where

```
E[ P_m | state ] = (R · x)_m ,      x_j = <P_j> of the prepared state.
```

The optimization

```
min_R  1/2 Σ_k w_k ( Σ_j R_{m_k,j} x_{k,j} − (1 − 2 p1_k) )²
s.t.   R is the PTM of a CPTP process
```

is solved with an accelerated projected-gradient method (**FISTA**) whose CPTP
projection is computed by the **Dykstra** cyclic alternating-projection
algorithm (TP affine space ∩ PSD Choi cone). Linear algebra is provided by
[`faer`](https://crates.io/crates/faer) and the heavy inner loops are
parallelised with [`rayon`](https://crates.io/crates/rayon); interactive
plotting uses [`plotly`](https://crates.io/crates/plotly).

There are **no C / Fortran / BLAS dependencies**.

## Features

- Arbitrary number of qubits (full QPT cost grows as `4^n`, as expected).
- Pure-Rust solver core — easy to cross-compile.
- String-based input for preparation states and measurement Pauli strings.
- Diagnostics: trace-preserving error, minimum Choi eigenvalue, fit RMSE,
  data coverage.
- Plotly heat maps of the PTM (`to_plotly`, `show_plot`, `save_html`).
- Ideal-PTM constructors for validation: `PtmMatrix::identity`,
  `PtmMatrix::from_unitary`.

## Installation

```toml
[dependencies]
cz-qtool = "0.1"

# Add these only if you need them in your own code:
faer = "0.24"   # build unitaries or inspect the returned matrices
plotly = "0.14" # drive the Plot objects returned by to_plotly yourself
```

## Quick tour of the public API

All public types live in the `qpt` module:

```rust
use cz_qtool::qpt::*;
```

| Item | Purpose |
| --- | --- |
| `QptDataset::new(n)` | Empty data set for `n` qubits. |
| `dataset.add_data(prep, meas, p1)` | Record one experiment. |
| `dataset.add_weight_data(prep, meas, p1, w)` | Weighted experiment (`w > 0`). |
| `QptSolver::new(n, config)` / `.solve(&dataset)` | Run the FISTA reconstruction. |
| `QptResult.ptm` | Reconstructed `PtmMatrix`. |
| `QptResult.diagnostics` | TP error, min Choi eigenvalue, RMSE, coverage, Choi matrix. |
| `QptResult.meta` | Solve time, iteration count, status. |
| `PtmMatrix::identity(n)` | Ideal identity-gate PTM. |
| `PtmMatrix::from_unitary(n, &u)` | Ideal PTM of a unitary `u` (`faer::Mat<c64>`). |
| `ptm.get("meas", "prep")` | Look up one PTM element by Pauli strings. |
| `ptm.fidelity(&ideal)` | `(process_fidelity, average_fidelity)` vs an ideal PTM. |
| `ptm.display_table()` | Plain-text table (first 16×16). |
| `ptm.to_plotly(..)` / `.show_plot(..)` / `.save_html(..)` | Plotly heat map. |
| `QuantumState` / `PauliString` | Parsed from strings; `FromStr` / `Display`. |
| `QuantumState::pauli_vector()` | Pauli-basis expectation vector of a state. |

String conventions:

- Preparation states — one token per qubit from the Bloch extremes
  `Z+` (|0⟩), `Z-` (|1⟩), `X+`, `X-`, `Y+`, `Y-`, joined by spaces/commas.
- Measurements — an `n`-character Pauli string over `I, X, Y, Z`,
  e.g. `"Z"`, `"XX"`, `"ZI"`.
- `p1` is the probability of detecting the excited state along the measurement
  axis; the expectation used internally is `E = 1 − 2·p1`.

## Examples

### Example 1 — one-qubit Pauli-X gate

```rust
use cz_qtool::qpt::{QptDataset, QptSolver};

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

    // Solve with the default configuration.
    let solver = QptSolver::new(1, None);
    let result = solver.solve(&dataset).expect("tomography failed");

    // Physical sanity checks exposed by QptResult::diagnostics.
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

The same recipe in `n = 2`: build the ideal `U_CZ = diag(1, 1, 1, −1)`,
synthesize a complete experiment set from it, reconstruct, and check fidelity.

```rust
use cz_qtool::qpt::{PtmMatrix, QptDataset, QptSolver, QuantumState};
use faer::{c64, Mat};

fn main() {
    let n = 2;
    let d = 16; // 4^n

    // Ideal CZ gate, U_CZ = diag(1, 1, 1, -1) in the computational basis.
    let mut u = Mat::<c64>::zeros(4, 4);
    for i in 0..4 {
        u[(i, i)] = if i == 3 { c64::new(-1.0, 0.0) } else { c64::new(1.0, 0.0) };
    }
    let ideal = PtmMatrix::from_unitary(n, &u);
    let basis = ideal.basis_order.clone(); // all Pauli strings, lexicographic order

    // A complete, noiseless experiment set: 36 preparations x 15 measurements.
    let mut dataset = QptDataset::new(n);
    let states = ["Z+", "Z-", "X+", "X-", "Y+", "Y-"];
    for s1 in states {
        for s2 in states {
            let prep = format!("{}, {}", s1, s2);
            let x = prep.parse::<QuantumState>().unwrap().pauli_vector();
            for (m_idx, meas) in basis.iter().enumerate().skip(1) {
                let mut exp = 0.0;
                for j in 0..d {
                    exp += ideal.mat[(m_idx, j)] * x[j];
                }
                dataset
                    .add_data(&prep, &meas.to_string(), (1.0 - exp) * 0.5)
                    .unwrap();
            }
        }
    }

    // Reconstruct and compare against the ideal CZ PTM.
    let result = QptSolver::new(n, None).solve(&dataset).expect("tomography failed");
    let (f_proc, f_avg) = result.ptm.fidelity(&ideal).unwrap();
    println!("process fidelity = {f_proc:.6}, average fidelity = {f_avg:.6}");
    assert!(f_proc > 0.99);
    assert!(result.diagnostics.trace_preserving_error < 1e-3);

    // Optionally export an interactive 16x16 heat map.
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
  near `0` means the CP constraint is satisfied.
- `fit_rmse` — weighted RMS error between model predictions and inputs.
- `data_coverage` — fraction of measurement basis elements present in the data.
- `choi_matrix` — the reconstructed complex Choi matrix (`faer::Mat<c64>`).

## License

MIT — see [LICENSE](LICENSE).
