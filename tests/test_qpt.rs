use faer::{c64, Mat};
use cz_qtool::qpt::{PtmMatrix, QptDataset, QptSolver, QptSolverConfig};
use rand::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

fn sample_complex_gaussian<R: Rng>(rng: &mut R) -> c64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    c64::new(r * theta.cos(), r * theta.sin())
}

fn random_unitary<R: Rng>(num_qubits: usize, rng: &mut R) -> Mat<c64> {
    let d = 1 << num_qubits;
    // Random complex Gaussian matrix
    let mut cols: Vec<Vec<c64>> = Vec::with_capacity(d);
    for _ in 0..d {
        let mut col = Vec::with_capacity(d);
        for _ in 0..d {
            col.push(sample_complex_gaussian(rng));
        }
        cols.push(col);
    }

    // Modified Gram-Schmidt orthonormalization
    let mut u_cols: Vec<Vec<c64>> = Vec::with_capacity(d);
    for i in 0..d {
        let mut v = cols[i].clone();
        for u in &u_cols {
            let mut dot = c64::new(0.0, 0.0);
            for k in 0..d {
                let uk = u[k];
                let vk = v[k];
                // <u, v> = sum_k u_k^* * v_k
                dot = dot + c64::new(uk.re * vk.re + uk.im * vk.im, uk.re * vk.im - uk.im * vk.re);
            }
            for k in 0..d {
                let uk = u[k];
                let prod = c64::new(dot.re * uk.re - dot.im * uk.im, dot.re * uk.im + dot.im * uk.re);
                v[k] = c64::new(v[k].re - prod.re, v[k].im - prod.im);
            }
        }
        let mut norm_sq = 0.0;
        for k in 0..d {
            norm_sq += v[k].re * v[k].re + v[k].im * v[k].im;
        }
        let norm = norm_sq.sqrt();
        for k in 0..d {
            v[k] = c64::new(v[k].re / norm, v[k].im / norm);
        }
        u_cols.push(v);
    }

    let mut u_mat = Mat::<c64>::zeros(d, d);
    for c in 0..d {
        for r in 0..d {
            u_mat[(r, c)] = u_cols[c][r];
        }
    }
    u_mat
}

fn quantum_states(num_qubits: usize) -> Vec<String> {
    // 1-2 qubits: standard overcomplete 6-state basis; 3 qubits: complete 4-state basis
    let single_basis = if num_qubits <= 2 {
        vec!["Z+", "Z-", "X+", "X-", "Y+", "Y-"]
    } else {
        vec!["Z+", "Z-", "X+", "Y+"]
    };

    let mut combinations = vec![String::new()];
    for _ in 0..num_qubits {
        let mut next = Vec::new();
        for prefix in &combinations {
            for state in &single_basis {
                if prefix.is_empty() {
                    next.push(state.to_string());
                } else {
                    next.push(format!("{prefix} {state}"));
                }
            }
        }
        combinations = next;
    }
    combinations
}

// =========================================================================
// Single-round reconstruction verification
// =========================================================================
fn test_single_round<R: Rng>(round_idx: usize, num_qubits: usize, rng: &mut R) -> bool {
    let t0 = Instant::now();


    let u = random_unitary(num_qubits, rng);
    let ideal_ptm = PtmMatrix::from_unitary(num_qubits, &u);
    let dim = ideal_ptm.dim;


    let basis = ideal_ptm.basis_order.clone();
    let prep_states = quantum_states(num_qubits);
    let mut dataset = QptDataset::new(num_qubits);

    for prep_str in &prep_states {
        let x = prep_str
            .parse::<cz_qtool::qpt::QuantumState>()
            .unwrap()
            .pauli_vector();

        for (m_idx, meas_pauli) in basis.iter().enumerate().skip(1) {
            // Ideal expectation value <M> = sum_j R_{m, j} * x_j
            let mut exp_val = 0.0;
            for j in 0..dim {
                exp_val += ideal_ptm.mat[(m_idx, j)] * x[j];
            }
            // Excited-state probability p1 = (1 - <M>) / 2
            let p1 = ((1.0 - exp_val) * 0.5).clamp(0.0, 1.0);
            dataset
                .add_data(prep_str, &meas_pauli.to_string(), p1)
                .unwrap();
        }
    }


    let solver = QptSolver::new(
        num_qubits,
        Some(QptSolverConfig {
            tol_convergence: 1e-6,
            max_iter: 200,
            verbose: false,
            max_dykstra_iter: 10,
        }),
    );
    let solve_res = match solver.solve(&dataset) {
        Ok(res) => res,
        Err(e) => {
            println!("Round [{round_idx:02}] FAILED: solver error: {e}");
            return false;
        }
    };


    let (f_process, f_average) = solve_res.ptm.fidelity(&ideal_ptm).unwrap();

    let mut max_matrix_err = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            let err = (solve_res.ptm.mat[(i, j)] - ideal_ptm.mat[(i, j)]).abs();
            if err > max_matrix_err {
                max_matrix_err = err;
            }
        }
    }

    let elapsed = t0.elapsed();

    // Pass/fail criteria
    let fidelity_pass = f_process >= 0.99;
    let rmse_pass = solve_res.diagnostics.fit_rmse <= 1e-3;
    let tp_pass = solve_res.diagnostics.trace_preserving_error <= 1e-4;
    let cp_pass = solve_res.diagnostics.min_choi_eigenvalue >= -1e-5;
    let matrix_pass = max_matrix_err <= 0.05;

    let round_passed = fidelity_pass && rmse_pass && tp_pass && cp_pass && matrix_pass;

    println!(
        "Round [{:02}] | Qubits: {} | Time: {:>6.2?} | F_proc: {:.6} | F_avg: {:.6} | RMSE: {:.2e} | MaxPTMErr: {:.4} | Status: {}",
        round_idx,
        num_qubits,
        elapsed,
        f_process,
        f_average,
        solve_res.diagnostics.fit_rmse,
        max_matrix_err,
        if round_passed { "\x1b[32mPASS\x1b[0m" } else { "\x1b[31mFAIL\x1b[0m" }
    );

    if !round_passed {
        eprintln!(
            "  -> Diagnostics detail: TP_Err={:.2e}, MinChoiEig={:.2e}",
            solve_res.diagnostics.trace_preserving_error,
            solve_res.diagnostics.min_choi_eigenvalue
        );
    }

    round_passed
}

// =========================================================================
// Driver running N randomized repetitions
// =========================================================================
pub fn run_n_tests(n: usize) -> bool {
    let mut rng = rand::rng();

    println!("================================================================================");
    println!("Starting QPT Inversion Accuracy Test (N = {n} repetitions, using rand crate)");
    println!("Randomly testing 1-qubit, 2-qubit, and 3-qubit arbitrary quantum processes");
    println!("================================================================================");

    let mut all_passed = true;

    for i in 1..=n {
        // First 3 rounds test 1, 2 and 3 qubits; later rounds pick 1..=3 at random
        let num_qubits = if i <= 3 {
            i
        } else {
            rng.random_range(1..=3)
        };

        let passed = test_single_round(i, num_qubits, &mut rng);
        if !passed {
            all_passed = false;
        }
    }

    println!("================================================================================");
    if all_passed {
        println!("\x1b[32;1mAll {n} test rounds passed successfully! QPT algorithm verified.\x1b[0m");
    } else {
        println!("\x1b[31;1mSome test rounds failed. Please check the logs above.\x1b[0m");
    }
    println!("================================================================================");

    all_passed
}

#[test]
fn test_qpt_random_processes() {
    let repeat_n = 10;
    assert!(
        run_n_tests(repeat_n),
        "QPT verification across {repeat_n} iterations"
    );
}