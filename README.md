# qtool

Analysis toolbox for quantum computation, in Rust — with a Python binding.

**Superconducting-qubit calibration** (`qtool::superconductor`) — fit the standard device
experiments, from the resonator down to the gates, and render each one as a self-contained
interactive plotly report: `s21`, `qspec`, `rabi`, `ramsey`, `t1`, `t2_echo`, the `drag`
chain and `bloch`. Shipped with a `plot` feature (on by default).

Everything numeric is Rust: `faer` for linear algebra, `lmfit` for the fits, `rayon` for
the batch paths — so a `*_fit_batch` call spreads the batch over every core *from inside the
call*, with no process pool to build, chunk or pickle (see
[Performance](#performance)). The reports are HTML fragments that assume only that the host
page loads `plotly.js` — no Python, no notebook, no server.

## Installation

```bash
cargo add qtool                # Rust: the plot feature is on by default
pip install qtool-py           # Python bindings (uv add qtool-py)
```

Rust users who only want the numerics can turn the reports off:

```toml
qtool = { version = "0.2", default-features = false }
```

## The reports

One function per experiment returns a **self-contained HTML fragment**; you paste it into
a page that loads `plotly.js` (the crate re-exports the CDN tag as `PLOTLY_JS_CDN`) and the
figure is live: hover, legend toggles, zoom, and — for the Bloch sphere — orbit rotation
and scroll zoom. The images below are screenshots of exactly those pages; every one of
them is produced by the demo of the same name (`cargo run --release --example <name>`).

<!-- Images use absolute raw URLs: crates.io resolves relative paths but PyPI does not (the camo proxy), so one set of URLs works on both -->

<details>
<summary><b>s21</b> — resonator spectroscopy, 11-parameter complex model</summary>

![s21 fit](https://raw.githubusercontent.com/xncz/qtool/master/docs/s21.png)

Fits `S21(f)` with the full microwave model: resonance frequency, loaded and coupling
quality factors, the asymmetry angle, plus the cable delay, the background slope and the
resonance centre in the IQ plane (11 parameters, analytic Jacobian, weights optional).
The card carries |S21|, phase, the IQ plane and the normalized notch — data, fit and
residuals — over the parameter table.

`cargo run --release --example s21_demo` · `qtool.s21.fit(...)` · [live page](https://xncz.github.io/qtool/s21_demo.html)
</details>

<details>
<summary><b>s21 vs power</b> — the same fit, swept over drive power</summary>

![s21 vs power](https://raw.githubusercontent.com/xncz/qtool/master/docs/s21_power.png)

Each power level is fitted independently and the parameters are laid out against power,
which is how you spot the drive-dependent shift of the resonance and the onset of
nonlinearity.

This demo reads a real measurement (`data/s21/fit_input.bin`), which is **not** part of
the repository — pass your own file, or see the docstring for the format.

`cargo run --release --example s21_power_demo <file>`
</details>

<details>
<summary><b>qspec</b> — qubit spectroscopy (Lorentzian)</summary>

![qspec](https://raw.githubusercontent.com/xncz/qtool/master/docs/qspec.png)

The qubit's own spectroscopy: a Lorentzian in the projected P1, with the orientation of
the IQ projection either taken from calibrated state centres or fitted from the data
itself (the `uncalibrated axis` case).

`cargo run --release --example qspec_demo` · `qtool.qspec.fit(...)` · [live page](https://xncz.github.io/qtool/qspec_demo.html)
</details>

<details>
<summary><b>qspec vs Z</b> — the flux sweep</summary>

![qspec vs Z](https://raw.githubusercontent.com/xncz/qtool/master/docs/qspec_z.png)

One Lorentzian per flux point; the report overlays the fitted frequency and linewidth
against the coupler/flux bias, which is where the sweet spot and the flux-noise
sensitivity become visible.

`cargo run --release --example qspec_z_demo` · `qtool.qspec.z_plot(...)` · [live page](https://xncz.github.io/qtool/qspec_z_demo.html)
</details>

<details>
<summary><b>rabi</b> — Rabi amplitude, cosine anchored at P1(0)=0</summary>

![rabi](https://raw.githubusercontent.com/xncz/qtool/master/docs/rabi.png)

Amplitude sweep of the drive. The fitted cosine is anchored at zero (a zero-amplitude
pulse does nothing) and its half period is reported as the π amplitude `a_pi = 1/(2f)`,
which is what the calibration table wants.

`cargo run --release --example rabi_demo` · `qtool.rabi.fit(...)` · [live page](https://xncz.github.io/qtool/rabi_demo.html)
</details>

<details>
<summary><b>iq</b> — readout clouds: density regions and pairwise separability</summary>

![iq](https://raw.githubusercontent.com/xncz/qtool/master/docs/iq.png)

Readout quality from the raw IQ clouds: per-state density regions at the 0.68/0.95/0.99/1.0
levels with their areas and radii, and for every pair the separation, the discriminant
threshold, the assignment error rate, the AUC and the SNR. Any number of states.

`cargo run --release --example iq_demo` · `qtool.iq.iq_stats(...)` · [live page](https://xncz.github.io/qtool/iq_demo.html)
</details>

<details>
<summary><b>ramsey</b> — T2* from a damped cosine</summary>

![ramsey](https://raw.githubusercontent.com/xncz/qtool/master/docs/ramsey.png)

The fringe of a Ramsey delay sweep: `offset + amplitude·exp(−τ/decay)·cos(2πfτ + phase)`.
The decay is the dephasing time; the frequency is what the drive detuning is read from.

`cargo run --release --example ramsey_demo` · `qtool.ramsey.fit(...)` · [live page](https://xncz.github.io/qtool/ramsey_demo.html)
</details>

<details>
<summary><b>t1</b> — energy relaxation</summary>

![t1](https://raw.githubusercontent.com/xncz/qtool/master/docs/t1.png)

Excitation decays back to the ground state; the exponential's time constant is T1.

`cargo run --release --example t1_demo` · `qtool.t1.fit(...)` · [live page](https://xncz.github.io/qtool/t1_demo.html)
</details>

<details>
<summary><b>t2_echo</b> — echo dephasing</summary>

![t2_echo](https://raw.githubusercontent.com/xncz/qtool/master/docs/t2_echo.png)

The echo sequence (π/2 — τ/2 — π — τ/2 — π/2) cancels the static detuning, so what
decays is T2 echo rather than T2*. The convention here is the **total free evolution
time**.

`cargo run --release --example t2_echo_demo` · `qtool.t2_echo.fit(...)` · [live page](https://xncz.github.io/qtool/t2_echo_demo.html)
</details>

<details>
<summary><b>drag — amplitude</b> — first order (cosine) and the escalated valley fits</summary>

![drag amplitude](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_amplitude.png)

The DRAG amplitude calibration: at low order the leakage-vs-amplitude curve is a Rabi
cosine whose first minimum sits at `1/freq` (**not** the single-pulse `a_pi`); at higher
orders each stage re-windows around the previous valley and fits an inverted Lorentzian.
The report draws every order in its own colour (hue = quantity, depth = order) plus a
line chart of the four valley parameters against the number of pulse pairs.

`cargo run --release --example drag_amplitude_demo` · `qtool.drag.amplitude.fit(...)` · [live page](https://xncz.github.io/qtool/drag_amplitude_demo.html)
</details>

<details>
<summary><b>drag — coefficient</b></summary>

![drag coeff](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_coeff.png)

The leakage optimum is the phase-error optimum **times two** — the report fits the
valley, the doubling is the driver's business (`qtool.drag.coeff.valley_fit(...)` returns
the valley centre).

`cargo run --release --example drag_coeff_demo` · [live page](https://xncz.github.io/qtool/drag_coeff_demo.html)
</details>

<details>
<summary><b>drag — detuning</b> — carrier residual</summary>

![drag detuning](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_detuning.png)

Sweeps the drive's carrier detuning; the valley centre is the residual carrier error to
correct. A zero-detuning reference line is drawn on the three panels that share the
frequency axis, so "how far off is the working frequency" is readable at a glance.

`cargo run --release --example drag_detuning_demo` · [live page](https://xncz.github.io/qtool/drag_detuning_demo.html)
</details>

<details>
<summary><b>bloch</b> — three-basis readout → Bloch trajectory</summary>

![bloch](https://raw.githubusercontent.com/xncz/qtool/master/docs/bloch.gif)

Feed it the X/Y/Z-basis readout of a sweep: it projects the three series once (one axis
for the whole trajectory, so the radius means something), then draws the trajectory as a
3D Bloch sphere — orbit-rotatable, scroll-zoomable — next to its YZ/XZ/XY projections.
Points are coloured by P1 (blue at `|0>`, red at `|1>`), hovering a point on the sphere
lights up the matching point in all three projections.

The sphere carries no solid surface on purpose: plotly picks 3D hover **by depth**, so any
filled ball (even translucent) would occlude every point inside it.

`cargo run --release --example bloch_demo` · `qtool.bloch.bloch_vector(...)` · [live page](https://xncz.github.io/qtool/bloch_demo.html)
</details>

## Data export

Every card carries its payload **inside the HTML fragment** — the same numbers the figure is
drawn from — and adds six buttons to the plotly modebar: `csv`, `txt`, `npz`, `mat`, `xlsx`
and `arrow`. A click downloads `<div_id>.<ext>` (`t1-0.npz`, `qspec-z.arrow.zip`, …), with no
server and no notebook in the loop.

The payload is self-describing: one table per scan, column names tagged with their SI unit,
complex numbers kept complex, and `ref` columns that name the scan sub-tables. Each ecosystem
has one format it is expected to read:

| who | reads | why |
|---|---|---|
| Python | `.npz` | numpy is already there; structured arrays carry the names, units and dtypes |
| MATLAB / Julia | `.mat` | `load` + struct field access, native complex, nothing to install |
| Rust / R | `.arrow.zip` | one Feather (Arrow IPC) file per table; the schema holds the units |
| humans | `.xlsx` | double-click; one sheet per table, `<column> [unit]` headers |

`.csv` / `.txt` need no library anywhere and remain the fallback.

**[data.md](data.md)** documents the payload field by field, the six containers, and one
verified reader per ecosystem — the real commands and their real output for all 14 payloads.

## Quick start

### Rust

```rust
use qtool::superconductor::StateCenters;
use qtool::superconductor::t1::{t1_fit, t1_plot::t1_plot_div};

// taus in seconds, iq = the shot-averaged complex readout, states = calibrated centres
let states = StateCenters::new(vec![g0, g1]);
let fit = t1_fit(&taus, &iq, Some(&states), None)?;

let div = t1_plot_div(&taus, &iq, Some(&states), None, Ok(&fit), "t1-q0", Some("T1 — Q0"));
```

Drop `div` into any page that loads `plotly.js`; the crate re-exports the CDN tag:

```rust
use qtool::superconductor::t1::t1_plot::PLOTLY_JS_CDN;
let page = format!("<!doctype html><meta charset=\"utf-8\">{PLOTLY_JS_CDN}{div}");
```

The `examples/*_demo.rs` files are the executable version of every card above — each one
synthesises data with a known truth, fits it, checks the numbers it prints, and writes
`plt/<name>.html`. Start from the one closest to your experiment.

### Python

```python
import qtool

states = qtool.StateCenters([0.30 + 0.10j, 0.42 + 0.30j])
fit = qtool.t1.fit(taus, iq, states=states)          # raises qtool.t1.T1Error on failure
qtool.t1.plot(taus, iq, states=states, fit=fit)      # last line of a cell → the report shows up
```

The returned object is an `Html` (a `str` subclass): it renders itself in a notebook,
behaves like a string in a script, and writes a standalone page on demand:

```python
from pathlib import Path

qtool.t1.plot(taus, iq, states=states, fit=fit).persist(Path("reports/q0"))
# → reports/q0/t1.html
```

Nothing has to be loaded by hand: the page carries its own `plotly.js` tag, so this works
in JupyterLab, in classic Notebook and in VS Code without `init_notebook_mode`, a renderer
setting, or a manual `<script>`. Arrays cross the boundary as numpy arrays; `fit_batch`
returns one entry per line and hands failures back as exception **instances** (they can be
inspected, or fed straight back into `plot`, which re-raises them).

## API at a glance

| Module | Entry points |
| --- | --- |
| `superconductor` | `StateCenters`, `p1`, `p1_sigma`, `direction` — the IQ → P1 vocabulary every experiment shares |
| `superconductor::s21` | `s21_fit` / `s21_fit_batch`, `s21_fit_plot_div`, `s21_power_plot_div` + `PowerLine` |
| `superconductor::qspec` | `qspec_fit` / `_batch`, `qspec_fit_plot_div`, `qspec_z_plot_div` + `QspecZLine` |
| `superconductor::rabi` | `rabi_amp_fit` / `_batch`, `rabi_amp_plot_div` |
| `superconductor::iq` | `iq_stats`, `pair_projection`, `iq_plot_div` |
| `superconductor::ramsey` | `ramsey_fit` / `_batch`, `ramsey_plot_div` |
| `superconductor::t1` · `t2_echo` | `t1_fit` / `t2_echo_fit` (+ `_batch`), `t1_plot_div` / `t2_echo_plot_div` |
| `superconductor::drag::{amplitude,coeff,detuning}` | `factor_one_fit` / `factor_n_fit` (amplitude), `valley_fit` (coeff, detuning) — each with a `_batch` twin; `drag_*_plot_div` + `plot::OrderScan` |
| `superconductor::bloch` | `bloch_vector` + `Trajectory`, `bloch_plot_div` |

Every `*_fit` shares one shape — `(axis, iq, states, sigma)` in, `Result<Fit, Error>` out,
with a `*_fit_batch` twin that runs the lines in parallel through `rayon`. `states` is
`Option<&StateCenters>` everywhere except the DRAG family, where it is required; `s21` is
the one fit that works on the raw complex IQ and takes no `states` at all. The plot entry
points that draw a fit take it as `Result<&Fit, &Error>`, so a failed fit still renders a
card that says what went wrong.

## Performance

`fit_batch` is the reason the core is Rust: one call, every core, nothing for you to set
up. Both benches solve the **identical 3000 problems**, and the model under them is checked
to agree before anything is timed:

```bash
cargo run --release --example s21_batch_bench                    # the Rust rows (repo root)
cd qtool-py && uv run python bench/bench_s21_batch.py --lines 3000   # the Python rows
```

### Test machine

| | |
| --- | --- |
| CPU | AMD Ryzen 9 9950X — 16 cores / 32 threads, 4.3 GHz, 16 MB L2, 64 MB L3 |
| Memory | 61.6 GB |
| OS | Windows 11 Pro for Workstations 10.0.26200, x86-64 |
| Rust | rustc 1.96.0, `--release`; `rayon` on 32 threads |
| Python | CPython 3.14.3, numpy 2.5.3, scipy 1.18.1 |

### Method

- **The problems.** 3000 S21 notches, each on the same 51-point, 10 MHz window centred on
  6.8982 GHz. Per line `fr` is jittered by ±3 MHz (≈3 linewidths), `ql` by ±25%, `qc` by
  ±30% and `θ` by ±0.4 rad; the parameters and the complex noise both come from a
  deterministic LCG seeded per line, so the two benches solve **bit-for-bit the same
  problems**. The numpy model is checked against the Rust kernel's `model_at` before
  anything is timed — worst relative deviation 9.2e-14, i.e. round-off — and the two sides
  land on the same optimum: fitted `fr` and `ql` agree to 3e-16 relative.
- **What is timed.** The fits, and nothing else: problem generation (0.2 s), interpreter
  and module import, and the process pool's start-up stay outside the clock or are
  reported on their own line.
- **Repetitions.** Three rounds of each bench, run **alternately** — Rust then Python — so
  both sides see the same state of the machine; every row is the fastest of its three, with
  no core pinning or affinity games. The long rows are the steadiest (the 95 s serial
  pipeline repeats to 1.2%, the 5.3 s rayon one to 7.4%) and the short parallel rows carry
  the most relative noise, so read the ratios, not the last digit.
- **Fair pairing.** The same-task rows share the data, the start point *and* the Jacobian:
  both sides are handed a hand-written analytic one (the Rust Jacobian ships with the
  crate, the numpy one is checked against central differences to 4.6e-06 before anything
  is timed). The single row that uses scipy's default finite differences is labelled as
  such. `curve_fit` also gets the same complex-to-real split lmfit uses; the order inside
  that vector is a permutation and does not change the least-squares problem. The pipeline
  rows are a strictly bigger job, which is why they are printed separately.
- **Caveat.** The fixture's noise is ~1e-12 of the model magnitude, so these are *easy*
  fits: every time below is a **lower bound**. The ranking is the point, not the
  milliseconds.

### Same task, same start point — 3000 fits from `p0` = the true parameters

With the same Jacobian and the same start, both sides converge in **two residual
evaluations** — the fit is the same short sequence of model sweeps and Jacobians on either
side, and that is what the table compares:

| path | Jacobian | 3000 fits | per fit | vs analytic scipy |
| --- | --- | --- | --- | --- |
| `scipy.optimize.curve_fit`, serial | finite differences — scipy's default | 2.28 s | 0.760 ms | 0.42× |
| `scipy.optimize.curve_fit`, serial | **analytic, written by hand** | 0.96 s | 0.321 ms | 1× |
| **`qtool` core (Rust), serial** — what `s21_fit` runs | analytic, ships with the crate | **0.157 s** | **0.052 ms** | **6.2×** |
| `scipy.optimize.curve_fit`, 32-process pool (1 chunk/worker) | analytic | 0.09 s | 0.030 ms | 10.7× |
| **`qtool`, `rayon` 32 threads** | analytic | **0.011 s** | **0.0037 ms** | **86×** |

Read the second column before the last one. With the Jacobian taken out of the picture,
what is left is the interpreter: a numpy sweep of the 51 points the Rust kernel does in
3.0 µs (60 ns/point) costs 42 µs, and a numpy Jacobian 67 µs. Two of each is 219 µs of the
0.321 ms; the rest is scipy's own machinery. Against scipy's **default** finite differences
the same two rows read **14.6×** (serial) and **204×** (rayon).

Start from a realistic point instead of the optimum — estimator-grade: `fr` off by a
linewidth (1 MHz), `ql` by 25%, `qc` by 30%, `θ` by 0.3 rad — and a single fit costs 15.3 ms
in scipy with **3.7% of the lines never converging** (24.4 ms and 6% with the default
differences), against 6.7 ms in the core, which loses none.

### The whole pipeline — 3000 lines, no start point given

`s21_fit` estimates the initial values from the data, fits up to eight candidates per line
and keeps the best:

| path | 3000 lines | per line | vs serial |
| --- | --- | --- | --- |
| `s21_fit` serial — estimation + up to 8 fits | 95.0 s | 31.7 ms | — |
| `s21_fit_batch` (rayon, 32 threads), the same work | 5.3 s | 1.78 ms | 17.9× |

scipy has no entry point that does this — the estimator is ~90 lines of geometry you would
have to port yourself — so there is no fair Python row to print beside it, and none is
needed: the same-task table is what composes into this one.

### The GIL, and what it costs to work around it

The GIL blocks *threads*, not processes — so a Python user who wants the other 31 cores has
to reach for `multiprocessing`, and pays for it on every run:

- **A pool has to be built.** Constructing a 32-process pool costs **1.0–1.5 s** on Windows
  (**2.7–3.4 s** for the first one in a process, where every child re-imports numpy and
  scipy) — before a single fit has run.
- **The chunk size has to be guessed, and the answer moves.** Across three rounds one chunk
  per worker gave 0.09–0.25 s of map time, four chunks per worker 0.18–0.56 s. Too coarse
  and the workers idle; too fine and the pickling is the job.
- **Every chunk crosses a pipe.** The IQ in, the results and the exception instances back
  out — all pickled, on top of whatever the fit itself costs.
- **It does not win end to end.** Map plus start-up is **1.1–1.7 s** for the 3000 lines,
  against the **0.96 s** the same single-threaded fit takes when it is simply handed the
  analytic Jacobian. The pool beats scipy's *default* serial loop (2.28 s) and nothing else.

And when the pool does work, it reaches **10.7×** — 0.030 ms per fit. `rayon`, which has
none of those four costs, reaches **13.9×** at 0.0037 ms. Same task, same Jacobian:
**8.1×** apart in absolute terms, from a call that takes no arguments for it.

## License

MIT — see [LICENSE](LICENSE).
