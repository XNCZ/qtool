# qtool-py

Python bindings for **[qtool](https://github.com/xncz/qtool)** — superconducting-qubit
calibration analysis: fit the standard spectroscopy / Rabi / Ramsey / T1 / T2-echo /
DRAG sweeps, and render each one as an interactive plotly report.

The fitting core, the plotting and the HTML generation all run in Rust — no part of a fit
enters the interpreter, and `fit_batch` uses every core; Python only holds the API.

That second half is the point. Using 32 cores *from Python* means a `multiprocessing` pool
you have to construct (**1.0–1.5 s** on Windows, 2.7–3.4 s cold), a chunk size you have to
guess, and a pickle per chunk across a pipe — and at 3000 lines it still ends up slower than
a single-threaded scipy fit that was merely handed the analytic Jacobian. `fit_batch` spreads
the same work over the same cores from inside the call, at **17.9×** the serial rate. Numbers
below; method and caveats in the
[main README](https://github.com/xncz/qtool#performance).

## Install

```bash
pip install qtool-py          # or: uv add qtool-py
```

## Quickstart

```python
import qtool

# a T1 relaxation sweep: taus (s), iq = complex readout, states = calibrated centres
states = qtool.StateCenters([0.30 + 0.10j, 0.42 + 0.30j])
fit = qtool.t1.fit(taus, iq, states=states)

print(fit.model.t1, dict(fit.params)["t1"])   # value and stderr
qtool.t1.plot(taus, iq, states=states, fit=fit)   # last line of a cell → the report shows up
```

**Nothing has to be loaded by hand.** The returned object is an `Html` (a `str`
subclass): in a notebook it renders itself (the page carries its own plotly.js tag
inside a `srcdoc` iframe), in a script it is just a string, and to save a standalone
page:

```python
from pathlib import Path

qtool.t1.plot(taus, iq, states=states, fit=fit).persist(Path("reports/q0"))
# → reports/q0/t1.html, self-contained
```

## What is in the box

| Module | Fit |
| --- | --- |
| `qtool.s21` | resonator `S21(f)` (11-parameter complex model), plus the power-sweep report |
| `qtool.qspec` | qubit spectroscopy (Lorentzian), single line and vs. flux |
| `qtool.rabi` | Rabi amplitude (cosine, anchored at `P1(0)=0`) |
| `qtool.iq` | readout IQ clouds: per-state density regions, pairwise separability |
| `qtool.ramsey` / `qtool.t1` / `qtool.t2_echo` | damped cosine and exponentials |
| `qtool.drag.{amplitude,coeff,detuning}` | the DRAG calibration chain (first order and the valley fits) |
| `qtool.bloch` | three-basis readout → Bloch-vector trajectory (3D report) |

Every fitting module exposes the same shape: `fit(...)` (raises on failure),
`fit_batch(...)` (one result per line, failures come back as exception **instances**),
and `plot(...)` returning an `Html`. `qtool.iq` and `qtool.bloch` are analysis plus plot —
their entry points are `iq_stats`/`pair_projection` and `bloch_vector`. Array-valued results
cross the boundary as numpy arrays; `params` is a list of `(name, (value, stderr))` tuples,
so `dict(fit.params)["t1"]` is how you look one up.

## What the reports look like

Screenshots of the live figures — hover, legend toggles, zoom, and the Bloch sphere's orbit
controls all work in the page `plot(...)` returns:

**s21** — resonator spectroscopy, 11-parameter complex model

![s21 fit](https://raw.githubusercontent.com/xncz/qtool/master/docs/s21.png)

**s21 vs power** — the same fit swept over drive power

![s21 vs power](https://raw.githubusercontent.com/xncz/qtool/master/docs/s21_power.png)

**qspec** — qubit spectroscopy (Lorentzian)

![qspec](https://raw.githubusercontent.com/xncz/qtool/master/docs/qspec.png)

**qspec vs Z** — the flux sweep

![qspec vs Z](https://raw.githubusercontent.com/xncz/qtool/master/docs/qspec_z.png)

**rabi** — Rabi amplitude, cosine anchored at `P1(0)=0`

![rabi](https://raw.githubusercontent.com/xncz/qtool/master/docs/rabi.png)

**iq** — readout clouds: density regions and pairwise separability

![iq](https://raw.githubusercontent.com/xncz/qtool/master/docs/iq.png)

**ramsey** — T2* from a damped cosine

![ramsey](https://raw.githubusercontent.com/xncz/qtool/master/docs/ramsey.png)

**t1** — energy relaxation

![t1](https://raw.githubusercontent.com/xncz/qtool/master/docs/t1.png)

**t2_echo** — echo dephasing

![t2_echo](https://raw.githubusercontent.com/xncz/qtool/master/docs/t2_echo.png)

**drag — amplitude** — first order and the escalated valley fits

![drag amplitude](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_amplitude.png)

**drag — coefficient**

![drag coeff](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_coeff.png)

**drag — detuning** — carrier residual

![drag detuning](https://raw.githubusercontent.com/xncz/qtool/master/docs/drag_detuning.png)

**bloch** — three-basis readout → Bloch trajectory (orbit-rotatable, hover cross-highlights
the three projections)

![bloch](https://raw.githubusercontent.com/xncz/qtool/master/docs/bloch.gif)

## Performance

3000 S21 lines (51 frequency points each) on a 32-thread Ryzen 9 9950X, best of three runs.
Both sides solve the identical problems, and the same-task rows are given the same start
point *and* a hand-written analytic Jacobian, so what is left on the Python side is the
interpreter:

| path | 3000 fits | per fit |
| --- | --- | --- |
| `scipy.optimize.curve_fit`, serial — **scipy's default** (finite differences) | 2.28 s | 0.760 ms |
| `scipy.optimize.curve_fit`, serial — analytic Jacobian | 0.96 s | 0.321 ms |
| **`qtool` core (Rust), serial** — Jacobian ships with the crate | **0.157 s** | **0.052 ms** |
| `scipy.optimize.curve_fit`, 32-process pool — analytic | 0.09 s | 0.030 ms |
| **`qtool` core (Rust), `rayon` 32 threads** | **0.011 s** | **0.0037 ms** |

The whole pipeline, no start point given — `fit_batch` estimates the initial values from the
data, fits up to eight candidates per line and keeps the best:

| path | 3000 lines | per line |
| --- | --- | --- |
| `qtool.s21.fit`, one line at a time — estimation + up to 8 fits | 95.0 s | 31.7 ms |
| `qtool.s21.fit_batch` (32 threads), the same work | 5.3 s | 1.78 ms (**17.9×**, 0 failures) |

From a realistic (estimator-grade) start a single scipy fit costs 15.3 ms and **3.7% of the
lines never converge**; the core takes 6.7 ms and loses none. Machine, method, controls and
caveats are in the [main README](https://github.com/xncz/qtool#performance).

Full documentation and the Rust API live in the
[main README](https://github.com/xncz/qtool).

## License

MIT
