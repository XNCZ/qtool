# Reading the exported data

Every report card carries an **embedded data payload** and five export buttons in the plotly
modebar: `csv`, `txt`, `npz`, `mat`, `xlsx`. All five carry the same payload in a different
container, named after the report's `div_id` (`t1-0.csv`, `power-res0.mat`, `iq-99.xlsx`, …).

**Every table and column of every report, in every format, with the commands that read them and
what those commands print.** §3 covers the five formats across eight readers; §4 walks all 13
reports file by file.

---

## 1. What the payload holds

```json
{ "version": "0.2.3", "timestamp": {"utc": "...", "local": "..."}, "name": "t1-0",
  "tables": [ { "name": "data",
                "columns": [ {"key": "tau", "unit": "s", "kind": "real"},
                             {"key": "iq",  "unit": "a.u.", "kind": "complex"} ],
                "data": { "tau": [...], "iq": [[re, im], ...] } } ] }
```

* `kind` is `real`, `complex` or `ref`.
  * `real` / `complex` say how the exporter lays a value out.
  * **`ref` is a foreign key**: the value **is the table name** of the sibling sub-table
    (`row` = `0` ⇒ table `"0"`), so a lookup is `tables[str(value)]` with no prefix to build.
    `.mat` is the one format that escapes it, writing `row_0` — see §2.
* Missing values are `null`; in csv/txt they are empty fields; in every numeric container, `NaN`.
* Units are SI. Dimensionless is `1`; arbitrary units is `a.u.`.

### Table layout shared by all reports

| table | one row per | contents |
|---|---|---|
| `data` | scan point | the swept axis, raw complex IQ, per-point σ, the projected `p1`, the fitted `model` |
| `fits` | fit | every fitted parameter and its stderr (`<name>`, `<name>_stderr`) |
| `params` | report | report-level scalars (e.g. `chosen`) |
| `states` | readout state | the calibration centres |
| `0`, `1`, … | scan point of scan *i* | per-scan detail, **named by the scan index**; `fits.row` holds that name (`.mat` spells it `row_<i>`) |
| `flux` | report | flux-tuning fit (`qspec vs Z` only) |
| `density`, `pairs` | state / state pair | density regions and discrimination stats (`iq` only) |

A single-scan report (`t1`, `t2_echo`, `ramsey`, `rabi`, `s21`, `qspec`) has `data` + `fits` +
`states`. A multi-scan report (`s21 vs power`, `qspec vs Z`, `drag-*`) puts one row per scan in
`fits` and one numerically-named sub-table per scan — **no `data` table**, the per-point rows live
in the scan sub-tables. `iq` has no fit; `bloch` has a single `data` table.

---

## 2. The five formats — and **where the column names live**

| | `.csv` | `.txt` | `.npz` | `.mat` | `.xlsx` |
|---|---|---|---|---|---|
| container | text, comma | text, tab | zip of `.npy` | MAT v5 | xlsx |
| tables | **only the first** as real rows, the rest `# `-commented | same | **one structured array per table** | **one struct per table**, plus a `units` struct | one sheet per table |
| **column names** | header row; each commented table repeats them on a `# columns:` line | same | **`z["fits"].dtype.names`** | **`fieldnames(m.fits)`** | row 1, `<column> [unit]` |
| **units** | on the `# columns:` line | same | **inside the field name: `'t1 [s]'`** | **`m.units.fits.t1`** | inside the header cell |
| complex | `re±imj` text | same | `complex128` | complex double | complex cell |
| `ref` column | number | number | **`int64`** | `double` | number |
| **table names** | as-is (`# ---- 0 ----`) | same | **as-is (`"0"`)** | **`row_0`** — see below | as-is (`0`) |
| missing | empty field | empty field | `NaN` | `NaN` | empty cell |

Because the names carry everything, the **`.npz` is fully self-describing — no side-car member**,
and the **`.mat` needs no lookup either**. Three details:

* **Only `.mat` escapes the numeric table names.** MATLAB identifiers cannot start with a digit —
  a variable named `0` makes `load` reject the *whole file* (verified:
  `MATLAB:AddField:InvalidFieldName`), so the writer prefixes them: table `0` → `m.row_0`, and the
  same escape is applied inside `m.units`. Every other container keeps the bare index, so
  `[n for n in z.files if n.isdigit()]` gives the scan indices and `str(row_value)` is the key.
* **Field names come in pairs.** Each npz field is written as `(('key', 'key [unit]'), '<dtype>')`:
  the *name* is the labelled form (`'t1 [s]'`, what `dtype.names` shows) and the *title* is the
  clean form. numpy indexes by either, so both `z["fits"]["t1"]` and `z["fits"]["t1 [s]"]` work.
* **MATLAB forbids spaces and brackets in struct field names** (verified: `'t1 [s]'` →
  *"Invalid field name"*), so the `.mat` keeps clean field names and carries the units in a
  parallel `units` struct instead.

csv/txt start with a 4-line preamble:

```
# qtool 0.2.3 | 2026-10-09T22:18:56+08:00 | t1-0
# complex columns: data.iq, states.center  (literal <re><sign><im>j; pandas: converters={<col>: complex})
# ---- data ----
# columns: tau [s], iq [a.u.], iq_sigma [a.u.], p1 [1], p1_sigma [1], model [1]
```

---

## 3. One reader per ecosystem

Each ecosystem gets the one format that fits it — there is no cross-reading ladder any more (Rust is
not expected to read the `.npz`, Python is not expected to read the `.mat`). Every block below was
run against the exported `qspec-z.*` files (the 24-table `qspec vs Z` card); the `#` comments are the
real output.

One card produces five sibling files — `qspec-z.csv` `qspec-z.txt` `qspec-z.npz` `qspec-z.mat`
`qspec-z.xlsx` — and a report with two cards adds `qspec-z-window.*` alongside them.

### 3.1 MATLAB — `.mat` (nothing to install)

```matlab
m = load("qspec-z.mat");

fieldnames(m)          % 25 tables: {'fits';'flux';'states';'row_0';…;'row_20';'units'}
fieldnames(m.fits)     % {'row';'z';'fq';'fq_stderr';'fwhm';'fwhm_stderr';'amp';'amp_stderr';'offset';'offset_stderr'}

m.fits.fq(1)           % 5.036781e+09
m.flux.f_max           % 5.050032e+09
m.fits.row'            % [0 1 2 … 20]           the ref column: its value IS the sub-table's index
m.row_0.iq(1)          % 0.309314 + 0.0565525i  native complex
m.states.center        % [0.3+0.1i, 0.42+0.3i]
size(m.row_0.freq)     % 121  1
m.units.fits.fq        % 'Hz'
```

Only `.mat` spells the scan tables `row_<i>` (MATLAB identifiers cannot start with a digit — see §2);
`m.units` uses the same spelling, so `m.units.row_0.freq` is `'Hz'`.

### 3.2 Python — `.npz` (numpy only, no other package needed)

```python
import numpy as np
z = np.load("qspec-z.npz", allow_pickle=False)

z.files                          # 24 tables: ['fits', 'flux', 'states', '0', '1', …, '20']
[int(n) for n in z.files if n.isdigit()]      # [0, 1, …, 20]   the scan indices, no parsing

z["fits"].dtype.names            # ('row [1]', 'z [V]', 'fq [Hz]', 'fq_stderr [Hz]', 'fwhm [Hz]', …)
z["fits"]["fq"][0]               # 5036781280.342374    ← clean field title
z["fits"]["fq [Hz]"][0]          # the same value       ← labelled field name
z["flux"]["f_max"][0]            # 5050032216.098451
z["0"]["freq"][0]                # 5020000000.0         ← table named by the scan index
z["0"]["iq"][0]                  # (0.30931418381332726+0.05655247203434104j)
z["states"]["center"]            # [0.3+0.1j 0.42+0.3j]
z["fits"]["row"].dtype           # dtype('int64')

for k in z["fits"]["row"]:       # the ref value is the key — nothing to build
    scan = z[str(int(k))]
    fq_k = z["fits"]["fq [Hz]"][k]
```

The whole payload as a plain `{table: {column: array}}`:

```python
def load(path, clean=False):
    """clean=False keeps the labelled names ('fq [Hz]'), True the bare titles ('fq')."""
    z = np.load(path, allow_pickle=False)
    out = {}
    for table in z.files:
        a = z[table]
        cols = {}
        for n in a.dtype.names:          # iterate names, not fields — a titled field is listed twice there
            match a.dtype.fields[n]:
                case (_, _, title):      # 3-tuple: the field carries a title
                    key = title if clean else n
                case (_, _):             # 2-tuple: no title
                    key = n
            cols[key] = a[n]
        out[table] = cols
    return out

d = load("qspec-z.npz")
d["fits"]["fq [Hz]"][0]          # 5036781280.342374
d["0"]["iq [a.u.]"][0]           # (0.30931418381332726+0.05655247203434104j)
load("qspec-z.npz", clean=True)["fits"]["fq"][0]      # same number, bare name
```

### 3.3 Julia — `.mat` (MAT.jl)

```julia
using MAT
m = matread("qspec-z.mat")

length(m)                        # 25
m["fits"]["fq"][1]               # 5.036781280342374e9
m["flux"]["f_max"][1]            # 5.050032216098451e9
m["fits"]["row"][1:3]            # [0.0, 1.0, 2.0]                     the ref column
m["row_0"]["iq"][1]              # 0.30931418381332726 + 0.05655247203434104im
size(m["row_20"]["freq"])        # (121, 1)
m["states"]["center"]            # ComplexF64[0.3 + 0.1im, 0.42 + 0.3im]
m["units"]["fits"]["fq"]         # "Hz"
```

Into a DataFrame, if you want one (the `vec` is required — MAT.jl hands back `n×1` matrices):

```julia
using DataFrames
DataFrame(Dict(k => vec(v) for (k, v) in m["row_0"]))     # 121×4
```

### 3.4 R — `.arrow` *(planned)*

The `.arrow` export is not in this build yet; today R reads the `.csv` (§3.6) or the `.mat` through
`R.matlab`, which rewrites `_` to `.` in names and flattens the nested `units` struct. The intended
path, once the export ships:

```r
library(arrow)
d <- tempfile(); unzip("qspec-z.arrow.zip", exdir = d)
list.files(d)                     # "fits.arrow" "flux.arrow" "states.arrow" "0.arrow" …

fits <- read_feather(file.path(d, "fits.arrow"))
names(fits)                       # "row" "z" "fq" "fq_stderr" …   exact, nothing mangled
fits$fq[1]                        # 5.036781e+09

r0 <- read_feather(file.path(d, "0.arrow"))
r0$iq <- r0$iq_re + 1i * r0$iq_im            # Arrow has no complex type — recombine
st <- read_feather(file.path(d, "states.arrow"))

sch <- read_feather(file.path(d, "fits.arrow"), as_data_frame = FALSE)$schema
sch$GetFieldByName("fq")$metadata            # unit -> "Hz"
```

### 3.5 Rust — `.arrow` *(planned)*

Same caveat as R; today Rust reads `.csv` (the `.npz` and `.mat` are both unreadable from Rust — see
§5). Once the export ships, one zip entry per table:

```rust
use arrow::ipc::reader::FileReader;
use std::{fs::File, io::{Cursor, Read}};
use zip::ZipArchive;

let mut archive = ZipArchive::new(File::open("qspec-z.arrow.zip")?)?;
for i in 0..archive.len() {
    let mut entry = archive.by_index(i)?;
    let name = entry.name()?.to_string();          // "0.arrow", "fits.arrow", …
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf)?;                  // ZipFile has no Seek; FileReader needs Read + Seek
    let reader = FileReader::try_new(Cursor::new(buf), None)?;
    for f in reader.schema().fields() {
        // fq: Float64  unit=Hz ; row: Int64  unit=1
        println!("{} {:?} unit={}", f.name(), f.data_type(),
                 f.metadata().get("unit").map(String::as_str).unwrap_or("-"));
    }
    for batch in reader { /* batch.column_by_name("fq") … */ }
}
```

### 3.6 Humans, and the two text formats

`.xlsx` opens by double-click: one sheet per table (sheet names as in the payload — `fits`, `flux`,
`0`…`20`), header row 1 as `<column> [unit]`, complex values as real complex cells, missing values as
empty cells.

`.csv` / `.txt` need no library in any language. They carry **the first table only** as real rows;
the remaining tables follow as `# `-commented blocks with their own `# columns:` header line, so
read them with the comment character declared: MATLAB `CommentStyle"#"`, Julia `comments=true`,
R `comment.char="#"`, Rust `csv::ReaderBuilder::comment`. Without that the 4-line preamble is parsed
as data.

---

## 4. Every report, every table

Generated from the 13 demo reports in `plt/_export/`: every table each card carries, its shape and
its columns. `†` marks a complex column, `→ref` a foreign key to a scan sub-table (the value is that
table's name). Download names are `<div_id>.<ext>`; a report with two cards has a second `div_id`.

<!-- generated by plt/_export/gen_section4.py -->

### t1 — T1 energy decay

`div_id = t1-0` · files `t1-0.csv` `t1-0.txt` `t1-0.npz` `t1-0.mat` `t1-0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `t1[s]`, `t1_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

### t2_echo — T2 echo decay

`div_id = t2echo-0` · files `t2echo-0.csv` `t2echo-0.txt` `t2echo-0.npz` `t2echo-0.mat` `t2echo-0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `t2_echo[s]`, `t2_echo_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

### ramsey — T2* (Ramsey)

`div_id = ramsey-0` · files `ramsey-0.csv` `ramsey-0.txt` `ramsey-0.npz` `ramsey-0.mat` `ramsey-0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 10 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `frequency[Hz]`, `frequency_stderr[Hz]`, `phase[rad]`, `phase_stderr[rad]`, `decay[s]`, `decay_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

### rabi — Rabi amplitude

`div_id = rabi-0` · files `rabi-0.csv` `rabi-0.txt` `rabi-0.npz` `rabi-0.mat` `rabi-0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 41 × 6 | `amp[1]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `freq[1]`, `freq_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `a_pi[1]`, `a_pi_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |

### qspec — qubit spectroscopy

`div_id = qspec-0` · files `qspec-0.csv` `qspec-0.txt` `qspec-0.npz` `qspec-0.mat` `qspec-0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 51 × 6 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 8 | `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |

### qspec vs Z — spectroscopy vs flux bias

`div_id = qspec-z` · files `qspec-z.csv` `qspec-z.txt` `qspec-z.npz` `qspec-z.mat` `qspec-z.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 10 | `row[1]`→ref, `z[V]`, `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `flux` | 1 × 10 | `f_max[Hz]`, `f_max_stderr[Hz]`, `z_offset[V]`, `z_offset_stderr[V]`, `z_period[V]`, `z_period_stderr[V]`, `eta[Hz]`, `eta_stderr[Hz]`, `asymmetry[1]`, `asymmetry_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |
| `0` … `20` *(one per scan)* | 121 × 4 | `freq[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

*second card* `div_id = qspec-z-window` · same five files under that name

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 10 | `row[1]`→ref, `z[V]`, `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `flux` | 1 × 10 | `f_max[Hz]`, `f_max_stderr[Hz]`, `z_offset[V]`, `z_offset_stderr[V]`, `z_period[V]`, `z_period_stderr[V]`, `eta[Hz]`, `eta_stderr[Hz]`, `asymmetry[1]`, `asymmetry_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |
| `0` … `20` *(one per scan)* | 31 × 4 | `freq[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

### s21 — resonator spectroscopy

`div_id = demo-ok` · files `demo-ok.csv` `demo-ok.txt` `demo-ok.npz` `demo-ok.mat` `demo-ok.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 51 × 4 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `model[a.u.]`† |
| `fits` | 1 × 26 | `fr[Hz]`, `fr_stderr[Hz]`, `ql[1]`, `ql_stderr[1]`, `qc[1]`, `qc_stderr[1]`, `theta[rad]`, `theta_stderr[rad]`, `ap[1]`, `ap_stderr[1]`, `tau[s]`, `tau_stderr[s]`, `a[1]`, `a_stderr[1]`, `b[1]`, `b_stderr[1]`, `phi[rad]`, `phi_stderr[rad]`, `zc_re[a.u.]`, `zc_re_stderr[a.u.]`, `zc_im[a.u.]`, `zc_im_stderr[a.u.]`, `qi[1]`, `qi_stderr[1]`, `kappa_ex[Hz]`, `kappa_ex_stderr[Hz]` |

### s21 vs power — resonator vs pump power

`div_id = power-res0` · files `power-res0.csv` `power-res0.txt` `power-res0.npz` `power-res0.mat` `power-res0.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 28 | `row[1]`→ref, `power[a.u.]`, `fr[Hz]`, `fr_stderr[Hz]`, `ql[1]`, `ql_stderr[1]`, `qc[1]`, `qc_stderr[1]`, `theta[rad]`, `theta_stderr[rad]`, `ap[1]`, `ap_stderr[1]`, `tau[s]`, `tau_stderr[s]`, `a[1]`, `a_stderr[1]`, `b[1]`, `b_stderr[1]`, `phi[rad]`, `phi_stderr[rad]`, `zc_re[a.u.]`, `zc_re_stderr[a.u.]`, `zc_im[a.u.]`, `zc_im_stderr[a.u.]`, `qi[1]`, `qi_stderr[1]`, `kappa_ex[Hz]`, `kappa_ex_stderr[Hz]` |
| `0` … `20` *(one per scan)* | 51 × 4 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `model[a.u.]`† |

### iq — readout discrimination

`div_id = iq-99` · files `iq-99.csv` `iq-99.txt` `iq-99.npz` `iq-99.mat` `iq-99.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `states` | 3 × 3 | `state[1]`, `row[1]`→ref, `center[a.u.]`† |
| `density` | 12 × 4 | `state[1]`, `level[1]`, `area[a.u.^2]`, `radius[a.u.]` |
| `pairs` | 3 × 7 | `p[1]`, `q[1]`, `separation[a.u.]`, `threshold[a.u.]`†, `error_rate[1]`, `auc[1]`, `snr[1]` |
| `0` … `2` *(one per scan)* | 6000 × 1 | `iq[a.u.]`† |

### bloch — three-axis readout

`div_id = bloch` · files `bloch.csv` `bloch.txt` `bloch.npz` `bloch.mat` `bloch.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 61 × 4 | `axis[a.u.]`, `x[1]`, `y[1]`, `z[1]` |

### drag — amplitude scan

`div_id = drag-3` · files `drag-3.csv` `drag-3.txt` `drag-3.npz` `drag-3.mat` `drag-3.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 2 × 10 | `row[1]`→ref, `pairs[1]`, `centre[1]`, `centre_stderr[1]`, `fwhm[1]`, `fwhm_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 3 | `a_pi[1]`, `a_pi_stderr[1]`, `chosen[1]` |
| `0` … `2` *(one per scan)* | 31 × 4 | `amp[1]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

### drag — coeff scan

`div_id = drag-coeff-3` · files `drag-coeff-3.csv` `drag-coeff-3.txt` `drag-coeff-3.npz` `drag-coeff-3.mat` `drag-coeff-3.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 3 × 10 | `row[1]`→ref, `pairs[1]`, `centre[1]`, `centre_stderr[1]`, `fwhm[1]`, `fwhm_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 1 | `chosen[1]` |
| `0` … `2` *(one per scan)* | 31 × 4 | `coeff[1]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

### drag — detuning scan

`div_id = drag-detuning-3` · files `drag-detuning-3.csv` `drag-detuning-3.txt` `drag-detuning-3.npz` `drag-detuning-3.mat` `drag-detuning-3.xlsx`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 3 × 10 | `row[1]`→ref, `pairs[1]`, `centre[Hz]`, `centre_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 1 | `chosen[Hz]` |
| `0` … `2` *(one per scan)* | 31 × 4 | `detuning[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

---

## 5. Verification status

Everything above was run against the files the report pages themselves produced: 13 demo reports,
14 payloads (`qspec vs Z` has two cards), 5 formats — 70 files. Names, units, shapes and values were
cross-checked against the payload embedded in the HTML, and every table name was checked against the
tables actually present in the container.

| ecosystem | format | read with | verified |
|---|---|---|---|
| MATLAB R2020b (9.9) | `.mat` | `load` | yes — names, units, native complex, all 24 tables |
| Python 3.14 / numpy 2.5 | `.npz` | `np.load` | yes — field titles and labelled names, `complex128`, `int64` refs |
| Julia 1.12 | `.mat` | `MAT.jl` 0.12 | yes — `ComplexF64`, units, every sub-table |
| R 4.6 | `.arrow` *(planned)* | `arrow` | format not shipped yet; today `R.matlab` / `read.csv` / `openxlsx` |
| Rust 1.96 | `.arrow` *(planned)* | `arrow` (`ipc`) | format not shipped yet; today `csv` / `calamine` |
| Excel | `.xlsx` | double-click | one sheet per table, `<column> [unit]` headers |

### Gotchas found while verifying

1. **`#` must be declared as a comment** in every text reader, or the 4-line preamble is parsed as
   data: MATLAB `CommentStyle"#"`, Julia `comments=true`, R `comment.char="#"`, Rust
   `csv::ReaderBuilder::comment`. Without it MATLAB returns 115 rows instead of 102 for `t1`.
2. **Complex literals in csv/txt/xlsx use the `j` suffix** (`re±imj`). Python's `complex()` and
   MATLAB's `readmatrix` take them as-is; R needs `sub("j$", "i", x)`; Julia needs
   `parse(ComplexF64, x)`.
3. **npz field names come in pairs**: name `'fq [Hz]'` (what `dtype.names` shows) and title `'fq'`
   (indexing by either works). Getting the bare title needs `dtype.fields`, and iterating
   `dtype.fields` double-counts — iterate `dtype.names` instead.
4. **Digit-leading table names only survive outside `.mat`.** MATLAB cannot have a variable named
   `0`; `load` rejects the *entire file* with `MATLAB:AddField:InvalidFieldName`, so the `.mat`
   writer prefixes the scan tables: table `0` → `m.row_0`, and `m.units` follows the same spelling.
5. **`.mat` field names cannot hold the unit** — MATLAB rejects spaces and brackets in field names,
   so units live in the parallel `units` struct (`m.units.fits.fq`).
6. **`R.matlab` rewrites `_` to `.`** in variable names (`row_0` → `m$row.0`) and flattens the nested
   `units` struct; **`openxlsx` turns spaces in the header row into dots** (`t1 [s]` → `t1.[s]`) and
   hands back the complex column of the `data` sheet as empty cells — R's weakest path is the xlsx.
7. **Rust cannot read the `.npz` or `.mat`**: `ndarray-npy` rejects structured descriptors (with an
   explicit error), `npyz` rejects the field-title descriptor (`list entry must contain a string for
   id`), and `matfile` skips struct variables silently (a struct-carrying file reports 0 variables).
   Rust users take `.csv` / `.xlsx` today, `.arrow` once it ships.
8. **A complex column no longer complexifies its table**: with the structured descriptors `iq` is
   `complex128` while `tau`/`p1` stay `float64`.
9. **The `ref` column differs by container**: `int64` in `.npz`, `double` in `.mat`, plain number in
   the text formats — all three carry the same values, only the storage differs.
