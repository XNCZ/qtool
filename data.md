# Reading the exported data

Every report card carries an **embedded data payload** and six export buttons in the plotly
modebar: `csv`, `txt`, `npz`, `mat`, `xlsx`, `arrow`. All six carry the same payload in a
different container, named after the report's `div_id` (`t1-0.csv`, `power-res0.mat`,
`iq-99.xlsx`, `qspec-z.arrow.zip`, …).

**Each ecosystem has exactly one format it is expected to read** — no cross-reading ladder:

| who | reads | why |
|---|---|---|
| Python | `.npz` | numpy is already there; structured arrays carry names, units and dtypes |
| MATLAB | `.mat` | `load` + struct field access, native complex, nothing to install |
| Julia | `.mat` | MAT.jl, same file as MATLAB reads |
| Rust | `.arrow` | `arrow` crate; `zip` + `FileReader` per table |
| R | `.arrow` | `arrow::read_feather`; same Feather files Rust reads |
| humans | `.xlsx` | double-click; one sheet per table, `<column> [unit]` headers |

`.csv` / `.txt` still ship (no library needed anywhere, and they stay the fallback for a language
whose package is missing), but they are **not** part of the ladder above.

§1 describes the payload, §2 the six containers, §3 gives one working reader per ecosystem (§3.7
is addressed to an AI reader), §4 walks **all 14 payloads** with the exact commands and their real
output, §5 records what was run where, §6 collects the gotchas.

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
* Units are SI and always sit in brackets after the column name: `tau [s]`, `fq [Hz]`, `z [V]`,
  `phase [rad]`. Two special ones:
  * **`[1]` means dimensionless** — it is the unit "one", nothing more. It does *not* say the values
    are normalised to 0–1: `p1 [1]` *is* a probability by definition (0 = the `|0>` end of the
    readout axis, 1 = the `|1>` end), but `ql [1]` (quality factor, thousands) and `snr [1]`
    (a few) carry the same label and are nowhere near 0–1. `state [1]`, `pairs [1]`, `chosen [1]`,
    `level [1]` are indices / counts / ratios, likewise dimensionless.
  * **`[a.u.]` = arbitrary units** — the number sits on a consistent but **unknown** scale (readout
    gain, DAC full-scale, …): `iq [a.u.]`, `center [a.u.]`, s21's `model [a.u.]`,
    `separation [a.u.]`, `threshold [a.u.]`; an area on that scale is `area [a.u.^2]` (`density`).
    Compare values **within one dataset**, not across reports — another gain means another scale.
  * **The two differ in kind, not in degree.** `[1]` says "this quantity has no unit at all": it
    survives any change of units or of machine and can go straight into a formula. `[a.u.]` says
    "there *is* a unit, we simply do not know it": self-consistent inside one dataset only. That is
    why `p1` is `[1]` although it comes from `a.u.` data — the projection divides by the
    centre-to-centre distance, and the unknown scale cancels.

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

```
single scan (t1, t2_echo, ramsey, rabi, s21, qspec)    multi-scan (s21 vs power, qspec vs Z, drag-*)
├── data    per-point rows                             ├── 0       per-point rows of scan 0
├── fits    one row                                    ├── 1       per-point rows of scan 1
└── states  calibration centres                        ├── …       …
                                                       ├── fits    one row per scan; row → sub-table
                                                       └── states  calibration centres
```

---

## 2. The six formats — and **where the column names live**

| | `.csv` | `.txt` | `.npz` | `.mat` | `.xlsx` | `.arrow.zip` |
|---|---|---|---|---|---|---|
| container | text, comma | text, tab | zip of `.npy` | MAT v5 | xlsx | zip of Feather (Arrow IPC file) |
| tables | **only the first** as real rows, the rest `# `-commented | same | **one structured array per table** | **one struct per table**, plus a `units` struct | one sheet per table | **one `.arrow` file per table** |
| **column names** | header row; each commented table repeats them on a `# columns:` line | same | **`z["fits"].dtype.names`** | **`fieldnames(m.fits)`** | row 1, `<column> [unit]` | **`schema$names`** / `field.name()` |
| **units** | on the `# columns:` line | same | **inside the field name: `'t1 [s]'`** | **`m.units.fits.t1`** | inside the header cell | **field metadata `unit`** (`f.metadata.get("unit")`) |
| complex | `re±imj` text | same | `complex128` | complex double | complex cell | **split: `<key>_re` + `<key>_im`** (Arrow has no complex type) |
| `ref` column | number | number | **`int64`** | `double` | number | **`Int64`** |
| **table names** | as-is (`# ---- 0 ----`) | same | **as-is (`"0"`)** | **`row_0`** — see below | as-is (`0`) | **as-is (`0.arrow`)** |
| missing | empty field | empty field | `NaN` | `NaN` | empty cell | `NaN` |

Because the names carry everything, the **`.npz` is self-describing — no side-car member**, the
**`.mat` needs no lookup either**, and the **`.arrow` schema is the column list** (names + units +
types travel in the IPC metadata). Three details:

* **Only `.mat` escapes the numeric table names.** MATLAB identifiers cannot start with a digit —
  a variable named `0` makes `load` reject the *whole file* (verified:
  `MATLAB:AddField:InvalidFieldName`), so the writer prefixes them: table `0` → `m.row_0`, and the
  same escape is applied inside `m.units`. Every other container keeps the bare index, so
  `[n for n in z.files if n.isdigit()]` gives the scan indices and `str(row_value)` is the key.
* **Field names come in pairs** (`.npz` only). Each npz field is written as
  `(('key', 'key [unit]'), '<dtype>')`: the *name* is the labelled form (`'t1 [s]'`, what
  `dtype.names` shows) and the *title* is the clean form. numpy indexes by either, so both
  `z["fits"]["t1"]` and `z["fits"]["t1 [s]"]` work. `.arrow` keeps the clean name and puts the
  unit in metadata instead — Arrow field names are plain strings, so nothing is mangled there.
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

Every block below was **run** on this machine against the generated files — **except MATLAB**: its
blocks are marked *not run* (no MATLAB here; the `row_`/`units` conventions come from an earlier
lab-machine run, and their numbers are pending a re-run there — §5). The `→` lines are copied from
what the commands actually printed.

One card produces six sibling files — `qspec-z.csv` `qspec-z.txt` `qspec-z.npz` `qspec-z.mat`
`qspec-z.xlsx` `qspec-z.arrow.zip` — and a report with two cards adds `qspec-z-window.*`
alongside them.

### 3.1 Python — `.npz` (numpy only)

```python
import numpy as np
z = np.load("qspec-z.npz", allow_pickle=False)

z.files                                       # the table list, in payload order
[int(n) for n in z.files if n.isdigit()]      # scan indices, nothing to parse
z["fits"].dtype.names                         # ('row [1]', 'z [V]', 'fq [Hz]', …)
z["fits"]["fq"][0]                            # clean field title
z["fits"]["fq [Hz]"][0]                       # the same value, labelled field name
z["fits"]["row"].dtype                        # dtype('int64')
for k in z["fits"]["row"]:                    # the ref value IS the key
    scan = z[str(int(k))]
```

To get a plain `{table: {column: array}}` dict:

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
```

### 3.2 MATLAB — `.mat`

```matlab
m = load("qspec-z.mat");

fieldnames(m)          % {'fits';'flux';'states';'row_0';…;'row_20';'units'}
fieldnames(m.fits)     % {'row';'z';'fq';'fq_stderr';…}

m.fits.fq(1)           % 5.036781e+09
m.fits.row'            % [0 1 2 … 20]     the ref column: its value IS the sub-table's index
m.row_0.iq(1)          % native complex
size(m.row_0.freq)     % 121  1
m.units.fits.fq        % 'Hz'
```

Only `.mat` spells the scan tables `row_<i>` (MATLAB identifiers cannot start with a digit — see
§2); `m.units` uses the same spelling, so `m.units.row_0.freq` is `'Hz'`.

### 3.3 Julia — `.mat` (MAT.jl)

```julia
using MAT
m = matread("qspec-z.mat")

length(m)                        # 25
m["fits"]["fq"][1]               # 5.036781280342331e9
m["fits"]["row"][1:3]            # [0.0, 1.0, 2.0]
m["row_0"]["iq"][1]              # ComplexF64
size(m["row_20"]["freq"])        # (121, 1)
m["units"]["fits"]["fq"]         # "Hz"
```

Into a DataFrame (the `vec` is required — MAT.jl hands back `n×1` matrices):

```julia
using DataFrames
DataFrame(Dict(k => vec(v) for (k, v) in m["row_0"]))     # 121×4
```

### 3.4 Rust — `.arrow.zip` (`arrow` + `zip`)

```rust
use arrow::ipc::reader::FileReader;
use std::{fs::File, io::{Cursor, Read}};
use zip::ZipArchive;

let mut archive = ZipArchive::new(File::open("qspec-z.arrow.zip")?)?;
for i in 0..archive.len() {
    let mut entry = archive.by_index(i)?;
    let name = entry.name().to_string();           // "0.arrow", "fits.arrow", …
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf)?;                  // ZipFile has no Seek; FileReader needs Read + Seek
    let reader = FileReader::try_new(Cursor::new(buf), None)?;
    for f in reader.schema().fields() {
        // fq: Float64 unit=Hz ; row: Int64 unit=1
        println!("{} {:?} unit={}", f.name(), f.data_type(),
                 f.metadata().get("unit").map(String::as_str).unwrap_or("-"));
    }
    for batch in reader { /* batch.column_by_name("fq") … */ }
}
```

Complex columns arrive as two columns (`iq_re`, `iq_im`) — recombine on the consumer side.

### 3.5 R — `.arrow.zip` (`arrow`)

```r
library(arrow)
d <- tempfile(); unzip("qspec-z.arrow.zip", exdir = d)

fits <- read_feather(file.path(d, "fits.arrow"))
names(fits)                       # "row" "z" "fq" "fq_stderr" …   exact, nothing mangled
fits$fq[1]                        # 5.036781e+09

r0 <- read_feather(file.path(d, "0.arrow"))
r0$iq <- r0$iq_re + 1i * r0$iq_im            # Arrow has no complex type — recombine

sch <- read_feather(file.path(d, "fits.arrow"), as_data_frame = FALSE)$schema
sch$GetFieldByName("fq")$metadata[["unit"]]  # "Hz"
```

### 3.6 Humans, and the two text formats

`.xlsx` opens by double-click: one sheet per table (sheet names as in the payload — `fits`,
`flux`, `0`…`20`), header row 1 as `<column> [unit]`, complex values as real complex cells,
missing values as empty cells. Verified with LibreOffice 25.2 headless as well as openpyxl.

`.csv` / `.txt` need no library in any language. They carry **the first table only** as real rows;
the remaining tables follow as `# `-commented blocks with their own `# columns:` header line, so
read them with the comment character declared: MATLAB `CommentStyle"#"`, Julia `comments=true`,
R `comment.char="#"`, Rust `csv::ReaderBuilder::comment`. Without that the 4-line preamble is parsed
as data.

### 3.7 AIs

If you are a language model asked to load, plot, fit or explain any of this data, this section is the
shortest correct path from a file to an answer. It is written so that nothing has to be assumed.

**You may not need the binary files at all.** Every report page carries the payload as JSON —
`<script type="application/json" id="<div_id>-data">…</script>` — and all six exporters are just
containers around that same JSON. With the HTML in hand, parse the tag (real output shown):

```python
import json, re

html = open("ramsey_demo.html").read()
payload = json.loads(re.search(r'id="[^"]+-data">(.*?)</script>', html, re.S).group(1))

payload["name"]                            # 'ramsey-0'
[t["name"] for t in payload["tables"]]     # ['data', 'fits', 'states']
payload["tables"][1]["columns"][0]         # {'key': 'offset', 'kind': 'real', 'unit': '1'}
payload["tables"][1]["data"]["frequency"]  # [149586.56405352696]  (Hz)
```

In this JSON a real value is a number, a complex value is a `[re, im]` pair, a missing value is
`null`, and each entry of `columns` carries `key`, `kind` (`real` / `complex` / `ref`) and `unit` —
nothing has to be guessed, and `kind` tells you which lists are complex before you touch them.

**With a file, read the container bound to your language** (the table at the top of this file). Do
not cross-read — the payload is identical in all six, so the bound container always works. For your
own scratch analysis, Python + `.npz` is the shortest path:

* **Python** — `z = np.load("<div_id>.npz", allow_pickle=False)`; tables are `z.files`; column names
  are `z[table].dtype.names`, each in the labelled form `'<key> [<unit>]'`; the bare key is the field
  *title*, so `z["fits"]["t1"]` and `z["fits"]["t1 [s]"]` are the same column. A `ref` column is
  `int64` and its **value is the sub-table name**: `z[str(int(row))]`. Iterate `dtype.names` —
  iterating `dtype.fields` lists titled fields twice (§6.3).
* **MATLAB / Julia** — the same `.mat`: scan sub-tables are spelled `row_<i>`, units sit in the
  parallel `units` struct (`m.units.fits.t1`), complex is native; in Julia the fields come back as
  `n×1` matrices — `vec` them.
* **Rust / R** — the same `.arrow.zip`: one Feather per table, complex split into `<key>_re` /
  `<key>_im`, units in the field metadata (`f.metadata.get("unit")`).
* **csv / txt, only when no library exists** — the first table as rows, the rest `#`-commented,
  complex as `re±imj`, and the comment character must be declared (§3.6) or the preamble parses as
  data.

Establish three facts from the data itself before computing anything — and never infer them from the
report's name, because the reports disagree on purpose (`s21` has 26 fit columns, `ramsey` has 10):
**which tables exist, what columns each has, what unit each column carries.** `shape` plus
`dtype.names` per table is one cheap call and kills the wrong-column class of bugs.

Then the rules that decide whether your answer is right:

* **Units are part of the value.** `[1]` means genuinely dimensionless — *not* "normalised to 0–1":
  `p1 [1]` is a probability, while `ql [1]` runs to thousands and `snr [1]` sits at a few. `[a.u.]` means
  the scale is unknown but self-consistent — such values are comparable **within one payload only**.
  Carry the unit into every answer: "T1 = 40.1 µs from `fits.t1 [s]`", not "T1 = 40.1".
* **Missing is `null` / `NaN` / an empty cell — never 0.** Count what you drop, and say so.
* **Complex stays complex** until the question is about magnitude: `iq` is the raw readout and
  `s21`'s `model` is complex too; in the text containers the trailing `j` is literal (`re±imj`).
* **`model` is the fitted curve evaluated at the axis points** (in `data` and in the scan
  sub-tables), so a residual is `p1 − model` — or `iq − model` where `model` is complex. `fits` is
  the parameter table; `<name>_stderr` is its 1σ. A multi-scan report has **no `data` table**:
  iterate `fits` and join `row` → sub-table (`.mat`: `row_<i>`).
* **A value slightly outside a "physical" range is usually not a bug.** `p1` is a linear projection:
  the shipped `t2_echo` payload has `p1 = −0.0388` at `τ = 0`. Check the `_stderr` column before
  declaring anything wrong, and never clamp or drop silently.

---

## 4. Every report: what is inside, and how each language reads it

<!-- section4:start -->
Each payload below comes with two things: a **table list** (what this data holds) and **one reader section per language** (each section is a code block; the output right below the code is a real run). `†` marks complex columns; `→ref` is a foreign key into the scan sub-tables (the value **is** that table's name).

Payloads: [t1 — T1 energy decay](#t1--t1-energy-decay) · [t2_echo — T2 echo decay](#t2_echo--t2-echo-decay) · [ramsey — T2\* (Ramsey)](#ramsey--t2-ramsey) · [rabi — Rabi amplitude](#rabi--rabi-amplitude) · [qspec — qubit spectroscopy](#qspec--qubit-spectroscopy) · [qspec vs Z — spectroscopy vs flux bias](#qspec-vs-z--spectroscopy-vs-flux-bias) · [qspec vs Z (the moving-window card)](#qspec-vs-z-the-moving-window-card) · [s21 — resonator spectroscopy](#s21--resonator-spectroscopy) · [s21 vs power — resonator vs pump power](#s21-vs-power--resonator-vs-pump-power) · [iq — readout discrimination](#iq--readout-discrimination) · [bloch — three-axis readout](#bloch--three-axis-readout) · [drag — amplitude scan](#drag--amplitude-scan) · [drag — coeff scan](#drag--coeff-scan) · [drag — detuning scan](#drag--detuning-scan)

### t1 — T1 energy decay

<details>
<summary><b>t1-0</b> — single scan: data point by point + one fits row + states</summary>

`div_id = t1-0` · six files: `t1-0.csv` `t1-0.txt` `t1-0.npz` `t1-0.mat` `t1-0.xlsx` `t1-0.arrow.zip` · single scan: `data` point by point + one `fits` row + `states`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `t1[s]`, `t1_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("t1-0.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits', 'states']
print(z["data"].shape)
# (101,)
print(z["data"].dtype.names)
# ('tau [s]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'p1 [1]', 'p1_sigma [1]', 'model [1]')
print(z["data"][:2])   # first 2 rows only
# [(0.e+00, 0.41454223+0.27910948j, 0.00466476, 0.91115743, 0.02, 0.95079663)
#  (2.e-06, 0.40966686+0.28323217j, 0.00466476, 0.91555988, 0.02, 0.90602442)]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('offset [1]', 'offset_stderr [1]', 'amplitude [1]', 'amplitude_stderr [1]', 't1 [s]', 't1_stderr [s]')
print(z["fits"])
# [(0.0302147, 0.00414604, 0.92058193, 0.0091513, 4.01146088e-05, 8.74451364e-07)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
```

#### MATLAB

```matlab
m = load("t1-0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)
fieldnames(m.states)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.tau

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("t1-0.mat")

keys(m)
# 4-element Vector{String}:
#  "data"
#  "fits"
#  "states"
#  "units"
m["data"]
#   iq = 101×1 ComplexF64, first 2 rows: 0.41454222994423734 + 0.27910948304282146im, 0.4096668609537991 + 0.283232170172207im
#   iq_sigma = 101×1 Float64, first 2 rows: 0.00466476151587624, 0.00466476151587624
#   model = 101×1 Float64, first 2 rows: 0.9507966304038435, 0.9060244218637487
#   p1 = 101×1 Float64, first 2 rows: 0.911157430181485, 0.9155598777370826
#   … the remaining 2 fields share one shape
m["fits"]
#   amplitude = 0.9205819306704167
#   amplitude_stderr = 0.009151297199603402
#   offset = 0.03021469973342679
#   offset_stderr = 0.004146035647370506
#   … the remaining 2 fields share one shape
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
m.units.data
# Dict{String, Any} with 6 entries:
#   "model" => "1"
#   "p1_sigma" => "1"
#   "p1" => "1"
#   "tau" => "s"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("t1-0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): tau:Float64 unit=s, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., p1:Float64 unit=1, p1_sigma:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 101 rows (only the first 4 columns; the full list is on the fields line above): tau=0 iq_re=0.41454222994423734 iq_im=0.27910948304282146 iq_sigma=0.00466476151587624
//   index 1 (fits.arrow)
//     schema.fields(): offset:Float64 unit=1, offset_stderr:Float64 unit=1, amplitude:Float64 unit=1, amplitude_stderr:Float64 unit=1, t1:Float64 unit=s, t1_stderr:Float64 unit=s
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): offset=0.03021469973342679 offset_stderr=0.004146035647370506 amplitude=0.9205819306704167 amplitude_stderr=0.009151297199603402
//   index 2 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("t1-0.arrow.zip", exdir = d)

entries
# [1] "data.arrow"   "fits.arrow"   "states.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 101   7
names(data)
# [1] "tau"      "iq_re"    "iq_im"    "iq_sigma" "p1"       "p1_sigma" "model"   
head(data, 2)   # first 2 rows only
#     tau     iq_re     iq_im    iq_sigma        p1 p1_sigma     model
# 1 0e+00 0.4145422 0.2791095 0.004664762 0.9111574     0.02 0.9507966
# 2 2e-06 0.4096669 0.2832322 0.004664762 0.9155599     0.02 0.9060244
sch$GetFieldByName("tau")$metadata
# [1] "s"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 1 6
names(fits)
# [1] "offset"           "offset_stderr"    "amplitude"        "amplitude_stderr"
# [5] "t1"               "t1_stderr"       
fits
#      offset offset_stderr amplitude amplitude_stderr           t1    t1_stderr
# 1 0.0302147   0.004146036 0.9205819      0.009151297 4.011461e-05 8.744514e-07
sch$GetFieldByName("offset")$metadata
# [1] "1"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### t2_echo — T2 echo decay

<details>
<summary><b>t2echo-0</b> — single scan, same shape as t1</summary>

`div_id = t2echo-0` · six files: `t2echo-0.csv` `t2echo-0.txt` `t2echo-0.npz` `t2echo-0.mat` `t2echo-0.xlsx` `t2echo-0.arrow.zip` · single scan, same shape as t1

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `t2_echo[s]`, `t2_echo_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("t2echo-0.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits', 'states']
print(z["data"].shape)
# (101,)
print(z["data"].dtype.names)
# ('tau [s]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'p1 [1]', 'p1_sigma [1]', 'model [1]')
print(z["data"][:2])   # first 2 rows only
# [(0.e+00, 0.30054223+0.08910948j, 0.00466476, -0.03884257, 0.02, -0.00862685)
#  (2.e-06, 0.30566415+0.10989432j, 0.00466476,  0.04887063, 0.02,  0.0322289 )]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('offset [1]', 'offset_stderr [1]', 'amplitude [1]', 'amplitude_stderr [1]', 't2_echo [s]', 't2_echo_stderr [s]')
print(z["fits"])
# [(0.49886529, 0.00290749, -0.50749213, 0.01136654, 2.38291327e-05, 9.44049184e-07)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
```

#### MATLAB

```matlab
m = load("t2echo-0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)
fieldnames(m.states)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.tau

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("t2echo-0.mat")

keys(m)
# 4-element Vector{String}:
#  "data"
#  "fits"
#  "states"
#  "units"
m["data"]
#   iq = 101×1 ComplexF64, first 2 rows: 0.30054222994423735 + 0.08910948304282142im, 0.30566415170572214 + 0.10989432142541201im
#   iq_sigma = 101×1 Float64, first 2 rows: 0.00466476151587624, 0.00466476151587624
#   model = 101×1 Float64, first 2 rows: -0.008626847927922454, 0.032228901507516616
#   p1 = 101×1 Float64, first 2 rows: -0.038842569818515346, 0.04887063400310773
#   … the remaining 2 fields share one shape
m["fits"]
#   amplitude = -0.5074921337054582
#   amplitude_stderr = 0.011366539690489537
#   offset = 0.49886528577753575
#   offset_stderr = 0.002907488861022497
#   … the remaining 2 fields share one shape
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
m.units.data
# Dict{String, Any} with 6 entries:
#   "model" => "1"
#   "p1_sigma" => "1"
#   "p1" => "1"
#   "tau" => "s"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("t2echo-0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): tau:Float64 unit=s, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., p1:Float64 unit=1, p1_sigma:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 101 rows (only the first 4 columns; the full list is on the fields line above): tau=0 iq_re=0.30054222994423735 iq_im=0.08910948304282142 iq_sigma=0.00466476151587624
//   index 1 (fits.arrow)
//     schema.fields(): offset:Float64 unit=1, offset_stderr:Float64 unit=1, amplitude:Float64 unit=1, amplitude_stderr:Float64 unit=1, t2_echo:Float64 unit=s, t2_echo_stderr:Float64 unit=s
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): offset=0.49886528577753575 offset_stderr=0.002907488861022497 amplitude=-0.5074921337054582 amplitude_stderr=0.011366539690489537
//   index 2 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("t2echo-0.arrow.zip", exdir = d)

entries
# [1] "data.arrow"   "fits.arrow"   "states.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 101   7
names(data)
# [1] "tau"      "iq_re"    "iq_im"    "iq_sigma" "p1"       "p1_sigma" "model"   
head(data, 2)   # first 2 rows only
#     tau     iq_re      iq_im    iq_sigma          p1 p1_sigma        model
# 1 0e+00 0.3005422 0.08910948 0.004664762 -0.03884257     0.02 -0.008626848
# 2 2e-06 0.3056642 0.10989432 0.004664762  0.04887063     0.02  0.032228902
sch$GetFieldByName("tau")$metadata
# [1] "s"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 1 6
names(fits)
# [1] "offset"           "offset_stderr"    "amplitude"        "amplitude_stderr"
# [5] "t2_echo"          "t2_echo_stderr"  
fits
#      offset offset_stderr  amplitude amplitude_stderr      t2_echo
# 1 0.4988653   0.002907489 -0.5074921       0.01136654 2.382913e-05
#   t2_echo_stderr
# 1   9.440492e-07
sch$GetFieldByName("offset")$metadata
# [1] "1"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### ramsey — T2\* (Ramsey)

<details>
<summary><b>ramsey-0</b> — single scan, five fit parameters</summary>

`div_id = ramsey-0` · six files: `ramsey-0.csv` `ramsey-0.txt` `ramsey-0.npz` `ramsey-0.mat` `ramsey-0.xlsx` `ramsey-0.arrow.zip` · single scan, five fit parameters

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 101 × 6 | `tau[s]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 10 | `offset[1]`, `offset_stderr[1]`, `amplitude[1]`, `amplitude_stderr[1]`, `frequency[Hz]`, `frequency_stderr[Hz]`, `phase[rad]`, `phase_stderr[rad]`, `decay[s]`, `decay_stderr[s]` |
| `states` | 2 × 1 | `center[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("ramsey-0.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits', 'states']
print(z["data"].shape)
# (101,)
print(z["data"].dtype.names)
# ('tau [s]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'p1 [1]', 'p1_sigma [1]', 'model [1]')
print(z["data"][:2])   # first 2 rows only
# [(0.e+00, 0.39725465+0.25029686j, 0.00466476, 0.76709431, 0.02, 0.80980063)
#  (6.e-07, 0.37505296+0.22554234j, 0.00466476, 0.62711072, 0.02, 0.61013103)]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('offset [1]', 'offset_stderr [1]', 'amplitude [1]', 'amplitude_stderr [1]', 'frequency [Hz]', 'frequency_stderr [Hz]', 'phase [rad]', 'phase_stderr [rad]', 'decay [s]', 'decay_stderr [s]')
print(z["fits"])
# [(0.50120538, 0.00212262, -0.41514382, 0.01095367, 149586.56405353, 302.5657429, -2.40885419, 0.024307, 1.92000678e-05, 7.27235163e-07)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
```

#### MATLAB

```matlab
m = load("ramsey-0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)
fieldnames(m.states)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.tau

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("ramsey-0.mat")

keys(m)
# 4-element Vector{String}:
#  "data"
#  "fits"
#  "states"
#  "units"
m["data"]
#   iq = 101×1 ComplexF64, first 2 rows: 0.3972546549338928 + 0.2502968580255805im, 0.3750529621887545 + 0.22554233889713254im
#   iq_sigma = 101×1 Float64, first 2 rows: 0.00466476151587624, 0.00466476151587624
#   model = 101×1 Float64, first 2 rows: 0.8098006317613072, 0.6101310265765489
#   p1 = 101×1 Float64, first 2 rows: 0.7670943050952801, 0.6271107213617105
#   … the remaining 2 fields share one shape
m["fits"]
#   amplitude = -0.41514381857820326
#   amplitude_stderr = 0.010953673355630397
#   decay = 1.9200067836289136e-5
#   decay_stderr = 7.272351634106979e-7
#   … the remaining 6 fields share one shape
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
m.units.data
# Dict{String, Any} with 6 entries:
#   "model" => "1"
#   "p1_sigma" => "1"
#   "p1" => "1"
#   "tau" => "s"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("ramsey-0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): tau:Float64 unit=s, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., p1:Float64 unit=1, p1_sigma:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 101 rows (only the first 4 columns; the full list is on the fields line above): tau=0 iq_re=0.3972546549338928 iq_im=0.2502968580255805 iq_sigma=0.00466476151587624
//   index 1 (fits.arrow)
//     schema.fields(): offset:Float64 unit=1, offset_stderr:Float64 unit=1, amplitude:Float64 unit=1, amplitude_stderr:Float64 unit=1, frequency:Float64 unit=Hz, frequency_stderr:Float64 unit=Hz, phase:Float64 unit=rad, phase_stderr:Float64 unit=rad, decay:Float64 unit=s, decay_stderr:Float64 unit=s
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): offset=0.5012053789269696 offset_stderr=0.002122621688876113 amplitude=-0.41514381857820326 amplitude_stderr=0.010953673355630397
//   index 2 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("ramsey-0.arrow.zip", exdir = d)

entries
# [1] "data.arrow"   "fits.arrow"   "states.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 101   7
names(data)
# [1] "tau"      "iq_re"    "iq_im"    "iq_sigma" "p1"       "p1_sigma" "model"   
head(data, 2)   # first 2 rows only
#     tau     iq_re     iq_im    iq_sigma        p1 p1_sigma     model
# 1 0e+00 0.3972547 0.2502969 0.004664762 0.7670943     0.02 0.8098006
# 2 6e-07 0.3750530 0.2255423 0.004664762 0.6271107     0.02 0.6101310
sch$GetFieldByName("tau")$metadata
# [1] "s"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1]  1 10
names(fits)
#  [1] "offset"           "offset_stderr"    "amplitude"        "amplitude_stderr"
#  [5] "frequency"        "frequency_stderr" "phase"            "phase_stderr"    
#  [9] "decay"            "decay_stderr"    
fits
#      offset offset_stderr  amplitude amplitude_stderr frequency
# 1 0.5012054   0.002122622 -0.4151438       0.01095367  149586.6
#   frequency_stderr     phase phase_stderr        decay decay_stderr
# 1         302.5657 -2.408854     0.024307 1.920007e-05 7.272352e-07
sch$GetFieldByName("offset")$metadata
# [1] "1"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### rabi — Rabi amplitude

<details>
<summary><b>rabi-0</b> — single scan, the π amplitude is fits.a_pi</summary>

`div_id = rabi-0` · six files: `rabi-0.csv` `rabi-0.txt` `rabi-0.npz` `rabi-0.mat` `rabi-0.xlsx` `rabi-0.arrow.zip` · single scan, the π amplitude is `fits.a_pi`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 41 × 6 | `amp[1]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 6 | `freq[1]`, `freq_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `a_pi[1]`, `a_pi_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("rabi-0.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits', 'states']
print(z["data"].shape)
# (41,)
print(z["data"].dtype.names)
# ('amp [1]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'p1 [1]', 'p1_sigma [1]', 'model [1]')
print(z["data"][:2])   # first 2 rows only
# [(0.   , 0.30054223+0.08910948j, 0.00466476, -0.03884257, 0.02, 0.        )
#  (0.025, 0.31251011+0.12130426j, 0.00466476,  0.10592031, 0.02, 0.09578223)]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('freq [1]', 'freq_stderr [1]', 'amp [1]', 'amp_stderr [1]', 'a_pi [1]', 'a_pi_stderr [1]')
print(z["fits"])
# [(4.00113938, 0.00264827, 0.50124611, 0.00277621, 0.1249644, 8.271124e-05)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
```

#### MATLAB

```matlab
m = load("rabi-0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)
fieldnames(m.states)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.amp

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("rabi-0.mat")

keys(m)
# 4-element Vector{String}:
#  "data"
#  "fits"
#  "states"
#  "units"
m["data"]
#   amp = 41×1 Float64, first 2 rows: 0.0, 0.025
#   iq = 41×1 ComplexF64, first 2 rows: 0.30054222994423735 + 0.08910948304282142im, 0.31251011282642344 + 0.12130425662658084im
#   iq_sigma = 41×1 Float64, first 2 rows: 0.00466476151587624, 0.00466476151587624
#   model = 41×1 Float64, first 2 rows: 0.0, 0.09578222574382965
#   … the remaining 2 fields share one shape
m["fits"]
#   a_pi = 0.12496440457402602
#   a_pi_stderr = 8.271123996251817e-5
#   amp = 0.5012461133709237
#   amp_stderr = 0.0027762072788375435
#   … the remaining 2 fields share one shape
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
m.units.data
# Dict{String, Any} with 6 entries:
#   "model" => "1"
#   "amp" => "1"
#   "p1" => "1"
#   "p1_sigma" => "1"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("rabi-0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): amp:Float64 unit=1, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., p1:Float64 unit=1, p1_sigma:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 41 rows (only the first 4 columns; the full list is on the fields line above): amp=0 iq_re=0.30054222994423735 iq_im=0.08910948304282142 iq_sigma=0.00466476151587624
//   index 1 (fits.arrow)
//     schema.fields(): freq:Float64 unit=1, freq_stderr:Float64 unit=1, amp:Float64 unit=1, amp_stderr:Float64 unit=1, a_pi:Float64 unit=1, a_pi_stderr:Float64 unit=1
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): freq=4.001139378084353 freq_stderr=0.0026482677234265267 amp=0.5012461133709237 amp_stderr=0.0027762072788375435
//   index 2 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("rabi-0.arrow.zip", exdir = d)

entries
# [1] "data.arrow"   "fits.arrow"   "states.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 41  7
names(data)
# [1] "amp"      "iq_re"    "iq_im"    "iq_sigma" "p1"       "p1_sigma" "model"   
head(data, 2)   # first 2 rows only
#     amp     iq_re      iq_im    iq_sigma          p1 p1_sigma      model
# 1 0.000 0.3005422 0.08910948 0.004664762 -0.03884257     0.02 0.00000000
# 2 0.025 0.3125101 0.12130426 0.004664762  0.10592031     0.02 0.09578223
sch$GetFieldByName("amp")$metadata
# [1] "1"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 1 6
names(fits)
# [1] "freq"        "freq_stderr" "amp"         "amp_stderr"  "a_pi"       
# [6] "a_pi_stderr"
fits
#       freq freq_stderr       amp  amp_stderr      a_pi  a_pi_stderr
# 1 4.001139 0.002648268 0.5012461 0.002776207 0.1249644 8.271124e-05
sch$GetFieldByName("freq")$metadata
# [1] "1"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### qspec — qubit spectroscopy

<details>
<summary><b>qspec-0</b> — a single spectrum</summary>

`div_id = qspec-0` · six files: `qspec-0.csv` `qspec-0.txt` `qspec-0.npz` `qspec-0.mat` `qspec-0.xlsx` `qspec-0.arrow.zip` · a single spectrum

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 51 × 6 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `p1[1]`, `p1_sigma[1]`, `model[1]` |
| `fits` | 1 × 8 | `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("qspec-0.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits', 'states']
print(z["data"].shape)
# (51,)
print(z["data"].dtype.names)
# ('freq [Hz]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'p1 [1]', 'p1_sigma [1]', 'model [1]')
print(z["data"][:2])   # first 2 rows only
# [(4.99700e+09, 0.32309395+0.12669569j, nan, 0.14908846, nan, 0.19088996)
#  (4.99712e+09, 0.32480261+0.14179175j, nan, 0.2083578 , nan, 0.20095667)]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('fq [Hz]', 'fq_stderr [Hz]', 'fwhm [Hz]', 'fwhm_stderr [Hz]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"])
# [(4.99998864e+09, 9459.33430477, 2397774.79372069, 59461.29044985, 0.99877391, 0.01184426, 0.05244545, 0.01215384)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
```

#### MATLAB

```matlab
m = load("qspec-0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)
fieldnames(m.states)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.freq

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("qspec-0.mat")

keys(m)
# 4-element Vector{String}:
#  "data"
#  "fits"
#  "states"
#  "units"
m["data"]
#   freq = 51×1 Float64, first 2 rows: 4.997e9, 4.99712e9
#   iq = 51×1 ComplexF64, first 2 rows: 0.3230939540821684 + 0.12669568993937314im, 0.3248026117788611 + 0.14179175488064363im
#   iq_sigma = 51×1 Float64, first 2 rows: NaN, NaN
#   model = 51×1 Float64, first 2 rows: 0.1908899630956913, 0.20095667263576578
#   … the remaining 2 fields share one shape
m["fits"]
#   amp = 0.9987739070704332
#   amp_stderr = 0.011844261117054607
#   fq = 4.999988635218431e9
#   fq_stderr = 9459.334304770151
#   … the remaining 4 fields share one shape
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
m.units.data
# Dict{String, Any} with 6 entries:
#   "model" => "1"
#   "p1_sigma" => "1"
#   "p1" => "1"
#   "freq" => "Hz"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("qspec-0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): freq:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., p1:Float64 unit=1, p1_sigma:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 51 rows (only the first 4 columns; the full list is on the fields line above): freq=4997000000 iq_re=0.3230939540821684 iq_im=0.12669568993937314 iq_sigma=NaN
//   index 1 (fits.arrow)
//     schema.fields(): fq:Float64 unit=Hz, fq_stderr:Float64 unit=Hz, fwhm:Float64 unit=Hz, fwhm_stderr:Float64 unit=Hz, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): fq=4999988635.218431 fq_stderr=9459.334304770151 fwhm=2397774.7937206915 fwhm_stderr=59461.290449854794
//   index 2 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("qspec-0.arrow.zip", exdir = d)

entries
# [1] "data.arrow"   "fits.arrow"   "states.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 51  7
names(data)
# [1] "freq"     "iq_re"    "iq_im"    "iq_sigma" "p1"       "p1_sigma" "model"   
head(data, 2)   # first 2 rows only
#         freq     iq_re     iq_im iq_sigma        p1 p1_sigma     model
# 1 4997000000 0.3230940 0.1266957      NaN 0.1490885      NaN 0.1908900
# 2 4997120000 0.3248026 0.1417918      NaN 0.2083578      NaN 0.2009567
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 1 8
names(fits)
# [1] "fq"            "fq_stderr"     "fwhm"          "fwhm_stderr"  
# [5] "amp"           "amp_stderr"    "offset"        "offset_stderr"
fits
#           fq fq_stderr    fwhm fwhm_stderr       amp amp_stderr     offset
# 1 4999988635  9459.334 2397775    59461.29 0.9987739 0.01184426 0.05244545
#   offset_stderr
# 1    0.01215384
sch$GetFieldByName("fq")$metadata
# [1] "Hz"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### qspec vs Z — spectroscopy vs flux bias

<details>
<summary><b>qspec-z</b> — multi-row: one spectrum per Z (fits 21 rows + 0…20 sub-tables + flux)</summary>

`div_id = qspec-z` · six files: `qspec-z.csv` `qspec-z.txt` `qspec-z.npz` `qspec-z.mat` `qspec-z.xlsx` `qspec-z.arrow.zip` · multi-row: one spectrum per Z (`fits` 21 rows + `0`…`20` sub-tables + `flux`)

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 10 | `row[1]`→ref, `z[V]`, `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `flux` | 1 × 10 | `f_max[Hz]`, `f_max_stderr[Hz]`, `z_offset[V]`, `z_offset_stderr[V]`, `z_period[V]`, `z_period_stderr[V]`, `eta[Hz]`, `eta_stderr[Hz]`, `asymmetry[1]`, `asymmetry_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |
| `0` … `20` *(one scan per row, same shape)* | 121 × 4 | `freq[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

#### Python

```python
import numpy as np
z = np.load("qspec-z.npz", allow_pickle=False)

print(z.files)
# ['fits', 'flux', 'states', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
print(z["fits"].shape)
# (21,)
print(z["fits"].dtype.names)
# ('row [1]', 'z [V]', 'fq [Hz]', 'fq_stderr [Hz]', 'fwhm [Hz]', 'fwhm_stderr [Hz]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"][:2])   # first 2 rows only
# [(0, -0.02 , 5.03678128e+09, 65661.1990869 , 2065861.50040653, 201918.74431375, 1.04699388, 0.06657645, 0.07272972, 0.01113011)
#  (1, -0.018, 5.03926453e+09, 66908.44716862, 2518168.40343208, 209870.62276364, 0.9738074 , 0.05177741, 0.04186813, 0.00992407)]
print(z["flux"].shape)
# (1,)
print(z["flux"].dtype.names)   # column names first, then the values
# ('f_max [Hz]', 'f_max_stderr [Hz]', 'z_offset [V]', 'z_offset_stderr [V]', 'z_period [V]', 'z_period_stderr [V]', 'eta [Hz]', 'eta_stderr [Hz]', 'asymmetry [1]', 'asymmetry_stderr [1]')
print(z["flux"])
# [(5.05003222e+09, 24375.46514899, 2.09505037e-05, 2.02924106e-05, -0.59792977, 0.00112207, -2.2e+08, nan, 0.1, nan)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
print(z["0"].shape)
# (121,)
print(z["0"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["0"][0])   # the row-0 value
# (5020000000.0, 0.30931418+0.05655247j, -0.13918756499140664, 0.07668151056674619)
print(z["1"].shape)
# (121,)
print(z["1"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["1"][0])   # the row-0 value
# (5020000000.0, 0.30453553+0.12445971j, 0.09993024091777251, 0.04601017774095219)
# the remaining 19 sub-tables (z["2"] … z["20"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("qspec-z.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.flux)
fieldnames(m.states)
fieldnames(m.row_0)

% size and a value:
size(m.fits.z)
m.fits.z(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.z

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("qspec-z.mat")

keys(m)
# 25-element Vector{String}:
#  "fits"
#  "flux"
#  "row_0"
#  "row_1"
#  "row_10"
#  "row_11"
#  "row_12"
#  "row_13"
#  "row_14"
#  "row_15"
#  "row_16"
#  "row_17"
#  "row_18"
#  "row_19"
#  "row_2"
#  "row_20"
#  "row_3"
#  "row_4"
#  "row_5"
#  "row_6"
#  "row_7"
#  "row_8"
#  "row_9"
#  "states"
#  "units"
m["fits"]
#   amp = 21×1 Float64, first 2 rows: 1.0469938798094323, 0.973807399437088
#   amp_stderr = 21×1 Float64, first 2 rows: 0.0665764498447261, 0.05177741014456852
#   fq = 21×1 Float64, first 2 rows: 5.036781280342331e9, 5.039264525219668e9
#   fq_stderr = 21×1 Float64, first 2 rows: 65661.19908690403, 66908.44716861677
#   … the remaining 6 fields share one shape
m["flux"]
#   asymmetry = 0.1
#   asymmetry_stderr = NaN
#   eta = -2.2e8
#   eta_stderr = NaN
#   … the remaining 6 fields share one shape
m["row_0"]
#   freq = 121×1 Float64, first 2 rows: 5.02e9, 5.020333333333333e9
#   iq = 121×1 ComplexF64, first 2 rows: 0.30931418381332726 + 0.05655247203434104im, 0.31188310145268344 + 0.1220755120005147im
#   model = 121×1 Float64, first 2 rows: 0.07668151056674619, 0.07684267167109579
#   p1 = 121×1 Float64, first 2 rows: -0.13918756499140664, 0.10737269438281163
m["row_1"]
#   freq = 121×1 Float64, first 2 rows: 5.02e9, 5.020333333333333e9
#   iq = 121×1 ComplexF64, first 2 rows: 0.30453553188681814 + 0.12445970639754324im, 0.34399742402853506 + 0.10252515485325413im
#   model = 121×1 Float64, first 2 rows: 0.04601017774095219, 0.04615667702278639
#   p1 = 121×1 Float64, first 2 rows: 0.09993024091777251, 0.106336798788144
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
# the remaining 19 sub-tables (row_2 … row_20) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 10 entries:
#   "fq_stderr" => "Hz"
#   "offset" => "1"
#   "row" => "1"
#   "amp" => "1"
#   "offset_stderr" => "1"
#   "fwhm" => "Hz"
#   "z" => "V"
#   "amp_stderr" => "1"
#   "fq" => "Hz"
#   "fwhm_stderr" => "Hz"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("qspec-z.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "3.arrow", "4.arrow", "5.arrow", "6.arrow", "7.arrow", "8.arrow", "9.arrow", "10.arrow", "11.arrow", "12.arrow", "13.arrow", "14.arrow", "15.arrow", "16.arrow", "17.arrow", "18.arrow", "19.arrow", "20.arrow", "fits.arrow", "flux.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): freq:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., p1:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 121 rows (only the first 4 columns; the full list is on the fields line above): freq=5020000000 iq_re=0.30931418381332726 iq_im=0.05655247203434104 p1=-0.13918756499140664
//   index 21 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, z:Float64 unit=V, fq:Float64 unit=Hz, fq_stderr:Float64 unit=Hz, fwhm:Float64 unit=Hz, fwhm_stderr:Float64 unit=Hz, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 21 rows (only the first 4 columns; the full list is on the fields line above): row=0 z=-0.02 fq=5036781280.342331 fq_stderr=65661.19908690403
//   index 22 (flux.arrow)
//     schema.fields(): f_max:Float64 unit=Hz, f_max_stderr:Float64 unit=Hz, z_offset:Float64 unit=V, z_offset_stderr:Float64 unit=V, z_period:Float64 unit=V, z_period_stderr:Float64 unit=V, eta:Float64 unit=Hz, eta_stderr:Float64 unit=Hz, asymmetry:Float64 unit=1, asymmetry_stderr:Float64 unit=1
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): f_max=5050032216.083149 f_max_stderr=24375.465148985117 z_offset=0.00002095050373312709 z_offset_stderr=0.000020292410571805908
//   index 23 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
//   index …: the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("qspec-z.arrow.zip", exdir = d)

entries
#  [1] "0.arrow"      "1.arrow"      "10.arrow"     "11.arrow"     "12.arrow"    
#  [6] "13.arrow"     "14.arrow"     "15.arrow"     "16.arrow"     "17.arrow"    
# [11] "18.arrow"     "19.arrow"     "2.arrow"      "20.arrow"     "3.arrow"     
# [16] "4.arrow"      "5.arrow"      "6.arrow"      "7.arrow"      "8.arrow"     
# [21] "9.arrow"      "fits.arrow"   "flux.arrow"   "states.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 121   5
names(r0)
# [1] "freq"  "iq_re" "iq_im" "p1"    "model"
head(r0, 2)   # first 2 rows only
#         freq     iq_re      iq_im         p1      model
# 1 5020000000 0.3093142 0.05655247 -0.1391876 0.07668151
# 2 5020333333 0.3118831 0.12207551  0.1073727 0.07684267
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 121   5
names(r1)
# [1] "freq"  "iq_re" "iq_im" "p1"    "model"
head(r1, 2)   # first 2 rows only
#         freq     iq_re     iq_im         p1      model
# 1 5020000000 0.3045355 0.1244597 0.09993024 0.04601018
# 2 5020333333 0.3439974 0.1025252 0.10633680 0.04615668
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 21 10
names(fits)
#  [1] "row"           "z"             "fq"            "fq_stderr"    
#  [5] "fwhm"          "fwhm_stderr"   "amp"           "amp_stderr"   
#  [9] "offset"        "offset_stderr"
head(fits, 2)   # first 2 rows only
#   row      z         fq fq_stderr    fwhm fwhm_stderr       amp amp_stderr
# 1   0 -0.020 5036781280  65661.20 2065862    201918.7 1.0469939 0.06657645
# 2   1 -0.018 5039264525  66908.45 2518168    209870.6 0.9738074 0.05177741
#       offset offset_stderr
# 1 0.07272972   0.011130111
# 2 0.04186813   0.009924069
sch$GetFieldByName("row")$metadata
# [1] "1"
flux <- read_feather(file.path(d, "flux.arrow"))
dim(flux)   # rows × cols
# [1]  1 10
names(flux)
#  [1] "f_max"            "f_max_stderr"     "z_offset"         "z_offset_stderr" 
#  [5] "z_period"         "z_period_stderr"  "eta"              "eta_stderr"      
#  [9] "asymmetry"        "asymmetry_stderr"
flux
#        f_max f_max_stderr    z_offset z_offset_stderr   z_period
# 1 5050032216     24375.47 2.09505e-05    2.029241e-05 -0.5979298
#   z_period_stderr      eta eta_stderr asymmetry asymmetry_stderr
# 1      0.00112207 -2.2e+08        NaN       0.1              NaN
sch$GetFieldByName("f_max")$metadata
# [1] "Hz"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
# the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### qspec vs Z (the moving-window card)

<details>
<summary><b>qspec-z-window</b> — same layout; each row carries its own frequency axis; sub-tables have 31 points</summary>

`div_id = qspec-z-window` · six files: `qspec-z-window.csv` `qspec-z-window.txt` `qspec-z-window.npz` `qspec-z-window.mat` `qspec-z-window.xlsx` `qspec-z-window.arrow.zip` · same layout; each row carries its own frequency axis; sub-tables have 31 points

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 10 | `row[1]`→ref, `z[V]`, `fq[Hz]`, `fq_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `flux` | 1 × 10 | `f_max[Hz]`, `f_max_stderr[Hz]`, `z_offset[V]`, `z_offset_stderr[V]`, `z_period[V]`, `z_period_stderr[V]`, `eta[Hz]`, `eta_stderr[Hz]`, `asymmetry[1]`, `asymmetry_stderr[1]` |
| `states` | 2 × 1 | `center[a.u.]`† |
| `0` … `20` *(one scan per row, same shape)* | 31 × 4 | `freq[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

#### Python

```python
import numpy as np
z = np.load("qspec-z-window.npz", allow_pickle=False)

print(z.files)
# ['fits', 'flux', 'states', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
print(z["fits"].shape)
# (21,)
print(z["fits"].dtype.names)
# ('row [1]', 'z [V]', 'fq [Hz]', 'fq_stderr [Hz]', 'fwhm [Hz]', 'fwhm_stderr [Hz]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"][:2])   # first 2 rows only
# [(0, -0.02 , 5.03693509e+09, 188046.32004741, 3467890.98116615, 631064.9765653, 0.8198734, 0.08800862,  0.0191028 , 0.02672145)
#  (1, -0.018, 5.03954824e+09, 198459.77909455, 3271065.84840291, 661585.0209221, 0.9830316, 0.11830125, -0.00896084, 0.03413902)]
print(z["flux"].shape)
# (1,)
print(z["flux"].dtype.names)   # column names first, then the values
# ('f_max [Hz]', 'f_max_stderr [Hz]', 'z_offset [V]', 'z_offset_stderr [V]', 'z_period [V]', 'z_period_stderr [V]', 'eta [Hz]', 'eta_stderr [Hz]', 'asymmetry [1]', 'asymmetry_stderr [1]')
print(z["flux"])
# [(5.05004262e+09, 42536.36035585, -5.87601084e-05, 3.55212328e-05, 0.59890901, 0.00196772, -2.2e+08, nan, 0.1, nan)]
print(z["states"].shape)
# (2,)
print(z["states"].dtype.names)   # column names first, then the values
# ('center [a.u.]',)
print(z["states"])
# [(0.3 +0.1j,) (0.42+0.3j,)]
print(z["0"].shape)
# (31,)
print(z["0"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["0"][0])   # the row-0 value
# (5021885229.145344, 0.30454818+0.10819558j, 0.04016353984321623, 0.029843326594350864)
print(z["1"].shape)
# (31,)
print(z["1"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["1"][0])   # the row-0 value
# (5024377879.496459, 0.30452945+0.08535874j, -0.043836731992955326, 0.002333876799366563)
# the remaining 19 sub-tables (z["2"] … z["20"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("qspec-z-window.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.flux)
fieldnames(m.states)
fieldnames(m.row_0)

% size and a value:
size(m.fits.z)
m.fits.z(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.z

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("qspec-z-window.mat")

keys(m)
# 25-element Vector{String}:
#  "fits"
#  "flux"
#  "row_0"
#  "row_1"
#  "row_10"
#  "row_11"
#  "row_12"
#  "row_13"
#  "row_14"
#  "row_15"
#  "row_16"
#  "row_17"
#  "row_18"
#  "row_19"
#  "row_2"
#  "row_20"
#  "row_3"
#  "row_4"
#  "row_5"
#  "row_6"
#  "row_7"
#  "row_8"
#  "row_9"
#  "states"
#  "units"
m["fits"]
#   amp = 21×1 Float64, first 2 rows: 0.8198734027409226, 0.9830315953100197
#   amp_stderr = 21×1 Float64, first 2 rows: 0.08800861994617014, 0.11830125141032777
#   fq = 21×1 Float64, first 2 rows: 5.036935093821951e9, 5.039548242475719e9
#   fq_stderr = 21×1 Float64, first 2 rows: 188046.32004740788, 198459.77909454986
#   … the remaining 6 fields share one shape
m["flux"]
#   asymmetry = 0.1
#   asymmetry_stderr = NaN
#   eta = -2.2e8
#   eta_stderr = NaN
#   … the remaining 6 fields share one shape
m["row_0"]
#   freq = 31×1 Float64, first 2 rows: 5.021885229145344e9, 5.022885229145344e9
#   iq = 31×1 ComplexF64, first 2 rows: 0.3045481788285533 + 0.10819557554022284im, 0.3000654436309281 + 0.08766729536686593im
#   model = 31×1 Float64, first 2 rows: 0.029843326594350864, 0.0314028983572344
#   p1 = 31×1 Float64, first 2 rows: 0.04016353984321623, -0.0451964649065339
m["row_1"]
#   freq = 31×1 Float64, first 2 rows: 5.024377879496459e9, 5.025377879496459e9
#   iq = 31×1 ComplexF64, first 2 rows: 0.30452944937191256 + 0.08535873927476861im, 0.31297394805566836 + 0.09684007739764243im
#   model = 31×1 Float64, first 2 rows: 0.002333876799366563, 0.003962561155226181
#   p1 = 31×1 Float64, first 2 rows: -0.043836731992955326, 0.017001640555306795
m["states"]
#   center = 2×1 Matrix{ComplexF64}:
#   0.3 + 0.1im
#  0.42 + 0.3im
# the remaining 19 sub-tables (row_2 … row_20) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 10 entries:
#   "fq_stderr" => "Hz"
#   "offset" => "1"
#   "row" => "1"
#   "amp" => "1"
#   "offset_stderr" => "1"
#   "fwhm" => "Hz"
#   "z" => "V"
#   "amp_stderr" => "1"
#   "fq" => "Hz"
#   "fwhm_stderr" => "Hz"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("qspec-z-window.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "3.arrow", "4.arrow", "5.arrow", "6.arrow", "7.arrow", "8.arrow", "9.arrow", "10.arrow", "11.arrow", "12.arrow", "13.arrow", "14.arrow", "15.arrow", "16.arrow", "17.arrow", "18.arrow", "19.arrow", "20.arrow", "fits.arrow", "flux.arrow", "states.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): freq:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., p1:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 31 rows (only the first 4 columns; the full list is on the fields line above): freq=5021885229.145344 iq_re=0.3045481788285533 iq_im=0.10819557554022284 p1=0.04016353984321623
//   index 21 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, z:Float64 unit=V, fq:Float64 unit=Hz, fq_stderr:Float64 unit=Hz, fwhm:Float64 unit=Hz, fwhm_stderr:Float64 unit=Hz, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 21 rows (only the first 4 columns; the full list is on the fields line above): row=0 z=-0.02 fq=5036935093.821951 fq_stderr=188046.32004740788
//   index 22 (flux.arrow)
//     schema.fields(): f_max:Float64 unit=Hz, f_max_stderr:Float64 unit=Hz, z_offset:Float64 unit=V, z_offset_stderr:Float64 unit=V, z_period:Float64 unit=V, z_period_stderr:Float64 unit=V, eta:Float64 unit=Hz, eta_stderr:Float64 unit=Hz, asymmetry:Float64 unit=1, asymmetry_stderr:Float64 unit=1
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): f_max=5050042619.745449 f_max_stderr=42536.36035585069 z_offset=-0.000058760108354869036 z_offset_stderr=0.0000355212327778251
//   index 23 (states.arrow)
//     schema.fields(): center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 2 rows (= all columns): center_re=0.3 center_im=0.1
//   index …: the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("qspec-z-window.arrow.zip", exdir = d)

entries
#  [1] "0.arrow"      "1.arrow"      "10.arrow"     "11.arrow"     "12.arrow"    
#  [6] "13.arrow"     "14.arrow"     "15.arrow"     "16.arrow"     "17.arrow"    
# [11] "18.arrow"     "19.arrow"     "2.arrow"      "20.arrow"     "3.arrow"     
# [16] "4.arrow"      "5.arrow"      "6.arrow"      "7.arrow"      "8.arrow"     
# [21] "9.arrow"      "fits.arrow"   "flux.arrow"   "states.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 31  5
names(r0)
# [1] "freq"  "iq_re" "iq_im" "p1"    "model"
head(r0, 2)   # first 2 rows only
#         freq     iq_re     iq_im          p1      model
# 1 5021885229 0.3045482 0.1081956  0.04016354 0.02984333
# 2 5022885229 0.3000654 0.0876673 -0.04519646 0.03140290
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 31  5
names(r1)
# [1] "freq"  "iq_re" "iq_im" "p1"    "model"
head(r1, 2)   # first 2 rows only
#         freq     iq_re      iq_im          p1       model
# 1 5024377879 0.3045294 0.08535874 -0.04383673 0.002333877
# 2 5025377879 0.3129739 0.09684008  0.01700164 0.003962561
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 21 10
names(fits)
#  [1] "row"           "z"             "fq"            "fq_stderr"    
#  [5] "fwhm"          "fwhm_stderr"   "amp"           "amp_stderr"   
#  [9] "offset"        "offset_stderr"
head(fits, 2)   # first 2 rows only
#   row      z         fq fq_stderr    fwhm fwhm_stderr       amp amp_stderr
# 1   0 -0.020 5036935094  188046.3 3467891      631065 0.8198734 0.08800862
# 2   1 -0.018 5039548242  198459.8 3271066      661585 0.9830316 0.11830125
#         offset offset_stderr
# 1  0.019102801    0.02672145
# 2 -0.008960839    0.03413902
sch$GetFieldByName("row")$metadata
# [1] "1"
flux <- read_feather(file.path(d, "flux.arrow"))
dim(flux)   # rows × cols
# [1]  1 10
names(flux)
#  [1] "f_max"            "f_max_stderr"     "z_offset"         "z_offset_stderr" 
#  [5] "z_period"         "z_period_stderr"  "eta"              "eta_stderr"      
#  [9] "asymmetry"        "asymmetry_stderr"
flux
#        f_max f_max_stderr      z_offset z_offset_stderr z_period
# 1 5050042620     42536.36 -5.876011e-05    3.552123e-05 0.598909
#   z_period_stderr      eta eta_stderr asymmetry asymmetry_stderr
# 1     0.001967722 -2.2e+08        NaN       0.1              NaN
sch$GetFieldByName("f_max")$metadata
# [1] "Hz"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 2 2
names(states)
# [1] "center_re" "center_im"
states
#   center_re center_im
# 1      0.30       0.1
# 2      0.42       0.3
sch$GetFieldByName("center_re")$metadata
# [1] "a.u."
# the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### s21 — resonator spectroscopy

<details>
<summary><b>demo-ok</b> — complex-domain fit, 26 parameter columns</summary>

`div_id = demo-ok` · six files: `demo-ok.csv` `demo-ok.txt` `demo-ok.npz` `demo-ok.mat` `demo-ok.xlsx` `demo-ok.arrow.zip` · complex-domain fit, 26 parameter columns

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 51 × 4 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `model[a.u.]`† |
| `fits` | 1 × 26 | `fr[Hz]`, `fr_stderr[Hz]`, `ql[1]`, `ql_stderr[1]`, `qc[1]`, `qc_stderr[1]`, `theta[rad]`, `theta_stderr[rad]`, `ap[1]`, `ap_stderr[1]`, `tau[s]`, `tau_stderr[s]`, `a[1]`, `a_stderr[1]`, `b[1]`, `b_stderr[1]`, `phi[rad]`, `phi_stderr[rad]`, `zc_re[a.u.]`, `zc_re_stderr[a.u.]`, `zc_im[a.u.]`, `zc_im_stderr[a.u.]`, `qi[1]`, `qi_stderr[1]`, `kappa_ex[Hz]`, `kappa_ex_stderr[Hz]` |

#### Python

```python
import numpy as np
z = np.load("demo-ok.npz", allow_pickle=False)

print(z.files)
# ['data', 'fits']
print(z["data"].shape)
# (51,)
print(z["data"].dtype.names)
# ('freq [Hz]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'model [a.u.]')
print(z["data"][:2])   # first 2 rows only
# [(6.8932e+09, 3.70310489e+09-2.50003243e+09j, 50000000., 3.71498585e+09-2.48771558e+09j)
#  (6.8934e+09, 4.30685042e+09-1.25896324e+09j, 50000000., 4.29461285e+09-1.26074402e+09j)]
print(z["fits"].shape)
# (1,)
print(z["fits"].dtype.names)   # column names first, then the values
# ('fr [Hz]', 'fr_stderr [Hz]', 'ql [1]', 'ql_stderr [1]', 'qc [1]', 'qc_stderr [1]', 'theta [rad]', 'theta_stderr [rad]', 'ap [1]', 'ap_stderr [1]', 'tau [s]', 'tau_stderr [s]', 'a [1]', 'a_stderr [1]', 'b [1]', 'b_stderr [1]', 'phi [rad]', 'phi_stderr [rad]', 'zc_re [a.u.]', 'zc_re_stderr [a.u.]', 'zc_im [a.u.]', 'zc_im_stderr [a.u.]', 'qi [1]', 'qi_stderr [1]', 'kappa_ex [Hz]', 'kappa_ex_stderr [Hz]')
print(z["fits"])
# [(6.89818998e+09, 5690.91586475, 7973.89191307, 42.17648171, -13962.42459542, 57.59871093, 2.88390234, 0.00409963, 0.08564584, 0.0070869, -2.40014383e-07, 6.41609094e-12, 4.39849919e+09, 3199397.03420149, 4.40605094e+09, 3016163.22913313, 10395.96080314, 0.27804727, -6368890.2599211, 2152305.22346249, 5778951.51435177, 2210386.86915022, 17808.38904129, 166.03330947, -494053.87496567, 2038.06506481)]
```

#### MATLAB

```matlab
m = load("demo-ok.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)
fieldnames(m.fits)

% size and a value:
size(m.data.iq)
m.data.iq(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.data.freq

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("demo-ok.mat")

keys(m)
# 3-element Vector{String}:
#  "data"
#  "fits"
#  "units"
m["data"]
#   freq = 51×1 Float64, first 2 rows: 6.8932e9, 6.8934e9
#   iq = 51×1 ComplexF64, first 2 rows: 3.703104890186604e9 - 2.5000324329574966e9im, 4.306850417809932e9 - 1.2589632428833923e9im
#   iq_sigma = 51×1 Float64, first 2 rows: 5.0e7, 5.0e7
#   model = 51×1 ComplexF64, first 2 rows: 3.7149858499346175e9 - 2.4877155791039543e9im, 4.2946128464905205e9 - 1.2607440229510005e9im
m["fits"]
#   a = 4.398499191493687e9
#   a_stderr = 3.199397034201487e6
#   ap = 0.08564584087154581
#   ap_stderr = 0.007086897414396107
#   … the remaining 22 fields share one shape
m.units.data
# Dict{String, Any} with 4 entries:
#   "model" => "a.u."
#   "freq" => "Hz"
#   "iq" => "a.u."
#   "iq_sigma" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("demo-ok.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow", "fits.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): freq:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., model_re:Float64 unit=a.u., model_im:Float64 unit=a.u.
//     batch row 0 of 51 rows (only the first 4 columns; the full list is on the fields line above): freq=6893200000 iq_re=3703104890.186604 iq_im=-2500032432.9574966 iq_sigma=50000000
//   index 1 (fits.arrow)
//     schema.fields(): fr:Float64 unit=Hz, fr_stderr:Float64 unit=Hz, ql:Float64 unit=1, ql_stderr:Float64 unit=1, qc:Float64 unit=1, qc_stderr:Float64 unit=1, theta:Float64 unit=rad, theta_stderr:Float64 unit=rad, ap:Float64 unit=1, ap_stderr:Float64 unit=1, tau:Float64 unit=s, tau_stderr:Float64 unit=s, a:Float64 unit=1, a_stderr:Float64 unit=1, b:Float64 unit=1, b_stderr:Float64 unit=1, phi:Float64 unit=rad, phi_stderr:Float64 unit=rad, zc_re:Float64 unit=a.u., zc_re_stderr:Float64 unit=a.u., zc_im:Float64 unit=a.u., zc_im_stderr:Float64 unit=a.u., qi:Float64 unit=1, qi_stderr:Float64 unit=1, kappa_ex:Float64 unit=Hz, kappa_ex_stderr:Float64 unit=Hz
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): fr=6898189975.284475 fr_stderr=5690.9158647530485 ql=7973.891913070236 ql_stderr=42.17648170876852
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("demo-ok.arrow.zip", exdir = d)

entries
# [1] "data.arrow" "fits.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 51  6
names(data)
# [1] "freq"     "iq_re"    "iq_im"    "iq_sigma" "model_re" "model_im"
head(data, 2)   # first 2 rows only
#         freq      iq_re       iq_im iq_sigma   model_re    model_im
# 1 6893200000 3703104890 -2500032433    5e+07 3714985850 -2487715579
# 2 6893400000 4306850418 -1258963243    5e+07 4294612846 -1260744023
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1]  1 26
names(fits)
#  [1] "fr"              "fr_stderr"       "ql"              "ql_stderr"      
#  [5] "qc"              "qc_stderr"       "theta"           "theta_stderr"   
#  [9] "ap"              "ap_stderr"       "tau"             "tau_stderr"     
# [13] "a"               "a_stderr"        "b"               "b_stderr"       
# [17] "phi"             "phi_stderr"      "zc_re"           "zc_re_stderr"   
# [21] "zc_im"           "zc_im_stderr"    "qi"              "qi_stderr"      
# [25] "kappa_ex"        "kappa_ex_stderr"
fits
#           fr fr_stderr       ql ql_stderr        qc qc_stderr    theta
# 1 6898189975  5690.916 7973.892  42.17648 -13962.42  57.59871 2.883902
#   theta_stderr         ap   ap_stderr           tau   tau_stderr          a
# 1  0.004099632 0.08564584 0.007086897 -2.400144e-07 6.416091e-12 4398499191
#   a_stderr          b b_stderr      phi phi_stderr    zc_re zc_re_stderr
# 1  3199397 4406050945  3016163 10395.96  0.2780473 -6368890      2152305
#     zc_im zc_im_stderr       qi qi_stderr  kappa_ex kappa_ex_stderr
# 1 5778952      2210387 17808.39  166.0333 -494053.9        2038.065
sch$GetFieldByName("fr")$metadata
# [1] "Hz"
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### s21 vs power — resonator vs pump power

<details>
<summary><b>power-res0</b> — multi-row: one row per power step (fits 21 rows + 0…20 sub-tables)</summary>

`div_id = power-res0` · six files: `power-res0.csv` `power-res0.txt` `power-res0.npz` `power-res0.mat` `power-res0.xlsx` `power-res0.arrow.zip` · multi-row: one row per power step (`fits` 21 rows + `0`…`20` sub-tables)

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 21 × 28 | `row[1]`→ref, `power[a.u.]`, `fr[Hz]`, `fr_stderr[Hz]`, `ql[1]`, `ql_stderr[1]`, `qc[1]`, `qc_stderr[1]`, `theta[rad]`, `theta_stderr[rad]`, `ap[1]`, `ap_stderr[1]`, `tau[s]`, `tau_stderr[s]`, `a[1]`, `a_stderr[1]`, `b[1]`, `b_stderr[1]`, `phi[rad]`, `phi_stderr[rad]`, `zc_re[a.u.]`, `zc_re_stderr[a.u.]`, `zc_im[a.u.]`, `zc_im_stderr[a.u.]`, `qi[1]`, `qi_stderr[1]`, `kappa_ex[Hz]`, `kappa_ex_stderr[Hz]` |
| `0` … `20` *(one scan per row, same shape)* | 51 × 4 | `freq[Hz]`, `iq[a.u.]`†, `iq_sigma[a.u.]`, `model[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("power-res0.npz", allow_pickle=False)

print(z.files)
# ['fits', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
print(z["fits"].shape)
# (21,)
print(z["fits"].dtype.names)
# ('row [1]', 'power [a.u.]', 'fr [Hz]', 'fr_stderr [Hz]', 'ql [1]', 'ql_stderr [1]', 'qc [1]', 'qc_stderr [1]', 'theta [rad]', 'theta_stderr [rad]', 'ap [1]', 'ap_stderr [1]', 'tau [s]', 'tau_stderr [s]', 'a [1]', 'a_stderr [1]', 'b [1]', 'b_stderr [1]', 'phi [rad]', 'phi_stderr [rad]', 'zc_re [a.u.]', 'zc_re_stderr [a.u.]', 'zc_im [a.u.]', 'zc_im_stderr [a.u.]', 'qi [1]', 'qi_stderr [1]', 'kappa_ex [Hz]', 'kappa_ex_stderr [Hz]')
print(z["fits"][:2])   # first 2 rows only
# [(0, 0.01      , 6.89819823e+09, 5781.53464936, 8046.47541599, 43.46855837, -14078.53425737, 59.35378866, 2.88077946, 0.00419294, 0.10407035, 0.00727983, -2.39990467e-07, 9.33766905e-12, 4.40730854e+09, 3036986.79292986, 4.40221917e+09, 3315879.73042389, 10394.92346065, 0.40461932, -4338427.68779695, 2218126.67464055, 7522593.58466956, 2277729.65575777, 17969.42279023, 173.55318311, -489979.8587882 , 2065.69561284)
#  (1, 0.01216042, 6.89818659e+09, 6133.31836384, 7714.66194439, 44.8398147 , -13900.93706223, 66.56949114, 2.88485637, 0.00429124, 0.10282359, 0.00734815, -2.40013404e-07, 2.05280291e-11, 4.39709644e+09, 3360439.57207613, 4.39913675e+09, 3165336.56315091, 10395.91720838, 0.88969648, -4603069.61767989, 2208887.63860995, 3864357.16572462, 2271825.60847906, 16654.57889987, 155.15381648, -496238.96300542, 2376.3828872 )]
print(z["0"].shape)
# (51,)
print(z["0"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'model [a.u.]')
print(z["0"][0])   # the row-0 value
# (6893200000.0, 3.72385507e+09-2.45733439e+09j, 50000000.0, 3.72452771e+09-2.48137995e+09j)
print(z["1"].shape)
# (51,)
print(z["1"].dtype.names)   # column names first, then the values
# ('freq [Hz]', 'iq [a.u.]', 'iq_sigma [a.u.]', 'model [a.u.]')
print(z["1"][0])   # the row-0 value
# (6893200000.0, 3.71024958e+09-2.5042912e+09j, 50000000.0, 3.72011324e+09-2.48314124e+09j)
# the remaining 19 sub-tables (z["2"] … z["20"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("power-res0.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.row_0)

% size and a value:
size(m.fits.power)
m.fits.power(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.power

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("power-res0.mat")

keys(m)
# 23-element Vector{String}:
#  "fits"
#  "row_0"
#  "row_1"
#  "row_10"
#  "row_11"
#  "row_12"
#  "row_13"
#  "row_14"
#  "row_15"
#  "row_16"
#  "row_17"
#  "row_18"
#  "row_19"
#  "row_2"
#  "row_20"
#  "row_3"
#  "row_4"
#  "row_5"
#  "row_6"
#  "row_7"
#  "row_8"
#  "row_9"
#  "units"
m["fits"]
#   a = 21×1 Float64, first 2 rows: 4.407308537788437e9, 4.397096443153756e9
#   a_stderr = 21×1 Float64, first 2 rows: 3.036986792929857e6, 3.3604395720761344e6
#   ap = 21×1 Float64, first 2 rows: 0.10407034511799494, 0.10282358963337963
#   ap_stderr = 21×1 Float64, first 2 rows: 0.007279833896293302, 0.007348150794566306
#   … the remaining 24 fields share one shape
m["row_0"]
#   freq = 51×1 Float64, first 2 rows: 6.8932e9, 6.8934e9
#   iq = 51×1 ComplexF64, first 2 rows: 3.7238550707959013e9 - 2.4573343940343947e9im, 4.321913042780413e9 - 1.2653867274814339e9im
#   iq_sigma = 51×1 Float64, first 2 rows: 5.0e7, 5.0e7
#   model = 51×1 ComplexF64, first 2 rows: 3.724527710964271e9 - 2.48137994554845e9im, 4.3034050550315695e9 - 1.254329244822545e9im
m["row_1"]
#   freq = 51×1 Float64, first 2 rows: 6.8932e9, 6.8934e9
#   iq = 51×1 ComplexF64, first 2 rows: 3.7102495755576587e9 - 2.504291196984876e9im, 4.298464042922678e9 - 1.2369513170779479e9im
#   iq_sigma = 51×1 Float64, first 2 rows: 5.0e7, 5.0e7
#   model = 51×1 ComplexF64, first 2 rows: 3.7201132441031775e9 - 2.483141244617232e9im, 4.297371299050392e9 - 1.2570970568234956e9im
# the remaining 19 sub-tables (row_2 … row_20) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 28 entries:
#   "qc_stderr" => "1"
#   "kappa_ex_stderr" => "Hz"
#   "zc_re_stderr" => "a.u."
#   "b" => "1"
#   "row" => "1"
#   "phi_stderr" => "rad"
#   "tau" => "s"
#   "a" => "1"
#   "qi_stderr" => "1"
#   "kappa_ex" => "Hz"
#   "theta" => "rad"
#   "tau_stderr" => "s"
#   "zc_re" => "a.u."
#   "phi" => "rad"
#   "ql_stderr" => "1"
#   "ql" => "1"
#   "qc" => "1"
#   "ap_stderr" => "1"
#   "theta_stderr" => "rad"
#   "fr" => "Hz"
#   "power" => "a.u."
#   "a_stderr" => "1"
#   "b_stderr" => "1"
#   "qi" => "1"
#   "fr_stderr" => "Hz"
#   "ap" => "1"
#   "zc_im_stderr" => "a.u."
#   "zc_im" => "a.u."
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("power-res0.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "3.arrow", "4.arrow", "5.arrow", "6.arrow", "7.arrow", "8.arrow", "9.arrow", "10.arrow", "11.arrow", "12.arrow", "13.arrow", "14.arrow", "15.arrow", "16.arrow", "17.arrow", "18.arrow", "19.arrow", "20.arrow", "fits.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): freq:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., iq_sigma:Float64 unit=a.u., model_re:Float64 unit=a.u., model_im:Float64 unit=a.u.
//     batch row 0 of 51 rows (only the first 4 columns; the full list is on the fields line above): freq=6893200000 iq_re=3723855070.7959013 iq_im=-2457334394.0343947 iq_sigma=50000000
//   index 21 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, power:Float64 unit=a.u., fr:Float64 unit=Hz, fr_stderr:Float64 unit=Hz, ql:Float64 unit=1, ql_stderr:Float64 unit=1, qc:Float64 unit=1, qc_stderr:Float64 unit=1, theta:Float64 unit=rad, theta_stderr:Float64 unit=rad, ap:Float64 unit=1, ap_stderr:Float64 unit=1, tau:Float64 unit=s, tau_stderr:Float64 unit=s, a:Float64 unit=1, a_stderr:Float64 unit=1, b:Float64 unit=1, b_stderr:Float64 unit=1, phi:Float64 unit=rad, phi_stderr:Float64 unit=rad, zc_re:Float64 unit=a.u., zc_re_stderr:Float64 unit=a.u., zc_im:Float64 unit=a.u., zc_im_stderr:Float64 unit=a.u., qi:Float64 unit=1, qi_stderr:Float64 unit=1, kappa_ex:Float64 unit=Hz, kappa_ex_stderr:Float64 unit=Hz
//     batch row 0 of 21 rows (only the first 4 columns; the full list is on the fields line above): row=0 power=0.01 fr=6898198227.3726015 fr_stderr=5781.534649356943
//   index …: the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("power-res0.arrow.zip", exdir = d)

entries
#  [1] "0.arrow"    "1.arrow"    "10.arrow"   "11.arrow"   "12.arrow"  
#  [6] "13.arrow"   "14.arrow"   "15.arrow"   "16.arrow"   "17.arrow"  
# [11] "18.arrow"   "19.arrow"   "2.arrow"    "20.arrow"   "3.arrow"   
# [16] "4.arrow"    "5.arrow"    "6.arrow"    "7.arrow"    "8.arrow"   
# [21] "9.arrow"    "fits.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 51  6
names(r0)
# [1] "freq"     "iq_re"    "iq_im"    "iq_sigma" "model_re" "model_im"
head(r0, 2)   # first 2 rows only
#         freq      iq_re       iq_im iq_sigma   model_re    model_im
# 1 6893200000 3723855071 -2457334394    5e+07 3724527711 -2481379946
# 2 6893400000 4321913043 -1265386727    5e+07 4303405055 -1254329245
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 51  6
names(r1)
# [1] "freq"     "iq_re"    "iq_im"    "iq_sigma" "model_re" "model_im"
head(r1, 2)   # first 2 rows only
#         freq      iq_re       iq_im iq_sigma   model_re    model_im
# 1 6893200000 3710249576 -2504291197    5e+07 3720113244 -2483141245
# 2 6893400000 4298464043 -1236951317    5e+07 4297371299 -1257097057
sch$GetFieldByName("freq")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1] 21 28
names(fits)
#  [1] "row"             "power"           "fr"              "fr_stderr"      
#  [5] "ql"              "ql_stderr"       "qc"              "qc_stderr"      
#  [9] "theta"           "theta_stderr"    "ap"              "ap_stderr"      
# [13] "tau"             "tau_stderr"      "a"               "a_stderr"       
# [17] "b"               "b_stderr"        "phi"             "phi_stderr"     
# [21] "zc_re"           "zc_re_stderr"    "zc_im"           "zc_im_stderr"   
# [25] "qi"              "qi_stderr"       "kappa_ex"        "kappa_ex_stderr"
head(fits, 2)   # first 2 rows only
#   row      power         fr fr_stderr       ql ql_stderr        qc qc_stderr
# 1   0 0.01000000 6898198227  5781.535 8046.475  43.46856 -14078.53  59.35379
# 2   1 0.01216042 6898186593  6133.318 7714.662  44.83981 -13900.94  66.56949
#      theta theta_stderr        ap   ap_stderr           tau   tau_stderr
# 1 2.880779  0.004192943 0.1040703 0.007279834 -2.399905e-07 9.337669e-12
# 2 2.884856  0.004291244 0.1028236 0.007348151 -2.400134e-07 2.052803e-11
#            a a_stderr          b b_stderr      phi phi_stderr    zc_re
# 1 4407308538  3036987 4402219168  3315880 10394.92  0.4046193 -4338428
# 2 4397096443  3360440 4399136746  3165337 10395.92  0.8896965 -4603070
#   zc_re_stderr   zc_im zc_im_stderr       qi qi_stderr  kappa_ex
# 1      2218127 7522594      2277730 17969.42  173.5532 -489979.9
# 2      2208888 3864357      2271826 16654.58  155.1538 -496239.0
#   kappa_ex_stderr
# 1        2065.696
# 2        2376.383
sch$GetFieldByName("row")$metadata
# [1] "1"
# the remaining 19 sub-tables (2.arrow … 20.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### iq — readout discrimination

<details>
<summary><b>iq-99</b> — no fit: full single-shot per state + states/density/pairs</summary>

`div_id = iq-99` · six files: `iq-99.csv` `iq-99.txt` `iq-99.npz` `iq-99.mat` `iq-99.xlsx` `iq-99.arrow.zip` · no fit: full single-shot per state + `states`/`density`/`pairs`

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `states` | 3 × 3 | `state[1]`, `row[1]`→ref, `center[a.u.]`† |
| `density` | 12 × 4 | `state[1]`, `level[1]`, `area[a.u.^2]`, `radius[a.u.]` |
| `pairs` | 3 × 7 | `p[1]`, `q[1]`, `separation[a.u.]`, `threshold[a.u.]`†, `error_rate[1]`, `auc[1]`, `snr[1]` |
| `0` … `2` *(one scan per row, same shape)* | 6000 × 1 | `iq[a.u.]`† |

#### Python

```python
import numpy as np
z = np.load("iq-99.npz", allow_pickle=False)

print(z.files)
# ['states', 'density', 'pairs', '0', '1', '2']
print(z["states"].shape)
# (3,)
print(z["states"].dtype.names)
# ('state [1]', 'row [1]', 'center [a.u.]')
print(z["states"])
# [(0., 0, -0.50009326-0.19939427j) (1., 1,  0.00106072+0.24925813j)
#  (2., 2,  0.49974082-0.15032486j)]
print(z["density"].shape)
# (12,)
print(z["density"].dtype.names)   # column names first, then the values
# ('state [1]', 'level [1]', 'area [a.u.^2]', 'radius [a.u.]')
print(z["density"][0])   # the row-0 value
# (0.0, 0.68, 0.02150874663123999, 0.08274325768391037)
print(z["pairs"].shape)
# (3,)
print(z["pairs"].dtype.names)   # column names first, then the values
# ('p [1]', 'q [1]', 'separation [a.u.]', 'threshold [a.u.]', 'error_rate [1]', 'auc [1]', 'snr [1]')
print(z["pairs"])
# [(0., 1., 0.67263979, -0.26388743+0.01206631j, 0., 1.,  8.01876398)
#  (0., 2., 1.00103746,  0.03967487-0.17290377j, 0., 1., 12.73676649)
#  (1., 2., 0.63902144,  0.27163579+0.03245141j, 0., 1.,  7.60548926)]
print(z["0"].shape)
# (6000,)
print(z["0"].dtype.names)   # column names first, then the values
# ('iq [a.u.]',)
print(z["0"][0])   # the row-0 value
# (-0.41063576-0.13061962j,)
print(z["1"].shape)
# (6000,)
print(z["1"].dtype.names)   # column names first, then the values
# ('iq [a.u.]',)
print(z["1"][0])   # the row-0 value
# (0.09224753+0.27719613j,)
# the remaining 1 sub-tables (z["2"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("iq-99.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.states)
fieldnames(m.density)
fieldnames(m.pairs)
fieldnames(m.row_0)

% size and a value:
size(m.states.state)
m.states.center(1)     % complex numbers are native double complex

% units live in the parallel units struct:
m.units.states.state

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("iq-99.mat")

keys(m)
# 7-element Vector{String}:
#  "density"
#  "pairs"
#  "row_0"
#  "row_1"
#  "row_2"
#  "states"
#  "units"
m["density"]
#   area = 12×1 Float64, first 2 rows: 0.02150874663123999, 0.05659803201630561
#   level = 12×1 Float64, first 2 rows: 0.68, 0.95
#   radius = 12×1 Float64, first 2 rows: 0.08274325768391037, 0.13422262525124734
#   state = 12×1 Float64, first 2 rows: 0.0, 0.0
m["pairs"]
#   auc = 3×1 Matrix{Float64}:
#  1.0
#  1.0
#  1.0
#   error_rate = 3×1 Matrix{Float64}:
#  0.0
#  0.0
#  0.0
#   p = 3×1 Matrix{Float64}:
#  0.0
#  0.0
#  1.0
#   q = 3×1 Matrix{Float64}:
#  1.0
#  2.0
#  2.0
#   … the remaining 3 fields share one shape
m["row_0"]
#   iq = 6000×1 ComplexF64, first 2 rows: -0.41063575930873164 - 0.1306196169956319im, -0.5431306485399195 - 0.21975202109617883im
m["row_1"]
#   iq = 6000×1 ComplexF64, first 2 rows: 0.09224753449331234 + 0.2771961271305407im, -0.07870210886453854 + 0.280733967705238im
m["states"]
#   center = 3×1 Matrix{ComplexF64}:
#    -0.5000932641939138 - 0.19939426797060153im
#  0.0010607216927039586 + 0.24925812649789825im
#     0.4997408190124789 - 0.1503248583084832im
#   row = 3×1 Matrix{Float64}:
#  0.0
#  1.0
#  2.0
#   state = 3×1 Matrix{Float64}:
#  0.0
#  1.0
#  2.0
# the remaining 1 sub-tables (row_2) share one shape: one scan each, same column names as row_0
m.units.density
# Dict{String, Any} with 4 entries:
#   "area" => "a.u.^2"
#   "radius" => "a.u."
#   "level" => "1"
#   "state" => "1"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("iq-99.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "states.arrow", "density.arrow", "pairs.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u.
//     batch row 0 of 6000 rows (only the first 4 columns; the full list is on the fields line above): iq_re=-0.41063575930873164 iq_im=-0.1306196169956319
//   index 3 (states.arrow)
//     schema.fields(): state:Float64 unit=1, row:Int64 unit=1, center_re:Float64 unit=a.u., center_im:Float64 unit=a.u.
//     batch row 0 of 3 rows (only the first 4 columns; the full list is on the fields line above): state=0 row=0 center_re=-0.5000932641939138 center_im=-0.19939426797060153
//   index 4 (density.arrow)
//     schema.fields(): state:Float64 unit=1, level:Float64 unit=1, area:Float64 unit=a.u.^2, radius:Float64 unit=a.u.
//     batch row 0 of 12 rows (only the first 4 columns; the full list is on the fields line above): state=0 level=0.68 area=0.02150874663123999 radius=0.08274325768391037
//   index 5 (pairs.arrow)
//     schema.fields(): p:Float64 unit=1, q:Float64 unit=1, separation:Float64 unit=a.u., threshold_re:Float64 unit=a.u., threshold_im:Float64 unit=a.u., error_rate:Float64 unit=1, auc:Float64 unit=1, snr:Float64 unit=1
//     batch row 0 of 3 rows (only the first 4 columns; the full list is on the fields line above): p=0 q=1 separation=0.6726397911455749 threshold_re=-0.2638874344902054
//   index …: the remaining 1 sub-tables (2.arrow … 2.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("iq-99.arrow.zip", exdir = d)

entries
# [1] "0.arrow"       "1.arrow"       "2.arrow"       "density.arrow"
# [5] "pairs.arrow"   "states.arrow" 
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 6000    2
names(r0)
# [1] "iq_re" "iq_im"
head(r0, 2)   # first 2 rows only
#        iq_re      iq_im
# 1 -0.4106358 -0.1306196
# 2 -0.5431306 -0.2197520
sch$GetFieldByName("iq_re")$metadata
# [1] "a.u."
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 6000    2
names(r1)
# [1] "iq_re" "iq_im"
head(r1, 2)   # first 2 rows only
#         iq_re     iq_im
# 1  0.09224753 0.2771961
# 2 -0.07870211 0.2807340
sch$GetFieldByName("iq_re")$metadata
# [1] "a.u."
density <- read_feather(file.path(d, "density.arrow"))
dim(density)   # rows × cols
# [1] 12  4
names(density)
# [1] "state"  "level"  "area"   "radius"
head(density, 2)   # first 2 rows only
#   state level       area     radius
# 1     0  0.68 0.02150875 0.08274326
# 2     0  0.95 0.05659803 0.13422263
sch$GetFieldByName("state")$metadata
# [1] "1"
pairs <- read_feather(file.path(d, "pairs.arrow"))
dim(pairs)   # rows × cols
# [1] 3 8
names(pairs)
# [1] "p"            "q"            "separation"   "threshold_re" "threshold_im"
# [6] "error_rate"   "auc"          "snr"         
pairs
#   p q separation threshold_re threshold_im error_rate auc       snr
# 1 0 1  0.6726398  -0.26388743   0.01206631          0   1  8.018764
# 2 0 2  1.0010375   0.03967487  -0.17290377          0   1 12.736766
# 3 1 2  0.6390214   0.27163579   0.03245141          0   1  7.605489
sch$GetFieldByName("p")$metadata
# [1] "1"
states <- read_feather(file.path(d, "states.arrow"))
dim(states)   # rows × cols
# [1] 3 4
names(states)
# [1] "state"     "row"       "center_re" "center_im"
states
#   state row    center_re  center_im
# 1     0   0 -0.500093264 -0.1993943
# 2     1   1  0.001060722  0.2492581
# 3     2   2  0.499740819 -0.1503249
sch$GetFieldByName("state")$metadata
# [1] "1"
# the remaining 1 sub-tables (2.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### bloch — three-axis readout

<details>
<summary><b>bloch</b> — no fit: a single data table</summary>

`div_id = bloch` · six files: `bloch.csv` `bloch.txt` `bloch.npz` `bloch.mat` `bloch.xlsx` `bloch.arrow.zip` · no fit: a single `data` table

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `data` | 61 × 4 | `axis[a.u.]`, `x[1]`, `y[1]`, `z[1]` |

#### Python

```python
import numpy as np
z = np.load("bloch.npz", allow_pickle=False)

print(z.files)
# ['data']
print(z["data"].shape)
# (61,)
print(z["data"].dtype.names)
# ('axis [a.u.]', 'x [1]', 'y [1]', 'z [1]')
print(z["data"][:2])   # first 2 rows only
# [(0.  , 0.08330857,  0.0260277 , 0.97635625)
#  (0.05, 0.13219882, -0.15601616, 0.86961559)]
```

#### MATLAB

```matlab
m = load("bloch.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.data)

% size and a value:
size(m.data.axis)
m.data.axis(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.data.axis

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("bloch.mat")

keys(m)
# 2-element Vector{String}:
#  "data"
#  "units"
m["data"]
#   axis = 61×1 Float64, first 2 rows: 0.0, 0.05
#   x = 61×1 Float64, first 2 rows: 0.08330856524365116, 0.1321988219849316
#   y = 61×1 Float64, first 2 rows: 0.02602770178917846, -0.15601615872173968
#   z = 61×1 Float64, first 2 rows: 0.9763562531350645, 0.8696155858036185
m.units.data
# Dict{String, Any} with 4 entries:
#   "axis" => "a.u."
#   "x" => "1"
#   "z" => "1"
#   "y" => "1"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("bloch.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["data.arrow"]
// what each iteration prints:
//   index 0 (data.arrow)
//     schema.fields(): axis:Float64 unit=a.u., x:Float64 unit=1, y:Float64 unit=1, z:Float64 unit=1
//     batch row 0 of 61 rows (only the first 4 columns; the full list is on the fields line above): axis=0 x=0.08330856524365116 y=0.02602770178917846 z=0.9763562531350645
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("bloch.arrow.zip", exdir = d)

entries
# [1] "data.arrow"
data <- read_feather(file.path(d, "data.arrow"))
dim(data)   # rows × cols
# [1] 61  4
names(data)
# [1] "axis" "x"    "y"    "z"   
head(data, 2)   # first 2 rows only
#   axis          x          y         z
# 1 0.00 0.08330857  0.0260277 0.9763563
# 2 0.05 0.13219882 -0.1560162 0.8696156
sch$GetFieldByName("axis")$metadata
# [1] "a.u."
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### drag — amplitude scan

<details>
<summary><b>drag-3</b> — multi-row: the escalation chain (fits + params + 0…2 sub-tables)</summary>

`div_id = drag-3` · six files: `drag-3.csv` `drag-3.txt` `drag-3.npz` `drag-3.mat` `drag-3.xlsx` `drag-3.arrow.zip` · multi-row: the escalation chain (`fits` + `params` + `0`…`2` sub-tables)

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 2 × 10 | `row[1]`→ref, `pairs[1]`, `centre[1]`, `centre_stderr[1]`, `fwhm[1]`, `fwhm_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 3 | `a_pi[1]`, `a_pi_stderr[1]`, `chosen[1]` |
| `0` … `2` *(one scan per row, same shape)* | 31 × 4 | `amp[1]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

#### Python

```python
import numpy as np
z = np.load("drag-3.npz", allow_pickle=False)

print(z.files)
# ['fits', 'params', '0', '1', '2']
print(z["fits"].shape)
# (2,)
print(z["fits"].dtype.names)
# ('row [1]', 'pairs [1]', 'centre [1]', 'centre_stderr [1]', 'fwhm [1]', 'fwhm_stderr [1]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"])
# [(1, 2., 0.0500156 , 6.14321392e-05, 0.0174305, 0.00076558, -1.59783115, 0.05296281, 1.57019386, 0.05788911)
#  (2, 4., 0.04996161, 3.32572364e-05, 0.0088371, 0.0004237 , -1.58594372, 0.05822333, 1.57465831, 0.06352255)]
print(z["params"].shape)
# (1,)
print(z["params"].dtype.names)   # column names first, then the values
# ('a_pi [1]', 'a_pi_stderr [1]', 'chosen [1]')
print(z["params"])
# [(0.04996959, 7.20763456e-05, 0.04996161)]
print(z["0"].shape)
# (31,)
print(z["0"].dtype.names)   # column names first, then the values
# ('amp [1]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["0"][0])   # the row-0 value
# (0.0, 0.29392643+0.09231418j, -0.041654282621825654, 0.0)
print(z["1"].shape)
# (31,)
print(z["1"].dtype.names)   # column names first, then the values
# ('amp [1]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["1"][0])   # the row-0 value
# (0.03746959092463774, 0.4194373+0.3030772j, 1.010072001384905, 1.050116377567035)
# the remaining 1 sub-tables (z["2"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("drag-3.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.params)
fieldnames(m.row_0)

% size and a value:
size(m.fits.pairs)
m.fits.pairs(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.pairs

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("drag-3.mat")

keys(m)
# 6-element Vector{String}:
#  "fits"
#  "params"
#  "row_0"
#  "row_1"
#  "row_2"
#  "units"
m["fits"]
#   amp = 2×1 Matrix{Float64}:
#  -1.5978311456708396
#  -1.5859437151777183
#   amp_stderr = 2×1 Matrix{Float64}:
#  0.05296280677262448
#  0.058223332784387204
#   centre = 2×1 Matrix{Float64}:
#  0.050015599317852016
#  0.04996161326425327
#   centre_stderr = 2×1 Matrix{Float64}:
#  6.143213915780104e-5
#  3.325723637419831e-5
#   … the remaining 6 fields share one shape
m["params"]
#   a_pi = 0.04996959092463774
#   a_pi_stderr = 7.207634555874306e-5
#   chosen = 0.04996161326425327
m["row_0"]
#   amp = 31×1 Float64, first 2 rows: 0.0, 0.0033333333333333335
#   iq = 31×1 ComplexF64, first 2 rows: 0.29392643200620944 + 0.09231417592313776im, 0.2986916647193367 + 0.1044497838252724im
#   model = 31×1 Float64, first 2 rows: 0.0, 0.04319091816152095
#   p1 = 31×1 Float64, first 2 rows: -0.041654282621825654, 0.013473465650273651
m["row_1"]
#   amp = 31×1 Float64, first 2 rows: 0.03746959092463774, 0.03830292425797107
#   iq = 31×1 ComplexF64, first 2 rows: 0.4194373013777846 + 0.3030772035500234im, 0.41002347790529714 + 0.300684399981616im
#   model = 31×1 Float64, first 2 rows: 1.050116377567035, 1.0007894226630936
#   p1 = 31×1 Float64, first 2 rows: 1.010072001384905, 0.9805091423705674
# the remaining 1 sub-tables (row_2) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 10 entries:
#   "centre_stderr" => "1"
#   "offset" => "1"
#   "row" => "1"
#   "amp" => "1"
#   "offset_stderr" => "1"
#   "fwhm" => "1"
#   "centre" => "1"
#   "pairs" => "1"
#   "amp_stderr" => "1"
#   "fwhm_stderr" => "1"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("drag-3.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "fits.arrow", "params.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): amp:Float64 unit=1, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., p1:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 31 rows (only the first 4 columns; the full list is on the fields line above): amp=0 iq_re=0.29392643200620944 iq_im=0.09231417592313776 p1=-0.041654282621825654
//   index 3 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, pairs:Float64 unit=1, centre:Float64 unit=1, centre_stderr:Float64 unit=1, fwhm:Float64 unit=1, fwhm_stderr:Float64 unit=1, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 2 rows (only the first 4 columns; the full list is on the fields line above): row=1 pairs=2 centre=0.050015599317852016 centre_stderr=0.00006143213915780104
//   index 4 (params.arrow)
//     schema.fields(): a_pi:Float64 unit=1, a_pi_stderr:Float64 unit=1, chosen:Float64 unit=1
//     batch row 0 of 1 rows (only the first 4 columns; the full list is on the fields line above): a_pi=0.04996959092463774 a_pi_stderr=0.00007207634555874306 chosen=0.04996161326425327
//   index …: the remaining 1 sub-tables (2.arrow … 2.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("drag-3.arrow.zip", exdir = d)

entries
# [1] "0.arrow"      "1.arrow"      "2.arrow"      "fits.arrow"   "params.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 31  5
names(r0)
# [1] "amp"   "iq_re" "iq_im" "p1"    "model"
head(r0, 2)   # first 2 rows only
#           amp     iq_re      iq_im          p1      model
# 1 0.000000000 0.2939264 0.09231418 -0.04165428 0.00000000
# 2 0.003333333 0.2986917 0.10444978  0.01347347 0.04319092
sch$GetFieldByName("amp")$metadata
# [1] "1"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 31  5
names(r1)
# [1] "amp"   "iq_re" "iq_im" "p1"    "model"
head(r1, 2)   # first 2 rows only
#          amp     iq_re     iq_im        p1    model
# 1 0.03746959 0.4194373 0.3030772 1.0100720 1.050116
# 2 0.03830292 0.4100235 0.3006844 0.9805091 1.000789
sch$GetFieldByName("amp")$metadata
# [1] "1"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1]  2 10
names(fits)
#  [1] "row"           "pairs"         "centre"        "centre_stderr"
#  [5] "fwhm"          "fwhm_stderr"   "amp"           "amp_stderr"   
#  [9] "offset"        "offset_stderr"
fits
#   row pairs     centre centre_stderr        fwhm  fwhm_stderr       amp
# 1   1     2 0.05001560  6.143214e-05 0.017430500 0.0007655772 -1.597831
# 2   2     4 0.04996161  3.325724e-05 0.008837103 0.0004237005 -1.585944
#   amp_stderr   offset offset_stderr
# 1 0.05296281 1.570194    0.05788911
# 2 0.05822333 1.574658    0.06352255
sch$GetFieldByName("row")$metadata
# [1] "1"
params <- read_feather(file.path(d, "params.arrow"))
dim(params)   # rows × cols
# [1] 1 3
names(params)
# [1] "a_pi"        "a_pi_stderr" "chosen"     
params
#         a_pi  a_pi_stderr     chosen
# 1 0.04996959 7.207635e-05 0.04996161
sch$GetFieldByName("a_pi")$metadata
# [1] "1"
# the remaining 1 sub-tables (2.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### drag — coeff scan

<details>
<summary><b>drag-coeff-3</b> — same, the axis is the dimensionless coefficient</summary>

`div_id = drag-coeff-3` · six files: `drag-coeff-3.csv` `drag-coeff-3.txt` `drag-coeff-3.npz` `drag-coeff-3.mat` `drag-coeff-3.xlsx` `drag-coeff-3.arrow.zip` · same, the axis is the dimensionless coefficient

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 3 × 10 | `row[1]`→ref, `pairs[1]`, `centre[1]`, `centre_stderr[1]`, `fwhm[1]`, `fwhm_stderr[1]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 1 | `chosen[1]` |
| `0` … `2` *(one scan per row, same shape)* | 31 × 4 | `coeff[1]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

#### Python

```python
import numpy as np
z = np.load("drag-coeff-3.npz", allow_pickle=False)

print(z.files)
# ['fits', 'params', '0', '1', '2']
print(z["fits"].shape)
# (3,)
print(z["fits"].dtype.names)
# ('row [1]', 'pairs [1]', 'centre [1]', 'centre_stderr [1]', 'fwhm [1]', 'fwhm_stderr [1]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"])
# [(0, 1., 0.08000929, 0.00024531, 0.05478801, 0.00139937, -0.96893555, 0.01151951, 0.97321214, 0.01099431)
#  (1, 2., 0.08010856, 0.0001462 , 0.03421606, 0.00085354, -1.01300513, 0.01177382, 1.00376291, 0.01145903)
#  (2, 4., 0.07977009, 0.00010073, 0.01971431, 0.00058891, -0.99169773, 0.0138099 , 0.99967337, 0.01345646)]
print(z["params"].shape)
# (1,)
print(z["params"].dtype.names)   # column names first, then the values
# ('chosen [1]',)
print(z["params"])
# [(0.07977009,)]
print(z["0"].shape)
# (31,)
print(z["0"].dtype.names)   # column names first, then the values
# ('coeff [1]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["0"][0])   # the row-0 value
# (0.0, 0.40072637+0.27031407j, 0.8483451741656152, 0.871544346614291)
print(z["1"].shape)
# (31,)
print(z["1"].dtype.names)   # column names first, then the values
# ('coeff [1]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["1"][0])   # the row-0 value
# (0.03405602074372262, 0.40463929+0.27841385j, 0.8867552118335885, 0.8809170244451201)
# the remaining 1 sub-tables (z["2"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("drag-coeff-3.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.params)
fieldnames(m.row_0)

% size and a value:
size(m.fits.pairs)
m.fits.pairs(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.pairs

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("drag-coeff-3.mat")

keys(m)
# 6-element Vector{String}:
#  "fits"
#  "params"
#  "row_0"
#  "row_1"
#  "row_2"
#  "units"
m["fits"]
#   amp = 3×1 Matrix{Float64}:
#  -0.9689355492605289
#  -1.0130051256838364
#  -0.9916977285063262
#   amp_stderr = 3×1 Matrix{Float64}:
#  0.011519511589443967
#  0.011773822535375128
#  0.013809897575681837
#   centre = 3×1 Matrix{Float64}:
#  0.08000929152111295
#  0.08010855859573329
#  0.07977009055495261
#   centre_stderr = 3×1 Matrix{Float64}:
#  0.0002453076414315927
#  0.00014619636317169196
#  0.00010073371274044351
#   … the remaining 6 fields share one shape
m["params"]
#   chosen = 0.07977009055495261
m["row_0"]
#   coeff = 31×1 Float64, first 2 rows: 0.0, 0.005
#   iq = 31×1 ComplexF64, first 2 rows: 0.40072636682070234 + 0.2703140672806259im, 0.39870987162994753 + 0.27114679534295716im
#   model = 31×1 Float64, first 2 rows: 0.871544346614291, 0.8591867892600563
#   p1 = 31×1 Float64, first 2 rows: 0.8483451741656152, 0.8469585232386975
m["row_1"]
#   coeff = 31×1 Float64, first 2 rows: 0.03405602074372262, 0.03711957212888197
#   iq = 31×1 ComplexF64, first 2 rows: 0.40463928663162657 + 0.27841384563976007im, 0.39455552285836465 + 0.2749044749033951im
#   model = 31×1 Float64, first 2 rows: 0.8809170244451201, 0.8652633992598726
#   p1 = 31×1 Float64, first 2 rows: 0.8867552118335885, 0.8516095169794629
# the remaining 1 sub-tables (row_2) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 10 entries:
#   "centre_stderr" => "1"
#   "offset" => "1"
#   "row" => "1"
#   "amp" => "1"
#   "offset_stderr" => "1"
#   "fwhm" => "1"
#   "centre" => "1"
#   "pairs" => "1"
#   "amp_stderr" => "1"
#   "fwhm_stderr" => "1"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("drag-coeff-3.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "fits.arrow", "params.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): coeff:Float64 unit=1, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., p1:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 31 rows (only the first 4 columns; the full list is on the fields line above): coeff=0 iq_re=0.40072636682070234 iq_im=0.2703140672806259 p1=0.8483451741656152
//   index 3 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, pairs:Float64 unit=1, centre:Float64 unit=1, centre_stderr:Float64 unit=1, fwhm:Float64 unit=1, fwhm_stderr:Float64 unit=1, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 3 rows (only the first 4 columns; the full list is on the fields line above): row=0 pairs=1 centre=0.08000929152111295 centre_stderr=0.0002453076414315927
//   index 4 (params.arrow)
//     schema.fields(): chosen:Float64 unit=1
//     batch row 0 of 1 rows (= all columns): chosen=0.07977009055495261
//   index …: the remaining 1 sub-tables (2.arrow … 2.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("drag-coeff-3.arrow.zip", exdir = d)

entries
# [1] "0.arrow"      "1.arrow"      "2.arrow"      "fits.arrow"   "params.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 31  5
names(r0)
# [1] "coeff" "iq_re" "iq_im" "p1"    "model"
head(r0, 2)   # first 2 rows only
#   coeff     iq_re     iq_im        p1     model
# 1 0.000 0.4007264 0.2703141 0.8483452 0.8715443
# 2 0.005 0.3987099 0.2711468 0.8469585 0.8591868
sch$GetFieldByName("coeff")$metadata
# [1] "1"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 31  5
names(r1)
# [1] "coeff" "iq_re" "iq_im" "p1"    "model"
head(r1, 2)   # first 2 rows only
#        coeff     iq_re     iq_im        p1     model
# 1 0.03405602 0.4046393 0.2784138 0.8867552 0.8809170
# 2 0.03711957 0.3945555 0.2749045 0.8516095 0.8652634
sch$GetFieldByName("coeff")$metadata
# [1] "1"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1]  3 10
names(fits)
#  [1] "row"           "pairs"         "centre"        "centre_stderr"
#  [5] "fwhm"          "fwhm_stderr"   "amp"           "amp_stderr"   
#  [9] "offset"        "offset_stderr"
fits
#   row pairs     centre centre_stderr       fwhm  fwhm_stderr        amp
# 1   0     1 0.08000929  0.0002453076 0.05478801 0.0013993689 -0.9689355
# 2   1     2 0.08010856  0.0001461964 0.03421606 0.0008535414 -1.0130051
# 3   2     4 0.07977009  0.0001007337 0.01971431 0.0005889122 -0.9916977
#   amp_stderr    offset offset_stderr
# 1 0.01151951 0.9732121    0.01099431
# 2 0.01177382 1.0037629    0.01145903
# 3 0.01380990 0.9996734    0.01345646
sch$GetFieldByName("row")$metadata
# [1] "1"
params <- read_feather(file.path(d, "params.arrow"))
dim(params)   # rows × cols
# [1] 1 1
names(params)
# [1] "chosen"
params
#       chosen
# 1 0.07977009
sch$GetFieldByName("chosen")$metadata
# [1] "1"
# the remaining 1 sub-tables (2.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

### drag — detuning scan

<details>
<summary><b>drag-detuning-3</b> — same, the axis is Hz</summary>

`div_id = drag-detuning-3` · six files: `drag-detuning-3.csv` `drag-detuning-3.txt` `drag-detuning-3.npz` `drag-detuning-3.mat` `drag-detuning-3.xlsx` `drag-detuning-3.arrow.zip` · same, the axis is Hz

| table | shape | columns (`key[unit]`) |
|---|---|---|
| `fits` | 3 × 10 | `row[1]`→ref, `pairs[1]`, `centre[Hz]`, `centre_stderr[Hz]`, `fwhm[Hz]`, `fwhm_stderr[Hz]`, `amp[1]`, `amp_stderr[1]`, `offset[1]`, `offset_stderr[1]` |
| `params` | 1 × 1 | `chosen[Hz]` |
| `0` … `2` *(one scan per row, same shape)* | 31 × 4 | `detuning[Hz]`, `iq[a.u.]`†, `p1[1]`, `model[1]` |

#### Python

```python
import numpy as np
z = np.load("drag-detuning-3.npz", allow_pickle=False)

print(z.files)
# ['fits', 'params', '0', '1', '2']
print(z["fits"].shape)
# (3,)
print(z["fits"].dtype.names)
# ('row [1]', 'pairs [1]', 'centre [Hz]', 'centre_stderr [Hz]', 'fwhm [Hz]', 'fwhm_stderr [Hz]', 'amp [1]', 'amp_stderr [1]', 'offset [1]', 'offset_stderr [1]')
print(z["fits"])
# [(0, 1., 253704.368933  , 4817.27107253, 1086402.86874285, 26352.25820764, -0.96999265, 0.01109525, 0.97109525, 0.01022071)
#  (1, 2., 252090.71812026, 2740.58829234,  641372.02697594, 15998.26222819, -1.01287072, 0.01177097, 1.00365153, 0.01145493)
#  (2, 4., 245694.56171217, 1886.12174402,  369125.4664661 , 11026.21132091, -0.99170293, 0.01380932, 0.9996607 , 0.01345541)]
print(z["params"].shape)
# (1,)
print(z["params"].dtype.names)   # column names first, then the values
# ('chosen [Hz]',)
print(z["params"])
# [(245694.56171217,)]
print(z["0"].shape)
# (31,)
print(z["0"].dtype.names)   # column names first, then the values
# ('detuning [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["0"][0])   # the row-0 value
# (-1500000.0, 0.40268944+0.27358585j, 0.8647040988810646, 0.886179182775028)
print(z["1"].shape)
# (31,)
print(z["1"].dtype.names)   # column names first, then the values
# ('detuning [Hz]', 'iq [a.u.]', 'p1 [1]', 'model [1]')
print(z["1"][0])   # the row-0 value
# (-607819.3973147757, 0.40453239+0.27823568j, 0.8858643909209256, 0.8799841852028474)
# the remaining 1 sub-tables (z["2"]) share one shape: one scan each, same column names as the two above
```

#### MATLAB

```matlab
m = load("drag-detuning-3.mat");

% the table list; scan sub-tables are spelled row_0, row_1 … here (MATLAB variable names cannot start with a digit)
fieldnames(m)
% the column names of every table (same-shaped scan sub-tables: only row_0 is listed):
fieldnames(m.fits)
fieldnames(m.params)
fieldnames(m.row_0)

% size and a value:
size(m.fits.pairs)
m.fits.pairs(1)     % for a single number, take its first point

% units live in the parallel units struct:
m.units.fits.pairs

% (no MATLAB on this machine — this block is **not run**: the `row_` prefix and the `units` struct layout come from the earlier run on the lab machine; per-report numbers are to be re-run there)
```

#### Julia

```julia
using MAT
m = matread("drag-detuning-3.mat")

keys(m)
# 6-element Vector{String}:
#  "fits"
#  "params"
#  "row_0"
#  "row_1"
#  "row_2"
#  "units"
m["fits"]
#   amp = 3×1 Matrix{Float64}:
#  -0.9699926472649698
#  -1.0128707156257413
#  -0.9917029284775616
#   amp_stderr = 3×1 Matrix{Float64}:
#  0.011095249837242785
#  0.011770966493732962
#  0.01380932382532855
#   centre = 3×1 Matrix{Float64}:
#  253704.36893300057
#  252090.7181202648
#  245694.5617121749
#   centre_stderr = 3×1 Matrix{Float64}:
#  4817.271072526896
#  2740.588292342551
#  1886.1217440153907
#   … the remaining 6 fields share one shape
m["params"]
#   chosen = 245694.5617121749
m["row_0"]
#   detuning = 31×1 Float64, first 2 rows: -1.5e6, -1.4e6
#   iq = 31×1 ComplexF64, first 2 rows: 0.40268943778655625 + 0.2735858522237158im, 0.4010101765369858 + 0.27498063685468743im
#   model = 31×1 Float64, first 2 rows: 0.886179182775028, 0.8766292719670586
#   p1 = 31×1 Float64, first 2 rows: 0.8647040988810646, 0.866127730797349
m["row_1"]
#   detuning = 31×1 Float64, first 2 rows: -607819.3973147757, -550384.479564924
#   iq = 31×1 ComplexF64, first 2 rows: 0.40453238812210707 + 0.2782356814572275im, 0.39442867906957985 + 0.2746930685887538im
#   model = 31×1 Float64, first 2 rows: 0.8799841852028474, 0.8641733451735655
#   p1 = 31×1 Float64, first 2 rows: 0.8858643909209256, 0.8505524854062563
# the remaining 1 sub-tables (row_2) share one shape: one scan each, same column names as row_0
m.units.fits
# Dict{String, Any} with 10 entries:
#   "centre_stderr" => "Hz"
#   "offset" => "1"
#   "row" => "1"
#   "amp" => "1"
#   "offset_stderr" => "1"
#   "fwhm" => "Hz"
#   "centre" => "Hz"
#   "pairs" => "1"
#   "amp_stderr" => "1"
#   "fwhm_stderr" => "Hz"
```

#### Rust

```rust
let mut archive = ZipArchive::new(File::open("drag-detuning-3.arrow.zip")?)?;

for index in 0..archive.len() {                     // one entry per iteration: one Feather file = one table
    let mut entry = archive.by_index(index)?;       // entry = ZipFile
    let name = entry.name().to_string();            // "data.arrow" / "fits.arrow" / …
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;                 // ZipFile has no Seek; read it into memory first
    let reader = FileReader::try_new(Cursor::new(bytes), None)?;   // Arrow IPC file

    let schema = reader.schema();                   // field name : type; the unit sits in field metadata
    let batches = reader.collect::<Result<Vec<_>, _>>()?;   // ← this is where batch comes from
    let batch = &batches[0];                        // record batch 0 = the whole table

    // it println!s the items below one by one (written out in full in §3.4)
}

// the tables in the zip (archive.len() entries):
//   ["0.arrow", "1.arrow", "2.arrow", "fits.arrow", "params.arrow"]
// what each iteration prints:
//   index 0 (0.arrow)
//     schema.fields(): detuning:Float64 unit=Hz, iq_re:Float64 unit=a.u., iq_im:Float64 unit=a.u., p1:Float64 unit=1, model:Float64 unit=1
//     batch row 0 of 31 rows (only the first 4 columns; the full list is on the fields line above): detuning=-1500000 iq_re=0.40268943778655625 iq_im=0.2735858522237158 p1=0.8647040988810646
//   index 3 (fits.arrow)
//     schema.fields(): row:Int64 unit=1, pairs:Float64 unit=1, centre:Float64 unit=Hz, centre_stderr:Float64 unit=Hz, fwhm:Float64 unit=Hz, fwhm_stderr:Float64 unit=Hz, amp:Float64 unit=1, amp_stderr:Float64 unit=1, offset:Float64 unit=1, offset_stderr:Float64 unit=1
//     batch row 0 of 3 rows (only the first 4 columns; the full list is on the fields line above): row=0 pairs=1 centre=253704.36893300057 centre_stderr=4817.271072526896
//   index 4 (params.arrow)
//     schema.fields(): chosen:Float64 unit=Hz
//     batch row 0 of 1 rows (= all columns): chosen=245694.5617121749
//   index …: the remaining 1 sub-tables (2.arrow … 2.arrow) share one shape — one scan each
```

#### R

```r
library(arrow)
d <- tempfile(); unzip("drag-detuning-3.arrow.zip", exdir = d)

entries
# [1] "0.arrow"      "1.arrow"      "2.arrow"      "fits.arrow"   "params.arrow"
r0 <- read_feather(file.path(d, "0.arrow"))
dim(r0)   # rows × cols
# [1] 31  5
names(r0)
# [1] "detuning" "iq_re"    "iq_im"    "p1"       "model"   
head(r0, 2)   # first 2 rows only
#   detuning     iq_re     iq_im        p1     model
# 1 -1500000 0.4026894 0.2735859 0.8647041 0.8861792
# 2 -1400000 0.4010102 0.2749806 0.8661277 0.8766293
sch$GetFieldByName("detuning")$metadata
# [1] "Hz"
r1 <- read_feather(file.path(d, "1.arrow"))
dim(r1)   # rows × cols
# [1] 31  5
names(r1)
# [1] "detuning" "iq_re"    "iq_im"    "p1"       "model"   
head(r1, 2)   # first 2 rows only
#    detuning     iq_re     iq_im        p1     model
# 1 -607819.4 0.4045324 0.2782357 0.8858644 0.8799842
# 2 -550384.5 0.3944287 0.2746931 0.8505525 0.8641733
sch$GetFieldByName("detuning")$metadata
# [1] "Hz"
fits <- read_feather(file.path(d, "fits.arrow"))
dim(fits)   # rows × cols
# [1]  3 10
names(fits)
#  [1] "row"           "pairs"         "centre"        "centre_stderr"
#  [5] "fwhm"          "fwhm_stderr"   "amp"           "amp_stderr"   
#  [9] "offset"        "offset_stderr"
fits
#   row pairs   centre centre_stderr      fwhm fwhm_stderr        amp amp_stderr
# 1   0     1 253704.4      4817.271 1086402.9    26352.26 -0.9699926 0.01109525
# 2   1     2 252090.7      2740.588  641372.0    15998.26 -1.0128707 0.01177097
# 3   2     4 245694.6      1886.122  369125.5    11026.21 -0.9917029 0.01380932
#      offset offset_stderr
# 1 0.9710953    0.01022071
# 2 1.0036515    0.01145493
# 3 0.9996607    0.01345541
sch$GetFieldByName("row")$metadata
# [1] "1"
params <- read_feather(file.path(d, "params.arrow"))
dim(params)   # rows × cols
# [1] 1 1
names(params)
# [1] "chosen"
params
#     chosen
# 1 245694.6
sch$GetFieldByName("chosen")$metadata
# [1] "Hz"
# the remaining 1 sub-tables (2.arrow) share one shape: one scan each
```

For humans, read the `.xlsx`: double-click it — one sheet per table (names as above), header cells are `<column> [unit]`, complex values sit in complex cells — see §3.6.

</details>

<!-- section4:end -->

## 5. Verification status

Every line above was produced by running the reader on the file the report page itself produced.
The files come from the 14 demo payloads (`plt/_export/` runs the report's own export script in
node — same script text the browser runs, only the CDN `import` is swapped for the npm package),
84 files in total (14 payloads × 6 formats).

| ecosystem | format | read with | verified on | result |
|---|---|---|---|---|
| Python 3.13.5 / numpy 2.5.3 | `.npz` | `np.load` | this machine, 2026-10-10 | 14/14 payloads: field titles + labelled names, `complex128`, `int64` refs |
| Julia 1.11.9 / MAT.jl 0.12.1 | `.mat` | `matread` | this machine, 2026-10-10 | 14/14: `ComplexF64`, `n×1` matrices, `row_<i>`, units struct |
| Rust 1.x / arrow 56.2.1 + zip 2 | `.arrow.zip` | `FileReader` | this machine, 2026-10-10 | 14/14: per-table schemas, `Int64` refs, `unit` field metadata, row counts |
| R 4.5.3 / arrow 25.0.0 | `.arrow.zip` | `read_feather` | this machine, 2026-10-10 | 14/14: names intact, values, `unit` metadata, complex recombined by hand |
| LibreOffice 25.2 + openpyxl | `.xlsx` | double-click / `load_workbook` | this machine, 2026-10-10 | 14/14: sheet list, `<column> [unit]` headers, `a+bj` complex cells |
| MATLAB R2020b (9.9) | `.mat` | `load` | lab machine, 2026-10-09 | `.mat` shape verified (`row_` prefix and `units` struct come from that run); per-report numbers to be re-run there |

The `s21 vs power` card needed the lab's `data/s21/fit_input.bin` (not in the repo), so a synthetic
21-line power scan was generated for the run above — same code path, different numbers.

---

## 6. Gotchas found while verifying

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
6. **Arrow has no complex type**: the writer splits every complex column into `<key>_re` and
   `<key>_im` (`iq` → `iq_re`, `iq_im`; `threshold` → `threshold_re`, `threshold_im`). R/Rust
   recombine on read. Everything else keeps its name exactly — Arrow is the only numeric container
   whose column names survive verbatim.
7. **Rust cannot read the `.npz` or `.mat`**: `ndarray-npy` rejects structured descriptors (with an
   explicit error), `npyz` rejects the field-title descriptor (`list entry must contain a string for
   id`), and `matfile` skips struct variables silently. That is why Rust is bound to `.arrow`.
8. **The Arrow writer must not use `new Table(schema, ...vectors)`** — apache-arrow's JS Table
   constructor treats a Vector as an iterable and flattens it into scalars, so the table silently
   comes back with **0 rows** (and the IPC file carries an empty batch). Build it as
   `new Table(new RecordBatch(schema, makeData({type: new Struct(fields), children: …})))`, which
   is what `EXPORT_SCRIPT` does.
9. **A complex column no longer complexifies its table**: with the structured descriptors `iq` is
   `complex128` while `tau`/`p1` stay `float64`.
10. **The `ref` column differs by container**: `int64` in `.npz` and `.arrow`, `double` in `.mat`,
    plain number in the text formats — all four carry the same values, only the storage differs.
11. **`R.matlab` rewrites `_` to `.`** in variable names (`row_0` → `m$row.0`) and flattens the
    nested `units` struct — one more reason R is bound to `.arrow`, not `.mat`.
