//! 报告内蕴数据：载荷的构建、序列化，与浏览器端的导出脚本。
//!
//! 每张报告把"喂进拟合的输入 + 拟合结果 + 标定中心"内蕴成一份 JSON（`<script type="application/json"
//! id="{div_id}-data">`）随页面交付；五枚 modebar 按钮在浏览器端把它转成 csv / txt / npz / mat / xlsx。
//! 数据只在这里定义一次，各报告只负责给出自己的列——格式差异是数据（一张表的列表），不是分支代码。
//!
//! **列存**：一张表 = 列定义 + 每列一个值数组；复数是 `[re, im]` 两元数组（与 `complex128`、
//! MATLAB complex double 的语义一致，导出时再拍成交错缓冲），缺值是 `null`。
//!
//! **表的分工**：`data` 逐点长表、`fits` 逐拟合一行的参数表、`params` 全局标量、`states` 标定
//! 中心；多行报告另有逐行明细子表 `row_<i>`，由 `fits` 里 `kind = "ref"` 的 `row` 列指向
//! （值就是那个 `i`，表名即 `row_` 加它）。行号是实数，所以五个导出格式都照数值列落格，
//! 表名也全是合法标识符（不必再净化）。
//!
//! 单位一律 SI（无量纲写 `1`），与各分析与报告的口径一致。

use crate::utils::heatmap::escape_html;
use plotly::Plot;
use serde_json::Value;

/// 一个值。
#[derive(Debug, Clone)]
pub(crate) enum Datum {
    /// 实数
    Real(f64),
    /// 复数 (实部, 虚部)
    Complex(f64, f64),
    /// 缺值（JSON 里写 null；JSON 没有 NaN）
    Missing,
}

/// 列的类型标签：导出端按它决定落格方式（只有复数需要特殊照料；指向子表的行号本身是实数）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    /// 实数
    Real,
    /// 复数
    Complex,
    /// 指向另一张表的行号（`row` = i ⇒ 表 `row_<i>`）
    Ref,
}

impl Kind {
    /// 输出的类型名（JSON 里的 `kind` 字段）。
    ///
    /// 返回值:
    ///     `"real"` / `"complex"` / `"ref"`
    fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Complex => "complex",
            Self::Ref => "ref",
        }
    }
}

/// 一列的定义。
#[derive(Debug, Clone)]
pub(crate) struct Column {
    key: String,
    unit: String,
    kind: Kind,
}

impl Column {
    /// 实数列。
    ///
    /// 形参:
    ///     key: 列名（csv 表头 / mat 字段 / npz 的列名清单）
    ///     unit: SI 单位；无量纲写 `1`
    ///
    /// 返回值:
    ///     列定义
    pub(crate) fn real(key: &str, unit: &str) -> Self {
        Self {
            key: key.to_string(),
            unit: unit.to_string(),
            kind: Kind::Real,
        }
    }

    /// 复数列。
    ///
    /// 形参:
    ///     key: 列名
    ///     unit: SI 单位；无量纲写 `1`
    ///
    /// 返回值:
    ///     列定义
    pub(crate) fn complex(key: &str, unit: &str) -> Self {
        Self {
            key: key.to_string(),
            unit: unit.to_string(),
            kind: Kind::Complex,
        }
    }

    /// 指向另一张表的行号列（`kind = "ref"`）：值是该行在报告里的行号，表名由 `row_`
    /// 加这个数得到。值仍是实数，五个导出格式都照数值列落格。
    ///
    /// 形参:
    ///     key: 列名
    ///     unit: 行号无量纲，写 `1`
    ///
    /// 返回值:
    ///     列定义
    pub(crate) fn reference(key: &str, unit: &str) -> Self {
        Self {
            key: key.to_string(),
            unit: unit.to_string(),
            kind: Kind::Ref,
        }
    }
}

/// 一张表：列定义 + 列存的值数组（`data[i]` 属于 `columns[i]`）。
#[derive(Debug, Clone)]
pub(crate) struct Table {
    name: String,
    columns: Vec<Column>,
    data: Vec<Vec<Datum>>,
}

impl Table {
    /// 空表：列定义决定列数，行由 `push` 逐条压入。
    ///
    /// 形参:
    ///     name: 表名（csv 的分块名 / npz 的数组名 / mat 的变量名 / xlsx 的 sheet 名）
    ///     columns: 列定义，顺序即输出顺序
    ///
    /// 返回值:
    ///     尚无行的表
    pub(crate) fn new(name: &str, columns: Vec<Column>) -> Self {
        Self {
            name: name.to_string(),
            data: columns.iter().map(|_| Vec::new()).collect(),
            columns,
        }
    }

    /// 压一行：`row` 的长度必须与列数一致（内部构建，不一致就是调用处写错了）。
    ///
    /// 形参:
    ///     row: 一行各列的单元格，顺序与列定义一致
    ///
    /// 返回值:
    ///     无
    pub(crate) fn push(&mut self, row: &[Datum]) {
        assert_eq!(
            row.len(),
            self.columns.len(),
            "table {}: row has {} cells but {} columns",
            self.name,
            row.len(),
            self.columns.len()
        );
        for (slot, cell) in self.data.iter_mut().zip(row.iter()) {
            slot.push(cell.clone());
        }
    }

    /// 一行的 JSON（列存）。
    ///
    /// 返回值:
    ///     `{"name": …, "columns": [...], "data": {列名: [值, …]}}`
    fn to_json(&self) -> Value {
        let columns: Vec<Value> = self
            .columns
            .iter()
            .map(|column| {
                serde_json::json!({
                    "key": column.key,
                    "unit": column.unit,
                    "kind": column.kind.as_str(),
                })
            })
            .collect();
        let mut data = serde_json::Map::new();
        for (column, values) in self.columns.iter().zip(self.data.iter()) {
            let cells: Vec<Value> = values.iter().map(datum_json).collect();
            data.insert(column.key.clone(), Value::Array(cells));
        }
        serde_json::json!({
            "name": self.name,
            "columns": columns,
            "data": Value::Object(data),
        })
    }
}

/// 参数名 → 单位：表里没有的名字按无量纲落 `1`（单位表必须写全，缺项说明这张表没列完）。
///
/// 形参:
///     units: `(参数名, 单位)` 表
///     name: 参数名
///
/// 返回值:
///     SI 单位串（无量纲写 `1`）
pub(crate) fn param_unit(units: &[(&'static str, &'static str)], name: &str) -> &'static str {
    match units.iter().find(|(key, _)| *key == name) {
        Some((_, unit)) => unit,
        None => "1",
    }
}

/// 一行拟合：先落调用方给的前导列（该表的行键，如 `row`/`z`/`pairs`），再按参数表落
/// `<参数名>` 与 `<参数名>_stderr` 两列。
///
/// 列定义与该行取值出自同一份参数表，次序必然一致。多行报告用首个成功拟合的列定义建表、
/// 其余各行取自己的取值即可——同一个模型，参数名与次序相同。
///
/// 形参:
///     params: 拟合出的参数表
///     units: 参数名 → 单位，见 [`param_unit`]
///     keys: 该行的前导列与取值，成对给出
///
/// 返回值:
///     (列定义, 该行各格)
pub(crate) fn fit_row(
    params: &lmfit::Parameters,
    units: &[(&'static str, &'static str)],
    keys: &[(Column, Datum)],
) -> (Vec<Column>, Vec<Datum>) {
    let mut columns: Vec<Column> = Vec::with_capacity(keys.len() + 2 * params.len());
    let mut row: Vec<Datum> = Vec::with_capacity(keys.len() + 2 * params.len());
    for (column, cell) in keys {
        columns.push(column.clone());
        row.push(cell.clone());
    }
    for parameter in params.iter() {
        let unit = param_unit(units, &parameter.name);
        columns.push(Column::real(&parameter.name, unit));
        columns.push(Column::real(&format!("{}_stderr", parameter.name), unit));
        row.push(Datum::Real(parameter.value));
        row.push(match parameter.stderr {
            Some(value) => Datum::Real(value),
            None => Datum::Missing,
        });
    }
    (columns, row)
}

/// 单元格 → JSON：实数与非有限值分别落成数字与 null（JSON 没有 NaN/inf），复数落成两元数组。
///
/// 形参:
///     datum: 单元格
///
/// 返回值:
///     JSON 值
fn datum_json(datum: &Datum) -> Value {
    match datum {
        Datum::Real(value) => match serde_json::Number::from_f64(*value) {
            Some(number) => Value::Number(number),
            None => Value::Null,
        },
        Datum::Complex(re, im) => {
            let (Some(re), Some(im)) = (
                serde_json::Number::from_f64(*re),
                serde_json::Number::from_f64(*im),
            ) else {
                return Value::Null;
            };
            Value::Array(vec![Value::Number(re), Value::Number(im)])
        }
        Datum::Missing => Value::Null,
    }
}

/// 报告生成时刻的三种写法——**同一次取钟**，所以三个字符串必然指同一瞬间。
#[derive(Debug, Clone)]
pub(crate) struct Stamp {
    /// 载荷里的 UTC（RFC3339，`Z` 结尾）
    pub(crate) utc: String,
    /// 载荷里的本地时刻（RFC3339，带偏移）——导出文件里用的就是它
    pub(crate) local: String,
    /// 页脚显示用的写法：`2026-10-09 17:03:48 timezone: +08:00`
    pub(crate) display: String,
}

impl Stamp {
    /// 取当前时刻的三个写法。
    ///
    /// 返回值:
    ///     时刻三连
    pub(crate) fn now() -> Self {
        let local = chrono::Local::now();
        Self {
            utc: local
                .with_timezone(&chrono::Utc)
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            local: local.to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
            display: format!(
                "{} timezone: {}",
                local.format("%Y-%m-%d %H:%M:%S"),
                local.format("%:z")
            ),
        }
    }
}

/// 页脚要显示的时刻：报告带载荷时用载荷那一份（与导出文件里同源同值），否则现取一次。
///
/// 形参:
///     payload: 该报告的载荷；无载荷的报告传 None
///
/// 返回值:
///     `2026-10-09 17:03:48 timezone: +08:00` 形式的时刻
pub(crate) fn footer_stamp(payload: Option<&Payload>) -> String {
    match payload {
        Some(data) => data.stamp().display.clone(),
        None => Stamp::now().display,
    }
}

/// 一份报告的载荷：元信息 + 若干张表。
#[derive(Debug, Clone)]
pub(crate) struct Payload {
    name: String,
    stamp: Stamp,
    tables: Vec<Table>,
}

impl Payload {
    /// 空载荷；时刻在构造时定下（此后不随序列化时刻变）。
    ///
    /// 形参:
    ///     name: 报告名（div_id），同时是下载文件的基名
    ///
    /// 返回值:
    ///     尚无表的载荷
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stamp: Stamp::now(),
            tables: Vec::new(),
        }
    }

    /// 本次生成时刻（三种写法）。
    ///
    /// 返回值:
    ///     时刻三连
    pub(crate) fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    /// 追加一张表；输出顺序即追加顺序（主表在前）。
    ///
    /// 形参:
    ///     table: 表
    ///
    /// 返回值:
    ///     无
    pub(crate) fn table(&mut self, table: Table) {
        self.tables.push(table);
    }

    /// 载荷 JSON：版本（crate 版本）、时间戳（UTC 与本地各一份）、报告名、各表。
    ///
    /// 返回值:
    ///     JSON 文本
    pub(crate) fn to_json(&self) -> String {
        let tables: Vec<Value> = self.tables.iter().map(Table::to_json).collect();
        serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp": { "utc": self.stamp.utc, "local": self.stamp.local },
            "name": self.name,
            "tables": tables,
        })
        .to_string()
    }

    /// 内蕴数据的 `<script>` 片段。
    ///
    /// JSON 里的字符串（列名、单位、表名）全部由本模块与调用方给出，不含 `</script>`，无需转义。
    ///
    /// 返回值:
    ///     `<script type="application/json" id="{name}-data">` 片段
    pub(crate) fn script(&self) -> String {
        format!(
            "<script type=\"application/json\" id=\"{}-data\">{}</script>",
            escape_html(&self.name),
            self.to_json()
        )
    }
}

/// `Plotly.newPlot` 初始化片段：**自带 config**（crate 的内联模板把 config 塞在 figure 里，
/// 塞不进带函数的 modebar 按钮，也不确定 plotly 是否采用）。
///
/// 图 div 由 [`crate::utils::panels::block`] 出——两边的 id 约定相同，都是 `{name}-plot`。
///
/// 形参:
///     plot: plotly 图对象（`to_json` 给 `{data, layout, config, frames}`，这里只取 data/layout）
///     name: 块名（图 div 的 id 是 `{name}-plot`）
///
/// 返回值:
///     一段 `Plotly.newPlot(...)` 脚本
pub(crate) fn plot_script(plot: &Plot, name: &str) -> String {
    let figure = plot.to_json();
    format!(
        "<script type=\"text/javascript\">\
         (function(){{var figure={figure};\
             Plotly.newPlot(\"{id}-plot\", figure.data, figure.layout, {{responsive:false, doubleClick:false, scrollZoom:true, modeBarButtonsToAdd: (window.qtoolExport ? window.qtoolExport.buttons : [])}});\
         }})();\
         </script>",
        id = escape_html(name),
    )
}

/// 浏览器端的导出脚本（每张报告内联一份，`window.qtoolExport` 幂等注册）。
///
/// 四个格式库走 CDN 的 ESM 动态 `import`（首次点击时才拉）；csv 与 txt 不需要任何库。载荷与
/// 五种格式之间的换算全部写在这里，报告侧只负责给列。
pub(crate) const EXPORT_SCRIPT: &str = r##"<script type="text/javascript">
(function () {
  if (window.qtoolExport) { return; }
  var QE = {};
  var CDN = {
    npyjs: "https://cdn.jsdelivr.net/npm/npyjs@1.2.0/+esm",
    fflate: "https://cdn.jsdelivr.net/npm/fflate@0.8.2/+esm",
    sheetjs: "https://cdn.jsdelivr.net/npm/xlsx@0.18.5/+esm"
  };

  /* 首次用到时才拉库；同一页里多个报告共用同一份（module 说明符按 URL 去重） */
  var libsPromise = null;
  function libs() {
    if (!libsPromise) {
      libsPromise = Promise.all([
        import(CDN.npyjs), import(CDN.fflate), import(CDN.sheetjs)
      ]).then(function (mods) {
        QE.libs = { dump: mods[0].dump, zipSync: mods[1].zipSync,
                    XLSX: mods[2].default || mods[2] };
      });
    }
    return libsPromise;
  }

  /* 载荷：从图 div 一路上溯，取最近一个带载荷的卡片里的 JSON 脚本。报告自己的图第一层就命中；
     嵌在宿主报告里的行面板（气泡里钉住的那份）自身不带载荷，命中的是宿主报告那一份——同一份
     数据只存一遍，按钮也就不会点了没反应。 */
  function payload(gd) {
    var node = gd.parentElement;
    while (node) {
      var found = node.querySelector('script[type="application/json"]');
      if (found) { return JSON.parse(found.textContent); }
      node = node.parentElement;
    }
    return null;
  }

  /* 单元格 → 文本：复数写字面量 re±|im|j（numpy、Python、MATLAB 三条都能读），其余原样 */
  function cellText(cell, kind) {
    if (cell === null || cell === undefined) { return ""; }
    if (kind === "complex") {
      var im = cell[1];
      return String(cell[0]) + (im < 0 ? "-" : "+") + String(Math.abs(im)) + "j";
    }
    return String(cell);
  }
  function tableRows(table) {
    return table.columns.map(function (c) { return table.data[c.key] || []; });
  }
  function rowCount(table) { return table.columns.length ? tableRows(table)[0].length : 0; }

  function header(payload) {
    var complexColumns = [];
    payload.tables.forEach(function (table) {
      table.columns.forEach(function (column) {
        if (column.kind === "complex") { complexColumns.push(table.name + "." + column.key); }
      });
    });
    /* 前言固定四行（版本行、复数列提示、块标记、列清单）——numpy 的 genfromtxt 要按行数
       跳过前言，行数恒定它才能写成一行；所以复数列提示行即使为空也照写 */
    var lines = ["# qtool " + payload.version + " | " + payload.timestamp.local +
                 " | " + payload.name,
                 "# complex columns: " + (complexColumns.length ? complexColumns.join(", ") : "(none)") +
                 "  (literal <re><sign><im>j; pandas: converters={<col>: complex})"];
    return lines.join("\n");
  }
  function textFile(payload, sep, ext, mime) {
    /* 主表（第一张）是**真 CSV**行；其余表整块注释掉——read_csv(comment="#") 与
       read_csv(comment_prefix="#") 因此只拿到主表，辅助表在文件里仍可读 */
    var blocks = payload.tables.map(function (table, index) {
      var units = table.columns.map(function (c) { return c.key + " [" + c.unit + "]"; });
      var body = [table.columns.map(function (c) { return c.key; }).join(sep)];
      var rows = tableRows(table);
      var count = rowCount(table);
      for (var i = 0; i < count; i++) {
        body.push(table.columns.map(function (c, j) {
          return cellText(rows[j][i], c.kind);
        }).join(sep));
      }
      if (index > 0) {
        body = body.map(function (line) { return "# " + line; });
      }
      return "# ---- " + table.name + " ----\n# columns: " + units.join(", ") + "\n" +
             body.join("\n");
    });
    /* 块之间不留空行：polars 的 CSV 读法会把空行读成一行 NaN（与全行注释叠在一起时直接报错），
       两种读法都靠 `#` 前缀认注释，不留空行就都干净 */
    return { ext: ext, mime: mime,
             blob: new Blob([header(payload) + "\n" + blocks.join("\n") + "\n"], { type: mime }) };
  }

  /* 一表一个数值矩阵：有复数列时按 [re,im] 交错（complex128），否则一列一个数
     （文本/缺值落 NaN）；shape 恒为 [行数, 列数] */
  function matrix(table) {
    var rows = tableRows(table);
    var count = rowCount(table);
    var complex = table.columns.some(function (c) { return c.kind === "complex"; });
    var flat = [];
    for (var i = 0; i < count; i++) {
      for (var j = 0; j < table.columns.length; j++) {
        var value = rows[j][i];
        var pair = (table.columns[j].kind === "complex" && value !== null && value !== undefined)
          ? value : (typeof value === "number" ? [value, 0] : [NaN, 0]);
        if (complex) { flat.push(pair[0], pair[1]); } else { flat.push(pair[0]); }
      }
    }
    return { values: Float64Array.from(flat), shape: [count, table.columns.length],
             complex: complex, dtype: complex ? "c16" : "f8" };
  }

  /* MAT v5（level 5）二进制：一个变量一个 miMATRIX，元素是 double（有复数列时按复数落）。
     编码按 MathWorks 的格式文档，字段次序与 scipy.io.savemat 的产物逐字节一致：
     128 字节头（文本 + 8 字节子系统指针 + 版本 00 01 + 'IM'）之后，每个变量 = 标签(miMATRIX)
     + 数组标志(miUINT32，低位=类别 6、0x0800 位=复数) + 维度(miINT32，先行后列) + 变量名(miINT8)
     + pr/pi(miDOUBLE)，各子元素的数据按 8 字节对齐补齐。 */
  function u32(value) {
    var bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setUint32(0, value, true);
    return bytes;
  }
  function i32(value) {
    var bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setInt32(0, value, true);
    return bytes;
  }
  function f64(values) {
    var bytes = new Uint8Array(8 * values.length);
    var view = new DataView(bytes.buffer);
    for (var i = 0; i < values.length; i++) { view.setFloat64(8 * i, values[i], true); }
    return bytes;
  }
  function matTag(type, size) {
    var bytes = new Uint8Array(8);
    var view = new DataView(bytes.buffer);
    view.setUint32(0, type, true);
    view.setUint32(4, size, true);
    return bytes;
  }
  function concat(parts) {
    var total = 0;
    parts.forEach(function (part) { total += part.length; });
    var out = new Uint8Array(total);
    var offset = 0;
    parts.forEach(function (part) { out.set(part, offset); offset += part.length; });
    return out;
  }
  function pad8(bytes) {
    var rest = bytes.length % 8;
    return rest === 0 ? bytes : concat([bytes, new Uint8Array(8 - rest)]);
  }
  /* 矩阵按**列主序**装（MATLAB 的存储顺序）：matrix() 给的是行主序（numpy 的口径），
     这里按列遍历取值，两种顺序各自出现在该出现的地方 */
  function matVariable(name, box) {
    var nameBytes = new TextEncoder().encode(name);
    var body = [
      concat([matTag(6, 8), u32(box.complex ? 0x0806 : 0x0006), u32(0)]),
      concat([matTag(5, 8), i32(box.shape[0]), i32(box.shape[1])]),
      concat([matTag(1, nameBytes.length), pad8(nameBytes)])
    ];
    var rows = box.shape[0], cols = box.shape[1];
    var re = [], im = [];
    for (var j = 0; j < cols; j++) {
      for (var i = 0; i < rows; i++) {
        var at = box.complex ? 2 * (i * cols + j) : i * cols + j;
        re.push(box.values[at]);
        if (box.complex) { im.push(box.values[at + 1]); }
      }
    }
    body.push(concat([matTag(9, 8 * re.length), f64(re)]));
    if (box.complex) { body.push(concat([matTag(9, 8 * im.length), f64(im)])); }
    var payload = concat(body);
    return concat([matTag(14, payload.length), payload]);
  }
  function matBytes(payload) {
    var header = new Uint8Array(128);
    var text = new TextEncoder().encode("MATLAB 5.0 MAT-file, Platform: JS, Created by qtool");
    header.set(text.subarray(0, 116), 0);
    header[124] = 0x00; header[125] = 0x01;               /* 版本 0x0100，固定大头字节序 */
    header[126] = 0x49; header[127] = 0x4d;               /* 'IM'：小端 */
    var parts = [header];
    payload.tables.forEach(function (table) {
      parts.push(matVariable(varName(table.name), matrix(table)));
    });
    return concat(parts);
  }

  function sheetName(name) { return name.replace(/[\[\]:*?\/\\]/g, "_").slice(0, 31); }
  function varName(name) { return name.replace(/[^0-9A-Za-z_]/g, "_").replace(/^([0-9])/, "v$1"); }

  QE.format = {
    csv: function (payload) { return textFile(payload, ",", "csv", "text/csv"); },
    txt: function (payload) { return textFile(payload, "\t", "txt", "text/plain"); },
    npz: function (payload) {
      /* zipSync 只认 Uint8Array 叶子：numpy 的字节要转一层，列名清单要自己编码 */
      var entries = {};
      var meta = [];
      payload.tables.forEach(function (table) {
        var box = matrix(table);
        entries[table.name + ".npy"] =
          new Uint8Array(QE.libs.dump(box.values, box.shape, { dtype: box.dtype }));
        meta.push(table.name + ": " + table.columns.map(function (c) {
          return c.key + "[" + c.unit + "," + c.kind + "]";
        }).join(", "));
      });
      entries["columns.txt"] = new TextEncoder().encode(meta.join("\n") + "\n");
      return { ext: "npz", mime: "application/zip",
               blob: new Blob([QE.libs.zipSync(entries)], { type: "application/zip" }) };
    },
    mat: function (payload) {
      return { ext: "mat", mime: "application/octet-stream",
               blob: new Blob([matBytes(payload)], { type: "application/octet-stream" }) };
    },
    xlsx: function (payload) {
      var book = QE.libs.XLSX.utils.book_new();
      payload.tables.forEach(function (table) {
        /* 单行表头 `<列名> [<单位>]`：再插一行纯列名会让解析库把单位行当成数据 */
        var sheet = [table.columns.map(function (c) { return c.key + " [" + c.unit + "]"; })];
        var rows = tableRows(table);
        var count = rowCount(table);
        for (var i = 0; i < count; i++) {
          /* 实数落数字格（Excel 里还能继续算），复数与文本落字符串格 */
          sheet.push(table.columns.map(function (c, j) {
            var value = rows[j][i];
            return (typeof value === "number") ? value : cellText(value, c.kind);
          }));
        }
        QE.libs.XLSX.utils.book_append_sheet(book, QE.libs.XLSX.utils.aoa_to_sheet(sheet),
                                             sheetName(table.name));
      });
      var bytes = QE.libs.XLSX.write(book, { bookType: "xlsx", type: "array" });
      return { ext: "xlsx",
               mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
               blob: new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }) };
    }
  };

  QE.payload = payload;

  QE.download = function (gd, fmt) {
    var data = payload(gd);
    if (!data) { return; }
    var ready = (fmt === "csv" || fmt === "txt") ? Promise.resolve() : libs();
    ready.then(function () {
      var file = QE.format[fmt](data);
      var url = URL.createObjectURL(file.blob);
      var link = document.createElement("a");
      link.href = url;
      link.download = data.name + "." + file.ext;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 0);
    });
  };

  /* 五枚按钮：下载箭头 + 各自的文件记号（自绘，不用任何商标 logo） */
  var ICON = {
    csv: "M11.4 2.6h1.2v8.2h-1.2zM12 14.6 6.4 9.3h11.2zM6 16.2h5.2v1.2H6zM12.8 16.2H18v1.2h-5.2zM6 18.6h4v1.2H6zM11.2 18.6H18v1.2h-6.8zM6 21h6v1.2H6z",
    txt: "M11.4 2.6h1.2v8.2h-1.2zM12 14.6 6.4 9.3h11.2zM6 16.2h12v1.2H6zM6 18.6h12v1.2H6zM6 21h12v1.2H6z",
    npz: "M11.4 2.6h1.2v8.2h-1.2zM12 14.6 6.4 9.3h11.2zM6 16.2h3.2v2.4H6zM10.4 16.2h3.2v2.4h-3.2zM14.8 16.2H18v2.4h-3.2zM6 19.8h3.2V22H6zM10.4 19.8h3.2V22h-3.2zM14.8 19.8H18V22h-3.2z",
    mat: "M11.4 2.6h1.2v8.2h-1.2zM12 14.6 6.4 9.3h11.2zM5.6 16h1.2v6H5.6zM5.6 16h2.8v1.2H5.6zM5.6 20.8h2.8V22H5.6zM17.2 16h1.2v6h-1.2zM15.6 16h2.8v1.2h-2.8zM15.6 20.8h2.8V22h-2.8zM9.6 17.4h2v2H9.6zM12.9 17.4h2v2h-2z",
    xlsx: "M11.4 2.6h1.2v8.2h-1.2zM12 14.6 6.4 9.3h11.2zM6 16h12v1.1H6zM6 20.9h12V22H6zM6 16h1.1v6H6zM16.9 16H18v6h-1.1zM10.6 16h1.1v6h-1.1zM14 16h1.1v6H14z"
  };
  QE.buttons = ["csv", "txt", "npz", "mat", "xlsx"].map(function (fmt) {
    return { name: fmt, title: "download " + fmt, icon: { width: 24, height: 24, path: ICON[fmt] },
             click: function (gd) { QE.download(gd, fmt); } };
  });
  window.qtoolExport = QE;
})();
</script>"##;
