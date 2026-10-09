//! 通用二维图模板（feature = "plot"）：热图 + 右侧/下方边线图。
//!
//! 本模块是 crate 内部件（`utils` 下的模块都不作为外部接口），只服务仓库内的报告。
//!
//! 二维图占据左上区域；**下方**边线图显示点击点所在的 **row**——用该行自己的采样对
//! `(x, z[row][·])` 连成线（行内是连续的一段扫描，并集轴上的空格不进这里），**右侧**边线图
//! 显示点击点所在的 **col**（`z[·][col]` vs y，行没覆盖的频率上留空），两者带 continuous
//! error bar（`error_y.type = "data"`，逐点数组）。初始显示中间一行 / 中间一列。
//!
//! 热图**每行一条 trace**：plotly 的 x 是 per-trace 的，各行因此可以自带 x 坐标——动窗
//! 扫描（每行只扫一小段）按 (x, 值) 成对画出来就是连续的一条带，没覆盖的频率上整行不着色。
//! 稠密矩阵那套（并集列轴 + NaN 空格）只服务边线切片与点击取列，两者互不干扰。
//!
//! 交互靠片段内嵌的一小段 JS：`plotly_click` → 取最近的行列号 → `Plotly.restyle`
//! 更新两条边线；x/y 轴用 `matches` 联动缩放。宿主页面仍需自行加载 plotly.js。

use plotly::common::{
    AxisSide, ColorBar, ColorScale, ColorScaleElement, ErrorData, ErrorType, ExponentFormat, Font,
    Marker, Mode, Title,
};
use plotly::configuration::{Configuration, DoubleClick};
use plotly::layout::{Axis, Layout, Margin, TicksDirection};
use plotly::{Plot, Scatter, Trace};
use serde::Serialize;

/// 热图色板。
#[derive(Clone, Copy)]
pub(crate) enum Palette {
    /// matplotlib 的 `turbo`（蓝→青→绿→黄→橙→红，两端不发暗）。
    Turbo,
    /// ColorBrewer RdBu-11，蓝（低）→ 红（高）。plotly.js 内置的同名色板是另一套 6 档
    /// 配色，必须显式给出色标（见 [`rdbu`]）。
    RdBu,
}

/// 把等距色标展开成 plotly 的显式色板：位置按序号均分 [0, 1]。
///
/// plotly.py 在交给 plotly.js 之前就是这么展开色板名字的，所以只有照它的色标表写，
/// 才能和 Python 端渲染出同样的颜色（plotly.js 内置的几个同名色板档位/配色并不一样）。
fn even_scale(stops: &[&str]) -> ColorScale {
    let last = (stops.len() - 1) as f64;
    ColorScale::Vector(
        stops
            .iter()
            .enumerate()
            .map(|(index, color)| ColorScaleElement(index as f64 / last, (*color).to_string()))
            .collect(),
    )
}

/// matplotlib 的 `turbo`：plotly 没有内置，取 matplotlib `colormaps['turbo']` 在 0～1 上
/// 等距的 11 个采样点。
fn turbo() -> ColorScale {
    even_scale(&[
        "rgb(48,18,59)",
        "rgb(69,89,203)",
        "rgb(62,155,254)",
        "rgb(25,213,205)",
        "rgb(70,248,132)",
        "rgb(164,252,60)",
        "rgb(225,221,55)",
        "rgb(254,164,49)",
        "rgb(240,91,18)",
        "rgb(195,37,3)",
        "rgb(122,4,3)",
    ])
}

/// ColorBrewer `RdBu`-11，蓝（低）→ 红（高）。
///
/// 色标取自 plotly.py 的 `RdBu`（plotly.js 内置的同名 `RdBu` 是另一套 6 档配色：
/// `rgb(5,10,172)`→`rgb(178,10,28)`、灰中间，必须显式给出）；方向按"红=高"的习惯
/// 相对 plotly.py 的排列翻转。
fn rdbu() -> ColorScale {
    even_scale(&[
        "rgb(5,48,97)",
        "rgb(33,102,172)",
        "rgb(67,147,195)",
        "rgb(146,197,222)",
        "rgb(209,229,240)",
        "rgb(247,247,247)",
        "rgb(253,219,199)",
        "rgb(244,165,130)",
        "rgb(214,96,77)",
        "rgb(178,24,43)",
        "rgb(103,0,31)",
    ])
}

/// 一张热图能切换的一种看法：取值矩阵 + 自己的误差矩阵、色板与色标标题。
///
/// 几个视图共用同一对轴，切换只换这一层的取值与配色，所以色标范围、
/// 色板、色标标题都随视图走。
pub(crate) struct View<'a> {
    /// 选项文字，也是图名行里 " vs xxx" 前面的那半句。
    pub(crate) name: &'a str,
    /// 色标（colorbar）标题，可带单位。
    pub(crate) value_title: &'a str,
    /// 取值矩阵 `z[row][col]`。
    pub(crate) z: &'a [Vec<f64>],
    /// 与 `z` 同形的误差矩阵（边线图的 error bar）。
    pub(crate) z_err: Option<&'a [Vec<f64>]>,
    pub(crate) palette: Palette,
}

/// 二维图数据：行轴 / 列轴 + 一个或多个可切换的看法。
///
/// [`Grid2d::views`] 里**首个即默认显示的那个**（下拉框的初始选项），不另设默认下标；
/// 只有一个视图时不出下拉框，图名行就是一行文字。
pub(crate) struct Grid2d<'a> {
    /// 列轴（热图 x）。
    pub(crate) x: &'a [f64],
    /// 行轴（热图 y）。
    pub(crate) y: &'a [f64],
    /// 可切换的看法，首个即默认。
    pub(crate) views: &'a [View<'a>],
    pub(crate) x_title: &'a str,
    pub(crate) y_title: &'a str,
    /// 图名行里 " vs " 后面那半句（"power" / "Z"）。
    pub(crate) value_axis: &'a str,
}

/// 色板 → plotly 的显式色标。
fn scale_of(palette: Palette) -> ColorScale {
    match palette {
        Palette::Turbo => turbo(),
        Palette::RdBu => rdbu(),
    }
}

/// 色标的 JSON：切视图时脚本拿它 restyle `colorscale`。序列化失败给空数组（少一档颜色），
/// 不让整块图跟着崩掉——与 [`RowBand::to_json`] 同一个取舍。
fn scale_json(palette: Palette) -> String {
    match serde_json::to_string(&scale_of(palette)) {
        Ok(json) => json,
        Err(_invalid) => String::from("[]"),
    }
}

/// 某视图在 `z[row][col]` 的取值；越界给 NaN。
fn view_value(view: &View<'_>, row: usize, column: usize) -> f64 {
    match view.z.get(row).and_then(|values| values.get(column)) {
        Some(value) => *value,
        None => f64::NAN,
    }
}

/// 某视图全部有限取值的范围；一个有限值都没有时给 `(0, 1)`（plotly 不收 ±inf 当范围）。
fn view_range(view: &View<'_>) -> (f64, f64) {
    let mut range = (f64::INFINITY, f64::NEG_INFINITY);
    for row in view.z {
        for value in row {
            match value.is_finite() {
                true => {
                    range.0 = range.0.min(*value);
                    range.1 = range.1.max(*value);
                }
                false => {}
            }
        }
    }
    match range.0.is_finite() && range.1.is_finite() {
        true => range,
        false => (0.0, 1.0),
    }
}

const LINE_COLOR: &str = "#d62728";

/// 热图里的一行：plotly 的 x 是 per-trace 的（逐行频率轴只能用多条 trace 表达），而 Rust
/// 的 plotly crate 没有 `y0` / `dy` 字段（单点 `y` 会退化成默认厚度、铺满整条轴），所以
/// 这一条按 plotly.js 的 schema 直接发 JSON；色板与 colorbar 复用 crate 的类型。
#[derive(Clone, Serialize)]
struct RowBand {
    /// plotly trace 类型
    #[serde(rename = "type")]
    kind: &'static str,
    /// 该行自己的 x 坐标（动窗扫描各行不同，共用轴时就是整条轴）
    x: Vec<f64>,
    /// 这一行在 y 轴上的中心
    y0: f64,
    /// 格子厚度（见 [`band_thickness`]）
    dy: f64,
    /// 单行取值，与 `x` 等长（plotly 收 `z[row][col]`，这里只有一行）
    z: Vec<Vec<f64>>,
    colorscale: ColorScale,
    zmin: f64,
    zmax: f64,
    #[serde(rename = "showscale")]
    show_scale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    colorbar: Option<ColorBar>,
    /// 热图不弹默认的坐标浮标：读数由气泡（整行面板）承担。"none" 只是不显示内容，
    /// hover / click 事件照常派发，气泡和"点一下取该行该列到边线图"都依赖它们
    /// （"skip" 会把整条 trace 从 hover 里摘掉）。
    #[serde(rename = "hoverinfo")]
    hover_info: &'static str,
}

impl Trace for RowBand {
    /// trace 的 JSON；字段全是实数，序列化不会失败——真出错时给一个空对象，页面上少画
    /// 一行色带，而不是整块图跟着崩掉。
    fn to_json(&self) -> String {
        match serde_json::to_string(self) {
            Ok(json) => json,
            Err(_invalid) => String::from("{}"),
        }
    }
}

/// 一行格子的厚度：本行中心向两侧各伸出半个行距（首末行只有一侧有邻居，取单侧间距），
/// 与 plotly 单 trace 时算格子外延的规则一致。
///
/// 形参:
///     y: 行中心轴 (n,)，须非空（调用方按行号调用）
///     index: 行号
///
/// 返回值:
///     该行的格子厚度；只有一行时退回 plotly 的默认厚度 1
fn band_thickness(y: &[f64], index: usize) -> f64 {
    let last = y.len() - 1;
    match index == 0 {
        true => match last == 0 {
            true => 1.0,
            false => y[1] - y[0],
        },
        false => match index == last {
            true => y[last] - y[last - 1],
            false => 0.5 * (y[index + 1] - y[index - 1]),
        },
    }
}

/// 渲染热图 + 右/下边线图的 HTML 片段（自包含 div，宿主需提供 plotly.js）。
///
/// 形参:
///     grid: 二维数据与轴标题
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一，且须是合法 id）
///
/// 返回值:
///     `<div class="qtool-2d">` 片段（含内嵌交互脚本）
pub(crate) fn heatmap(grid: &Grid2d<'_>, div_id: &str) -> String {
    let view = match grid.views.first() {
        Some(view) => view,
        None => return String::new(),
    };
    let rows = grid.y.len();
    let cols = grid.x.len();
    // 初始状态：中间一行 / 中间一列
    let mid_row = match rows > 0 {
        true => rows / 2,
        false => 0,
    };
    let mid_col = match cols > 0 {
        true => cols / 2,
        false => 0,
    };
    let (row_x, row_values, row_err) = row_slice(grid.x, view, mid_row);
    let col_values = slice_col(view.z, mid_col, rows);
    let col_err = match view.z_err {
        Some(matrix) => slice_col(matrix, mid_col, rows),
        None => vec![f64::NAN; rows],
    };

    let mut plot = Plot::new();
    // 边线两条先发：点击取行列的 JS 按固定下标 restyle 它们（热图的色带 trace 从 2 起）
    plot.add_trace(row_trace(
        row_x,
        row_values,
        row_err,
        view.value_title,
        "x2",
        "y2",
    ));
    plot.add_trace(col_trace(
        col_values,
        col_err,
        grid.y.to_vec(),
        view.value_title,
        "x3",
        "y3",
    ));

    // 各行整理成 (行号, 该行的列号)：动窗扫描的相邻行频率交错，落进并集轴后非空格子本来就
    // 不相邻——按列号筛出来，格子才回到连续的一条带。列集取各视图的**并集**：不假设几个
    // 视图的 NaN 位置一致，某个视图在某格没值时由脚本发 null（plotly 视为不画）。
    let mut bands: Vec<(usize, Vec<usize>)> = Vec::new();
    for row in 0..rows {
        let mut columns: Vec<usize> = Vec::new();
        for column in 0..cols {
            let finite = grid
                .views
                .iter()
                .any(|view| view_value(view, row, column).is_finite());
            match finite {
                true => columns.push(column),
                false => {}
            }
        }
        // 整行在哪个视图里都没有有限值：这行什么都不画（plotly 也不收空的 z）
        match columns.is_empty() {
            true => {}
            false => bands.push((row, columns)),
        }
    }
    // 色标范围按视图各自取自整张图，不能边发边算——否则靠前的行只有自己那一段参与定标，
    // 同一颜色在不同行代表不同的值。
    let ranges: Vec<(f64, f64)> = grid.views.iter().map(view_range).collect();
    for (position, (index, columns)) in bands.iter().enumerate() {
        plot.add_trace(Box::new(RowBand {
            kind: "heatmap",
            x: columns.iter().map(|column| grid.x[*column]).collect(),
            y0: grid.y[*index],
            dy: band_thickness(grid.y, *index),
            z: vec![columns
                .iter()
                .map(|column| view_value(view, *index, *column))
                .collect()],
            colorscale: scale_of(view.palette),
            zmin: ranges[0].0,
            zmax: ranges[0].1,
            show_scale: position == 0,
            colorbar: match position == 0 {
                true => Some(
                    ColorBar::new()
                        .title(Title::with_text(view.value_title))
                        .exponent_format(ExponentFormat::SI),
                ),
                false => None,
            },
            hover_info: "none",
        }));
    }
    plot.set_layout(layout(grid, view.value_title));
    plot.set_configuration(interactive_config());
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = crate::utils::data::plot_script(&plot, div_id);
    // 色标范围的 min/max 与 auto 排在 `newPlot` 之后：plotly 建图时会把目标 div 清空，先放进去
    // 会被抹掉。空白 = 自动（各自回落到该视图算出来的范围），输入框的值只由用户写、脚本从不回写；
    // auto 按钮不是另一套状态，只是"两个框一起清空"的快捷方式。
    // max 在上：色标本身就是上端取 max，两行按色标的朝向排。
    let range_html = format!(
        "<div class=\"qtool-2d-range\" id=\"{div_id}-range\">\
         <label for=\"{div_id}-zmax\">max</label>\
         <input id=\"{div_id}-zmax\" type=\"number\" step=\"any\" placeholder=\"auto\" \
         title=\"color range max — empty = auto\">\
         <label for=\"{div_id}-zmin\">min</label>\
         <input id=\"{div_id}-zmin\" type=\"number\" step=\"any\" placeholder=\"auto\" \
         title=\"color range min — empty = auto\">\
         <button id=\"{div_id}-auto\" type=\"button\">auto</button>\
         </div>"
    );
    let plot_block = crate::utils::panels::block(
        div_id,
        "qtool-2d-plot",
        (5.0, 4.0),
        &format!("{plot_html}{range_html}"),
    );
    let caption_css = title_bar_style();
    // 图名行：多视图时给一个下拉框（与线图同一套外观），单视图时就是一行文字
    let title = match grid.views.len() > 1 {
        true => {
            let options: Vec<String> = grid
                .views
                .iter()
                .map(|view| format!("<option value=\"{}\">{}</option>", view.name, view.name))
                .collect();
            format!(
                "<select id=\"{div_id}-view\">{}</select><span> vs {}</span>",
                options.join(""),
                escape_html(grid.value_axis)
            )
        }
        false => format!(
            "{} vs {}",
            escape_html(view.name),
            escape_html(grid.value_axis)
        ),
    };
    let json_x = json_array(grid.x);
    let json_y = json_array(grid.y);
    // 每个视图一整份矩阵：切视图时脚本拿它重铺色带，也拿它算点击取到的行与列
    let json_views: Vec<String> = grid
        .views
        .iter()
        .map(|view| {
            let range = view_range(view);
            let err = match view.z_err {
                Some(matrix) => json_matrix(matrix),
                None => "null".to_string(),
            };
            format!(
                "\"{}\":{{\"title\":\"{}\",\"scale\":{},\"zmin\":{},\"zmax\":{},\"z\":{},\"e\":{}}}",
                view.name,
                view.value_title,
                scale_json(view.palette),
                range.0,
                range.1,
                json_matrix(view.z),
                err
            )
        })
        .collect();
    let json_order: Vec<String> = grid
        .views
        .iter()
        .map(|view| format!("\"{}\"", view.name))
        .collect();

    format!(
        r#"<div class="qtool-2d" id="{div_id}">
{STYLE}{caption_css}
<div class="qtool-2d-cap">{title}</div>
{plot_block}
<script>
(function () {{
  var X = {json_x}, Y = {json_y};
  var VIEWS = {{{views}}};
  var ORDER = [{order}];
  var gd = document.getElementById("{plot_div_id}");
  var chooser = document.getElementById("{div_id}-view");
  if (!window.Plotly || !gd) {{ return; }}
  var current = chooser ? chooser.value : ORDER[0];
  // 色标范围的两个输入框：空 = 自动（回落到该视图算出来的范围）。逐端独立，只填一端就只覆盖那端。
  // 空串必须显式判——Number("") 是 0，忘了判就等于把范围钉在 0 上
  var rangeBox = document.getElementById("{div_id}-range");
  var zminInput = document.getElementById("{div_id}-zmin");
  var zmaxInput = document.getElementById("{div_id}-zmax");
  var autoButton = document.getElementById("{div_id}-auto");
  var bound = function (field, auto) {{
    return field && field.value !== "" ? Number(field.value) : auto;
  }};
  // 回自动 = 两个框一起清空："空 = 自动"只有这一套语义，auto 按钮与切视图共用它
  var clearBounds = function () {{
    if (zminInput) {{ zminInput.value = ""; }}
    if (zmaxInput) {{ zmaxInput.value = ""; }}
  }};
  // 输入框与 placeholder 共用的一套数字写法：大数/小数走科学计数，其余留 5 位有效数字。
  // 位数是被面板的宽度定的——面板宽 = 右截线图那一列，数字再长就被裁掉了
  var formatNumber = function (value) {{
    return Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-3)
      ? value.toExponential(2)
      : String(+value.toPrecision(5));
  }};
  // 把当前视图的自动范围写进 placeholder：输入框空着时，用户看到的就是"自动是多少"
  var seed = function (view) {{
    if (zminInput) {{ zminInput.placeholder = formatNumber(view.zmin); }}
    if (zmaxInput) {{ zmaxInput.placeholder = formatNumber(view.zmax); }}
  }};
  // 滚轮微调：按整条色标范围（|max − min|）的 10% 增减（上滚增大、下滚减小），挡掉浏览器默认
  // 的 ±1。两个框共用同一把尺子——范围宽就粗调、范围窄就细调，两端的步长才不会各走各的。
  // 只在输入框已聚焦时接管——与浏览器自身触发滚轮改值的条件一致，免得光标从面板上路过时
  // 把页面滚动吃掉。输入框空着（自动）时以该视图的自动值参与算范围，滚一下即成为显式值
  var scrollTune = function (field, key) {{
    if (!field) {{ return; }}
    field.addEventListener(
      "wheel",
      function (event) {{
        var view = VIEWS[current];
        if (!view || document.activeElement !== field || event.deltaY === 0) {{ return; }}
        event.preventDefault();
        // 先按改动前的两个框算范围，再动这一个框
        var step = Math.abs(bound(zmaxInput, view.zmax) - bound(zminInput, view.zmin)) * 0.1;
        // 加法而非乘法：值为负时 base * 1.1 反而是减小，±step 才是与符号无关的"增减"
        field.value = formatNumber(bound(field, view[key]) + Math.sign(-event.deltaY) * step);
        paint();
      }},
      {{ passive: false }}
    );
  }};
  // 面板 = 2×2 布局右下那一格：宽取右截线图那一列、高取下截线图那一行，四条边因此与两张截线图
  // 的轴框重合。绘图区盒子只能取 plotly 排完版后的 `_fullLayout._size`——它是**解析后**的坐标
  // （轴的 auto_margin 撑开的边距、colorbar 从右侧吃掉的那几十 px 都已算进去）；`_fullLayout.margin`
  // 只是配置值，照它算会横向差出整整一条 colorbar 的宽度
  var place = function () {{
    if (!rangeBox || !gd._fullLayout) {{ return; }}
    var size = gd._fullLayout._size;
    rangeBox.style.left = (size.l + size.w * {x_left}) + "px";
    rangeBox.style.top = (size.t + size.h * {y_top}) + "px";
    rangeBox.style.width = (size.w * {x_span}) + "px";
    rangeBox.style.height = (size.h * {y_span}) + "px";
  }};
  // 当前选中：行、列各一个下标（初始是中间那行/那列，与上面发出去的边线一致）
  var picked = {{ row: {mid_row}, col: {mid_col} }};
  var nearest = function (arr, value) {{
    var best = 0, bestDist = Infinity;
    for (var i = 0; i < arr.length; i++) {{
      var d = Math.abs(arr[i] - value);
      if (d < bestDist) {{ bestDist = d; best = i; }}
    }}
    return best;
  }};
  // 色带 trace 的下标按类型认，不按下标：边线两条也在这个 figure 里，首条色带并非 0
  var bandIndices = function () {{
    var out = [];
    for (var i = 0; i < gd.data.length; i++) {{
      if (gd.data[i].type === "heatmap") {{ out.push(i); }}
    }}
    return out;
  }};
  var repeat = function (value, count) {{
    var out = [];
    for (var i = 0; i < count; i++) {{ out.push(value); }}
    return out;
  }};
  // 某一行在色带里的格子列号：色带只带该行自己的 x，换成整矩阵的列号才取得到值
  var columnsOf = function (row, index, rows) {{
    var at = rows.indexOf(row);
    if (at < 0) {{ return null; }}
    return gd.data[index[at]].x.map(function (value) {{ return nearest(X, value); }});
  }};
  // 把当前视图与当前选中的行、列画上去：点击与切视图走同一条路径
  var paint = function () {{
    var view = VIEWS[current];
    if (!view) {{ return; }}
    seed(view);
    var index = bandIndices();
    var rows = index.map(function (i) {{ return nearest(Y, gd.data[i].y0); }});
    Plotly.restyle(
      gd,
      {{
        "z": index.map(function (i, k) {{
          var cells = columnsOf(rows[k], index, rows);
          return [cells.map(function (c) {{ return view.z[rows[k]][c]; }})];
        }}),
        "colorscale": repeat(view.scale, index.length),
        "zmin": repeat(bound(zminInput, view.zmin), index.length),
        "zmax": repeat(bound(zmaxInput, view.zmax), index.length)
      }},
      index
    );
    if (index.length) {{ Plotly.restyle(gd, {{ "colorbar.title.text": view.title }}, [index[0]]); }}
    // 横截图用选中那行自己的格子（并集轴上的空格不进这里）；纵截图取选中那一列跨所有行
    var columns = columnsOf(picked.row, index, rows);
    if (columns) {{
      Plotly.restyle(
        gd,
        {{
          "x": [columns.map(function (c) {{ return X[c]; }})],
          "y": [columns.map(function (c) {{ return view.z[picked.row][c]; }})],
          "error_y.array": [view.e ? columns.map(function (c) {{ return view.e[picked.row][c]; }}) : null]
        }},
        [0]
      );
    }}
    Plotly.restyle(
      gd,
      {{
        "x": [view.z.map(function (r) {{ return r[picked.col]; }})],
        "error_x.array": [view.e ? view.e.map(function (r) {{ return r[picked.col]; }}) : null]
      }},
      [1]
    );
  }};
  gd.on("plotly_click", function (ev) {{
    var points = ev.points || [];
    if (!points.length) {{ return; }}
    if (gd.data[points[0].curveNumber].type !== "heatmap") {{ return; }}
    picked = {{ row: nearest(Y, points[0].y), col: nearest(X, points[0].x) }};
    paint();
  }});
  if (chooser) {{
    chooser.onchange = function () {{
      current = chooser.value;
      // 范围不跨视图保留：相位上的数字摆到 |S21| 上没有意义，切视图即回自动
      clearBounds();
      paint();
    }};
  }}
  if (zminInput) {{ zminInput.oninput = paint; }}
  if (zmaxInput) {{ zmaxInput.oninput = paint; }}
  scrollTune(zminInput, "zmin");
  scrollTune(zmaxInput, "zmax");
  if (autoButton) {{
    autoButton.onclick = function () {{ clearBounds(); paint(); }};
  }}
  // 角块跟着 plotly 的排版走：首次建图、容器缩放（ResizeObserver → plotly resize）都落在这个事件上
  gd.on("plotly_afterplot", place);
  // 铺一次：浏览器可能恢复了上次的下拉框选择，提示数字与角块位置也都要按当前状态对齐
  paint();
  place();
}})();
</script>
</div>
"#,
        views = json_views.join(","),
        order = json_order.join(","),
        x_left = X_RIGHT[0],
        x_span = X_RIGHT[1] - X_RIGHT[0],
        y_top = 1.0 - Y_BOTTOM[1],
        y_span = Y_BOTTOM[1] - Y_BOTTOM[0]
    )
}

/// 版面分区（paper 分数）：热图占 [`X_MAP`]×[`Y_MAP`]，下方边线图 [X_MAP]×[`Y_BOTTOM`]，
/// 右侧边线图 [`X_RIGHT`]×[Y_MAP]；其余留给轴标题与 colorbar。
/// 热图的上边距，px（见 [`layout`]）。
const MAP_PLOT_TOP: usize = 24;

const X_MAP: [f64; 2] = [0.0, 0.78];
const Y_MAP: [f64; 2] = [0.26, 1.0];
const Y_BOTTOM: [f64; 2] = [0.0, 0.16];
const X_RIGHT: [f64; 2] = [0.86, 1.0];

fn layout(grid: &Grid2d<'_>, value_title: &str) -> Layout {
    Layout::new()
        .font(figure_font())
        .show_legend(false)
        // 上边距：顶部那条轴只剩刻度标签要占地方（轴标题已省），给一小条即可。默认边距在
        // 它上方留了一大片空白，图名行因此离图很远；automargin 会按需再撑。
        .margin(Margin::new().top(MAP_PLOT_TOP))
        // 热图轴显式贴着格子（首/末坐标各外扩半格）。不这么做的话，同子图里的折线
        // trace 会按 6% 余量参与算范围，把轴撑开、四周留下一圈空格子大小的白边。
        .x_axis(axis_style(
            Axis::new()
                .domain(&X_MAP)
                .anchor("y")
                // 热图的频率轴放顶部：底部那条留给下方边线图，两条轴不再挤在一起。
                // 顶部不标轴标题——图名行就在它上面，再写一遍"freq (Hz)"是重复
                .side(AxisSide::Top)
                .range(edge_range(grid.x)),
        ))
        .y_axis(axis_style(
            Axis::new()
                .domain(&Y_MAP)
                .anchor("x")
                .title(grid.y_title)
                .range(edge_range(grid.y)),
        ))
        .x_axis2(axis_style(
            Axis::new()
                .domain(&X_MAP)
                .anchor("y2")
                .matches("x")
                .title(grid.x_title),
        ))
        .y_axis2(axis_style(
            Axis::new()
                .domain(&Y_BOTTOM)
                .anchor("x2")
                .title(value_title),
        ))
        .x_axis3(axis_style(
            Axis::new()
                .domain(&X_RIGHT)
                .anchor("y3")
                .title(value_title),
        ))
        .y_axis3(axis_style(
            Axis::new().domain(&Y_MAP).anchor("x3").matches("y"),
        ))
}

/// 热图轴的紧贴范围：首尾坐标各向外扩"相邻间距的一半"，让最外圈的格子正好贴着轴框。
fn edge_range(values: &[f64]) -> Vec<f64> {
    match (values.first(), values.last(), values.len()) {
        (Some(first), Some(last), count) if count >= 2 => {
            let head = (values[1] - values[0]) / 2.0;
            let tail = (values[count - 1] - values[count - 2]) / 2.0;
            vec![first - head, last + tail]
        }
        (Some(first), Some(last), _) => vec![*first, *last],
        _ => vec![0.0, 1.0],
    }
}

/// 下方边线：值 vs x，误差沿 y。
fn row_trace(
    xs: Vec<f64>,
    values: Vec<f64>,
    errors: Vec<f64>,
    name: &str,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<dyn Trace> = Scatter::new(xs, values)
        .name(name)
        .mode(Mode::LinesMarkers)
        .line(plotly::common::Line::new().color(LINE_COLOR).width(1.5))
        .marker(Marker::new().color(LINE_COLOR).size(4))
        .error_y(error_data(errors))
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 右侧边线：值 vs y（值在 x 轴上），误差沿 x。
fn col_trace(
    values: Vec<f64>,
    errors: Vec<f64>,
    ys: Vec<f64>,
    name: &str,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<dyn Trace> = Scatter::new(values, ys)
        .name(name)
        .mode(Mode::LinesMarkers)
        .line(plotly::common::Line::new().color(LINE_COLOR).width(1.5))
        .marker(Marker::new().color(LINE_COLOR).size(4))
        .error_x(error_data(errors))
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// continuous error bar（逐点数组、无 cap、细线；NaN/非有限值自动成为断点）。
fn error_data(errors: Vec<f64>) -> ErrorData {
    ErrorData::new(ErrorType::Data)
        .array(errors)
        .visible(true)
        .thickness(1.0)
        .width(0)
}

/// 横截图的一条：该行自己的 (x, 取值, 误差)，丢掉非有限的取值。
///
/// 行内本来就是连续的一段扫描；并集轴（稠密矩阵的列轴）上那些 NaN 只是"这一行没扫到那里"
/// 的空格，直接拿去画会把连线打断成一串孤点。
///
/// 形参:
///     x: 列轴
///     view: 取哪一层的取值（初始发出去的是首个视图）
///     row: 行号
///
/// 返回值:
///     (该行自己的 x, 取值, 误差)，三者等长；行越界或该行全空时给三条空数组，
///     没有误差矩阵（或误差行缺失）时误差全为 NaN
fn row_slice(x: &[f64], view: &View<'_>, row: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut values = Vec::new();
    let mut errors = Vec::new();
    let row_values = match view.z.get(row) {
        Some(row_values) => row_values,
        None => return (xs, values, errors),
    };
    let row_errors = match view.z_err {
        Some(matrix) => matrix.get(row),
        None => None,
    };
    for (column, value) in row_values.iter().enumerate() {
        match value.is_finite() {
            true => {
                xs.push(x[column]);
                values.push(*value);
                errors.push(match row_errors {
                    Some(column_errors) => match column_errors.get(column) {
                        Some(error) => *error,
                        None => f64::NAN,
                    },
                    None => f64::NAN,
                });
            }
            false => {}
        }
    }
    (xs, values, errors)
}

fn slice_col(matrix: &[Vec<f64>], col: usize, height: usize) -> Vec<f64> {
    (0..height)
        .map(|row| match matrix.get(row).and_then(|values| values.get(col)) {
            Some(value) => *value,
            None => f64::NAN,
        })
        .collect()
}

// =========================================================================
// 共享小工具（s21_plot 也复用）
// =========================================================================

/// 统一轴风格：轴线四边镜像、刻度朝内、隐藏零线、标题/刻度自动让出边距；大数用 SI 前缀。
///
/// `automargin` 让容器变窄时 plotly 自己加大 margin，轴标题与刻度不会被裁掉。
/// `exponent_format(SI)` 是必须显式设的：plotly 默认的 "B"（billions）会把 6.9 GHz
/// 写成 `6.9B`，同一个页面里的线图若用 SI 就变成 `6.9G` —— 两处记号不一致。
pub(crate) fn axis_style(axis: Axis) -> Axis {
    axis.mirror(true)
        .ticks(TicksDirection::Inside)
        .show_line(true)
        .zero_line(false)
        .auto_margin(true)
        .exponent_format(ExponentFormat::SI)
}

// =========================================================================
// 图名（HTML 图名条 与 plotly 的 title / annotation 共用同一组参数）
// =========================================================================

/// 图内文字的字体：与页面 CSS 同一套（`system-ui`），plotly 的标题/刻度/图例和 HTML
/// 部分的字才是一套。
pub(crate) const FONT_FAMILY: &str = "system-ui,'Segoe UI',sans-serif";

/// 图名的字号与颜色：**对齐 plotly layout title 的默认外观**（17px、#444、常规字重、
/// 居中），HTML 图名条与四面板的 annotation 都照这套来。
pub(crate) const TITLE_SIZE: usize = 17;
pub(crate) const TITLE_COLOR: &str = "#444";

/// plotly 图的统一字体（layout 级：标题、轴标题、刻度、图例都吃它）。
pub(crate) fn figure_font() -> Font {
    Font::new().family(FONT_FAMILY)
}

/// 图名行（HTML 版的图名）的样式：热图的图名条（`.qtool-2d-cap`）与线图的图名行
/// （`.qtool-line-title`）共用同一组字体/字号/颜色，只是外边距各自不同 —— CSS 读不到
/// Rust 常量，所以在这里插值生成，避免两处各写一份。图名行里可以直接嵌控件（下拉框）。
pub(crate) fn title_bar_style() -> String {
    format!(
        "<style>\
         .qtool-2d .qtool-2d-cap{{text-align:center;font-family:{FONT_FAMILY};\
         font-weight:400;font-size:{TITLE_SIZE}px;color:{TITLE_COLOR};margin:10px 0 2px}}\
         .qtool-line-title{{text-align:center;font-family:{FONT_FAMILY};\
         font-weight:400;font-size:{TITLE_SIZE}px;color:{TITLE_COLOR};margin:0 0 6px}}\
         .qtool-line-title:empty{{display:none}}\
         .qtool-line-title select{{font:inherit;color:inherit;background:transparent;\
         border:1px solid transparent;border-radius:6px;padding:0 2px;cursor:pointer;\
         appearance:none;-webkit-appearance:none;-moz-appearance:none}}\
         .qtool-line-title select:hover,.qtool-line-title select:focus{{border-color:#cbd5e1;background:#ffffff}}\
         .qtool-2d-cap select{{font:inherit;color:inherit;background:transparent;\
         border:1px solid transparent;border-radius:6px;padding:0 2px;cursor:pointer;\
         appearance:none;-webkit-appearance:none;-moz-appearance:none}}\
         .qtool-2d-cap select:hover,.qtool-2d-cap select:focus{{border-color:#cbd5e1;background:#ffffff}}\
         </style>"
    )
}

/// 统一的 plotly 交互配置。
///
/// - `responsive=false`：尺寸变化由我们自己的 ResizeObserver 统一驱动（见 utils::resize），
///   避免一份报告里每张图各挂一个 window resize 处理器；
/// - `double_click=False`：双击留给业务（气泡用它"钉住"），不要触发 autoscale；
/// - `scroll_zoom=true`：滚轮/双指缩放。
pub(crate) fn interactive_config() -> Configuration {
    Configuration::new()
        .responsive(false)
        .double_click(DoubleClick::False)
        .scroll_zoom(true)
}

/// HTML 文本转义（标题、错误信息等可能来自调用方）。
pub(crate) fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// 极简 JSON 数组（f64）；非有限值写成 null（plotly 视为断点）。
pub(crate) fn json_array(values: &[f64]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        match value.is_finite() {
            true => out.push_str(&format!("{value:e}")),
            false => out.push_str("null"),
        }
    }
    out.push(']');
    out
}

/// 极简 JSON 矩阵（`&[Vec<f64>]`）。
pub(crate) fn json_matrix(rows: &[Vec<f64>]) -> String {
    let items: Vec<String> = rows.iter().map(|row| json_array(row)).collect();
    format!("[{}]", items.join(","))
}

/// 极简 JSON 字符串数组。
pub(crate) fn json_strings(values: &[String]) -> String {
    let items: Vec<String> = values
        .iter()
        .map(|value| {
            let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
            format!("\"{escaped}\"")
        })
        .collect();
    format!("[{}]", items.join(","))
}

const STYLE: &str = r#"<style>
.qtool-2d{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（5:4 = 约 770×620 的设计比例），宽屏不会拉成一条扁带；
   max-height 只兜底"视口太矮"的情况，此时比例会让位 */
.qtool-2d .qtool-2d-plot{position:relative;max-height:85vh}
/* 色标范围的 min/max 与 auto：整体一个浅框，占住 2×2 布局右下那一格——列宽与右截线图相同、
   行高与下截线图相同（四条边由脚本按 plotly 的 _size 对齐到两张截线图的框）。
   两行 = 标签列 + 取值列，标签右对齐所以两个数从同一条竖线起。
   三行之间的间距不写死：`space-evenly` 让三行均分这一格的高度，横向前后留白与列间距都走百分比——
   格子多大（还是由两张截线图定）间距就多大，没有 px */
.qtool-2d .qtool-2d-range{position:absolute;box-sizing:border-box;font-size:13px;
display:grid;grid-template-columns:auto 1fr;column-gap:.4em;
align-content:space-evenly;align-items:center;padding:0 .4em;
border:1px solid #e5e7eb;border-radius:6px;background:#fcfcfd}
.qtool-2d .qtool-2d-range label{font-size:inherit;color:#6b7280;text-align:right;cursor:pointer}
.qtool-2d .qtool-2d-range input{box-sizing:border-box;width:100%;min-width:0;font:inherit;font-size:13px;
color:#1f2328;background:#ffffff;border:1px solid #e5e7eb;border-radius:6px;
padding:1px 3px;appearance:textfield;-moz-appearance:textfield}
.qtool-2d .qtool-2d-range input::-webkit-inner-spin-button,.qtool-2d .qtool-2d-range input::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
.qtool-2d .qtool-2d-range input:hover,.qtool-2d .qtool-2d-range input:focus{border-color:#9aa1a9;outline:none}
.qtool-2d .qtool-2d-range input::placeholder{color:#b6bcc4}
.qtool-2d .qtool-2d-range button{grid-column:1 / -1;justify-self:center;font:inherit;
line-height:1.5;color:#444444;background:#ffffff;border:1px solid #e5e7eb;border-radius:6px;
padding:1px 1em;cursor:pointer}
.qtool-2d .qtool-2d-range button:hover{border-color:#9aa1a9;background:#f3f4f6}
/* 窄屏角块放不下两个框（会压到 colorbar），此时不提供色标范围调节 */
@media (max-width:760px){.qtool-2d .qtool-2d-range{display:none}}
</style>"#;
