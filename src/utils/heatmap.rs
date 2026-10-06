//! 通用二维图模板（feature = "plot"）：热图 + 右侧/下方边线图。
//!
//! 本模块是 crate 内部件（`utils` 下的模块都不作为外部接口），只服务仓库内的报告。
//!
//! 二维图占据左上区域；**下方**边线图显示点击点所在的 **row**（`z[row][·]` vs x），
//! **右侧**边线图显示点击点所在的 **col**（`z[·][col]` vs y），两者带 continuous
//! error bar（`error_y.type = "data"`，逐点数组）。初始显示中间一行 / 中间一列。
//!
//! 交互靠片段内嵌的一小段 JS：`plotly_click` → 取最近的行列号 → `Plotly.restyle`
//! 更新两条边线；x/y 轴用 `matches` 联动缩放。宿主页面仍需自行加载 plotly.js。

use plotly::common::{
    AxisSide, ColorBar, ColorScale, ColorScaleElement, ErrorData, ErrorType, HoverInfo, Marker,
    Mode, Title,
};
use plotly::configuration::{Configuration, DoubleClick};
use plotly::layout::{Axis, Layout, TicksDirection};
use plotly::{HeatMap, Plot, Scatter, Trace};

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

/// 二维图数据：行轴 / 列轴 + 矩阵（+ 可选误差矩阵）。
pub(crate) struct Grid2d<'a> {
    /// 列轴（热图 x）。
    pub(crate) x: &'a [f64],
    /// 行轴（热图 y）。
    pub(crate) y: &'a [f64],
    /// 矩阵 `z[row][col]`。
    pub(crate) z: &'a [Vec<f64>],
    /// 与 `z` 同形的误差矩阵（边线图的 error bar）。
    pub(crate) z_err: Option<&'a [Vec<f64>]>,
    pub(crate) x_title: &'a str,
    pub(crate) y_title: &'a str,
    pub(crate) value_title: &'a str,
    pub(crate) palette: Palette,
}

const LINE_COLOR: &str = "#d62728";

/// 渲染热图 + 右/下边线图的 HTML 片段（自包含 div，宿主需提供 plotly.js）。
///
/// 形参:
///     grid: 二维数据与轴标题
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一，且须是合法 id）
///
/// 返回值:
///     `<div class="qtool-2d">` 片段（含内嵌交互脚本）
pub(crate) fn heatmap(grid: &Grid2d<'_>, div_id: &str) -> String {
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
    let row_values = slice_row(grid.z, mid_row, cols);
    let row_err = match grid.z_err {
        Some(matrix) => slice_row(matrix, mid_row, cols),
        None => vec![f64::NAN; cols],
    };
    let col_values = slice_col(grid.z, mid_col, rows);
    let col_err = match grid.z_err {
        Some(matrix) => slice_col(matrix, mid_col, rows),
        None => vec![f64::NAN; rows],
    };

    let mut plot = Plot::new();
    plot.add_trace(
        HeatMap::new(grid.x.to_vec(), grid.y.to_vec(), grid.z.to_vec())
            .color_scale(match grid.palette {
                Palette::Turbo => turbo(),
                Palette::RdBu => rdbu(),
            })
            .color_bar(ColorBar::new().title(Title::with_text(grid.value_title)))
            // 热图不弹默认的坐标浮标：读数由气泡（整行面板）承担。注意用 None 而不是
            // Skip —— None 只是不显示内容，hover / click 事件照常派发，气泡和"点一下
            // 取该行该列到边线图"都依赖它们；Skip 会把整条 trace 从 hover 里摘掉。
            .hover_info(HoverInfo::None),
    );
    plot.add_trace(row_trace(
        grid.x.to_vec(),
        row_values,
        row_err,
        grid.value_title,
        "x2",
        "y2",
    ));
    plot.add_trace(col_trace(
        col_values,
        col_err,
        grid.y.to_vec(),
        grid.value_title,
        "x3",
        "y3",
    ));
    plot.set_layout(layout(grid));
    plot.set_configuration(interactive_config());
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = plot.to_inline_html(Some(plot_div_id.as_str()));

    let register_script = crate::utils::resize::register_script(div_id);
    let json_x = json_array(grid.x);
    let json_y = json_array(grid.y);
    let json_z = json_matrix(grid.z);
    let json_err = match grid.z_err {
        Some(matrix) => json_matrix(matrix),
        None => "null".to_string(),
    };

    format!(
        r#"<div class="qtool-2d" id="{div_id}">
{STYLE}
<div class="qtool-2d-plot">{plot_html}</div>
{register_script}
<script>
(function () {{
  var X = {json_x}, Y = {json_y}, Z = {json_z}, E = {json_err};
  var gd = document.getElementById("{plot_div_id}");
  if (!window.Plotly || !gd) {{ return; }}
  var nearest = function (arr, value) {{
    var best = 0, bestDist = Infinity;
    for (var i = 0; i < arr.length; i++) {{
      var d = Math.abs(arr[i] - value);
      if (d < bestDist) {{ bestDist = d; best = i; }}
    }}
    return best;
  }};
  gd.on("plotly_click", function (ev) {{
    var points = ev.points || [];
    if (!points.length || points[0].curveNumber !== 0) {{ return; }}
    var row = nearest(Y, points[0].y);
    var col = nearest(X, points[0].x);
    var rowVals = Z[row];
    var colVals = Z.map(function (r) {{ return r[col]; }});
    Plotly.restyle(gd, {{ "y": [rowVals], "error_y.array": E ? [E[row]] : [null] }}, [1]);
    Plotly.restyle(gd, {{ "x": [colVals], "error_x.array": E ? [E.map(function (r) {{ return r[col]; }})] : [null] }}, [2]);
  }});
}})();
</script>
</div>
"#
    )
}

/// 版面分区（paper 分数）：热图占 [`X_MAP`]×[`Y_MAP`]，下方边线图 [X_MAP]×[`Y_BOTTOM`]，
/// 右侧边线图 [`X_RIGHT`]×[Y_MAP]；其余留给轴标题与 colorbar。
const X_MAP: [f64; 2] = [0.0, 0.78];
const Y_MAP: [f64; 2] = [0.26, 1.0];
const Y_BOTTOM: [f64; 2] = [0.0, 0.16];
const X_RIGHT: [f64; 2] = [0.86, 1.0];

fn layout(grid: &Grid2d<'_>) -> Layout {
    Layout::new()
        .show_legend(false)
        // 热图轴显式贴着格子（首/末坐标各外扩半格）。不这么做的话，同子图里的折线
        // trace 会按 6% 余量参与算范围，把轴撑开、四周留下一圈空格子大小的白边。
        .x_axis(axis_style(
            Axis::new()
                .domain(&X_MAP)
                .anchor("y")
                // 热图的频率轴放顶部：底部那条留给下方边线图，两条轴不再挤在一起
                .side(AxisSide::Top)
                .title(grid.x_title)
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
                .title(grid.value_title),
        ))
        .x_axis3(axis_style(
            Axis::new()
                .domain(&X_RIGHT)
                .anchor("y3")
                .title(grid.value_title),
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

fn slice_row(matrix: &[Vec<f64>], row: usize, width: usize) -> Vec<f64> {
    match matrix.get(row) {
        Some(values) => match values.len() == width {
            true => values.clone(),
            false => values.iter().copied().chain(std::iter::repeat(f64::NAN)).take(width).collect(),
        },
        None => vec![f64::NAN; width],
    }
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

/// 统一轴风格：轴线四边镜像、刻度朝内、隐藏零线、标题/刻度自动让出边距。
///
/// `automargin` 让容器变窄时 plotly 自己加大 margin，轴标题与刻度不会被裁掉。
pub(crate) fn axis_style(axis: Axis) -> Axis {
    axis.mirror(true)
        .ticks(TicksDirection::Inside)
        .show_line(true)
        .zero_line(false)
        .auto_margin(true)
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
.qtool-2d .qtool-2d-plot{width:100%;aspect-ratio:5/4;max-height:85vh}
</style>"#;
