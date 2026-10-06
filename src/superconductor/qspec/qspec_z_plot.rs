//! qspec 对 Z 偏置扫描的二维报告（feature = "plot"）。
//!
//! - 上方热图：颜色 = P1，纵轴 = Z 偏置，由 [`heatmap`] 生成（点击取该点到右/下边线，带 error bar）；
//! - **hover 到某一行** → 跟随光标的对话气泡显示该 Z 处的四面板 qspec 报告（首次使用时才渲染）；
//! - **double click** → 把气泡**原地**钉成固定浮层（可多个、按行去重），每个浮层带 × 关闭；
//! - 下方线图：下拉框在洛伦兹的四个参数间切换，画该参数逐 Z 的**拟合值 + stderr**；
//! - 产物是自包含 div，宿主页面需提供 plotly.js（见
//!   [`qspec_plot::PLOTLY_JS_CDN`](crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN)）。
//!
//! flux 调谐模型（`flux_tunable`）的拟合在 `qspec_z.rs`，尚未移植；本模块只画数据，
//! 模型曲线与参数表等那边落地后，在下方线图上再叠一层即可（baseline 的
//! `qspec_vs_z_plot(zs, peaks, result=...)` 那一步）。

use crate::superconductor::qspec::qspec::{QspecError, QspecFit};
use crate::superconductor::qspec::qspec_plot::{DESIGN_WIDTH, FIT_COLOR, qspec_fit_plot_div};
use crate::superconductor::{StateCenters, p1};
use crate::utils::bubble::{Bubble, bubble, framize};
use crate::utils::heatmap::{
    Grid2d, Palette, axis_style, figure_font, heatmap, interactive_config, json_array,
};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{ErrorData, ErrorType, Line, Marker, Mode};
use plotly::layout::{Axis, Layout, Margin};
use plotly::{Plot, Scatter};

/// 气泡/浮层里面板的缩放比例（设计宽度按 [`DESIGN_WIDTH`] 原样布局，再整体缩到这么小）。
const SCALE: f64 = 0.62;

/// 线图的上边距，px：图名在 HTML 图名行里、图内也没有顶部轴，plotly 默认的 100px 全是空白，
/// 压到"图名下面一点"即可。自带工具栏（modebar）不占这里——它在 CSS 里被挪到图名行右端。
const LINE_PLOT_TOP: usize = 10;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
/* 不设 max-width：报告宽度跟着宿主容器走，宽屏就铺满（图由 ResizeObserver 跟尺寸重排） */
.qtool-qspec-z{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
.qtool-qspec-z .qtool-qspec-z-maps{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.qtool-qspec-z .qtool-qspec-z-maps>.qtool-2d{flex:1 1 420px;min-width:0}
/* 下方线图：flex-basis 100% 强制换到卡片内的下一行 */
.qtool-qspec-z .qtool-qspec-z-lines{flex:1 1 100%;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.qtool-qspec-z .qtool-qspec-z-lines>.qtool-line{flex:1 1 420px;min-width:0}
/* 线图比二维图扁：容器高度 = plotly 上边距 + 纸面 + 下边距 */
.qtool-qspec-z .qtool-line-plot{width:100%;aspect-ratio:18/5;max-height:60vh}
/* plotly 自带工具栏固定贴在容器右上角（top:2px），上边距压小后就会压在图上 ——
   挪到图名行右端（图名行高约 21px + 6px 外边距，工具栏高 19px，居中即 -26px） */
.qtool-qspec-z .js-plotly-plot .modebar{top:-26px}
</style>"#;

/// 一条 Z 偏置处的 qspec 谱线：偏置 + 数据 + 该处的洛伦兹拟合（`Err` 表示拟合失败）。
pub struct QspecZLine<'a> {
    /// Z 线偏置，V
    pub z: f64,
    /// 该偏置处的平均复数 IQ (n,)
    pub iq: &'a [Complex64],
    /// 该偏置处的洛伦兹拟合结果
    pub fit: Result<&'a QspecFit, &'a QspecError>,
}

/// 渲染 qspec vs Z 的扫描报告 div。
///
/// 形参:
///     freqs_hz: 公共 XY 驱动频率轴 (n,)，Hz
///     lines: 逐条谱线（按 Z 递增）
///     states: 各态标定中心；None 表示未标定，P1 由各线自身定轴（须与各行 `fit` 用的是
///             同一套中心，否则热图上的曲线不是被拟合的那条）
///     div_id: 外层 div 的 HTML id（须是合法 id）
///     frame: 可选外框：`Some(title)` 套上卡片框，`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-qspec-z">` 片段（热图 + 参数线图 + 跟随光标的气泡
///     + 可钉住的浮层）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn qspec_z_plot_div(
    freqs_hz: &[f64],
    lines: &[QspecZLine<'_>],
    states: Option<&StateCenters>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let zs: Vec<f64> = lines.iter().map(|line| line.z).collect();

    // 热图画的就是各线被拟合的那条 P1（与 [`qspec_fit_plot_div`] 的 P1 面板同一条曲线）；
    // 该线没有拟合结果时退回原始投影，至少让数据可见。
    let mut p1_matrix: Vec<Vec<f64>> = Vec::with_capacity(lines.len());
    for line in lines {
        let prob = match &line.fit {
            Ok(fit) => fit.p1.clone(),
            Err(_) => p1(line.iq, states),
        };
        p1_matrix.push(prob);
    }

    let map = heatmap(
        &Grid2d {
            x: freqs_hz,
            y: &zs,
            z: &p1_matrix,
            z_err: None,
            x_title: "freq (Hz)",
            y_title: "Z (V)",
            value_title: "P1",
            caption: "P1 vs Z",
            // 蓝（低）→ 红（高）：|0> 端偏蓝、|1> 端偏红，与 IQ 面板的两态配色同语义
            palette: Palette::RdBu,
        },
        &format!("{div_id}-map"),
    );
    let param_plot = param_line_plot(
        &zs,
        &param_series(lines),
        &format!("{div_id}-param"),
        FIT_COLOR,
    );

    // 每个 Z 的面板（[`qspec_fit_plot_div`] 的四面板图 + 参数表，含自身样式）嵌在
    // <template> 里惰性实例化；`{div_id}-panel{row}` 是这一行的 id 前缀，前端挂载时
    // 会整体改写成该实例专属的 id（同一行可能同时挂在气泡与浮层上，id 不能重复）。
    let mut templates = String::new();
    for (row, line) in lines.iter().enumerate() {
        let panel_base = format!("{div_id}-panel{row}");
        let panel_html = qspec_fit_plot_div(freqs_hz, line.iq, states, line.fit, &panel_base, None);
        templates.push_str(&format!(
            "<template id=\"{div_id}-tpl-{row}\"><div class=\"qtool-panel\">{panel_html}</div></template>\n"
        ));
    }

    // 行标签由 Z 偏置派生：气泡/浮层的标题栏与 hover 提示都用它
    let labels: Vec<String> = zs.iter().map(|z| format!("Z {z:.4} V")).collect();
    let bubble_html = bubble(&Bubble {
        div_id,
        triggers: &[format!("{div_id}-map-plot")],
        row_values: &zs,
        labels: &labels,
        panel_width: DESIGN_WIDTH,
        scale: SCALE,
    });

    // 图组（热图 + 下方线图）作为一个整体套框：`frame` 由调用方给
    let title_bar_css = crate::utils::heatmap::title_bar_style();
    let maps = framize(
        &format!(
            "<div class=\"qtool-qspec-z-maps\">{map}\
             <div class=\"qtool-qspec-z-lines\">{param_plot}</div></div>"
        ),
        frame,
    );

    format!(
        r##"<div class="qtool-qspec-z" id="{div_id}">
{STYLE}{title_bar_css}
{maps}
{bubble_html}
{templates}</div>
"##
    )
}

/// 一条参数曲线：逐 Z 的拟合值与 stderr（拟合失败或缺 stderr 的档是 NaN）。
struct ParamSeries {
    /// 参数名，同时是下拉框的 value 与 stderr 查表的键。
    name: &'static str,
    /// 下拉框显示名（带单位），同时用作 y 轴标题。
    label: String,
    value: Vec<f64>,
    stderr: Vec<f64>,
}

/// 可画的四个洛伦兹参数：名字、显示名、以及从模型取出该参数的取值器。
///
/// 用取值器而不是按名字去匹配字符串：四个字段的对应关系只写一遍，加参数时编译器会提醒。
const EXTRACTORS: [(&str, &str, fn(&crate::superconductor::qspec::Lorentz) -> f64); 4] = [
    ("fq", "fq (Hz)", |model| model.fq),
    ("fwhm", "fwhm (Hz)", |model| model.fwhm),
    ("amp", "amp", |model| model.amp),
    ("offset", "offset", |model| model.offset),
];

/// 逐 Z 收集四个参数的拟合值与标准误。
fn param_series(lines: &[QspecZLine<'_>]) -> Vec<ParamSeries> {
    EXTRACTORS
        .iter()
        .map(|(name, label, extract)| ParamSeries {
            name,
            label: label.to_string(),
            value: lines
                .iter()
                .map(|line| match &line.fit {
                    Ok(fit) => extract(&fit.result.model),
                    Err(_) => f64::NAN,
                })
                .collect(),
            stderr: lines
                .iter()
                .map(|line| match &line.fit {
                    Ok(fit) => match fit.result.params.get(name) {
                        Some(parameter) => match parameter.stderr {
                            Some(value) => value,
                            None => f64::NAN,
                        },
                        None => f64::NAN,
                    },
                    Err(_) => f64::NAN,
                })
                .collect(),
        })
        .collect()
}

/// "参数 vs Z" 线图：value + 拟合 stderr（error bar）+ 上方的参数下拉框。
///
/// 四条曲线一次全塞进页面，切下拉框只做 `Plotly.restyle`/`relayout`，不再回 Rust。
fn param_line_plot(zs: &[f64], series: &[ParamSeries], div_id: &str, color: &'static str) -> String {
    let first = match series.first() {
        Some(item) => item,
        None => return String::new(),
    };
    let plot_html = line_plot_html(
        div_id,
        zs,
        &first.value,
        Some(&first.stderr),
        &first.label,
        color,
    );
    let options: Vec<String> = series
        .iter()
        .map(|item| format!("<option value=\"{}\">{}</option>", item.name, item.label))
        .collect();
    // 图名行 = 下拉框 + 固定的 " vs Z"：换参数时整行跟着变
    let title_bar = format!(
        "<select id=\"{div_id}-select\">{}</select><span> vs Z</span>",
        options.join("")
    );
    let data: Vec<String> = series
        .iter()
        .map(|item| {
            format!(
                "\"{}\":{{\"label\":\"{}\",\"value\":{},\"err\":{}}}",
                item.name,
                item.label,
                json_array(&item.value),
                json_array(&item.stderr)
            )
        })
        .collect();
    // 图名在 HTML 里（plotly 的 title 放不下 <select>），所以这里只改 y 轴标题。
    // 下拉框宽度按当前选项实测文字宽度来定 —— 原生 select 会撑到最宽的那个选项，
    // 那样"fq (Hz)"和"vs Z"之间会空出一大截，看着不像一个标题。
    let script = format!(
        r#"<script>
(function () {{
  var select = document.getElementById("{div_id}-select");
  var gd = document.getElementById("{div_id}-plot");
  var SERIES = {{{}}};
  if (!select || !gd || !window.Plotly) {{ return; }}
  var ruler = document.createElement("canvas").getContext("2d");
  var fitWidth = function () {{
    var picked = select.options[select.selectedIndex];
    if (!picked || !ruler) {{ return; }}
    ruler.font = getComputedStyle(select).font;
    select.style.width = Math.ceil(ruler.measureText(picked.text).width + 12) + "px";
  }};
  fitWidth();
  select.onchange = function () {{
    var item = SERIES[select.value];
    if (!item) {{ return; }}
    Plotly.restyle(gd, {{ "y": [item.value], "error_y.array": [item.err], "error_y.visible": [true] }}, [0]);
    Plotly.relayout(gd, {{ "yaxis.title.text": item.label }});
    fitWidth();
  }};
}})();
</script>"#,
        data.join(",")
    );
    line_frame(div_id, &title_bar, &plot_html, &script)
}

/// 单条曲线的 plotly 图：折线 + 误差棒，x 轴是线性的 Z 偏置。
///
/// 拟合失败或缺 stderr 的档在数据里是 NaN（plotly 视为断点，曲线断开而不是折回去）。
fn line_plot_html(
    div_id: &str,
    zs: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    y_title: &str,
    color: &'static str,
) -> String {
    let mut scatter = Scatter::new(zs.to_vec(), values.to_vec())
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(color).width(1.5))
        .marker(Marker::new().color(color).size(6))
        .show_legend(false)
        .x_axis("x")
        .y_axis("y");
    match errors {
        Some(array) => {
            scatter = scatter.error_y(
                ErrorData::new(ErrorType::Data)
                    .array(array.to_vec())
                    .visible(true)
                    .thickness(1.0)
                    .width(0)
                    .color(color),
            );
        }
        None => {}
    }
    let mut plot = Plot::new();
    let trace: Box<dyn Trace> = scatter;
    plot.add_trace(trace);
    let layout = Layout::new()
        .font(figure_font())
        // 图名在 HTML 图名行里、图内也没有顶部轴，plotly 默认的 100px 上边距全是空白 ——
        // 压到最小，让图名贴着图（与面板标题到面板的距离一致）
        .margin(Margin::new().top(LINE_PLOT_TOP))
        .x_axis(axis_style(Axis::new().title("Z (V)")))
        .y_axis(axis_style(Axis::new().title(y_title)));
    plot.set_layout(layout);
    plot.set_configuration(interactive_config());
    let plot_div_id = format!("{div_id}-plot");
    plot.to_inline_html(Some(plot_div_id.as_str()))
}

/// 线图面板的外壳：图名行（可含下拉框）+ plot div + 缩放注册 + 附加脚本。
fn line_frame(div_id: &str, title_bar: &str, plot_html: &str, script: &str) -> String {
    let register_script = crate::utils::resize::register_script(div_id);
    format!(
        "<div class=\"qtool-line\" id=\"{div_id}\">\
         <div class=\"qtool-line-title\">{title_bar}</div>\
         <div class=\"qtool-line-plot\">{plot_html}</div>{register_script}{script}</div>\n"
    )
}
