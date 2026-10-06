//! S21 功率扫描的二维报告（feature = "plot"）：两张并排热图 + 行面板交互。
//!
//! - 热图一：颜色 = |S21|；热图二：颜色 = unwrap + detrend 后的相位（与
//!   [`s21_plot`](crate::superconductor::s21::s21_plot) 的相位面板同口径）；
//! - 两张热图都由 [`crate::utils::heatmap::heatmap`] 生成（点击取
//!   该点的 row / col 到右、下边线图，带 continuous error bar）；
//! - **hover 到某一行** → 跟随光标的对话气泡显示该功率的 S21 四面板（首次使用时才渲染）；
//! - **double click** → 把气泡**原地**钉成固定浮层（位置不再跟走、移开鼠标也不消失；
//!   可多个、按行去重、允许重叠），每个浮层带 × 关闭；
//! - 热图下方两张线图：左 = 实测谷位 `f_dip` vs power（与模型无关）；右 = 下拉框选一个
//!   参数（参数表里的 13 个），画该参数逐功率档的拟合值 + stderr（模型给的量，与 f_dip
//!   不是一回事，各画各的）；
//! - 产物是自包含 div，宿主页面需提供 plotly.js（见
//!   [`s21_plot::PLOTLY_JS_CDN`](crate::superconductor::s21::s21_plot::PLOTLY_JS_CDN)）。

use crate::superconductor::s21::s21::{Complex64, JAC_NAMES, S21Error, S21Model};
use crate::superconductor::s21::s21_plot::{
    DATA_COLOR, DESIGN_WIDTH, FIT_COLOR, s21_fit_plot_div, unit_of,
};
use crate::utils::bubble::{Bubble, bubble, framize};
use crate::utils::heatmap::{
    Grid2d, Palette, axis_style, figure_font, heatmap, interactive_config,
};
use crate::utils::{argmin, detrend, unwrap_phase};
use lmfit::ComplexResult;
use plotly::Trace;
use plotly::common::{Anchor, DashType, ErrorData, ErrorType, Line, Marker, Mode};
use plotly::layout::{Annotation, Axis, AxisType, Layout, Margin, Shape, ShapeLayer, ShapeLine, ShapeType};
use plotly::{Plot, Scatter};

/// 一条频率线：泵幅 + 数据 + 拟合结果（`Err` 表示该线拟合失败）。
pub struct PowerLine<'a> {
    pub power: f64,
    pub iq: &'a [Complex64],
    pub sigma: Option<&'a [f64]>,
    pub fit: Result<&'a ComplexResult<S21Model>, &'a S21Error>,
}

/// 渲染功率扫描报告 div。
///
/// 形参:
///     freqs_hz: 公共读出频率轴 (n,)，Hz
///     lines: 逐条频率线（按功率递增），每条含泵幅/数据/σ/拟合结果
///     div_id: 外层 div 的 HTML id（须是合法 id）
///
/// 返回值:
///     自包含的 `<div class="qtool-power">` 片段（两张热图 + 跟随光标的行面板气泡
///     + 可钉住的浮层）。行标签由 `PowerLine::power` 派生。
pub fn s21_power_plot_div(
    freqs_hz: &[f64],
    lines: &[PowerLine<'_>],
    div_id: &str,
    frame: Option<&str>,
) -> String {
    // 频率轴直接用 SI 基本单位 Hz（与数据、与 s21_plot 的面板一致）
    let powers: Vec<f64> = lines.iter().map(|line| line.power).collect();

    let mut amp = Vec::with_capacity(lines.len());
    let mut amp_err = Vec::with_capacity(lines.len());
    let mut phase = Vec::with_capacity(lines.len());
    let mut phase_err = Vec::with_capacity(lines.len());
    for line in lines {
        let magnitudes: Vec<f64> = line.iq.iter().map(|z| z.norm()).collect();
        let raw_phase: Vec<f64> = line.iq.iter().map(|z| z.arg()).collect();
        let (detrended, _, _) = detrend(freqs_hz, &unwrap_phase(&raw_phase));
        amp.push(magnitudes.clone());
        phase.push(detrended);
        match line.sigma {
            Some(sigma) => {
                amp_err.push(sigma.to_vec());
                phase_err.push(
                    sigma
                        .iter()
                        .zip(magnitudes.iter())
                        .map(|(value, magnitude)| match *magnitude > 0.0 {
                            true => value / magnitude,
                            false => f64::NAN,
                        })
                        .collect(),
                );
            }
            None => {
                amp_err.push(vec![f64::NAN; freqs_hz.len()]);
                phase_err.push(vec![f64::NAN; freqs_hz.len()]);
            }
        }
    }

    let amp_map = heatmap(
        &Grid2d {
            x: freqs_hz,
            y: &powers,
            z: &amp,
            z_err: Some(&amp_err),
            x_title: "freq (Hz)",
            y_title: "power",
            value_title: "|S21|",
            caption: "|S21| vs power",
            palette: Palette::Turbo,
        },
        &format!("{div_id}-amp"),
    );
    let phase_map = heatmap(
        &Grid2d {
            x: freqs_hz,
            y: &powers,
            z: &phase,
            z_err: Some(&phase_err),
            x_title: "freq (Hz)",
            y_title: "power",
            value_title: "phase (rad)",
            caption: "phase vs power",
            palette: Palette::RdBu,
        },
        &format!("{div_id}-phase"),
    );

    // 两张热图下方的一对线图：左边是实测谷位 f_dip（与模型无关）；右边由下拉框选参数
    // ——画该参数逐功率档的**拟合值 + stderr**（模型给的量，与 f_dip 不是一回事）。
    let dip_values: Vec<f64> = lines
        .iter()
        .map(|line| match dip_frequency(freqs_hz, line.iq) {
            Some(value) => value,
            None => f64::NAN,
        })
        .collect();
    let dip_plot = power_line_plot(
        &powers,
        &dip_values,
        &format!("{div_id}-dip"),
        "f_dip vs power",
        "f_dip (Hz)",
        DATA_COLOR,
        shift_lines(&dip_values).as_ref(),
    );
    let param_plot = param_line_plot(
        &powers,
        &param_series(lines),
        &format!("{div_id}-param"),
        FIT_COLOR,
    );

    // 每个功率行的面板（[`s21_fit_plot_div`] 的四面板图 + 参数表，含自身样式）嵌在
    // <template> 里惰性实例化；`{div_id}-panel{row}` 是这一行的 id 前缀，前端挂载时
    // 会整体改写成该实例专属的 id（同一行可能同时挂在气泡与浮层上，id 不能重复）。
    let mut templates = String::new();
    for (row, line) in lines.iter().enumerate() {
        let panel_base = format!("{div_id}-panel{row}");
        let panel_html = s21_fit_plot_div(freqs_hz, line.iq, line.sigma, line.fit, &panel_base, None);
        templates.push_str(&format!(
            "<template id=\"{div_id}-tpl-{row}\"><div class=\"qtool-panel\">{panel_html}</div></template>\n"
        ));
    }

    // 行标签由泵幅派生：气泡/浮层的标题栏、hover 提示都用它
    let labels: Vec<String> = powers
        .iter()
        .map(|power| format!("power {power:.4}"))
        .collect();
    let bubble_html = bubble(&Bubble {
        div_id,
        triggers: &[format!("{div_id}-amp-plot"), format!("{div_id}-phase-plot")],
        row_values: &powers,
        labels: &labels,
        panel_width: DESIGN_WIDTH,
        scale: SCALE,
    });

    // 图组（两张热图 + 下面两张线图）作为一个整体套框：`frame` 由调用方给
    let title_bar_css = crate::utils::heatmap::title_bar_style();
    let maps = framize(
        &format!(
            "<div class=\"qtool-power-maps\">{amp_map}{phase_map}\
             <div class=\"qtool-power-lines\">{dip_plot}{param_plot}</div></div>"
        ),
        frame,
    );

    format!(
        r##"<div class="qtool-power" id="{div_id}">
{STYLE}{title_bar_css}
{maps}
{bubble_html}
{templates}</div>
"##
    )
}

/// 一张"频率 vs 泵幅"的线图：x 用 log 轴（功率档是 geomspace，等 log 距），
/// 拟合失败或取不到谷位的档在数据里是 NaN（plotly 视为断点，曲线断开而不是折回去）。
fn power_line_plot(
    powers: &[f64],
    values: &[f64],
    div_id: &str,
    title: &str,
    y_title: &str,
    color: &'static str,
    reference: Option<&ShiftLines>,
) -> String {
    let plot_html = line_plot_html(div_id, powers, values, None, y_title, color, reference);
    line_frame(div_id, title, &plot_html, "")
}

/// 实测谷位：|S21| 的最低点，再用相邻三点抛物线求顶点做亚步长校正（把 200 kHz 的网格
/// 量化压到 ~30 kHz）。
///
/// **不是模型里的 `fr`**：Duffing 位移把实测谷位往下推（我们量到 800 kHz ≈ 0.8 线宽），
/// δ = f_dip − fr 正是要测的非线性量。谷落在扫频窗口边上、点数不足或出现非有限值时返回
/// `None`——不伪造一个值。
fn dip_frequency(freqs_hz: &[f64], iq: &[Complex64]) -> Option<f64> {
    let magnitude: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
    let usable = match magnitude.len() == freqs_hz.len() {
        true => magnitude.len() >= 3 && magnitude.iter().all(|value| value.is_finite()),
        false => false,
    };
    if !usable {
        return None;
    }
    let index = argmin(&magnitude);
    // 最低点落在首/末点：共振已经在扫频窗口外，这一档没有谷位可报
    if index == 0 || index + 1 == magnitude.len() {
        return None;
    }
    let (left, mid, right) = (magnitude[index - 1], magnitude[index], magnitude[index + 1]);
    // 抛物线顶点相对最低点的偏移（以网格步长为单位）；三点共线/上凸时退化为 0
    let denominator = left - 2.0 * mid + right;
    let offset = match denominator > 0.0 {
        true => 0.5 * (left - right) / denominator,
        false => 0.0,
    };
    let step = freqs_hz[index + 1] - freqs_hz[index];
    Some(freqs_hz[index] + offset * step)
}

/// 一条参数曲线：逐功率档的拟合值与 stderr（拟合失败或缺 stderr 的档是 NaN）。
struct ParamSeries {
    /// 参数名，同时是下拉框的 value 与 stderr 查表的键。
    name: &'static str,
    /// 下拉框显示名（带单位），同时用作 y 轴标题。
    label: String,
    value: Vec<f64>,
    stderr: Vec<f64>,
}

/// 参数表里可画的全部参数（11 个拟合参数 + 2 个派生量，顺序与参数表一致）。
fn param_series(lines: &[PowerLine<'_>]) -> Vec<ParamSeries> {
    let mut series: Vec<ParamSeries> = JAC_NAMES
        .iter()
        .copied()
        .chain(["qi", "kappa_ex"])
        .map(|name| ParamSeries {
            name,
            // 下拉框与 y 轴共用的显示名："fr (Hz)"；量纲为一的不加括号
            label: match unit_of(name) {
                "" => name.to_string(),
                unit => format!("{name} ({})", unit.trim()),
            },
            value: Vec::with_capacity(lines.len()),
            stderr: Vec::with_capacity(lines.len()),
        })
        .collect();
    for line in lines {
        match line.fit {
            Ok(result) => {
                // 前 11 个按 `to_array` 的顺序，后 2 个是模型上的派生量
                let values = result.model.to_array();
                let derived = [result.model.qi, result.model.kappa_ex];
                for (index, item) in series.iter_mut().enumerate() {
                    item.value.push(match index < values.len() {
                        true => values[index],
                        false => derived[index - values.len()],
                    });
                    item.stderr.push(match result.params.get(item.name) {
                        Some(parameter) => match parameter.stderr {
                            Some(stderr) => stderr,
                            None => f64::NAN,
                        },
                        None => f64::NAN,
                    });
                }
            }
            // 这一档没有拟合结果：整列留空（该档的面板里已经写了失败原因）
            Err(_error) => {
                for item in series.iter_mut() {
                    item.value.push(f64::NAN);
                    item.stderr.push(f64::NAN);
                }
            }
        }
    }
    series
}

/// "参数 vs power" 线图：value + 拟合 stderr（error bar）+ 上方的参数下拉框。
///
/// 13 条曲线一次全塞进页面，切下拉框只做 `Plotly.restyle`/`relayout`，不再回 Rust。
fn param_line_plot(
    powers: &[f64],
    series: &[ParamSeries],
    div_id: &str,
    color: &'static str,
) -> String {
    let first = match series.first() {
        Some(item) => item,
        None => return String::new(),
    };
    let plot_html = line_plot_html(div_id, powers, &first.value, Some(&first.stderr), &first.label, color, None);
    let options: Vec<String> = series
        .iter()
        .map(|item| format!("<option value=\"{}\">{}</option>", item.name, item.label))
        .collect();
    // 图名行 = 下拉框 + 固定的 " vs power"：换参数时整行跟着变
    let title_bar = format!(
        "<select id=\"{div_id}-select\">{}</select><span> vs power</span>",
        options.join("")
    );
    let data: Vec<String> = series
        .iter()
        .map(|item| {
            format!(
                "\"{}\":{{\"label\":\"{}\",\"value\":{},\"err\":{}}}",
                item.name,
                item.label,
                crate::utils::heatmap::json_array(&item.value),
                crate::utils::heatmap::json_array(&item.stderr)
            )
        })
        .collect();
    // 图名在 HTML 里（plotly 的 title 放不下 <select>），所以这里只改 y 轴标题。
    // 下拉框宽度按当前选项实测文字宽度来定 —— 原生 select 会撑到最宽的那个选项，
    // 那样"fr (Hz)"和"vs power"之间会空出一大截，看着不像一个标题。
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

/// 一张线图的 plotly 片段：x 用 log 轴（功率档是 geomspace，等 log 距）、y 用 SI 记法；
/// NaN（拟合失败/取不到谷位）视为断点，曲线断开而不是折回去。图名在 HTML 图名行里。
fn line_plot_html(
    div_id: &str,
    powers: &[f64],
    values: &[f64],
    errors: Option<&[f64]>,
    y_title: &str,
    color: &'static str,
    reference: Option<&ShiftLines>,
) -> String {
    let mut scatter = Scatter::new(powers.to_vec(), values.to_vec())
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
    let mut layout = Layout::new()
        .font(figure_font())
        // 图名在 HTML 图名行里、图内也没有顶部轴，plotly 默认的 100px 上边距全是空白 ——
        // 压到最小，让图名贴着图（与 s21_plot 里"面板标题到面板"的距离一致）
        .margin(Margin::new().top(LINE_PLOT_TOP))
        // 功率是 0.01–0.5：log 轴默认按 D2 记法给密刻度（"2 3 4 …"）。log 轴的 dtick
        // 以 log10 为单位，取 log10(2) ⇒ 刻度落在 1、2、4、8…（此处 0.01→0.32 共 6 档）
        .x_axis(axis_style(
            Axis::new()
                .title("power")
                .type_(AxisType::Log)
                .dtick(LOG_TICK_STEP)
                .tick_format(".2g"),
        ))
        .y_axis(axis_style(Axis::new().title(y_title)));
    match reference {
        Some(lines) => {
            // 最低/最高功率的谷位各画一条浅灰虚线（画在数据下面），差值标在两条线中间
            let mut shapes: Vec<Shape> = Vec::with_capacity(2);
            for level in [lines.low, lines.high] {
                shapes.push(
                    Shape::new()
                        .shape_type(ShapeType::Line)
                        .layer(ShapeLayer::Below)
                        .x_ref("paper")
                        .x0(0.0)
                        .x1(1.0)
                        .y_ref("y")
                        .y0(level)
                        .y1(level)
                        .line(
                            ShapeLine::new()
                                .color(SHIFT_LINE_COLOR)
                                .width(1.0)
                                .dash(DashType::Dash),
                        ),
                );
            }
            layout = layout.shapes(shapes).annotations(vec![
                Annotation::new()
                    .text(lines.label.clone())
                    .x(0.99)
                    .y((lines.low + lines.high) / 2.0)
                    .x_ref("paper")
                    .y_ref("y")
                    .x_anchor(Anchor::Right)
                    .y_anchor(Anchor::Middle)
                    .show_arrow(false)
                    .font(
                        plotly::common::Font::new()
                            .size(SHIFT_TEXT_SIZE)
                            .color(SHIFT_TEXT_COLOR),
                    ),
            ]);
        }
        None => {}
    }
    plot.set_layout(layout);
    plot.set_configuration(interactive_config());
    let plot_div_id = format!("{div_id}-plot");
    plot.to_inline_html(Some(plot_div_id.as_str()))
}

/// 线图面板的外壳：图名行（可以是空的，空则不占位）+ plot div + 缩放注册 + 附加脚本。
fn line_frame(div_id: &str, title_bar: &str, plot_html: &str, script: &str) -> String {
    let register_script = crate::utils::resize::register_script(div_id);
    format!(
        "<div class=\"qtool-line\" id=\"{div_id}\">\
         <div class=\"qtool-line-title\">{title_bar}</div>\
         <div class=\"qtool-line-plot\">{plot_html}</div>{register_script}{script}</div>\n"
    )
}

/// 面板在气泡/浮层里的缩放比例。
const SCALE: f64 = 0.62;

/// log 轴的刻度步长（`dtick` 在 log 轴上以 log10 为单位）：log10(2) ⇒ 刻度落在 1、2、4、8…。
const LOG_TICK_STEP: f64 = std::f64::consts::LOG10_2;

/// 线图的上边距，px：图名在 HTML 图名行里、图内也没有顶部轴，plotly 默认的 100px 全是空白，
/// 压到"图名下面一点"即可。自带工具栏（modebar）不占这里 —— 它在 CSS 里被挪到图名行右端。
const LINE_PLOT_TOP: usize = 10;

/// f_dip 参考线（最低/最高功率的谷位）与 Lamb shift 标注的样式。
const SHIFT_LINE_COLOR: &str = "#cbd5e1";
const SHIFT_TEXT_COLOR: &str = "#64748b";
const SHIFT_TEXT_SIZE: usize = 13;

/// f_dip 图上的一对参考线 + 两者之差（Lamb shift）。
struct ShiftLines {
    /// 最低功率档的谷位。
    low: f64,
    /// 最高功率档的谷位。
    high: f64,
    /// 标注文本（差值取绝对值）。
    label: String,
}

/// 由逐功率档的谷位取参考线：最低功率与最高功率**能读到**的那两档（中间若有取不到谷位
/// 的档就跳过，功率是递增的，所以这就是两端的有效值）。少一头就返回 None、不画。
fn shift_lines(dips: &[f64]) -> Option<ShiftLines> {
    let low = dips.iter().copied().find(|value| value.is_finite());
    let high = dips.iter().rev().copied().find(|value| value.is_finite());
    match (low, high) {
        (Some(low), Some(high)) => Some(ShiftLines {
            low,
            high,
            label: shift_label(low - high),
        }),
        (Some(_low), None) => None,
        (None, Some(_high)) => None,
        (None, None) => None,
    }
}

/// "Lamb shift <值>"，单位选与量级相称的那个（≥1 MHz 用 MHz，否则 kHz）。
fn shift_label(delta_hz: f64) -> String {
    let magnitude = delta_hz.abs();
    match magnitude >= 1.0e6 {
        true => format!("Lamb shift {:.3} MHz", magnitude / 1.0e6),
        false => format!("Lamb shift {:.1} kHz", magnitude / 1.0e3),
    }
}

const STYLE: &str = r#"<style>
/* 不设 max-width：报告宽度跟着宿主容器走，宽屏就铺满（图由 ResizeObserver 跟尺寸重排） */
.qtool-power{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
.qtool-power .qtool-power-maps{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
/* 每张图至少 420px：容器放不下两张时自动换行、各占一行（窄屏不再互相压扁） */
.qtool-power .qtool-power-maps>.qtool-2d{flex:1 1 420px;min-width:0}
/* 下方两张线图：flex-basis 100% 强制换到卡片内的下一行，内部再并排/换行 */
.qtool-power .qtool-power-lines{flex:1 1 100%;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.qtool-power .qtool-power-lines>.qtool-line{flex:1 1 420px;min-width:0}
/* 线图比二维图扁：容器高度 = plotly 上边距 + 纸面 + 下边距，18/5 是为"纸面高度与
   改动前一致"配的（756px 宽时容器 210px、纸面 120px） */
.qtool-power .qtool-line-plot{width:100%;aspect-ratio:18/5;max-height:60vh}
/* plotly 自带工具栏固定贴在容器右上角（top:2px），上边距压小后就会压在图上 ——
   挪到图名行右端（图名行高约 21px + 6px 外边距，工具栏高 19px，居中即 -26px） */
.qtool-power .js-plotly-plot .modebar{top:-26px}
</style>"#;
