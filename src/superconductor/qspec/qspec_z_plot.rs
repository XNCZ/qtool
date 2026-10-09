//! qspec 对 Z 偏置扫描的二维报告（feature = "plot"）。
//!
//! - 上方热图：颜色 = P1，纵轴 = Z 偏置，由 [`heatmap`] 生成（点击取该点到右/下边线，带 error bar）；
//! - **hover 到某一行** → 跟随光标的对话气泡显示该 Z 处的四面板 qspec 报告（首次使用时才渲染）；
//! - **double click** → 把气泡**原地**钉成固定浮层（可多个、按行去重），每个浮层带 × 关闭；
//! - 下方线图：下拉框在洛伦兹的四个参数间切换，画该参数逐 Z 的**拟合值 + stderr**；
//! - 产物是自包含 div，宿主页面需提供 plotly.js（见
//!   [`qspec_plot::PLOTLY_JS_CDN`](crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN)）。
//!
//! 各行的 XY 频率轴可以不同（动窗扫描：窗心逐偏置外推，每行只扫一小段）：行自带轴
//! （[`QspecZLine::freqs`]）时按行用，热图按各轴的**并集**铺开、某行没覆盖到的格子留空；
//! 不给则各行共用公共轴（[`qspec_z_plot_div`] 的第一个参数）。
//!
//! 通量调谐（f01 vs Z）的拟合在 [`super::flux`]；把它的结果交给 [`qspec_z_plot_div`] 的最后
//! 一个形参，报告最下方会多一张面板：实测峰位（点）叠拟合曲线，下面是参数表。

use crate::superconductor::qspec::flux::{FluxError, FluxFit};
use crate::superconductor::qspec::qspec::{QspecError, QspecFit};
use crate::superconductor::qspec::qspec_plot::{PARAM_UNITS, qspec_panel_div};
use crate::utils::data::{Column, Datum, Payload, Table, fit_row};
use crate::utils::panels::{
    CARD_WIDTH, Card, block, card, DATA_COLOR, FIT_COLOR, dense_grid};
use crate::utils::params::{ParamRow, params_table};
use crate::superconductor::{StateCenters, p1};
use crate::utils::bubble::{Bubble, bubble};
use crate::utils::heatmap::{
    Grid2d, Palette, axis_style, figure_font, heatmap, interactive_config, json_array,
};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{ErrorData, ErrorType, Line, Marker, Mode};
use plotly::layout::{Axis, Layout, Margin};
use plotly::{Plot, Scatter};

/// 气泡/浮层里面板的缩放比例（设计宽度按 [`GRID_2X2`] 原样布局，再整体缩到这么小）。
const SCALE: f64 = 0.62;

/// 线图的上边距，px：图名在 HTML 图名行里、图内也没有顶部轴，plotly 默认的 100px 全是空白，
/// 压到"图名下面一点"即可。自带工具栏（modebar）不占这里——它在 CSS 里被挪到图名行右端。
const LINE_PLOT_TOP: usize = 10;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
/* 宽度上限由卡片容器给（与其余报告同一个设计宽度），图仍由 ResizeObserver 跟尺寸重排 */
.qtool-qspec-z{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
.qtool-qspec-z .qtool-qspec-z-maps{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.qtool-qspec-z .qtool-qspec-z-maps>.qtool-2d{flex:1 1 420px;min-width:0}
/* 下方线图：flex-basis 100% 强制换到卡片内的下一行 */
.qtool-qspec-z .qtool-qspec-z-lines{flex:1 1 100%;min-width:0;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.qtool-qspec-z .qtool-qspec-z-lines>.qtool-line{flex:1 1 420px;min-width:0}
/* 线图比二维图扁：容器高度 = plotly 上边距 + 纸面 + 下边距 */
.qtool-qspec-z .qtool-line-plot{max-height:60vh}
/* 通量调谐面板：basis 100% 让它独占一行，摆在参数线图下方 */
.qtool-qspec-z .qtool-qspec-z-lines>.qtool-flux{flex:1 1 100%}
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
    /// 该行自己的 XY 驱动频率轴 (n,)，Hz；`None` 表示与其余行共用公共轴。
    ///
    /// 动窗扫描（窗心逐偏置外推、每行只扫一小段）用它把逐行的轴带进来；长度须与 `iq` 相同。
    pub freqs: Option<&'a [f64]>,
}

/// 并集轴与逐行对齐。
///
/// 各行的频率轴一般不同：把全部样本按频率摊平排序，同频的挤进同一列，一趟扫出并集轴与每行
/// 的列号（比逐行二分少一层"找不到"的出口）。某行没覆盖到的格子留 NaN——[`json_array`]
/// 把它写成 `null`，plotly 天然留空；矩阵因此始终是稠密矩形（heatmap 的点击/restyle 在
/// JS 里直接按 `Z[row][col]` 下标取值，吃不了参差行）。
///
/// 行轴与该行取值不等长时按 `zip` 截断到较短者。
///
/// 形参:
///     rows: 逐行的 (频率轴, 取值)
///
/// 返回值:
///     (并集轴升序无重复, 对照并集轴的稠密矩阵)
fn align_rows(rows: &[(&[f64], Vec<f64>)]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut samples: Vec<(f64, usize, f64)> = Vec::new();
    for (index, (freqs, values)) in rows.iter().enumerate() {
        for (freq, value) in freqs.iter().zip(values.iter()) {
            samples.push((*freq, index, *value));
        }
    }
    samples.sort_by(|left, right| match left.0.partial_cmp(&right.0) {
        Some(ordering) => ordering,
        None => std::cmp::Ordering::Equal,
    });

    let mut axis: Vec<f64> = Vec::new();
    let mut placements: Vec<(usize, usize, f64)> = Vec::with_capacity(samples.len());
    for (freq, row, value) in samples {
        match axis.last() {
            Some(last) => match *last == freq {
                true => {}
                false => axis.push(freq),
            },
            None => axis.push(freq),
        }
        placements.push((axis.len() - 1, row, value));
    }

    let mut matrix = vec![vec![f64::NAN; axis.len()]; rows.len()];
    for (column, row, value) in placements {
        matrix[row][column] = value;
    }
    (axis, matrix)
}

/// 渲染 qspec vs Z 的扫描报告 div。
///
/// 热图按各行频率轴的**并集**铺开（见 [`align_rows`]）：各行只在自己扫过的频率上有值，
/// 没覆盖到的格子留空；各行的四面板气泡仍用该行自己的轴。
///
/// 形参:
///     freqs_hz: 公共 XY 驱动频率轴 (n,)，Hz；某行自带轴（[`QspecZLine::freqs`]）时该行不用它
///     lines: 逐条谱线（按 Z 递增）
///     states: 各态标定中心；None 表示未标定，P1 由各线自身定轴（须与各行 `fit` 用的是
///             同一套中心，否则热图上的曲线不是被拟合的那条）
///     div_id: 外层 div 的 HTML id（须是合法 id）
///     frame: 可选外框：`Some(title)` 套上卡片框，`None` 裸图
///     flux: 通量调谐（f01 vs Z）的拟合结果——`Some(Ok(fit))` 时最下方多一张面板
///           （实测峰位点 + 拟合曲线 + 参数表），`Some(Err(error))` 时只画点并在表下说明，
///           `None` 表示这一格不做通量拟合；峰位取各行 `fit` 的 `fq`
///
/// 返回值:
///     自包含的 `<div class="qtool-qspec-z">` 片段（热图 + 参数线图 + 通量面板 + 跟随光标的
///     气泡 + 可钉住的浮层），并内蕴一份原始数据供下载（见 [`crate::utils::data`]）；
///     宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn qspec_z_plot_div(
    freqs_hz: &[f64],
    lines: &[QspecZLine<'_>],
    states: Option<&StateCenters>,
    div_id: &str,
    frame: Option<&str>,
    flux: Option<Result<&FluxFit, &FluxError>>,
) -> String {
    let zs: Vec<f64> = lines.iter().map(|line| line.z).collect();

    // 逐行解析出各自的轴与 P1：行自带轴就用它，没有就退回公共轴——解析之后所有行一视同仁，
    // 热图与面板不再关心轴从哪来。热图画的就是各线被拟合的那条 P1（与 [`qspec_fit_plot_div`]
    // 的 P1 面板同一条曲线）；该线没有拟合结果时退回原始投影，至少让数据可见。
    let mut rows: Vec<(&[f64], Vec<f64>)> = Vec::with_capacity(lines.len());
    for line in lines {
        let freqs = match line.freqs {
            Some(axis) => axis,
            None => freqs_hz,
        };
        let prob = match &line.fit {
            Ok(fit) => fit.p1.clone(),
            Err(_) => p1(line.iq, states),
        };
        rows.push((freqs, prob));
    }
    let (union, p1_matrix) = align_rows(&rows);

    let map = heatmap(
        &Grid2d {
            x: &union,
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
        let panel_html = qspec_panel_div(rows[row].0, line.iq, states, line.fit, &panel_base);
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
        panel_width: CARD_WIDTH,
        scale: SCALE,
    });

    // 通量调谐面板（可选）：横坐标与热图共用 Z，实测峰位取各行拟合出的 `fq`
    let flux_html = match flux {
        Some(result) => {
            let peaks: Vec<f64> = lines
                .iter()
                .map(|line| match &line.fit {
                    Ok(fit) => fit.result.model.fq,
                    Err(_not_fitted) => f64::NAN,
                })
                .collect();
            flux_panel(&format!("{div_id}-flux"), &zs, &peaks, &result)
        }
        None => String::new(),
    };

    // 图组（热图 + 下方线图 + 通量面板）作为一个整体套框：`frame` 由调用方给
    let style = format!("{STYLE}{}", crate::utils::heatmap::title_bar_style());
    let body = format!(
        "<div class=\"qtool-qspec-z-maps\">{map}\
         <div class=\"qtool-qspec-z-lines\">{param_plot}{flux_html}</div></div>{bubble_html}{templates}"
    );
    let report_data = payload(freqs_hz, lines, states, &flux, div_id);
    card(Card {
        class: "qtool-qspec-z",
        style: style.as_str(),
        div_id,
        body,
        payload: Some(&report_data),
        frame,
    })
}

/// 报告载荷：逐 Z 偏置的原始数据（该行的频率轴、复 IQ、P1、模型）各占一张 `row_<i>` 子表；
/// 逐行的洛伦兹拟合一行进 `fits`（`row` 列指向该行子表，`z` 是该行的偏置）；通量调谐的拟合
/// 结果单占一张 `flux` 表；标定中心进 `states`。
///
/// 行序即热图的行序（`lines` 的次序）。某行自带频率轴时用它，否则用公共轴——与热图、气泡
/// 面板同一口径。
///
/// 形参:
///     freqs_hz: 公共 XY 驱动频率轴 (n,)，Hz
///     lines: 逐条谱线（按 Z 递增）
///     states: 各态标定中心
///     flux: 通量调谐拟合结果；`Some(Ok(..))` 时多一张 `flux` 表
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（`fits` 在前，随后 `flux`、`states`、逐行一张 `row_<i>`）
fn payload(
    freqs_hz: &[f64],
    lines: &[QspecZLine<'_>],
    states: Option<&StateCenters>,
    flux: &Option<Result<&FluxFit, &FluxError>>,
    div_id: &str,
) -> Payload {
    let mut report = Payload::new(div_id);
    let mut fits: Vec<(Vec<Column>, Vec<Datum>)> = Vec::new();
    let mut series: Vec<Table> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let row_freqs: &[f64] = match line.freqs {
            Some(axis) => axis,
            None => freqs_hz,
        };
        let prob: Vec<f64> = match &line.fit {
            Ok(fit) => fit.p1.clone(),
            Err(_) => p1(line.iq, states),
        };
        let model = match &line.fit {
            Ok(fit) => Some(fit.result.model.at(row_freqs)),
            Err(_) => None,
        };
        let mut data = Table::new(
            &format!("row_{index}"),
            vec![
                Column::real("freq", "Hz"),
                Column::complex("iq", "a.u."),
                Column::real("p1", "1"),
                Column::real("model", "1"),
            ],
        );
        for point in 0..line.iq.len() {
            data.push(&[
                Datum::Real(row_freqs[point]),
                Datum::Complex(line.iq[point].re, line.iq[point].im),
                Datum::Real(prob[point]),
                match &model {
                    Some(values) => Datum::Real(values[point]),
                    None => Datum::Missing,
                },
            ]);
        }
        series.push(data);
        match &line.fit {
            Ok(fit) => {
                fits.push(fit_row(
                    &fit.result.params,
                    &PARAM_UNITS,
                    &[
                        (Column::reference("row", "1"), Datum::Real(index as f64)),
                        (Column::real("z", "V"), Datum::Real(line.z)),
                    ],
                ));
            }
            Err(_) => {}
        }
    }
    match fits.first() {
        Some((columns, _)) => {
            let mut table = Table::new("fits", columns.clone());
            for (_, row) in &fits {
                table.push(row);
            }
            report.table(table);
        }
        None => {}
    }
    match flux {
        Some(Ok(fit)) => {
            let (columns, row) = fit_row(&fit.result.params, &FLUX_UNITS, &[]);
            let mut table = Table::new("flux", columns);
            table.push(&row);
            report.table(table);
        }
        Some(Err(_)) => {}
        None => {}
    }
    match states {
        Some(centers) => {
            let mut table = Table::new("states", vec![Column::complex("center", "a.u.")]);
            for center in centers.as_slice() {
                table.push(&[Datum::Complex(center.re, center.im)]);
            }
            report.table(table);
        }
        None => {}
    }
    for table in series {
        report.table(table);
    }
    report
}

/// 通量调谐模型各参数的单位（甜点频率与充电能是 Hz，偏置是 V，不对称度无量纲）。
const FLUX_UNITS: [(&str, &str); 5] = [
    ("f_max", "Hz"),
    ("z_offset", "V"),
    ("z_period", "V"),
    ("eta", "Hz"),
    ("asymmetry", "1"),
];

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
    crate::utils::data::plot_script(&plot, div_id)
}

/// 线图面板的外壳：图名行（可含下拉框）+ plot div + 缩放注册 + 附加脚本。
fn line_frame(name: &str, title_bar: &str, plot_html: &str, script: &str) -> String {
    format!(
        "<div class=\"qtool-line\" id=\"{name}\">\
         <div class=\"qtool-line-title\">{title_bar}</div>\
         {}{script}</div>\n",
        block(name, "qtool-line-plot", (18.0, 5.0), plot_html)
    )
}

// =========================================================================
// 通量调谐面板
// =========================================================================

/// 通量调谐面板：实测峰位（点）叠拟合曲线，下面一张参数表。
///
/// 形参:
///     div_id: 面板的 HTML id 前缀（plot div 是 `{div_id}-plot`）
///     zs: 逐行的 Z 偏置 (n,)，V
///     peaks: 逐行的峰位 (n,)，Hz（该行没有拟合结果时给 NaN，plotly 按断点跳过）
///     flux: 通量调谐拟合结果；`Err` 时只画点，并在表下说明失败原因
///
/// 返回值:
///     面板片段（沿用 `qtool-line` 外壳，`qtool-flux` 让它独占一行）
fn flux_panel(
    div_id: &str,
    zs: &[f64],
    peaks: &[f64],
    flux: &Result<&FluxFit, &FluxError>,
) -> String {
    let (plot_html, table) = match flux {
        Ok(fit) => {
            // 拟合曲线铺在与偏置轴同范围的密集网格上（网格退化时退回数据本身）
            let grid = match dense_grid(zs) {
                Some(grid) => grid,
                None => zs.to_vec(),
            };
            let curve = fit.result.model.at(&grid);
            (
                flux_plot_html(div_id, zs, peaks, &grid, &curve),
                params_table(&flux_rows(fit), None),
            )
        }
        Err(error) => (
            flux_plot_html(div_id, zs, peaks, &[], &[]),
            params_table(&[], Some(&format!("not fitted: {error}"))),
        ),
    };
    format!(
        "<div class=\"qtool-line qtool-flux\" id=\"{div_id}\">\
         <div class=\"qtool-line-title\">通量调谐 f01 vs Z</div>\
         {}{table}</div>\n",
        block(div_id, "qtool-line-plot", (18.0, 5.0), &plot_html)
    )
}

/// 通量调谐参数表的各行：三个拟合参数带标准误，两个结构常数原样带上（没有标准误）。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     参数表各行，顺序即渲染顺序
fn flux_rows(fit: &FluxFit) -> Vec<ParamRow<'static>> {
    let model = &fit.result.model;
    // 名字与 `Flux` 的字段一一对应；标准误按名字去 params 里取，格式与 qspec 的参数表同规
    let rows: [(&'static str, &'static str, f64, f64, usize, &'static str); 5] = [
        ("f_max", "sweet-spot frequency", model.f_max, 1e-9, 6, " GHz"),
        ("z_offset", "sweet-spot bias", model.z_offset, 1.0, 5, " V"),
        ("z_period", "bias per flux quantum", model.z_period, 1.0, 5, " V"),
        ("eta", "charging energy Ec/h (fixed)", model.eta, 1e-9, 5, " GHz"),
        ("asymmetry", "junction asymmetry (fixed)", model.asymmetry, 1.0, 4, ""),
    ];
    rows.iter()
        .map(|(name, description, value, scale, digits, unit)| ParamRow {
            name,
            description,
            value: format!("{:.digits$}{unit}", value * scale),
            stderr: match fit.result.params.get(name) {
                Some(parameter) => match parameter.stderr {
                    Some(error) => format!("{:.digits$}{unit}", error * scale),
                    None => "—".to_string(),
                },
                None => "—".to_string(),
            },
        })
        .collect()
}

/// 通量调谐面板的图：实测峰位（点）与拟合曲线（线）。
///
/// 形参:
///     div_id: 图的 HTML id 前缀（plot div 是 `{div_id}-plot`）
///     zs: 实测偏置 (n,)，V
///     peaks: 实测峰位 (n,)，Hz
///     curve_zs: 拟合曲线的偏置网格 (m,)，V；空表示没有曲线可画
///     curve: 拟合曲线在 `curve_zs` 上的取值 (m,)，Hz
///
/// 返回值:
///     自包含的 plotly 图片段
fn flux_plot_html(
    div_id: &str,
    zs: &[f64],
    peaks: &[f64],
    curve_zs: &[f64],
    curve: &[f64],
) -> String {
    let points: Box<dyn Trace> = Scatter::new(zs.to_vec(), peaks.to_vec())
        .mode(Mode::Markers)
        .marker(Marker::new().color(DATA_COLOR).size(6))
        .show_legend(false)
        .x_axis("x")
        .y_axis("y");
    let mut plot = Plot::new();
    plot.add_trace(points);
    match curve.is_empty() {
        true => {}
        false => {
            let fitted: Box<dyn Trace> = Scatter::new(curve_zs.to_vec(), curve.to_vec())
                .mode(Mode::Lines)
                .line(Line::new().color(FIT_COLOR).width(1.5))
                .show_legend(false)
                .x_axis("x")
                .y_axis("y");
            plot.add_trace(fitted);
        }
    }
    let layout = Layout::new()
        .font(figure_font())
        .margin(Margin::new().top(LINE_PLOT_TOP))
        .x_axis(axis_style(Axis::new().title("Z (V)")))
        .y_axis(axis_style(Axis::new().title("f01 (Hz)")));
    plot.set_layout(layout);
    plot.set_configuration(interactive_config());
    crate::utils::data::plot_script(&plot, div_id)
}
