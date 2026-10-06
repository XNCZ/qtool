//! qspec 拟合结果的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ 平面、P1）+ 页脚的 HTML 片段，可直接插入汇总
//! 报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只生成 `<div>` 与
//! `Plotly.newPlot` 调用（[`plotly::Plot::to_inline_html`]）；宿主页面需自行加载 plotly.js，
//! 可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板内容对齐 baseline `qspec_analysis.py::qspec_plot`：|S21| 与相位只画数据（拟合在 P1
//! 空间进行，这两个量没有对应的模型曲线）；P1 面板叠洛伦兹拟合曲线与残差；IQ 面板画
//! [`p1`] 定轴所用的投影几何——样本点按 P1 着色、投影轴、以及归一化钉在 0 和 1 上的两个
//! 参考点。参数框放在 IQ 面板：那里数据是一条细线、留白最多，不像 P1 面板会被峰顶压住。

use crate::superconductor::qspec::qspec::{QspecError, QspecFit};
use crate::superconductor::{StateCenters, p1};
use crate::utils::bubble::framize;
use crate::utils::heatmap::{
    TITLE_COLOR, TITLE_SIZE, axis_style, escape_html, figure_font, interactive_config,
};
use crate::utils::params::{ParamRow, params_table};
use crate::utils::unwrap_phase;
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{
    Anchor, AxisSide, ColorScale, ColorScaleElement, DashType, Font, Line, Marker, MarkerSymbol,
    Mode, Position,
};
use plotly::layout::{Annotation, Axis, Layout, Legend};
use plotly::{Plot, Scatter};

/// plotly.js 的 CDN 引入标签；与 [`crate::superconductor::s21::s21_plot`] 用的是同一个版本，
/// 这里转出以便 qspec 报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::s21::s21_plot::PLOTLY_JS_CDN;

/// 四面板报告的设计宽度（px），同时定下 `.qtool-qspec` 的高宽比，也是气泡里面板的布局宽度。
pub(crate) const DESIGN_WIDTH: f64 = 1000.0;

/// 与 [`DESIGN_WIDTH`] 配套的设计高度（px）。
const DESIGN_HEIGHT: f64 = 760.0;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）；尺寸相关的两条
/// 规则由 [`size_style`] 单独生成。
const STYLE: &str = r#"<style>
.qtool-qspec{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即 DESIGN_WIDTH:DESIGN_HEIGHT） */
.qtool-qspec .qtool-plot{width:100%;max-height:85vh}
.qtool-qspec .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
.qtool-qspec .qtool-footer{margin-top:6px;text-align:right;font-size:11px;color:#9ca3af}
</style>"#;

/// 尺寸相关的 CSS（卡片宽度上限 + 图的高宽比）：CSS 读不到 Rust 常量，所以由
/// [`DESIGN_WIDTH`]/[`DESIGN_HEIGHT`] 插值生成，不在样式表里再写一份。
fn size_style() -> String {
    format!(
        "<style>.qtool-qspec{{max-width:{width}px}}.qtool-qspec .qtool-plot{{aspect-ratio:{width}/{height}}}</style>",
        width = DESIGN_WIDTH,
        height = DESIGN_HEIGHT
    )
}

/// 数据点与拟合曲线的颜色（与 s21 报告同一套：紫=数据、红=模型、金=残差）。
pub(crate) const DATA_COLOR: &str = "#663399";
pub(crate) const FIT_COLOR: &str = "#d62728";
const RESIDUAL_COLOR: &str = "#eab308";

/// 两态着色（沿用 baseline `iq_norm.py` 里的那对既有配色）：P1 = 0 端蓝、1 端红。
const ZERO_COLOR: &str = "royalblue";
const ONE_COLOR: &str = "crimson";

/// 投影轴的颜色：灰色——红色在本图里已经专指 |1> 端，再用会把两套语义混在一起。
const AXIS_COLOR: &str = "gray";

/// 密集模型网格的采样倍率：点数 = 数据点数 × 该值。
const DENSE_FACTOR: usize = 10;

/// 把一条 qspec 谱线渲染成自包含的 HTML div。
///
/// 形参:
///     freqs_hz: XY 驱动频率数组 (n,)，Hz，取绝对频率
///     iq: 该谱线的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定（须与
///             [`crate::superconductor::qspec::qspec_fit`] 用的是同一组中心，
///             否则画出来的曲线不是被拟合的那条）
///     fit: 拟合结果；`Ok` 时在 P1 面板叠模型曲线与残差、在 IQ 面板列参数，`Err` 时只画
///          数据、参数框位置显示错误文本
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-qspec">` 片段（图 + 页脚）；宿主页面需自行加载
///     plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn qspec_fit_plot_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    fit: Result<&QspecFit, &QspecError>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let overlay = overlay(&fit, freqs_hz);
    let plot = fit_plot(freqs_hz, iq, states, &fit, &overlay);
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = plot.to_inline_html(Some(plot_div_id.as_str()));

    let mut html = String::new();
    html.push_str(&format!(
        "<div class=\"qtool-qspec\" id=\"{}\">",
        escape_html(div_id)
    ));
    html.push_str(STYLE);
    html.push_str(&size_style());
    html.push_str(&format!("<div class=\"qtool-plot\">{plot_html}</div>"));
    html.push_str(&crate::utils::resize::register_script(div_id));
    html.push_str(&overlay.table_html());
    html.push_str(&format!(
        "<div class=\"qtool-footer\">Powered by qtool v{}</div>",
        env!("CARGO_PKG_VERSION")
    ));
    html.push_str("</div>\n");
    framize(&html, frame)
}

/// 拟合结果的叠加层：`Ok` 时给出模型曲线与参数文本，`Err` 时把错误交给页面。
///
/// 把 `Result` 收成一个值之后，画 trace 与排布局的代码都不必再分支——只有真正不同的
/// 两处（P1 面板的模型曲线、IQ 面板的标注框）去问它要东西。
enum Overlay<'a> {
    Some {
        /// 数据频率网格上的模型值（算残差用）
        model: Vec<f64>,
        /// 密集网格上的模型值（画平滑的拟合曲线用）
        dense_freqs: Vec<f64>,
        dense_model: Vec<f64>,
        /// 参数表各行
        rows: Vec<ParamRow<'a>>,
        /// 表下提示（未收敛、取向被翻转）
        note: Option<String>,
    },
    Absent {
        error: &'a QspecError,
    },
}

impl Overlay<'_> {
    /// 数据网格上的模型值；无拟合时为 None。
    fn model(&self) -> Option<&[f64]> {
        match self {
            Self::Some { model, .. } => Some(model),
            Self::Absent { .. } => None,
        }
    }

    /// 密集网格与模型值；无拟合时为 None。
    fn dense(&self) -> Option<(&[f64], &[f64])> {
        match self {
            Self::Some {
                dense_freqs,
                dense_model,
                ..
            } => Some((dense_freqs, dense_model)),
            Self::Absent { .. } => None,
        }
    }

    /// 图下方的参数表；无拟合时是错误条。
    fn table_html(&self) -> String {
        match self {
            Self::Some { rows, note, .. } => params_table(rows, note.as_deref()),
            Self::Absent { error } => format!(
                "<div class=\"qtool-error\">Fit failed: {}</div>",
                escape_html(&error.to_string())
            ),
        }
    }
}

/// 由拟合结果构造叠加层：模型在数据网格与密集网格上各求一次值。
///
/// 形参:
///     fit: 拟合结果
///     freqs_hz: 数据频率轴 (n,)，Hz
///
/// 返回值:
///     叠加层；无拟合或频点数不足以插值时，密集曲线一栏为空
fn overlay<'a>(fit: &'a Result<&QspecFit, &QspecError>, freqs_hz: &[f64]) -> Overlay<'a> {
    match fit {
        Ok(result) => {
            let model = result.result.model.at(freqs_hz);
            let (dense_freqs, dense_model) = match dense_grid(freqs_hz) {
                Some(grid) => {
                    let values = result.result.model.at(&grid);
                    (grid, values)
                }
                None => (Vec::new(), Vec::new()),
            };
            Overlay::Some {
                model,
                dense_freqs,
                dense_model,
                rows: param_rows(result),
                note: note_of(result),
            }
        }
        Err(error) => Overlay::Absent { error },
    }
}

/// 密集频率网格（[`DENSE_FACTOR`]× 数据点数，闭区间）；点数不足以插值时返回 None。
fn dense_grid(freqs_hz: &[f64]) -> Option<Vec<f64>> {
    let count = DENSE_FACTOR * freqs_hz.len();
    match (freqs_hz.first(), freqs_hz.last()) {
        (Some(first), Some(last)) => match count >= 2 {
            true => Some(
                (0..count)
                    .map(|k| first + (last - first) * k as f64 / (count - 1) as f64)
                    .collect(),
            ),
            false => None,
        },
        (None, _) => None,
        (_, None) => None,
    }
}

/// 参数表：峰中心、半高全宽、峰高、基线，各带标准误。
///
/// 标准误缺失（协方差不可用）或非有限时给占位符，不伪造一个 0 误差。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     参数表各行，顺序即渲染顺序
fn param_rows(fit: &QspecFit) -> Vec<ParamRow<'static>> {
    let model = &fit.result.model;
    // 名字与 `Lorentz` 的字段一一对应；标准误按名字去 params 里取
    let rows: [(&'static str, &'static str, f64, f64, usize, &'static str); 4] = [
        ("fq", "peak centre (qubit frequency)", model.fq, 1e-9, 6, " GHz"),
        (
            "fwhm",
            "full width at half maximum",
            model.fwhm,
            1e-6,
            3,
            " MHz",
        ),
        ("amp", "peak height above the baseline", model.amp, 1.0, 4, ""),
        (
            "offset",
            "baseline level away from resonance",
            model.offset,
            1.0,
            4,
            "",
        ),
    ];
    rows.iter()
        .map(|(name, description, value, scale, digits, unit)| ParamRow {
            name,
            description,
            value: format!("{:.digits$}{unit}", value * scale),
            stderr: match stderr_of(fit, name) {
                Some(error) => format!("{:.digits$}{unit}", error * scale),
                None => "—".to_string(),
            },
        })
        .collect()
}

/// 表下提示：求解器未收敛时给一行说明。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     提示文本；收敛时返回 None
fn note_of(fit: &QspecFit) -> Option<String> {
    match fit.result.success {
        true => None,
        false => Some(format!("not converged: {}", fit.result.message)),
    }
}

/// 某个参数的标准误；缺失或非有限时返回 None。
fn stderr_of(fit: &QspecFit, name: &str) -> Option<f64> {
    match fit.result.params.get(name) {
        Some(parameter) => match parameter.stderr {
            Some(value) => match value.is_finite() {
                true => Some(value),
                false => None,
            },
            None => None,
        },
        None => None,
    }
}

/// [`p1`] 在该数据上实际使用的两个参考点（baseline `iq_norm.projection_refs`）。
///
/// 给定各态标定中心时，参考点即前两态；未标定时投影轴由数据自身拟合直线得到
/// （见 [`direction`]），参考点取投影最小与最大的那两个样本——min-max 归一化正是把 0 与 1
/// 钉在它们上。两点连成的线段即投影轴（IQ 面板画的就是这条线）。
///
/// 形参:
///     iq: 复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定
///     prob: 已定向的 P1 曲线 (n,)，与 `iq` 等长；未标定时用它找两个端点
///
/// 返回值:
///     (|0> 端参考点, |1> 端参考点)；`iq` 为空时两点皆为 NaN
fn projection_refs(
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
) -> (Complex64, Complex64) {
    match states {
        Some(centers) => match centers.as_slice() {
            [zero, one, ..] => (*zero, *one),
            // 标定不足两态：与 [`p1`] 一致，参考点无定义
            [] | [_] => (
                Complex64::new(f64::NAN, f64::NAN),
                Complex64::new(f64::NAN, f64::NAN),
            ),
        },
        None => {
            // P1 最小与最大的那两个样本——归一化正是把 0 与 1 钉在它们上。曲线已定向，
            // 所以谁是最小谁就是 |0> 端，不必再去问投影轴的朝向
            let (mut ref_zero, mut ref_one) = (
                Complex64::new(f64::NAN, f64::NAN),
                Complex64::new(f64::NAN, f64::NAN),
            );
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for (z, value) in iq.iter().zip(prob.iter()) {
                match *value < lo {
                    true => {
                        lo = *value;
                        ref_zero = *z;
                    }
                    false => {}
                }
                match *value > hi {
                    true => {
                        hi = *value;
                        ref_one = *z;
                    }
                    false => {}
                }
            }
            (ref_zero, ref_one)
        }
    }
}

/// 四面板的名字与轴绑定；轴按列对应 x/x2/x3/x4。
#[derive(Clone, Copy)]
enum Panel {
    Magnitude,
    Phase,
    Iq,
    Norm,
}

impl Panel {
    const ALL: [Panel; 4] = [Panel::Magnitude, Panel::Phase, Panel::Iq, Panel::Norm];

    /// 本面板的 (x, y) 轴引用。
    fn axis_refs(self) -> (&'static str, &'static str) {
        match self {
            Self::Magnitude => ("x", "y"),
            Self::Phase => ("x2", "y2"),
            Self::Iq => ("x3", "y3"),
            Self::Norm => ("x4", "y4"),
        }
    }

    /// x 轴标题（写量，不写区域）。
    fn x_title(self) -> &'static str {
        match self {
            Self::Magnitude => "freq (Hz)",
            Self::Phase => "freq (Hz)",
            Self::Iq => "I",
            Self::Norm => "freq (Hz)",
        }
    }

    /// y 轴标题。
    fn y_title(self) -> &'static str {
        match self {
            Self::Magnitude => "|S21|",
            Self::Phase => "Phase (rad)",
            Self::Iq => "Q",
            Self::Norm => "P1",
        }
    }

    /// 面板名（写区域，不写量）。
    fn name(self) -> &'static str {
        match self {
            Self::Magnitude => "Magnitude",
            Self::Phase => "Phase",
            Self::Iq => "IQ plane",
            Self::Norm => "P1",
        }
    }
}

/// 2×2 网格的 paper 分数：左右列各 40%、上下排各 40%，其余留给轴标题。
const X_LEFT: [f64; 2] = [0.0, 0.40];
const X_RIGHT: [f64; 2] = [0.62, 1.0];
const Y_TOP: [f64; 2] = [0.58, 1.0];
const Y_BOTTOM: [f64; 2] = [0.0, 0.40];

/// 一个面板的标题：x 取列中点、y 取排顶边（`yanchor=Bottom` ⇒ 字落在面板上方），
/// 字号/颜色对齐 plotly layout title 的默认外观（见 [`TITLE_SIZE`]）。
fn panel_title(panel: Panel, x_domain: [f64; 2], y_domain: [f64; 2]) -> Annotation {
    Annotation::new()
        .text(panel.name())
        .x((x_domain[0] + x_domain[1]) / 2.0)
        .y(y_domain[1])
        .x_ref("paper")
        .y_ref("paper")
        .x_anchor(Anchor::Center)
        .y_anchor(Anchor::Bottom)
        .show_arrow(false)
        // font 未指定的属性（family）沿用 layout.font
        .font(Font::new().size(TITLE_SIZE).color(TITLE_COLOR))
}

/// 带虚线的数据点：折线看趋势，点看采样位置（与 baseline 的 `lines+markers` 一致）。
fn series(
    x: Vec<f64>,
    y: Vec<f64>,
    name: &str,
    color: &'static str,
    show_legend: bool,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        // 同名 trace 归入同一 legendgroup：点一次图例即可开关多个面板里的同类曲线
        .legend_group(name)
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(color).dash(DashType::Dash))
        .marker(Marker::new().color(color).size(7))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 纯数据点（P1 面板的样本）。
fn markers(
    x: Vec<f64>,
    y: Vec<f64>,
    name: &str,
    color: &'static str,
    show_legend: bool,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        .legend_group(name)
        .mode(Mode::Markers)
        .marker(Marker::new().color(color).size(7))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 曲线（拟合模型）。
fn curve(
    x: Vec<f64>,
    y: Vec<f64>,
    name: &str,
    color: &'static str,
    width: f64,
    show_legend: bool,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        .legend_group(name)
        .mode(Mode::Lines)
        .line(Line::new().color(color).width(width))
        .show_legend(show_legend)
        .hover_info(plotly::common::HoverInfo::Skip)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 残差标记：金色小点、画在右轴（`y_ref`）上。
fn residual_markers(
    x: Vec<f64>,
    y: Vec<f64>,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name("Residual")
        .legend_group("Residual")
        .mode(Mode::Markers)
        .marker(Marker::new().color(RESIDUAL_COLOR).size(5))
        .show_legend(false)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 按 P1 着色的样本点：|0> 端蓝、|1> 端红，状态沿投影轴的过渡一眼可读。
fn colored_samples(
    x: Vec<f64>,
    y: Vec<f64>,
    values: &[f64],
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    // P1 已经定向（0 端即基态），配色无需再翻：低端蓝、高端红
    let scale = vec![
        ColorScaleElement(0.0, ZERO_COLOR.to_string()),
        ColorScaleElement(1.0, ONE_COLOR.to_string()),
    ];
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .color_array(values.to_vec())
                .color_scale(ColorScale::Vector(scale))
                .cmin(0.0)
                .cmax(1.0)
                .show_scale(false)
                .size(8),
        )
        .show_legend(false)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 投影轴：跨两个参考点的灰色虚线（红色在本图里已专指 |1> 端）。
fn projection_axis(
    ref_zero: Complex64,
    ref_one: Complex64,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(
        vec![ref_zero.re, ref_one.re],
        vec![ref_zero.im, ref_one.im],
    )
    .mode(Mode::Lines)
    .line(Line::new().color(AXIS_COLOR).width(2.0).dash(DashType::Dash))
    .show_legend(false)
    .hover_info(plotly::common::HoverInfo::Skip)
    .x_axis(x_ref)
    .y_axis(y_ref);
    trace
}

/// 一个态参考点：`×` 号加标签。
fn ref_point(
    point: Complex64,
    label: &str,
    color: &'static str,
    position: Position,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(vec![point.re], vec![point.im])
        .mode(Mode::MarkersText)
        .text_array(vec![label.to_string()])
        .text_position(position)
        .marker(
            Marker::new()
                .color(color)
                .size(12)
                .symbol(MarkerSymbol::X),
        )
        .show_legend(false)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 某面板的全部 trace（数据 → 拟合 → 残差），无拟合时自动只剩数据。
fn panel_traces(
    panel: Panel,
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
    overlay: &Overlay,
) -> Vec<Box<dyn Trace>> {
    let (x_ref, y_ref) = panel.axis_refs();
    let x_hz = freqs_hz.to_vec();
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            let amp: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
            traces.push(series(x_hz, amp, "Data", DATA_COLOR, true, x_ref, y_ref));
        }
        Panel::Phase => {
            // 相位经 unwrap 解缠绕，否则跳变处会出现整圈假台阶
            let phase = unwrap_phase(&iq.iter().map(|z| z.arg()).collect::<Vec<f64>>());
            traces.push(series(x_hz, phase, "Data", DATA_COLOR, false, x_ref, y_ref));
        }
        Panel::Iq => {
            let (ref_zero, ref_one) = projection_refs(iq, states, prob);
            traces.push(colored_samples(
                iq.iter().map(|z| z.re).collect(),
                iq.iter().map(|z| z.im).collect(),
                prob,
                x_ref,
                y_ref,
            ));
            traces.push(projection_axis(ref_zero, ref_one, x_ref, y_ref));
            let zero_color = ZERO_COLOR;
            let one_color = ONE_COLOR;
            let zero_label = "|0>";
            let one_label = "|1>";
            traces.push(ref_point(
                ref_zero,
                zero_label,
                zero_color,
                Position::BottomRight,
                x_ref,
                y_ref,
            ));
            traces.push(ref_point(
                ref_one,
                one_label,
                one_color,
                Position::TopLeft,
                x_ref,
                y_ref,
            ));
        }
        Panel::Norm => {
            traces.push(markers(
                x_hz.clone(),
                prob.to_vec(),
                "Data",
                DATA_COLOR,
                false,
                x_ref,
                y_ref,
            ));
            match overlay.dense() {
                Some((dense_freqs, dense_model)) => traces.push(curve(
                    dense_freqs.to_vec(),
                    dense_model.to_vec(),
                    "Fit",
                    FIT_COLOR,
                    2.0,
                    true,
                    x_ref,
                    y_ref,
                )),
                None => {}
            }
            match overlay.model() {
                Some(model) => {
                    let residual: Vec<f64> = prob
                        .iter()
                        .zip(model.iter())
                        .map(|(value, fitted)| value - fitted)
                        .collect();
                    // 残差与 P1 差若干数量级，画在右轴 y5 上
                    traces.push(residual_markers(x_hz, residual, "x4", "y5"));
                }
                None => {}
            }
        }
    }
    traces
}

/// 四面板图：|S21| / 相位 / IQ 平面 / P1。
///
/// 形参:
///     freqs_hz: XY 驱动频率数组 (n,)，Hz
///     iq: 该谱线的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定
///     fit: 拟合结果；决定 P1 面板画哪条曲线（胜出取向下的那条）以及 IQ 面板的标签朝向
///     overlay: 拟合叠加层
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
fn fit_plot(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    fit: &Result<&QspecFit, &QspecError>,
    overlay: &Overlay,
) -> Plot {
    // 面板上画的那条 P1：有拟合时是胜出取向下的曲线（即被拟合的那条），否则是原始投影；
    // 两者在取向被翻转时互为 1 − P1，所以必须与拟合用的那条一致，不能另算一遍
    // 面板上画的那条 P1：有拟合时是它实际拟合的那条（已经定向），否则是原始投影
    let prob = match fit {
        Ok(result) => result.p1.clone(),
        Err(_) => p1(iq, states),
    };
    let mut plot = Plot::new();
    for panel in Panel::ALL {
        for trace in panel_traces(panel, freqs_hz, iq, states, &prob, overlay) {
            plot.add_trace(trace);
        }
    }
    plot.set_layout(layout(overlay));
    plot.set_configuration(interactive_config());
    plot
}

/// 2×2 网格布局：四个面板各占一角，P1 面板另配一条右轴画残差。
fn layout(overlay: &Overlay) -> Layout {
    let (prob_range, residual_range) = match overlay.model() {
        Some(model) => {
            let bounds = value_bounds(&[model]);
            let range = pad(bounds);
            (range, residual_axis_range(range))
        }
        None => (None, None),
    };
    Layout::new()
        .font(figure_font())
        .show_legend(true)
        // 四块面板各挂一个标题（纸面坐标：列中点 / 排顶边），缩放窗口时跟着面板走
        .annotations(vec![
            panel_title(Panel::Magnitude, X_LEFT, Y_TOP),
            panel_title(Panel::Phase, X_RIGHT, Y_TOP),
            panel_title(Panel::Iq, X_LEFT, Y_BOTTOM),
            panel_title(Panel::Norm, X_RIGHT, Y_BOTTOM),
        ])
        // 图例放四块图右侧顶部、上边距里（yanchor=Bottom 让图例往上长，而不是默认的
        // 从 paper y=1 往下长——那样会压住相位面板右上角的刻度）
        .legend(
            Legend::new()
                .x(1.0)
                .x_anchor(Anchor::Left)
                .y(1.0)
                .y_anchor(Anchor::Bottom),
        )
        .x_axis(axis_style(
            Axis::new().domain(&X_LEFT).anchor("y").title(Panel::Magnitude.x_title()),
        ))
        .y_axis(axis_style(
            Axis::new().domain(&Y_TOP).anchor("x").title(Panel::Magnitude.y_title()),
        ))
        .x_axis2(axis_style(
            Axis::new().domain(&X_RIGHT).anchor("y2").title(Panel::Phase.x_title()),
        ))
        .y_axis2(axis_style(
            Axis::new().domain(&Y_TOP).anchor("x2").title(Panel::Phase.y_title()),
        ))
        .x_axis3(axis_style(
            Axis::new().domain(&X_LEFT).anchor("y3").title(Panel::Iq.x_title()),
        ))
        .y_axis3(axis_style(
            Axis::new()
                .domain(&Y_BOTTOM)
                .anchor("x3")
                .title(Panel::Iq.y_title())
                .scale_anchor("x3"),
        ))
        .x_axis4(axis_style(
            Axis::new().domain(&X_RIGHT).anchor("y4").title(Panel::Norm.x_title()),
        ))
        .y_axis4(at_range(
            axis_style(
                Axis::new().domain(&Y_BOTTOM).anchor("x4").title(Panel::Norm.y_title()),
            ),
            prob_range,
        ))
        // 残差叠加轴不画网格：否则与数据轴网格叠成双层虚线
        .y_axis5(at_range(
            axis_style(
                Axis::new()
                    .overlaying("y4")
                    .side(AxisSide::Right)
                    .anchor("x4")
                    .title("Residual")
                    .show_grid(false),
            ),
            residual_range,
        ))
}

/// 一组序列的取值范围 [min, max]；全为空或无有限值时返回 None。
fn value_bounds(lists: &[&[f64]]) -> Option<[f64; 2]> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for list in lists {
        for value in list.iter() {
            if value.is_finite() {
                low = low.min(*value);
                high = high.max(*value);
            }
        }
    }
    match low <= high {
        true => Some([low, high]),
        false => None,
    }
}

/// 数据轴相对值域多留的边距（占 span 的比例）。
const AXIS_PAD: f64 = 0.05;

/// 数据轴范围：值域 ±[`AXIS_PAD`]；退化（span ≤ 0）时交回 plotly 自适应。
fn pad(bounds: Option<[f64; 2]>) -> Option<[f64; 2]> {
    match bounds {
        Some([low, high]) => {
            let span = high - low;
            match span > 0.0 {
                true => Some([low - AXIS_PAD * span, high + AXIS_PAD * span]),
                false => None,
            }
        }
        None => None,
    }
}

/// 残差 0 落在面板高度的这个比例处（右轴整体下移这么多）。
const RESIDUAL_ZERO_AT: f64 = 0.12;

/// 残差右轴范围：与数据轴同 span（同一 scale），整体下移使残差 0 落在面板
/// [`RESIDUAL_ZERO_AT`] 高度处。
fn residual_axis_range(data_range: Option<[f64; 2]>) -> Option<[f64; 2]> {
    match data_range {
        Some([low, high]) => {
            let span = high - low;
            match span > 0.0 {
                true => Some([-RESIDUAL_ZERO_AT * span, (1.0 - RESIDUAL_ZERO_AT) * span]),
                false => None,
            }
        }
        None => None,
    }
}

/// 可选地给轴设置 range。
fn at_range(axis: Axis, range: Option<[f64; 2]>) -> Axis {
    match range {
        Some([low, high]) => axis.range(vec![low, high]),
        None => axis,
    }
}
