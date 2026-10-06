//! S21 拟合结果的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ、归一化圆）+ 参数表 + 页脚的 HTML 片段，
//! 可直接插入汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只
//! 生成 `<div>` 与 `Plotly.newPlot` 调用（[`plotly::Plot::to_inline_html`]）；宿主页面
//! 需自行加载 plotly.js，可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板布局与配色对齐 baseline `s12_analysis.py::fit` 的四面板图；相位面板以**频率**为
//! 自变量做线性去趋势，且数据与拟合曲线共用同一条趋势线（baseline 各自独立 detrend，
//! 两条曲线可能整体错开；此处取同趋势以保证重合）。

use crate::utils::bubble::framize;
use crate::utils::heatmap::{
    TITLE_COLOR, TITLE_SIZE, axis_style, escape_html, figure_font, interactive_config,
};
use crate::utils::params::{ParamRow, params_table};
use crate::superconductor::s21::{
    Complex64, JAC_NAMES, S21Error, S21Model, background_at, model_at, notch_at,
};
use crate::utils::{detrend, linear_detrend, unwrap_phase};
use lmfit::ComplexResult;
use plotly::Trace;
use plotly::common::{Anchor, AxisSide, ErrorData, ErrorType, Font, HoverInfo, Line, Marker, Mode};
use plotly::layout::{Annotation, Axis, Layout, Legend};
use plotly::{Plot, Scatter};

/// plotly.js 的 CDN 引入标签（版本与 plotly crate 内嵌的一致）；宿主页面放一次即可。
pub const PLOTLY_JS_CDN: &str =
    r#"<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>"#;

/// 四面板报告的设计宽度（px）。气泡/浮层按这个宽度**原样布局**再整体 `transform: scale`
/// 缩放，所以它同时是 `s21_power_plot` 里面板宽度的来源，也定下 `.qtool-plot` 的高宽比。
pub(crate) const DESIGN_WIDTH: f64 = 1000.0;

/// 与 [`DESIGN_WIDTH`] 配套的设计高度（px）。
pub(crate) const DESIGN_HEIGHT: f64 = 760.0;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）；尺寸相关的两条
/// 规则由 [`size_style`] 单独生成。
const STYLE: &str = r#"<style>
.qtool-s21{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即 DESIGN_WIDTH:DESIGN_HEIGHT；气泡里宽度正好是设计宽度 ⇒ 仍是设计高度） */
.qtool-s21 .qtool-plot{width:100%;max-height:85vh}
.qtool-s21 .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
.qtool-s21 .qtool-footer{margin-top:6px;text-align:right;font-size:11px;color:#9ca3af}
</style>"#;

/// 尺寸相关的 CSS（卡片宽度上限 + 图的高宽比）：CSS 读不到 Rust 常量，所以由
/// [`DESIGN_WIDTH`]/[`DESIGN_HEIGHT`] 插值生成，不在样式表里再写一份。
fn size_style() -> String {
    format!(
        "<style>.qtool-s21{{max-width:{width}px}}.qtool-s21 .qtool-plot{{aspect-ratio:{width}/{height}}}</style>",
        width = DESIGN_WIDTH,
        height = DESIGN_HEIGHT
    )
}

pub(crate) const DATA_COLOR: &str = "#663399";
pub(crate) const FIT_COLOR: &str = "#d62728";
const RESIDUAL_COLOR: &str = "#eab308";

/// 把一条 S21 线渲染成自包含的 HTML div。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 该线的复数 IQ 数组 (n,)
///     sigma: 逐点测量不确定度（与 `s21_fit` 同一口径）；给定时在 |S21| 面板画 σ、
///            在相位面板画 σ/|S21| 的 error bar。长度与 `iq` 不一致时忽略
///     fit: 拟合结果；`Ok` 画拟合曲线（`success == false` 时表内标注未收敛），
///          `Err` 只画数据点、表位置显示错误文本、归一化面板留空
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），
///            `None` 裸图。嵌进气泡/浮层的行面板传 `None`
///
/// 返回值:
///     自包含的 `<div class="qtool-s21">` 片段（图 + 参数表 + 页脚）；
///     宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn s21_fit_plot_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let data = DataView::new(freqs_hz, iq, sigma);
    let overlay = overlay(fit, freqs_hz, &data);
    let plot = fit_plot(&data, &overlay);
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = plot.to_inline_html(Some(plot_div_id.as_str()));

    let mut html = String::new();
    html.push_str(&format!(
        "<div class=\"qtool-s21\" id=\"{}\">",
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

/// 拟合四面板的 Plotly 图对象（不含标题与参数表）。
///
/// 供需要自行组装页面或懒渲染的调用方使用：[`plotly::Plot::to_inline_html`] 可产出
/// 片段，[`s21_fit_plot_div`] 即在此基础上加标题与参数表。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 该线的复数 IQ 数组 (n,)
///     sigma: 逐点测量不确定度，同 [`s21_fit_plot_div`]
///     fit: 拟合结果，同 [`s21_fit_plot_div`]
///
/// 返回值:
///     四面板（|S21|、相位、IQ、归一化圆）图对象
pub fn s21_fit_plot(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
) -> Plot {
    let data = DataView::new(freqs_hz, iq, sigma);
    let overlay = overlay(fit, freqs_hz, &data);
    fit_plot(&data, &overlay)
}

/// 由数据视图与叠加层装配四面板图（供 div 与 plot 两个入口复用）。
fn fit_plot(data: &DataView, overlay: &FitOverlay<'_>) -> Plot {
    let mut plot = Plot::new();
    for panel in Panel::ALL {
        for trace in panel_traces(panel, data, overlay) {
            plot.add_trace(trace);
        }
    }
    plot.set_layout(layout(overlay, data));
    plot.set_configuration(interactive_config());
    plot
}

// =========================================================================
// 面板
// =========================================================================

/// 四面板之一，统一负责自己的轴绑定与轴标题。
#[derive(Clone, Copy)]
enum Panel {
    Magnitude,
    Phase,
    Iq,
    Norm,
}

impl Panel {
    const ALL: [Panel; 4] = [Panel::Magnitude, Panel::Phase, Panel::Iq, Panel::Norm];

    /// (x 轴引用, y 轴引用)，与 `Layout` 的 `x_axis*`/`y_axis*` 绑定。
    fn axis_refs(self) -> (&'static str, &'static str) {
        match self {
            Panel::Magnitude => ("x", "y"),
            Panel::Phase => ("x2", "y2"),
            Panel::Iq => ("x3", "y3"),
            Panel::Norm => ("x4", "y4"),
        }
    }

    fn x_title(self) -> &'static str {
        match self {
            Panel::Magnitude => "freq (Hz)",
            Panel::Phase => "freq (Hz)",
            Panel::Iq => "I",
            Panel::Norm => "I",
        }
    }

    fn y_title(self) -> &'static str {
        match self {
            Panel::Magnitude => "|S21|",
            Panel::Phase => "Phase (rad)",
            Panel::Iq => "Q",
            Panel::Norm => "Q",
        }
    }

    /// 面板标题（画在该面板正上方；与轴标题分工：标题说"这是哪一块"，轴标题说"画的是什么量"）。
    fn name(self) -> &'static str {
        match self {
            Panel::Magnitude => "Magnitude",
            Panel::Phase => "Phase",
            Panel::Iq => "IQ plane",
            Panel::Norm => "Normalized",
        }
    }
}

/// 原始数据的各视图（与拟合无关）。
struct DataView {
    x_hz: Vec<f64>,
    amp: Vec<f64>,
    phase: Vec<f64>,
    /// 相位趋势线（频率域直线 `slope*f + intercept`，rad/Hz）——数据点与拟合曲线共用
    phase_slope: f64,
    phase_intercept: f64,
    iq_re: Vec<f64>,
    iq_im: Vec<f64>,
    /// 幅值误差（= 传入的 σ），长度不匹配时为 None。
    sigma: Option<Vec<f64>>,
    /// 相位误差 σ/|S21|（rad）。
    phase_sigma: Option<Vec<f64>>,
}

impl DataView {
    fn new(freqs_hz: &[f64], iq: &[Complex64], sigma: Option<&[f64]>) -> Self {
        let x_hz = freqs_hz.to_vec();
        let amp: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
        let raw_phase: Vec<f64> = iq.iter().map(|z| z.arg()).collect();
        let (phase, phase_slope, phase_intercept) = detrend(&x_hz, &unwrap_phase(&raw_phase));
        let sigma = match sigma {
            Some(values) => match values.len() == iq.len() {
                true => Some(values.to_vec()),
                false => None,
            },
            None => None,
        };
        let phase_sigma = match &sigma {
            Some(values) => Some(
                values
                    .iter()
                    .zip(amp.iter())
                    .map(|(value, magnitude)| match *magnitude > 0.0 {
                        true => value / magnitude,
                        false => 0.0,
                    })
                    .collect(),
            ),
            None => None,
        };
        Self {
            x_hz,
            amp,
            phase,
            phase_slope,
            phase_intercept,
            iq_re: iq.iter().map(|z| z.re).collect(),
            iq_im: iq.iter().map(|z| z.im).collect(),
            sigma,
            phase_sigma,
        }
    }
}


// =========================================================================
// 拟合叠加层：成功/失败用同一接口回答"画什么曲线 / 表里写什么"
// =========================================================================

/// 密集拟合曲线（数据点数的 [`DENSE_FACTOR`] 倍）。
struct DenseCurves {
    freqs_hz: Vec<f64>,
    s21: Vec<Complex64>,
    phase: Vec<f64>,
    notch: Vec<Complex64>,
}

enum FitOverlay<'a> {
    Some {
        /// 数据网格上的模型值（残差用）。
        fit_on_data: Vec<Complex64>,
        /// 同上，已按数据趋势去趋势（相位残差用）。
        fit_on_data_phase: Vec<f64>,
        dense: Option<DenseCurves>,
        /// 归一化数据 `(iq − zc)/bg`。
        norm_data: Vec<Complex64>,
        rows: Vec<ParamRow<'a>>,
        note: Option<String>,
    },
    Absent {
        error: &'a S21Error,
    },
}

impl FitOverlay<'_> {
    fn fit_on_data(&self) -> Option<&[Complex64]> {
        match self {
            FitOverlay::Some { fit_on_data, .. } => Some(fit_on_data),
            FitOverlay::Absent { .. } => None,
        }
    }

    fn fit_on_data_phase(&self) -> Option<&[f64]> {
        match self {
            FitOverlay::Some {
                fit_on_data_phase, ..
            } => Some(fit_on_data_phase),
            FitOverlay::Absent { .. } => None,
        }
    }

    fn dense(&self) -> Option<&DenseCurves> {
        match self {
            FitOverlay::Some { dense, .. } => match dense {
                Some(curves) => Some(curves),
                None => None,
            },
            FitOverlay::Absent { .. } => None,
        }
    }

    fn norm_data(&self) -> Option<&[Complex64]> {
        match self {
            FitOverlay::Some { norm_data, .. } => Some(norm_data),
            FitOverlay::Absent { .. } => None,
        }
    }

    fn table_html(&self) -> String {
        match self {
            FitOverlay::Some { rows, note, .. } => params_table(rows, note.as_deref()),
            FitOverlay::Absent { error } => format!(
                "<div class=\"qtool-error\">Fit failed: {}</div>",
                escape_html(&error.to_string())
            ),
        }
    }
}

/// 由拟合结果构建叠加层；`Ok`/`Err` 只在这里分支一次。
fn overlay<'a>(
    fit: Result<&'a ComplexResult<S21Model>, &'a S21Error>,
    freqs_hz: &[f64],
    data: &DataView,
) -> FitOverlay<'a> {
    match fit {
        Ok(result) => {
            let params = &result.model;
            let fit_on_data: Vec<Complex64> =
                freqs_hz.iter().map(|f| model_at(*f, params)).collect();
            let raw_phase: Vec<f64> = fit_on_data.iter().map(|z| z.arg()).collect();
            let fit_on_data_phase =
                linear_detrend(freqs_hz, &unwrap_phase(&raw_phase), data.phase_slope, data.phase_intercept);

            let dense = match dense_grid(freqs_hz) {
                Some(dense_freqs) => {
                    let s21: Vec<Complex64> =
                        dense_freqs.iter().map(|f| model_at(*f, params)).collect();
                    let notch: Vec<Complex64> =
                        dense_freqs.iter().map(|f| notch_at(*f, params)).collect();
                    let raw: Vec<f64> = s21.iter().map(|z| z.arg()).collect();
                    let phase =
                        linear_detrend(&dense_freqs, &unwrap_phase(&raw), data.phase_slope, data.phase_intercept);
                    Some(DenseCurves {
                        freqs_hz: dense_freqs.clone(),
                        s21,
                        phase,
                        notch,
                    })
                }
                None => None,
            };

            let offset = Complex64::new(params.zc_re, params.zc_im);
            let norm_data: Vec<Complex64> = freqs_hz
                .iter()
                .zip(data.iq_re.iter().zip(data.iq_im.iter()))
                .map(|(f, (re, im))| {
                    (Complex64::new(*re, *im) - offset) / background_at(*f, params)
                })
                .collect();

            let note = match result.success {
                true => None,
                false => Some(format!("not converged: {}", result.message)),
            };

            FitOverlay::Some {
                fit_on_data,
                fit_on_data_phase,
                dense,
                norm_data,
                rows: param_rows(result),
                note,
            }
        }
        Err(error) => FitOverlay::Absent { error },
    }
}

/// 密集拟合曲线的采样倍率：点数 = 数据点数 × 该值。
const DENSE_FACTOR: usize = 10;

/// 密集频率网格（[`DENSE_FACTOR`]× 数据点数，闭区间）；点数不足以插值时返回 None。
fn dense_grid(freqs: &[f64]) -> Option<Vec<f64>> {
    let count = DENSE_FACTOR * freqs.len();
    match (freqs.first(), freqs.last()) {
        (Some(first), Some(last)) => match count >= 2 {
            true => Some(
                (0..count)
                    .map(|k| first + (last - first) * k as f64 / (count - 1) as f64)
                    .collect(),
            ),
            false => None,
        },
        (None, Some(_)) => None,
        (Some(_), None) => None,
        (None, None) => None,
    }
}

/// 参数表：11 个拟合参数 + 2 个派生量，值/stderr 按参数各自的单位格式化。
fn param_rows(result: &ComplexResult<S21Model>) -> Vec<ParamRow<'static>> {
    let values = result.model.to_array();
    let mut rows: Vec<ParamRow> = JAC_NAMES
        .iter()
        .zip(values.iter())
        .map(|(name, value)| ParamRow {
            name,
            description: describe(name),
            value: format_param(name, *value),
            stderr: stderr_text(result, name),
        })
        .collect();
    let derived = [
        ("qi", result.model.qi),
        ("kappa_ex", result.model.kappa_ex),
    ];
    for (name, value) in derived {
        rows.push(ParamRow {
            name,
            description: describe(name),
            value: format_param(name, value),
            stderr: stderr_text(result, name),
        });
    }
    rows
}

/// 参数含义（英文，渲染进报告表格）。
fn describe(name: &str) -> &'static str {
    match name {
        "fr" => "resonance frequency",
        "ql" => "loaded quality factor",
        "qc" => "coupling quality factor",
        "theta" => "notch phase",
        "ap" => "Duffing nonlinearity",
        "tau" => "background delay",
        "a" => "cos-branch background amplitude",
        "b" => "sin-branch background amplitude",
        "phi" => "background phase",
        "zc_re" => "constant complex offset (real)",
        "zc_im" => "constant complex offset (imag)",
        "qi" => "internal quality factor, 1/(1/ql - cos(theta)/qc)",
        "kappa_ex" => "external coupling rate, fr/qc",
        _ => "",
    }
}

/// 参数的单位后缀：一律 SI 基本单位（频率 Hz、时间 s、相位 rad），量纲为一的不带单位。
pub(crate) fn unit_of(name: &str) -> &'static str {
    match name {
        "fr" | "kappa_ex" => " Hz",
        "tau" => " s",
        "theta" | "phi" => " rad",
        _ => "",
    }
}

fn format_param(name: &str, value: f64) -> String {
    format!("{value:.6e}{}", unit_of(name))
}

/// stderr 文本；crate 在协方差不可得时给 None，显示为 "—"。
fn stderr_text(result: &ComplexResult<S21Model>, name: &str) -> String {
    match result.params.get(name) {
        Some(parameter) => match parameter.stderr {
            Some(value) => format_stderr(name, value),
            None => "—".to_string(),
        },
        None => "—".to_string(),
    }
}

/// stderr 恒用科学计数（量级通常远小于参数本身），单位与参数一致。
fn format_stderr(name: &str, value: f64) -> String {
    format!("{value:.2e}{}", unit_of(name))
}

// =========================================================================
// 曲线与布局
// =========================================================================

fn markers(
    x: Vec<f64>,
    y: Vec<f64>,
    error_y: Option<Vec<f64>>,
    name: &str,
    color: &'static str,
    size: usize,
    show_legend: bool,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let mut trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        // 同名 trace 归入同一 legendgroup：点一次图例即可同时开关四个面板里的同类曲线
        .legend_group(name)
        .mode(Mode::Markers)
        .marker(Marker::new().color(color).size(size))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    match error_y {
        Some(values) => {
            trace = trace.error_y(
                ErrorData::new(ErrorType::Data)
                    .array(values)
                    .visible(true)
                    .color(color)
                    .thickness(1.0)
                    .width(0),
            );
        }
        None => {}
    }
    trace
}

/// 归一化面板的固定坐标范围（配 [`CENTER`] 用）：text 标注画在正中，缩放面板时不会漂。
const UNIT_RANGE: [f64; 2] = [0.0, 1.0];

/// 面板正中。
const CENTER: f64 = 0.5;

/// 面板中央的纯文本标注（[`CENTER`]，配 x4/y4 的显式 [`UNIT_RANGE`]）。
fn text_label(text: &str, x_ref: &'static str, y_ref: &'static str) -> Box<dyn Trace> {
    let trace: Box<dyn Trace> = Scatter::new(vec![CENTER], vec![CENTER])
        .mode(Mode::Text)
        .text(text.to_string())
        .show_legend(false)
        .hover_info(HoverInfo::Skip)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 残差标记：金色小点、画在右轴（`y_ref`）上，hover 关闭——只让原始数据点响应 hover。
fn residual_markers(
    x: Vec<f64>,
    y: Vec<f64>,
    show_legend: bool,
    x_ref: &'static str,
    y_ref: &'static str,
) -> Box<dyn Trace> {
    let trace: Box<dyn Trace> = Scatter::new(x, y)
        .name("Residual")
        .legend_group("Residual")
        .mode(Mode::Markers)
        .marker(Marker::new().color(RESIDUAL_COLOR).size(5))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

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
    let trace: Box<dyn Trace> = Scatter::new(x, y)
        .name(name)
        .legend_group(name)
        .mode(Mode::Lines)
        .line(Line::new().color(color).width(width))
        .show_legend(show_legend)
        .hover_info(HoverInfo::Skip)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 某面板的全部 trace（数据点 → 拟合线 → 残差），失败时自动只剩数据点。
fn panel_traces(panel: Panel, data: &DataView, overlay: &FitOverlay) -> Vec<Box<dyn Trace>> {
    let (x_ref, y_ref) = panel.axis_refs();
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            traces.push(markers(
                data.x_hz.clone(),
                data.amp.clone(),
                data.sigma.clone(),
                "Data",
                DATA_COLOR,
                7,
                true,
                x_ref,
                y_ref,
            ));
            match overlay.dense() {
                Some(dense) => traces.push(curve(
                    dense.freqs_hz.clone(),
                    dense.s21.iter().map(|z| z.norm()).collect(),
                    "Fit",
                    FIT_COLOR,
                    2.0,
                    true,
                    x_ref,
                    y_ref,
                )),
                None => {}
            }
            match overlay.fit_on_data() {
                Some(fit) => {
                    let residual: Vec<f64> = data
                        .amp
                        .iter()
                        .zip(fit.iter())
                        .map(|(value, model)| value - model.norm())
                        .collect();
                    // 残差与 |S21| 差若干数量级，画在右轴 y5 上
                    traces.push(residual_markers(
                        data.x_hz.clone(),
                        residual,
                        true,
                        "x",
                        "y5",
                    ));
                }
                None => {}
            }
        }
        Panel::Phase => {
            traces.push(markers(
                data.x_hz.clone(),
                data.phase.clone(),
                data.phase_sigma.clone(),
                "Data",
                DATA_COLOR,
                7,
                false,
                x_ref,
                y_ref,
            ));
            match overlay.dense() {
                Some(dense) => traces.push(curve(
                    dense.freqs_hz.clone(),
                    dense.phase.clone(),
                    "Fit",
                    FIT_COLOR,
                    2.0,
                    false,
                    x_ref,
                    y_ref,
                )),
                None => {}
            }
            match overlay.fit_on_data_phase() {
                Some(fit_phase) => {
                    let residual: Vec<f64> = data
                        .phase
                        .iter()
                        .zip(fit_phase.iter())
                        .map(|(value, model)| value - model)
                        .collect();
                    traces.push(residual_markers(
                        data.x_hz.clone(),
                        residual,
                        false,
                        "x2",
                        "y6",
                    ));
                }
                None => {}
            }
        }
        Panel::Iq => {
            traces.push(markers(
                data.iq_re.clone(),
                data.iq_im.clone(),
                None,
                "Data",
                DATA_COLOR,
                7,
                false,
                x_ref,
                y_ref,
            ));
            match overlay.dense() {
                Some(dense) => traces.push(curve(
                    dense.s21.iter().map(|z| z.re).collect(),
                    dense.s21.iter().map(|z| z.im).collect(),
                    "Fit",
                    FIT_COLOR,
                    2.0,
                    false,
                    x_ref,
                    y_ref,
                )),
                None => {}
            }
        }
        Panel::Norm => {
            match overlay.norm_data() {
                Some(norm) => traces.push(markers(
                    norm.iter().map(|z| z.re).collect(),
                    norm.iter().map(|z| z.im).collect(),
                    None,
                    "Data",
                    DATA_COLOR,
                    7,
                    false,
                    x_ref,
                    y_ref,
                )),
                // 无拟合时本面板没有数据：用一条 text trace 标注（同时也让 plotly
                // 注册 x4/y4 轴——没有 trace 的轴不会被创建，标注会失效回落）
                None => traces.push(text_label("No fit result", x_ref, y_ref)),
            }
            match overlay.dense() {
                Some(dense) => traces.push(curve(
                    dense.notch.iter().map(|z| z.re).collect(),
                    dense.notch.iter().map(|z| z.im).collect(),
                    "Fit",
                    FIT_COLOR,
                    2.0,
                    false,
                    x_ref,
                    y_ref,
                )),
                None => {}
            }
        }
    }
    traces
}

/// 2×2 网格的 paper 分数：左右列各 40%、上下排各 40%，其余留给轴标题
/// （左列与下排同值只是设计巧合，两者各自可调）。
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

/// 2×2 网格布局；失败时在归一化面板上标注"无拟合结果"。
fn layout(overlay: &FitOverlay, data: &DataView) -> Layout {
    // 幅值/相位面板的值域（含拟合曲线），据此固定左右两轴的 range：
    // 左轴 = 值域 ±5%，右轴 = 同 span、下移（残差与数据可直接目视比量级）。
    let magnitude_bounds = match overlay.dense() {
        Some(dense) => {
            let fit_amp: Vec<f64> = dense.s21.iter().map(|z| z.norm()).collect();
            value_bounds(&[&data.amp, &fit_amp])
        }
        None => value_bounds(&[&data.amp]),
    };
    let phase_bounds = match overlay.dense() {
        Some(dense) => value_bounds(&[&data.phase, &dense.phase]),
        None => value_bounds(&[&data.phase]),
    };
    let magnitude_range = pad(magnitude_bounds);
    let phase_range = pad(phase_bounds);

    // 2×2 用显式 domain + anchor 定位：左右列各 40%，上下排各 40%（留出轴标题余量）；
    // 主数据用左轴（y / y2），残差用右轴叠加（overlaying y / y2 的 y5 / y6）。
    let layout = Layout::new()
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
        // 从 paper y=1 往下长——那样会压住相位面板右侧的残差轴刻度与标题）
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
        .y_axis(at_range(
            axis_style(
                Axis::new().domain(&Y_TOP).anchor("x").title(Panel::Magnitude.y_title()),
            ),
            magnitude_range,
        ))
        .x_axis2(axis_style(
            Axis::new().domain(&X_RIGHT).anchor("y2").title(Panel::Phase.x_title()),
        ))
        .y_axis2(at_range(
            axis_style(Axis::new().domain(&Y_TOP).anchor("x2").title(Panel::Phase.y_title())),
            phase_range,
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
        .y_axis4(axis_style(
            Axis::new()
                .domain(&Y_BOTTOM)
                .anchor("x4")
                .title(Panel::Norm.y_title())
                .scale_anchor("x4"),
        ))
        // 残差叠加轴不画网格：否则与数据轴网格叠成双层虚线
        .y_axis5(at_range(
            axis_style(
                Axis::new()
                    .overlaying("y")
                    .side(AxisSide::Right)
                    .anchor("x")
                    .title("Residual")
                    .show_grid(false),
            ),
            residual_axis_range(magnitude_range),
        ))
        .y_axis6(at_range(
            axis_style(
                Axis::new()
                    .overlaying("y2")
                    .side(AxisSide::Right)
                    .anchor("x2")
                    .title("Residual (rad)")
                    .show_grid(false),
            ),
            residual_axis_range(phase_range),
        ));
    match overlay {
        FitOverlay::Some { .. } => layout,
        // 失败时第四面板只有那条 text trace：给 x4/y4 固定 range [0,1]，
        // 让"无拟合结果"落在面板正中且坐标框稳定。
        FitOverlay::Absent { .. } => layout
            .x_axis4(axis_style(
                Axis::new()
                    .domain(&X_RIGHT)
                    .anchor("y4")
                    .title(Panel::Norm.x_title())
                    .range(UNIT_RANGE.to_vec()),
            ))
            .y_axis4(axis_style(
                Axis::new()
                    .domain(&Y_BOTTOM)
                    .anchor("x4")
                    .title(Panel::Norm.y_title())
                    .range(UNIT_RANGE.to_vec())
                    .scale_anchor("x4"),
            )),
    }
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

/// 残差右轴范围：与数据轴同 span（同一 scale），整体下移使残差 0 落在面板 [`RESIDUAL_ZERO_AT`] 高度处。
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

