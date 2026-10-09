//! S21 拟合结果的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ、归一化圆）+ 参数表 + 页脚的 HTML 片段，
//! 可直接插入汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只
//! 生成 `<div>` 与 `Plotly.newPlot` 调用（自带 config 的 `Plotly.newPlot` 片段）；宿主页面
//! 需自行加载 plotly.js，可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板布局与配色对齐 baseline `s12_analysis.py::fit` 的四面板图；相位面板以**频率**为
//! 自变量做线性去趋势，且数据与拟合曲线共用同一条趋势线（baseline 各自独立 detrend，
//! 两条曲线可能整体错开；此处取同趋势以保证重合）。

use crate::utils::heatmap::{escape_html, interactive_config};
use crate::utils::panels::{
    AxisOpts, CARD_WIDTH, Card, Cell, DATA_COLOR, FIT_COLOR, GRID_2X2, PanelSpec, ResidualAxis,
    Titles, axis_refs, block, card, curve, dense_grid, layout, markers, pad, residual_axis_range,
    residual_markers, value_bounds,
};
use crate::utils::data::{Column, Datum, Payload, Table, fit_row, plot_script};
use crate::utils::params::{ParamRow, params_table};
use crate::superconductor::s21::{
    Complex64, JAC_NAMES, S21Error, S21Model, background_at, model_at, notch_at,
};
use crate::utils::{detrend, linear_detrend, unwrap_phase};
use lmfit::ComplexResult;
use plotly::Trace;
use plotly::common::{HoverInfo, Mode};
use plotly::{Plot, Scatter};

/// plotly.js 的 CDN 引入标签（版本与 plotly crate 内嵌的一致）；宿主页面放一次即可。
pub const PLOTLY_JS_CDN: &str =
    r#"<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>"#;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
.qtool-s21{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高；气泡里宽度正好是设计宽度 ⇒ 仍是设计高度） */
.qtool-s21 .qtool-plot{width:100%;max-height:85vh}
.qtool-s21 .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
</style>"#;

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
///     自包含的 `<div class="qtool-s21">` 片段（图 + 参数表 + 页脚），并内蕴一份原始数据供
///     下载（见 [`crate::utils::data`]）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn s21_fit_plot_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let payload = payload(freqs_hz, iq, sigma, fit, div_id);
    s21_card(freqs_hz, iq, sigma, fit, div_id, frame, Some(&payload))
}

/// 行面板：嵌进 s21 vs power 的气泡/浮层里的那一份，与独立报告同一套版式，只是**不内蕴载荷**——
/// 宿主报告已经把这个功率档的数据放在自己的 `row_<i>` 子表里，再带一份就是同一份数据存两遍。
///
/// 形参:
///     freqs_hz: 读出频率轴 (n,)，Hz
///     iq: 该档的复数 IQ 数组 (n,)
///     sigma: 逐点测量不确定度
///     fit: 拟合结果
///     div_id: 面板 div 的 HTML id（宿主负责唯一）
///
/// 返回值:
///     裸面板片段（无载荷、无外框）
pub(crate) fn s21_panel_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
    div_id: &str,
) -> String {
    s21_card(freqs_hz, iq, sigma, fit, div_id, None, None)
}

/// 装配：四面板图 + 参数表 + 卡片。载荷由调用方给（行面板不带）。
///
/// 形参:
///     freqs_hz: 读出频率轴 (n,)，Hz
///     iq: 该档的复数 IQ 数组 (n,)
///     sigma: 逐点测量不确定度
///     fit: 拟合结果
///     div_id: 卡片 div 的 HTML id
///     frame: 可选外框，见 [`s21_fit_plot_div`]
///     payload: 内蕴数据；None 表示这份片段不随卡片带载荷
///
/// 返回值:
///     卡片片段
fn s21_card(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
    div_id: &str,
    frame: Option<&str>,
    payload: Option<&Payload>,
) -> String {
    let data = DataView::new(freqs_hz, iq, sigma);
    let overlay = overlay(fit, freqs_hz, &data);
    let plot = fit_plot(&data, &overlay);
    let plot_html = plot_script(&plot, div_id);
    let body = format!(
        "{}{}",
        block(div_id, "qtool-plot", (CARD_WIDTH, GRID_2X2.height), &plot_html),
        &overlay.table_html(),
    );
    card(Card {
        class: "qtool-s21",
        style: STYLE,
        div_id,
        body,
        payload,
        frame,
    })
}

/// 报告载荷：喂进拟合的输入（频率、复数 IQ、逐点 σ）+ 模型在数据点上的取值 + 参数表。
///
/// 形参:
///     freqs_hz: 读出频率数组 (n,)，Hz
///     iq: 该线的复数 IQ 数组 (n,)
///     sigma: 逐点测量不确定度；长度与 `iq` 不一致时按缺失落
///     fit: 拟合结果；`Err` 时只有数据表
///     div_id: 报告名（同时是下载文件的基名）
///
/// 返回值:
///     载荷（表按 `data`、`fits` 的次序）
fn payload(
    freqs_hz: &[f64],
    iq: &[Complex64],
    sigma: Option<&[f64]>,
    fit: Result<&ComplexResult<S21Model>, &S21Error>,
    div_id: &str,
) -> Payload {
    let iq_sigma = match sigma {
        Some(values) => match values.len() == iq.len() {
            true => Some(values),
            false => None,
        },
        None => None,
    };
    let model: Option<Vec<Complex64>> = match fit {
        Ok(result) => Some(freqs_hz.iter().map(|f| model_at(*f, &result.model)).collect()),
        Err(_) => None,
    };
    let mut data = Table::new(
        "data",
        vec![
            Column::real("freq", "Hz"),
            Column::complex("iq", "a.u."),
            Column::real("iq_sigma", "a.u."),
            Column::complex("model", "a.u."),
        ],
    );
    for index in 0..freqs_hz.len() {
        data.push(&[
            Datum::Real(freqs_hz[index]),
            Datum::Complex(iq[index].re, iq[index].im),
            match iq_sigma {
                Some(values) => Datum::Real(values[index]),
                None => Datum::Missing,
            },
            match &model {
                Some(values) => Datum::Complex(values[index].re, values[index].im),
                None => Datum::Missing,
            },
        ]);
    }
    let mut payload = Payload::new(div_id);
    payload.table(data);
    match fit {
        Ok(result) => {
            // 逐拟合一行：11 个拟合参数 + 2 个派生量，各带 `<参数>_stderr` 列
            let (columns, row) = fit_row(&result.params, &PARAM_UNITS, &[]);
            let mut fits = Table::new("fits", columns);
            fits.push(&row);
            payload.table(fits);
        }
        Err(_) => {}
    }
    payload
}

/// 拟合四面板的 Plotly 图对象（不含标题与参数表）。
///
/// 供需要自行组装页面的调用方使用：`plotly::Plot` 可自行序列化，[`s21_fit_plot_div`]
/// 即在此基础上加图块、参数表与页脚。
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
    for (index, panel) in Panel::ALL.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        for trace in panel_traces(*panel, x_ref.as_str(), y_ref.as_str(), data, overlay) {
            plot.add_trace(trace);
        }
    }
    plot.set_layout(layout(&GRID_2X2, &panel_specs(data, overlay)));
    plot.set_configuration(interactive_config());
    plot
}

// =========================================================================
// 面板
// =========================================================================

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

/// 参数的单位（SI 基本单位；量纲为一的写 `1`，任意单位写 `a.u.`）：载荷与显示同源。
pub(crate) const PARAM_UNITS: [(&str, &str); 13] = [
    ("fr", "Hz"),
    ("ql", "1"),
    ("qc", "1"),
    ("theta", "rad"),
    ("ap", "1"),
    ("tau", "s"),
    ("a", "1"),
    ("b", "1"),
    ("phi", "rad"),
    ("zc_re", "a.u."),
    ("zc_im", "a.u."),
    ("qi", "1"),
    ("kappa_ex", "Hz"),
];

/// 参数的单位后缀（显示用）：只有 Hz/s/rad 带后缀，量纲为一与任意单位都不显示。
///
/// 形参:
///     name: 参数名
///
/// 返回值:
///     `" Hz"` / `" s"` / `" rad"`；其余参数给空串
pub(crate) fn unit_of(name: &str) -> &'static str {
    match crate::utils::data::param_unit(&PARAM_UNITS, name) {
        "Hz" => " Hz",
        "s" => " s",
        "rad" => " rad",
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

/// 本实验的面板：四个各占一格。面板集合是各实验自己的事，这里只声明"我是谁、我占哪一格"。
#[derive(Clone, Copy)]
enum Panel {
    Magnitude,
    Phase,
    Iq,
    Norm,
}

impl Panel {
    const ALL: [Panel; 4] = [Panel::Magnitude, Panel::Phase, Panel::Iq, Panel::Norm];

    /// 本面板在网格里的格位（轴名与域都由它推出来，不在这里写）。
    fn cell(self) -> Cell {
        match self {
            Self::Magnitude => Cell::at(0, 0),
            Self::Phase => Cell::at(0, 1),
            Self::Iq => Cell::at(1, 0),
            Self::Norm => Cell::at(1, 1),
        }
    }
}

/// 某个面板的标题表（图名与两个轴标题各只出现一次）。
fn titles(panel: Panel) -> Titles {
    match panel {
        Panel::Magnitude => Titles {
            x: "freq (Hz)",
            y: "|S21|",
            name: "Magnitude",
        },
        Panel::Phase => Titles {
            x: "freq (Hz)",
            y: "Phase (rad)",
            name: "Phase",
        },
        Panel::Iq => Titles {
            x: "I",
            y: "Q",
            name: "IQ plane",
        },
        Panel::Norm => Titles {
            x: "I",
            y: "Q",
            name: "Normalized",
        },
    }
}

/// 归一化面板的固定坐标范围（配 [`CENTER`] 用）：text 标注画在正中，缩放面板时不会漂。
const UNIT_RANGE: [f64; 2] = [0.0, 1.0];

/// 面板正中。
const CENTER: f64 = 0.5;

/// 面板中央的纯文本标注（[`CENTER`]，配显式 [`UNIT_RANGE`]）。
fn text_label(text: &str, x_ref: &str, y_ref: &str) -> Box<dyn Trace> {
    let trace: Box<dyn Trace> = Scatter::new(vec![CENTER], vec![CENTER])
        .mode(Mode::Text)
        .text(text.to_string())
        .show_legend(false)
        .hover_info(HoverInfo::Skip)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 某面板的全部 trace（数据点 → 拟合线 → 残差），失败时自动只剩数据点。
///
/// 形参:
///     panel: 画哪一块
///     x_ref: 本面板的 x 轴名（由 [`panels::layout`] 按格位给出，见 `axis_refs`）
///     y_ref: 本面板的 y 轴名
fn panel_traces(
    panel: Panel,
    x_ref: &str,
    y_ref: &str,
    data: &DataView,
    overlay: &FitOverlay,
) -> Vec<Box<dyn Trace>> {
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

/// 本实验各面板的规格：标题 + 只有格位推不出来的轴选项。
///
/// 值域由拟合结果算出来后固定住（左轴 = 值域 ±5%，残差右轴 = 同 span 下移，两者可直接目视
/// 比量级）；无拟合时归一化面板没有数据，改用一条 text trace 标注，坐标范围钉死在
/// [`UNIT_RANGE`] 上，让"无拟合结果"落在面板正中且坐标框稳定。
fn panel_specs(data: &DataView, overlay: &FitOverlay) -> Vec<PanelSpec> {
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
    let (norm_x, norm_y) = match overlay {
        // 有拟合时归一化面板照常自适应；失败时它只剩那条 text trace
        FitOverlay::Some { .. } => (None, None),
        FitOverlay::Absent { .. } => (Some(UNIT_RANGE), Some(UNIT_RANGE)),
    };
    let opts = [
        AxisOpts {
            y_range: magnitude_range,
            residual: Some(ResidualAxis {
                title: "Residual",
                range: residual_axis_range(magnitude_range),
            }),
            ..AxisOpts::PLAIN
        },
        AxisOpts {
            y_range: phase_range,
            residual: Some(ResidualAxis {
                title: "Residual (rad)",
                range: residual_axis_range(phase_range),
            }),
            ..AxisOpts::PLAIN
        },
        AxisOpts {
            equal_aspect: true,
            ..AxisOpts::PLAIN
        },
        AxisOpts {
            x_range: norm_x,
            y_range: norm_y,
            equal_aspect: true,
            ..AxisOpts::PLAIN
        },
    ];
    Panel::ALL
        .iter()
        .zip(opts)
        .map(|(panel, opts)| PanelSpec {
            cell: panel.cell(),
            titles: titles(*panel),
            opts,
        })
        .collect()
}
