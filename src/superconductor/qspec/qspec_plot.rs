//! qspec 拟合结果的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ 平面、P1）+ 页脚的 HTML 片段，可直接插入汇总
//! 报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只生成 `<div>` 与
//! `Plotly.newPlot` 调用（自带 config 的 `Plotly.newPlot` 片段）；宿主页面需自行加载 plotly.js，
//! 可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板内容对齐 baseline `qspec_analysis.py::qspec_plot`：|S21| 与相位只画数据（拟合在 P1
//! 空间进行，这两个量没有对应的模型曲线）；P1 面板叠洛伦兹拟合曲线与残差；IQ 面板画
//! [`p1`] 定轴所用的投影几何——样本点按 P1 着色、投影轴、以及归一化钉在 0 和 1 上的两个
//! 参考点。参数框放在 IQ 面板：那里数据是一条细线、留白最多，不像 P1 面板会被峰顶压住。

use crate::superconductor::qspec::qspec::{QspecError, QspecFit};
use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::heatmap::{escape_html, interactive_config};
use crate::utils::panels::{
    AxisOpts, CARD_WIDTH, Card, Cell, DATA_COLOR, FIT_COLOR, GRID_2X2, ONE_COLOR, PanelSpec,
    ResidualAxis, Titles, ZERO_COLOR, axis_refs, block, card, color_samples, curve, dense_grid,
    layout, markers, pad, projection_axis, projection_refs, ref_point, residual_axis_range,
    residual_markers, series, value_bounds,
};
use crate::utils::data::{Column, Datum, Payload, Table, fit_row};

use crate::utils::params::{ParamRow, params_table};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::Position;
use plotly::Plot;

/// plotly.js 的 CDN 引入标签；与 [`crate::superconductor::s21::s21_plot`] 用的是同一个版本，
/// 这里转出以便 qspec 报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::s21::s21_plot::PLOTLY_JS_CDN;

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
.qtool-qspec{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高） */
.qtool-qspec .qtool-plot{width:100%;max-height:85vh}
.qtool-qspec .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
</style>"#;

/// 把一条 qspec 谱线渲染成自包含的 HTML div。
///
/// 形参:
///     freqs_hz: XY 驱动频率数组 (n,)，Hz，取绝对频率
///     iq: 该谱线的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定（须与
///             [`crate::superconductor::qspec::qspec_fit`] 用的是同一组中心，
///             否则画出来的曲线不是被拟合的那条）
///     sigma: **IQ 域**的逐点测量不确定度（与 [`qspec_fit`] 同一口径）；给定时在 P1 面板
///            画折算到 P1 空间的 error bar。长度与 `iq` 不一致时忽略、在 IQ 面板列参数，`Err` 时只画
///          数据、参数框位置显示错误文本
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-qspec">` 片段（图 + 页脚），并内蕴一份原始数据供下载
///     （见 [`crate::utils::data`]）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn qspec_fit_plot_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: Result<&QspecFit, &QspecError>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let payload = payload(freqs_hz, iq, states, sigma, &fit, div_id);
    qspec_card(freqs_hz, iq, states, sigma, fit, div_id, frame, Some(&payload))
}

/// 行面板：嵌进 qspec vs Z 的气泡/浮层里的那一份，与独立报告同一套版式，只是**不内蕴载荷**——
/// 宿主报告已经把这个 Z 点的数据放在自己的 `row_<i>` 子表里，再带一份就是同一份数据存两遍。
///
/// 形参:
///     freqs_hz: 驱动频率轴 (n,)，Hz
///     iq: 平均复数 IQ (n,)
///     states: 各态标定中心
///     fit: 拟合结果
///     div_id: 面板 div 的 HTML id（宿主负责唯一）
///
/// 返回值:
///     裸面板片段（无载荷、无外框、不画 error bar）
pub(crate) fn qspec_panel_div(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    fit: Result<&QspecFit, &QspecError>,
    div_id: &str,
) -> String {
    qspec_card(freqs_hz, iq, states, None, fit, div_id, None, None)
}

/// 装配：四面板图 + 参数表 + 卡片。σ 与载荷由调用方给（行面板两样都不带）。
///
/// 形参:
///     freqs_hz: 驱动频率轴 (n,)，Hz
///     iq: 平均复数 IQ (n,)
///     states: 各态标定中心
///     sigma: IQ 域逐点不确定度；None 表示不画 error bar
///     fit: 拟合结果
///     div_id: 卡片 div 的 HTML id
///     frame: 可选外框，见 [`qspec_fit_plot_div`]
///     payload: 内蕴数据；None 表示这份片段不随卡片带载荷
///
/// 返回值:
///     卡片片段
fn qspec_card(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: Result<&QspecFit, &QspecError>,
    div_id: &str,
    frame: Option<&str>,
    payload: Option<&Payload>,
) -> String {
    let overlay = overlay(&fit, freqs_hz);
    let plot = fit_plot(freqs_hz, iq, states, sigma, &fit, &overlay);
    let plot_html = crate::utils::data::plot_script(&plot, div_id);
    let body = format!(
        "{}{}",
        block(div_id, "qtool-plot", (CARD_WIDTH, GRID_2X2.height), &plot_html),
        &overlay.table_html(),
    );
    card(Card {
        class: "qtool-qspec",
        style: STYLE,
        div_id,
        body,
        payload,
        frame,
    })
}

/// 报告的载荷：喂进拟合的输入（频率轴、复 IQ、IQ 域 σ）+ 投影后的 P1 / P1σ + 模型在数据点上的
/// 值 + 拟合参数 + 标定中心。
///
/// P1 与图上同源：有拟合时取 `fit.p1`（实际拟合的那条，取向被翻转时它是翻转后的那条），否则现
/// 投影一次（[`p1`]）；模型是拟合曲线在数据点上的取值，无拟合时整列按缺失落。供显示用的量
/// （幅度、相位、残差、密集曲线）一概不入表——都能从这里的列推出来。
///
/// 形参:
///     freqs_hz: 驱动频率轴 (n,)，Hz
///     iq: 平均复数 IQ (n,)
///     states: 各态标定中心
///     sigma: IQ 域逐点不确定度；长度与 `iq` 不一致时按缺失处理（报告作图代码没有这一档校验，
///            载荷按 t1 的口径落表）
///     fit: 拟合结果
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（`data` + `fits`，给了 `states` 再带一张 `states`）
fn payload(
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: &Result<&QspecFit, &QspecError>,
    div_id: &str,
) -> Payload {
    let prob = match fit {
        Ok(result) => result.p1.clone(),
        Err(_) => p1(iq, states),
    };
    let iq_sigma = match sigma {
        Some(values) => match values.len() == iq.len() {
            true => Some(values),
            false => None,
        },
        None => None,
    };
    let prob_sigma = match iq_sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };
    let model = match fit {
        Ok(result) => Some(result.result.model.at(freqs_hz)),
        Err(_) => None,
    };
    let mut data = Table::new(
        "data",
        vec![
            Column::real("freq", "Hz"),
            Column::complex("iq", "a.u."),
            Column::real("iq_sigma", "a.u."),
            Column::real("p1", "1"),
            Column::real("p1_sigma", "1"),
            Column::real("model", "1"),
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
            Datum::Real(prob[index]),
            match &prob_sigma {
                Some(values) => Datum::Real(values[index]),
                None => Datum::Missing,
            },
            match &model {
                Some(values) => Datum::Real(values[index]),
                None => Datum::Missing,
            },
        ]);
    }
    let mut payload = Payload::new(div_id);
    payload.table(data);
    match fit {
        Ok(result) => {
            // 逐拟合一行：参数各占一列、标准误用 `<参数>_stderr` 列（单位按参数名查表）
            let (columns, row) = fit_row(&result.result.params, &PARAM_UNITS, &[]);
            let mut fits = Table::new("fits", columns);
            fits.push(&row);
            payload.table(fits);
        }
        Err(_) => {}
    }
    match states {
        Some(centers) => {
            let mut table = Table::new("states", vec![Column::complex("center", "a.u.")]);
            for center in centers.as_slice() {
                table.push(&[Datum::Complex(center.re, center.im)]);
            }
            payload.table(table);
        }
        None => {}
    }
    payload
}

/// 模型各参数的单位（无量纲写 `1`）；qspec vs Z 报告逐行复用同一张表。
pub(crate) const PARAM_UNITS: [(&str, &str); 4] = [
    ("fq", "Hz"),
    ("fwhm", "Hz"),
    ("amp", "1"),
    ("offset", "1"),
];

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
            x: "freq (Hz)",
            y: "P1",
            name: "P1",
        },
    }
}

/// 某面板的全部 trace（数据 → 拟合 → 残差），无拟合时自动只剩数据。
fn panel_traces(
    panel: Panel,
    x_ref: &str,
    y_ref: &str,
    freqs_hz: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
    prob_sigma: Option<&[f64]>,
    overlay: &Overlay,
) -> Vec<Box<dyn Trace>> {
    let x_hz = freqs_hz.to_vec();
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            let amp: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
            traces.push(series(x_hz, amp, "Data", DATA_COLOR, true, x_ref, y_ref));
        }
        Panel::Phase => {
            // 相位取主值 (−π, π]：不做 unwrap，跨割线处会有竖直落差，换来各条曲线电平可比
            let phase: Vec<f64> = iq.iter().map(|z| z.arg()).collect();
            traces.push(series(x_hz, phase, "Data", DATA_COLOR, false, x_ref, y_ref));
        }
        Panel::Iq => {
            let (ref_zero, ref_one) = projection_refs(iq, states, prob);
            traces.push(color_samples(
                iq.iter().map(|z| z.re).collect(),
                iq.iter().map(|z| z.im).collect(),
                prob,
                "Data",
                ZERO_COLOR,
                ONE_COLOR,
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
                prob_sigma.map(|values| values.to_vec()),
                "Data",
                DATA_COLOR,
                7,
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
                    traces.push(residual_markers(x_hz, residual, false, "x4", "y5"));
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
    sigma: Option<&[f64]>,
    fit: &Result<&QspecFit, &QspecError>,
    overlay: &Overlay,
) -> Plot {
    // 面板上画的那条 P1：有拟合时是胜出取向下的曲线（即被拟合的那条），否则是原始投影；
    // 两者在取向被翻转时互为 1 − P1，所以必须与拟合用的那条一致，不能另算一遍
    let prob = match fit {
        Ok(result) => result.p1.clone(),
        Err(_) => p1(iq, states),
    };
    // error bar 画的是 P1 域的 σ，与拟合所用的权重同源（[`p1_sigma`] 那一处除法）
    let prob_sigma = match sigma {
        Some(values) => Some(p1_sigma(iq, states, values)),
        None => None,
    };
    let mut plot = Plot::new();
    for (index, panel) in Panel::ALL.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        for trace in panel_traces(
            *panel,
            x_ref.as_str(),
            y_ref.as_str(),
            freqs_hz,
            iq,
            states,
            &prob,
            prob_sigma.as_deref(),
            overlay,
        ) {
            plot.add_trace(trace);
        }
    }
    plot.set_layout(layout(&GRID_2X2, &panel_specs(overlay)));
    plot.set_configuration(interactive_config());
    plot
}

/// 本实验各面板的规格：标题 + 只有格位推不出来的轴选项。
///
/// P1 的 y 轴按拟合曲线定值域并挂一条残差右轴（残差与 P1 差若干数量级，同轴画不出来）；
/// IQ 面板等比例锁定，否则投影轴会被拉成任意斜率。
fn panel_specs(overlay: &Overlay) -> Vec<PanelSpec> {
    let prob_range = match overlay.model() {
        Some(model) => pad(value_bounds(&[model])),
        None => None,
    };
    let opts = [
        AxisOpts::PLAIN,
        AxisOpts::PLAIN,
        AxisOpts {
            equal_aspect: true,
            ..AxisOpts::PLAIN
        },
        AxisOpts {
            y_range: prob_range,
            residual: Some(ResidualAxis {
                title: "Residual",
                range: residual_axis_range(prob_range),
            }),
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
