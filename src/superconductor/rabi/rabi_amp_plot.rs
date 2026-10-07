//! Rabi 幅度扫描的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出五面板图（|S21|、相位、IQ 平面、P1，第三行整宽放 P1 的幅度谱）+ 参数表 +
//! 页脚的 HTML 片段，可直接插入汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，
//! 本模块只生成 `<div>` 与 `Plotly.newPlot` 调用（[`plotly::Plot::to_inline_html`]）；宿主
//! 页面需自行加载 plotly.js，可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板内容对齐 baseline `rabi_amp_analysis.py::rabi_amp_plot`：|S21| 与相位只画数据（拟合在
//! P1 空间进行，这两个量没有对应的模型曲线）；P1 面板叠余弦拟合曲线与残差、画给定 σ 折算过来
//! 的 error bar，并标出这次扫描真正要产出的量——**π 脉冲幅的竖虚线**；IQ 面板画
//! [`crate::superconductor::p1`] 定轴所用的投影几何：样本点、投影轴、以及归一化钉在 0 和 1 上的
//! 两个参考点。
//!
//! 第三行的频谱不是新算的东西：它就是拟合初值候选里傅里叶那一路的谱（[`crate::utils::spectrum`]），
//! 画出来是为了让"初值落在哪"与"数据长什么样"在同一页上对得上——顺带让欠采样直接现形（谱峰贴到
//! 最右 bin，甚至折返）。横轴是**幅度的倒数**，是 Rabi 频率的单位，不是 Hz。

use crate::superconductor::rabi::rabi_amp::{RabiError, RabiFit};
use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::bubble::framize;
use crate::utils::heatmap::{escape_html, interactive_config};
use crate::utils::panels::{
    AxisOpts, Cell, DATA_COLOR, FIT_COLOR, GRID_2X3, ONE_COLOR, PanelSpec, ResidualAxis, Titles,
    ZERO_COLOR, axis_refs, color_samples, curve, dense_grid, layout, markers, pad, projection_axis,
    projection_refs, ref_point, report_div, residual_axis_range, residual_markers, series,
    value_bounds,
};
use crate::utils::params::{ParamRow, params_table};
use crate::utils::{spectrum, unwrap_phase};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{DashType, Position};
use plotly::layout::{Shape, ShapeLayer, ShapeLine, ShapeType};
use plotly::Plot;

/// plotly.js 的 CDN 引入标签；与 s21 / qspec 报告用的是同一个版本，这里转出以便 rabi 报告
/// 自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

/// π 脉冲幅竖虚线与频谱面板上频率标线的样式（baseline 用灰色细虚线）。
const PI_LINE_COLOR: &str = "gray";

/// P1 面板在 `Panel::ALL` 里的序号：π 脉冲幅的竖线画在它上面。
const NORM_INDEX: usize = 3;

/// 本实验的面板：三个实验共用的前四块（|S21| / 相位 / IQ / P1），外加第三行整宽的 P1 频谱。
///
/// 面板集合是各实验自己的事——多出来的那块不必挤进别处的枚举，别的实验也就不必替它写一条
/// 够不着的分支；这里每块只说两件事：画什么（`titles` / `panel_traces`）、占哪一格。
#[derive(Clone, Copy)]
enum Panel {
    Magnitude,
    Phase,
    Iq,
    Norm,
    /// 第三行整宽：P1 的幅度谱
    Spectrum,
}

impl Panel {
    const ALL: [Panel; 5] = [
        Panel::Magnitude,
        Panel::Phase,
        Panel::Iq,
        Panel::Norm,
        Panel::Spectrum,
    ];

    /// 本面板在网格里的格位（轴名与域都由它推出来，不在这里写）。
    fn cell(self) -> Cell {
        match self {
            Self::Magnitude => Cell::at(0, 0),
            Self::Phase => Cell::at(0, 1),
            Self::Iq => Cell::at(1, 0),
            Self::Norm => Cell::at(1, 1),
            // 第三行整宽：跨两列就是跨两列，不是一类特殊行
            Self::Spectrum => Cell::at(2, 0).span(1, 2),
        }
    }
}

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）；尺寸相关的两条
/// 规则由 [`crate::utils::panels::size_style`] 单独生成。
const STYLE: &str = r#"<style>
.qtool-rabi{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高） */
.qtool-rabi .qtool-plot{width:100%;max-height:85vh}
.qtool-rabi .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
.qtool-rabi .qtool-footer{margin-top:6px;text-align:right;font-size:11px;color:#9ca3af}
</style>"#;

/// 把一条 Rabi 幅度扫描渲染成自包含的 HTML div。
///
/// 形参:
///     amps: XY 驱动幅度 (n,)，取与 XY 通道 DAC 满幅的比值（无量纲）
///     iq: 该条扫描的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定（须与
///             [`crate::superconductor::rabi::rabi_amp_fit`] 用的是同一组中心，
///             否则画出来的曲线不是被拟合的那条）
///     sigma: **IQ 域**的逐点测量不确定度（与 [`rabi_amp_fit`] 同一口径，点数是 n 次单发
///            平均时传 `std(shots)/√n`）；给定时在 P1 面板画折算到 P1 空间的 error bar。
///            长度与 `iq` 不一致时忽略
///     fit: 拟合结果；`Ok` 时在 P1 面板叠余弦曲线、残差与 π 脉冲幅竖线，`Err` 时只画数据、
///          表位置显示错误文本
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-rabi">` 片段（图 + 参数表 + 页脚）；宿主页面需自行加载
///     plotly.js（见 [`PLOTLY_JS_CDN`]）
///
/// [`rabi_amp_fit`]: crate::superconductor::rabi::rabi_amp_fit
pub fn rabi_amp_plot_div(
    amps: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: Result<&RabiFit, &RabiError>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let overlay = overlay(&fit, amps);
    let plot = fit_plot(amps, iq, states, sigma, &fit, &overlay);
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = plot.to_inline_html(Some(plot_div_id.as_str()));

    let html = report_div(
        "qtool-rabi",
        STYLE,
        div_id,
        &GRID_2X3,
        &plot_html,
        &overlay.table_html(),
    );
    framize(&html, frame)
}

/// 拟合结果的叠加层：`Ok` 时给出模型曲线、π 脉冲幅与参数表，`Err` 时把错误交给页面。
///
/// 把 `Result` 收成一个值之后，画 trace 与排布局的代码都不必再分支——只有真正不同的三处
/// （P1 面板的模型曲线与竖线、表内容）去问它要东西。
enum Overlay<'a> {
    Some {
        /// 数据幅度网格上的模型值（算残差用）
        model: Vec<f64>,
        /// 密集网格上的模型值（画平滑的拟合曲线用）
        dense_amps: Vec<f64>,
        dense_model: Vec<f64>,
        /// π 脉冲幅（P1 面板上的竖虚线）
        a_pi: f64,
        /// 参数表各行
        rows: Vec<ParamRow<'a>>,
        /// 表下提示（未收敛）
        note: Option<String>,
    },
    Absent {
        error: &'a RabiError,
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
                dense_amps,
                dense_model,
                ..
            } => Some((dense_amps, dense_model)),
            Self::Absent { .. } => None,
        }
    }

    /// π 脉冲幅；无拟合时为 None。
    fn a_pi(&self) -> Option<f64> {
        match self {
            Self::Some { a_pi, .. } => Some(*a_pi),
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
///     amps: 数据幅度轴 (n,)
///
/// 返回值:
///     叠加层；无拟合或点数不足以插值时，密集曲线一栏为空
fn overlay<'a>(fit: &'a Result<&RabiFit, &RabiError>, amps: &[f64]) -> Overlay<'a> {
    match fit {
        Ok(result) => {
            let model = result.result.model.at(amps);
            let (dense_amps, dense_model) = match dense_grid(amps) {
                Some(grid) => {
                    let values = result.result.model.at(&grid);
                    (grid, values)
                }
                None => (Vec::new(), Vec::new()),
            };
            Overlay::Some {
                model,
                dense_amps,
                dense_model,
                a_pi: result.result.model.a_pi,
                rows: param_rows(result),
                note: note_of(result),
            }
        }
        Err(error) => Overlay::Absent { error },
    }
}

/// 参数表：π 脉冲幅、振荡频率、半幅，各带标准误。
///
/// 标准误缺失（协方差不可用）或非有限时给占位符，不伪造一个 0 误差。`a_pi` 的标准误由 lmfit
/// 按 delta 方法从 `freq` 的方差传播而来（与 baseline 手写的 `_pi_amp_sem` 同式）。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     参数表各行，顺序即渲染顺序
fn param_rows(fit: &RabiFit) -> Vec<ParamRow<'static>> {
    let model = &fit.result.model;
    // 名字与 `Cos` 的字段一一对应；标准误按名字去 params 里取
    let rows: [(&'static str, &'static str, f64, usize); 3] = [
        ("a_pi", "π pulse amplitude, 1/(2·freq)", model.a_pi, 6),
        ("freq", "Rabi frequency", model.freq, 6),
        ("amp", "half amplitude (peak is 2·amp)", model.amp, 4),
    ];
    rows.iter()
        .map(|(name, description, value, digits)| ParamRow {
            name,
            description,
            value: format!("{value:.digits$}"),
            stderr: match stderr_of(fit, name) {
                Some(error) => format!("{error:.digits$}"),
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
fn note_of(fit: &RabiFit) -> Option<String> {
    match fit.result.success {
        true => None,
        false => Some(format!("not converged: {}", fit.result.message)),
    }
}

/// 某个参数的标准误；缺失或非有限时返回 None。
fn stderr_of(fit: &RabiFit, name: &str) -> Option<f64> {
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

/// 某个面板的标题表（图名与两个轴标题各只出现一次）。
///
/// 横轴是扫描量——驱动幅度，三个面板同轴；IQ 面板是 I/Q；频谱面板换了变量，横轴是幅度的
/// 倒数（Rabi 频率），**不是 Hz**，标签只能照实写。
fn titles(panel: Panel) -> Titles {
    match panel {
        Panel::Magnitude => Titles {
            x: "XY amp (a.u.)",
            y: "|S21|",
            name: "Magnitude",
        },
        Panel::Phase => Titles {
            x: "XY amp (a.u.)",
            y: "Phase (rad)",
            name: "Phase",
        },
        Panel::Iq => Titles {
            x: "I",
            y: "Q",
            name: "IQ plane",
        },
        Panel::Norm => Titles {
            x: "XY amp (a.u.)",
            y: "P1",
            name: "P1",
        },
        Panel::Spectrum => Titles {
            x: "Rabi freq (1/a.u.)",
            y: "|Spectrum|",
            name: "P1 spectrum",
        },
    }
}

/// 某面板的全部 trace（数据 → 拟合 → 残差），无拟合时自动只剩数据。
///
/// 形参:
///     panel: 画哪一块
///     x_ref: 本面板的 x 轴名（由 `layout` 按格位给出，见 `axis_refs`）
///     y_ref: 本面板的 y 轴名
///     amps: 幅度轴 (n,)
///     iq: 该条扫描的平均复数 IQ (n,)
///     states: 各态标定中心；None 表示未标定
///     prob: 实际参与拟合的那条 P1 (n,)
///     prob_sigma: P1 域的逐点不确定度 (n,)；None 表示不画 error bar
///     overlay: 拟合叠加层
///
/// 返回值:
///     该面板的 trace 列表，按绘制顺序
fn panel_traces(
    panel: Panel,
    x_ref: &str,
    y_ref: &str,
    amps: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
    prob_sigma: Option<&[f64]>,
    overlay: &Overlay,
) -> Vec<Box<dyn Trace>> {
    let x_amps = amps.to_vec();
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            let magnitude: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
            traces.push(series(
                x_amps,
                magnitude,
                "Data",
                DATA_COLOR,
                true,
                x_ref,
                y_ref,
            ));
        }
        Panel::Phase => {
            // 相位经 unwrap 解缠绕，否则跳变处会出现整圈假台阶
            let phase = unwrap_phase(&iq.iter().map(|z| z.arg()).collect::<Vec<f64>>());
            traces.push(series(x_amps, phase, "Data", DATA_COLOR, false, x_ref, y_ref));
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
            traces.push(ref_point(
                ref_zero,
                "|0>",
                ZERO_COLOR,
                Position::BottomRight,
                x_ref,
                y_ref,
            ));
            traces.push(ref_point(
                ref_one,
                "|1>",
                ONE_COLOR,
                Position::TopLeft,
                x_ref,
                y_ref,
            ));
        }
        Panel::Norm => {
            traces.push(markers(
                x_amps.clone(),
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
                Some((dense_amps, dense_model)) => traces.push(curve(
                    dense_amps.to_vec(),
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
                    // 残差与 P1 差若干数量级，画在 P1 面板的右叠加轴（y6）上
                    traces.push(residual_markers(x_amps, residual, false, "x4", "y6"));
                }
                None => {}
            }
        }
        Panel::Spectrum => {
            let (freqs, spectrum_amps) = spectrum(amps, prob);
            traces.push(series(
                freqs,
                spectrum_amps,
                "Spectrum",
                DATA_COLOR,
                false,
                x_ref,
                y_ref,
            ));
        }
    }
    traces
}

/// 五面板图：|S21| / 相位 / IQ 平面 / P1，第三行整宽放 P1 的幅度谱。
///
/// 形参:
///     amps: XY 驱动幅度 (n,)
///     iq: 该条扫描的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定
///     sigma: IQ 域的逐点不确定度；长度与 `iq` 不一致时忽略
///     fit: 拟合结果；决定 P1 面板画哪条曲线（胜出取向下的那条）
///     overlay: 拟合叠加层
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
fn fit_plot(
    amps: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: &Result<&RabiFit, &RabiError>,
    overlay: &Overlay,
) -> Plot {
    // 面板上画的那条 P1：有拟合时是它实际拟合的那条（已经定向），否则是原始投影
    let prob = match fit {
        Ok(result) => result.p1.clone(),
        Err(_) => p1(iq, states),
    };
    // error bar 画的是 P1 域的 σ，与拟合所用的权重同源（[`p1_sigma`] 那一处除法）
    let prob_sigma = match sigma {
        Some(values) => match values.len() == iq.len() {
            true => Some(p1_sigma(iq, states, values)),
            false => None,
        },
        None => None,
    };
    let mut plot = Plot::new();
    for (index, panel) in Panel::ALL.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        for trace in panel_traces(
            *panel,
            x_ref.as_str(),
            y_ref.as_str(),
            amps,
            iq,
            states,
            &prob,
            prob_sigma.as_deref(),
            overlay,
        ) {
            plot.add_trace(trace);
        }
    }
    let mut settings = layout(&GRID_2X3, &panel_specs(overlay));
    match overlay.a_pi() {
        Some(a_pi) => settings = settings.shapes(vec![pi_line(a_pi)]),
        None => {}
    }
    plot.set_layout(settings);
    plot.set_configuration(interactive_config());
    plot
}

/// 本实验各面板的规格：标题 + 只有格位推不出来的轴选项。
///
/// P1 的 y 轴按拟合曲线定值域并挂一条残差右轴（残差与 P1 差若干数量级，同轴画不出来）；
/// IQ 面板等比例锁定，否则投影轴会被拉成任意斜率；频谱面板交给 plotly 自适应。
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
        AxisOpts::PLAIN,
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

/// π 脉冲幅的竖虚线：它是这条扫描真正要产出的量，标在 P1 面板上（与数据同轴）。
fn pi_line(a_pi: f64) -> Shape {
    let (x_ref, y_ref) = axis_refs(NORM_INDEX);
    Shape::new()
        .shape_type(ShapeType::Line)
        .layer(ShapeLayer::Below)
        .x_ref(x_ref.as_str())
        .y_ref(&format!("{y_ref} domain"))
        .x0(a_pi)
        .x1(a_pi)
        .y0(0.0)
        .y1(1.0)
        .line(
            ShapeLine::new()
                .color(PI_LINE_COLOR)
                .width(1.0)
                .dash(DashType::Dash),
        )
}