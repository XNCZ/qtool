//! T2 echo（回波退相位）的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ 平面、P1）+ 参数表 + 页脚的 HTML 片段，可直接插入
//! 汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只生成 `<div>` 与
//! `Plotly.newPlot` 调用（[`plotly::Plot::to_inline_html`]）；宿主页面需自行加载 plotly.js，
//! 可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 版式是 rabi / ramsey 那套的缩短版：那两份报告第三行整宽放 P1 频谱，是给傅里叶那一路初值
//! 做对照用的；回波的线型是单调指数、没有傅里叶初值（也不需要，见 [`super::t2_echo`]），那一
//! 行整个不出现。
//!
//! 面板内容：|S21| 与相位只画数据（拟合在 P1 空间进行，这两个量没有对应的模型曲线）；P1 面板
//! 叠指数拟合曲线与残差、画给定 σ 折算过来的 error bar，并标出这次扫描真正要产出的量——
//! **T2 echo 的竖虚线**；IQ 面板画 [`crate::superconductor::p1`] 定轴所用的投影几何。

use crate::superconductor::t2_echo::t2_echo::{T2EchoError, T2EchoFit};
use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::bubble::framize;
use crate::utils::heatmap::{escape_html, interactive_config};
use crate::utils::panels::{
    AxisOpts, Cell, DATA_COLOR, FIT_COLOR, GRID_2X2, ONE_COLOR, PanelSpec, ResidualAxis, Titles,
    ZERO_COLOR, axis_refs, colored_samples, curve, dense_grid, layout, markers, pad, projection_axis,
    projection_refs, ref_point, report_div, residual_axis_range, residual_markers, series,
    value_bounds,
};
use crate::utils::params::{ParamRow, params_table};
use crate::utils::unwrap_phase;
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{DashType, Position};
use plotly::layout::{Shape, ShapeLayer, ShapeLine, ShapeType};
use plotly::Plot;

/// plotly.js 的 CDN 引入标签；与 s21 / qspec / rabi / ramsey 报告用的是同一个版本，这里转出
/// 以便 T2 echo 报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

/// T2 echo 竖虚线的样式（与 rabi 那边 π 脉冲幅、ramsey 那边 T2\* 的灰细虚线同一套外观）。
const T2_LINE_COLOR: &str = "gray";

/// P1 面板在 `Panel::ALL` 里的序号：T2 echo 的竖线画在它上面。
const NORM_INDEX: usize = 3;

/// 本实验的面板：四个实验共用的那四块（|S21| / 相位 / IQ / P1）。
///
/// 面板集合是各实验自己的事——单调指数没有傅里叶初值，也就没有频谱那一块；这里每块只说两件
/// 事：画什么（`titles` / `panel_traces`）、占哪一格。
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

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）；尺寸相关的两条
/// 规则由 [`crate::utils::panels::size_style`] 单独生成。
const STYLE: &str = r#"<style>
.qtool-t2echo{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高） */
.qtool-t2echo .qtool-plot{width:100%;max-height:85vh}
.qtool-t2echo .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
.qtool-t2echo .qtool-footer{margin-top:6px;text-align:right;font-size:11px;color:#9ca3af}
</style>"#;

/// 把一条回波扫描渲染成自包含的 HTML div。
///
/// 形参:
///     taus: **总自由演化时间**轴 (n,)，单位 s（与 [`t2_echo_fit`] 同一口径）
///     iq: 该条扫描的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定（须与 [`t2_echo_fit`] 用的是同一组中心，否则画
///             出来的曲线不是被拟合的那条）
///     sigma: **IQ 域**的逐点测量不确定度（与 [`t2_echo_fit`] 同一口径，点数是 n 次单发平均
///             时传 `std(shots)/√n`）；给定时在 P1 面板画折算到 P1 空间的 error bar。长度与
///             `iq` 不一致时忽略
///     fit: 拟合结果；`Ok` 时在 P1 面板叠指数曲线、残差与 T2 echo 竖线，`Err` 时只画数据、
///          表位置显示错误文本
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-t2echo">` 片段（图 + 参数表 + 页脚）；宿主页面需自行加载
///     plotly.js（见 [`PLOTLY_JS_CDN`]）
///
/// [`t2_echo_fit`]: crate::superconductor::t2_echo::t2_echo_fit
pub fn t2_echo_plot_div(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: Result<&T2EchoFit, &T2EchoError>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let overlay = overlay(&fit, taus);
    let plot = fit_plot(taus, iq, states, sigma, &fit, &overlay);
    let plot_div_id = format!("{div_id}-plot");
    let plot_html = plot.to_inline_html(Some(plot_div_id.as_str()));

    let html = report_div(
        "qtool-t2echo",
        STYLE,
        div_id,
        &GRID_2X2,
        &plot_html,
        &overlay.table_html(),
    );
    framize(&html, frame)
}

/// 拟合结果的叠加层：`Ok` 时给出模型曲线、T2 echo 与参数表，`Err` 时把错误交给页面。
///
/// 把 `Result` 收成一个值之后，画 trace 与排布局的代码都不必再分支——只有真正不同的三处
/// （P1 面板的模型曲线与竖线、表内容）去问它要东西。
enum Overlay<'a> {
    Some {
        /// 数据延时网格上的模型值（算残差用）
        model: Vec<f64>,
        /// 密集网格上的模型值（画平滑的拟合曲线用）
        dense_taus: Vec<f64>,
        dense_model: Vec<f64>,
        /// T2 echo（P1 面板上的竖虚线）
        t2_echo: f64,
        /// 参数表各行
        rows: Vec<ParamRow<'static>>,
        /// 表下提示（未收敛、标准误不可用）
        note: Option<String>,
    },
    Absent {
        error: &'a T2EchoError,
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
                dense_taus,
                dense_model,
                ..
            } => Some((dense_taus, dense_model)),
            Self::Absent { .. } => None,
        }
    }

    /// T2 echo；无拟合时为 None。
    fn t2_echo(&self) -> Option<f64> {
        match self {
            Self::Some { t2_echo, .. } => Some(*t2_echo),
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
///     taus: 数据延时轴 (n,)
///
/// 返回值:
///     叠加层；无拟合或点数不足以插值时，密集曲线一栏为空
fn overlay<'a>(fit: &'a Result<&T2EchoFit, &T2EchoError>, taus: &[f64]) -> Overlay<'a> {
    match fit {
        Ok(result) => {
            let model = result.result.model.at(taus);
            let (dense_taus, dense_model) = match dense_grid(taus) {
                Some(grid) => {
                    let values = result.result.model.at(&grid);
                    (grid, values)
                }
                None => (Vec::new(), Vec::new()),
            };
            Overlay::Some {
                model,
                dense_taus,
                dense_model,
                t2_echo: result.result.model.t2_echo,
                rows: param_rows(result),
                note: note_of(result),
            }
        }
        Err(error) => Overlay::Absent { error },
    }
}

/// 参数表：T2 echo、本底、幅度，各带标准误。
///
/// 标准误缺失（协方差不可用）或非有限时给占位符，不伪造一个 0 误差。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     参数表各行，顺序即渲染顺序
fn param_rows(fit: &T2EchoFit) -> Vec<ParamRow<'static>> {
    let model = &fit.result.model;
    // (行名, lmfit 参数名, 解释, 值, 位数, 是否用科学计数)：科学计数是给时间留的——以秒为
    // 单位就是 1e-5 量级，定点写出来只有一串前导零
    let rows: [(&'static str, &'static str, &'static str, f64, usize, bool); 3] = [
        (
            "T2 echo",
            "t2_echo",
            "echo dephasing time (τ is the total free evolution time), the quantity this scan produces",
            model.t2_echo,
            4,
            true,
        ),
        (
            "offset",
            "offset",
            "level the curve settles at: readout misassignment (SPAM), not part of T2 echo",
            model.offset,
            4,
            false,
        ),
        (
            "amplitude",
            "amplitude",
            "amplitude of the decaying term: state preparation imperfection (SPAM); the curve rises from P1(0) = 0 to 0.5, so this comes out negative on the uncalibrated path",
            model.amplitude,
            4,
            false,
        ),
    ];
    let render = |value: f64, digits: usize, scientific: bool| match scientific {
        true => format!("{value:.digits$e}"),
        false => format!("{value:.digits$}"),
    };
    rows.iter()
        .map(|(name, key, description, value, digits, scientific)| ParamRow {
            name,
            description,
            value: render(*value, *digits, *scientific),
            stderr: match stderr_of(fit, key) {
                Some(error) => render(error, *digits, *scientific),
                None => "—".to_string(),
            },
        })
        .collect()
}

/// 表下提示：求解器未收敛、或 T2 echo 的标准误拿不到时给一行说明。
///
/// 后一条单列，是因为三参数指数在"窗口比时间常数还短"时会滑进"极长时间常数 + 极大幅度"的
/// 退化解：残差看着一样好、T2 echo 却毫无意义。协方差在那种解上通常已经退化，标准误这一栏
/// 会是 `—`——提示把它挑明，免得那个数被当成可信的读数。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     提示文本；收敛且 T2 echo 带得上标准误时返回 None
fn note_of(fit: &T2EchoFit) -> Option<String> {
    match fit.result.success {
        false => Some(format!("not converged: {}", fit.result.message)),
        true => match stderr_of(fit, "t2_echo") {
            Some(_) => None,
            None => Some(
                "T2 echo stderr unavailable (degenerate covariance): treat the fitted decay as unconstrained"
                    .to_string(),
            ),
        },
    }
}

/// 某个参数的标准误；缺失或非有限时返回 None。
fn stderr_of(fit: &T2EchoFit, name: &str) -> Option<f64> {
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
/// 横轴是扫描量——总自由演化时间，三个面板同轴；IQ 面板是 I/Q。
fn titles(panel: Panel) -> Titles {
    match panel {
        Panel::Magnitude => Titles {
            x: "Delay τ (s)",
            y: "|S21|",
            name: "Magnitude",
        },
        Panel::Phase => Titles {
            x: "Delay τ (s)",
            y: "Phase (rad)",
            name: "Phase",
        },
        Panel::Iq => Titles {
            x: "I",
            y: "Q",
            name: "IQ plane",
        },
        Panel::Norm => Titles {
            x: "Delay τ (s)",
            y: "P1",
            name: "P1",
        },
    }
}

/// 某面板的全部 trace（数据 → 拟合 → 残差），无拟合时自动只剩数据。
///
/// 形参:
///     panel: 画哪一块
///     x_ref: 本面板的 x 轴名（由 `layout` 按格位给出，见 `axis_refs`）
///     y_ref: 本面板的 y 轴名
///     taus: 延时轴 (n,)
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
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
    prob_sigma: Option<&[f64]>,
    overlay: &Overlay,
) -> Vec<Box<dyn Trace>> {
    let x_taus = taus.to_vec();
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            let magnitude: Vec<f64> = iq.iter().map(|z| z.norm()).collect();
            traces.push(series(
                x_taus,
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
            traces.push(series(x_taus, phase, "Data", DATA_COLOR, false, x_ref, y_ref));
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
                x_taus.clone(),
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
                Some((dense_taus, dense_model)) => traces.push(curve(
                    dense_taus.to_vec(),
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
                    // 残差与 P1 差若干数量级，画在 P1 面板的右叠加轴（y5）上
                    traces.push(residual_markers(x_taus, residual, false, "x4", "y5"));
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
///     taus: 延时轴 (n,)
///     iq: 该条扫描的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定
///     sigma: IQ 域的逐点不确定度；长度与 `iq` 不一致时忽略
///     fit: 拟合结果；决定 P1 面板画哪条曲线
///     overlay: 拟合叠加层
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
fn fit_plot(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: &Result<&T2EchoFit, &T2EchoError>,
    overlay: &Overlay,
) -> Plot {
    // 面板上画的那条 P1：有拟合时是它实际拟合的那条，否则是原始投影
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
            taus,
            iq,
            states,
            &prob,
            prob_sigma.as_deref(),
            overlay,
        ) {
            plot.add_trace(trace);
        }
    }
    let mut settings = layout(&GRID_2X2, &panel_specs(overlay, &prob));
    match overlay.t2_echo() {
        Some(t2_echo) => settings = settings.shapes(vec![t2_echo_line(t2_echo)]),
        None => {}
    }
    plot.set_layout(settings);
    plot.set_configuration(interactive_config());
    plot
}

/// 本实验各面板的规格：标题 + 只有格位推不出来的轴选项。
///
/// P1 的 y 轴按**拟合曲线与数据两者**定值域（只看曲线的话，拟合退化时曲线会跑到数据之外，
/// 值域跟着被拽偏、数据点还会被裁掉），并挂一条残差右轴（残差与 P1 差若干数量级，同轴画不
/// 出来）；IQ 面板等比例锁定，否则投影轴会被拉成任意斜率。
///
/// 形参:
///     overlay: 拟合叠加层
///     prob: 本面板画的那条 P1 (n,)，与 `prob` 同序
///
/// 返回值:
///     各面板的规格，次序即绘制次序
fn panel_specs(overlay: &Overlay, prob: &[f64]) -> Vec<PanelSpec> {
    let prob_range = match overlay.model() {
        Some(model) => pad(value_bounds(&[model, prob])),
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

/// T2 echo 的竖虚线：它是这条扫描真正要产出的量，标在 P1 面板上（与数据同轴）。
fn t2_echo_line(t2_echo: f64) -> Shape {
    let (x_ref, y_ref) = axis_refs(NORM_INDEX);
    Shape::new()
        .shape_type(ShapeType::Line)
        .layer(ShapeLayer::Below)
        .x_ref(x_ref.as_str())
        .y_ref(&format!("{y_ref} domain"))
        .x0(t2_echo)
        .x1(t2_echo)
        .y0(0.0)
        .y1(1.0)
        .line(
            ShapeLine::new()
                .color(T2_LINE_COLOR)
                .width(1.0)
                .dash(DashType::Dash),
        )
}
