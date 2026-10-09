//! T2\*（Ramsey）延时扫描的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出五面板图（|S21|、相位、IQ 平面、P1，第三行整宽放 P1 的幅度谱）+ 参数表 +
//! 页脚的 HTML 片段，可直接插入汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，
//! 本模块只生成 `<div>` 与 `Plotly.newPlot` 调用（自带 config 的 `Plotly.newPlot` 片段）；宿主
//! 页面需自行加载 plotly.js，可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 面板内容对齐 baseline `exps/t2_analysis.py::t2_plot` 的同族版式（rabi 报告的五面板骨架）：
//! |S21| 与相位只画数据（拟合在 P1 空间进行，这两个量没有对应的模型曲线）；P1 面板叠阻尼余弦
//! 拟合曲线与残差、画给定 σ 折算过来的 error bar，并标出这次扫描真正要产出的量——**T2\* 的竖
//! 虚线**；IQ 面板画 [`crate::superconductor::p1`] 定轴所用的投影几何：样本点、投影轴、以及
//! 归一化钉在 0 和 1 上的两个参考点。
//!
//! 第三行的频谱不是新算的东西：它就是拟合初值里傅里叶那一路的谱（[`crate::utils::spectrum`]），
//! 画出来是为了让"初值落在哪"与"数据长什么样"在同一页上对得上——顺带让欠采样直接现形（谱峰贴到
//! 最右 bin，甚至折返）。横轴的倒数（1/s）就是 **Hz**，与 rabi 那张"幅度的倒数"不同，标签照实写。

use crate::superconductor::ramsey::ramsey::{RamseyError, RamseyFit};
use crate::superconductor::{StateCenters, p1, p1_sigma};
use crate::utils::heatmap::{escape_html, interactive_config};
use crate::utils::panels::{
    AxisOpts, CARD_WIDTH, Card, Cell, DATA_COLOR, FIT_COLOR, GRID_2X3, ONE_COLOR, PanelSpec,
    ResidualAxis, Titles, ZERO_COLOR, axis_refs, block, card, color_samples, curve, dense_grid,
    layout, markers, pad, projection_axis, projection_refs, ref_point, residual_axis_range,
    residual_markers, series, value_bounds,
};
use crate::utils::data::{Column, Datum, Payload, Table, fit_row};
use crate::utils::params::{ParamRow, params_table};
use crate::utils::spectrum;
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::Position;
use plotly::Plot;

/// plotly.js 的 CDN 引入标签；与 s21 / qspec / rabi 报告用的是同一个版本，这里转出以便 ramsey
/// 报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

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

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
.qtool-ramsey{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 图块按设计比例渲染（比例由 block 的内联 aspect-ratio 给出），不设 max-height：
   压扁之后行距随之缩小、字号却不变，上一行的 x 轴标题会与下一行的面板标题叠在一起 */
.qtool-ramsey .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
</style>"#;

/// 把一条 Ramsey 延时扫描渲染成自包含的 HTML div。
///
/// 形参:
///     taus: 延时轴 (n,)，单位 s（须等间距，与 [`ramsey_fit`] 同一口径）
///     iq: 该条扫描的平均复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定（须与
///             [`crate::superconductor::ramsey::ramsey_fit`] 用的是同一组中心，
///             否则画出来的曲线不是被拟合的那条）
///     sigma: **IQ 域**的逐点测量不确定度（与 [`ramsey_fit`] 同一口径，点数是 n 次单发
///            平均时传 `std(shots)/√n`）；给定时在 P1 面板画折算到 P1 空间的 error bar。
///            长度与 `iq` 不一致时忽略
///     fit: 拟合结果；`Ok` 时在 P1 面板叠阻尼余弦曲线、残差与 T2\* 竖线，`Err` 时只画数据、
///          表位置显示错误文本
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-ramsey">` 片段（图 + 参数表 + 页脚），并内蕴一份原始数据供
///     下载（见 [`crate::utils::data`]）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
///
/// [`ramsey_fit`]: crate::superconductor::ramsey::ramsey_fit
pub fn ramsey_plot_div(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: Result<&RamseyFit, &RamseyError>,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let overlay = overlay(&fit, taus);
    let plot = fit_plot(taus, iq, states, sigma, &fit, &overlay);
    let plot_html = crate::utils::data::plot_script(&plot, div_id);
    let payload = payload(taus, iq, states, sigma, &fit, div_id);
    let body = format!(
        "{}{}",
        block(div_id, "qtool-plot", (CARD_WIDTH, GRID_2X3.height), &plot_html),
        &overlay.table_html(),
    );
    card(Card {
        class: "qtool-ramsey",
        style: STYLE,
        div_id,
        body,
        payload: Some(&payload),
        frame,
    })
}

/// 报告的载荷：喂进拟合的输入（延时轴、复 IQ、IQ 域 σ）+ 投影后的 P1 / P1σ + 模型在数据点上的
/// 值 + 拟合参数 + 标定中心。
///
/// P1 与图上同源：有拟合时取 `fit.p1`（实际拟合的那条），否则现投影一次（[`p1`]）；模型是拟合
/// 曲线在数据点上的取值，无拟合时整列按缺失落。供显示用的量（幅度、相位、残差、密集曲线、频谱）
/// 一概不入表——都能从这里的列推出来。
///
/// 形参:
///     taus: 延时轴 (n,)，s
///     iq: 平均复数 IQ (n,)
///     states: 各态标定中心
///     sigma: IQ 域逐点不确定度；长度与 `iq` 不一致时按缺失处理（与图上的口径一致）
///     fit: 拟合结果
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（`data` + `fits`，给了 `states` 再带一张 `states`）
fn payload(
    taus: &[f64],
    iq: &[Complex64],
    states: Option<&StateCenters>,
    sigma: Option<&[f64]>,
    fit: &Result<&RamseyFit, &RamseyError>,
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
        Ok(result) => Some(result.result.model.at(taus)),
        Err(_) => None,
    };
    let mut data = Table::new(
        "data",
        vec![
            Column::real("tau", "s"),
            Column::complex("iq", "a.u."),
            Column::real("iq_sigma", "a.u."),
            Column::real("p1", "1"),
            Column::real("p1_sigma", "1"),
            Column::real("model", "1"),
        ],
    );
    for index in 0..taus.len() {
        data.push(&[
            Datum::Real(taus[index]),
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

/// 模型各参数的单位（无量纲写 `1`）。
const PARAM_UNITS: [(&str, &str); 5] = [
    ("offset", "1"),
    ("amplitude", "1"),
    ("frequency", "Hz"),
    ("phase", "rad"),
    ("decay", "s"),
];

/// 拟合结果的叠加层：`Ok` 时给出模型曲线、T2\* 与参数表，`Err` 时把错误交给页面。
///
/// 把 `Result` 收成一个值之后，画 trace 与排布局的代码都不必再分支——只有真正不同的两处
/// （P1 面板的模型曲线、表内容）去问它要东西。
enum Overlay<'a> {
    Some {
        /// 数据延时网格上的模型值（算残差用）
        model: Vec<f64>,
        /// 密集网格上的模型值（画平滑的拟合曲线用）
        dense_taus: Vec<f64>,
        dense_model: Vec<f64>,
        /// 参数表各行
        rows: Vec<ParamRow<'a>>,
        /// 表下提示（未收敛）
        note: Option<String>,
    },
    Absent {
        error: &'a RamseyError,
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
fn overlay<'a>(fit: &'a Result<&RamseyFit, &RamseyError>, taus: &[f64]) -> Overlay<'a> {
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
                rows: param_rows(result),
                note: note_of(result),
            }
        }
        Err(error) => Overlay::Absent { error },
    }
}

/// 参数表：T2\*、条纹频率、常数项、幅度、相位，各带标准误。
///
/// 标准误缺失（协方差不可用）或非有限时给占位符，不伪造一个 0 误差。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     参数表各行，顺序即渲染顺序
fn param_rows(fit: &RamseyFit) -> Vec<ParamRow<'static>> {
    let model = &fit.result.model;
    // (行名, lmfit 参数名, 解释, 值, 位数, 是否用科学计数)：行名与参数名分开，是因为 T2\* 在线型
    // 里的字段名是 `decay`，而对外该叫它 T2\*；科学计数是给时间与频率留的——T2\* 以秒为单位就是
    // 1e-5 量级，定点写出来只有一串前导零
    let rows: [(&'static str, &'static str, &'static str, f64, usize, bool); 5] = [
        (
            "T2*",
            "decay",
            "dephasing time",
            model.decay,
            4,
            true,
        ),
        (
            "frequency",
            "frequency",
            "fringe frequency",
            model.frequency,
            6,
            true,
        ),
        (
            "offset",
            "offset",
            "level the fringes decay to",
            model.offset,
            4,
            false,
        ),
        (
            "amplitude",
            "amplitude",
            "peak amplitude of the fringe term",
            model.amplitude,
            4,
            false,
        ),
        ("phase", "phase", "fringe phase (rad)", model.phase, 4, false),
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

/// 表下提示：求解器未收敛、或 T2\* 的标准误拿不到时给一行说明。
///
/// 后一条单列，是因为五参数、无界的阻尼余弦在低信噪比下会滑进"极长衰减 + 极大幅度"的退化解：
/// 残差看着一样好、参数却毫无意义。协方差在那种解上通常已经退化，标准误这一栏会是 `—`——提示
/// 把它挑明，免得 T2\* 那个数被当成可信的读数。
///
/// 形参:
///     fit: 拟合结果
///
/// 返回值:
///     提示文本；收敛且 T2\* 带得上标准误时返回 None
fn note_of(fit: &RamseyFit) -> Option<String> {
    match fit.result.success {
        false => Some(format!("not converged: {}", fit.result.message)),
        true => match stderr_of(fit, "decay") {
            Some(_) => None,
            None => Some(
                "T2* stderr unavailable (degenerate covariance): treat the fitted decay as unconstrained"
                    .to_string(),
            ),
        },
    }
}

/// 某个参数的标准误；缺失或非有限时返回 None。
fn stderr_of(fit: &RamseyFit, name: &str) -> Option<f64> {
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
/// 横轴是扫描量——演化延时，三个面板同轴；IQ 面板是 I/Q；频谱面板换了变量，横轴是延时的倒数，
/// 单位 s⁻¹，**就是 Hz**，标签只能照实写。
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
        Panel::Spectrum => Titles {
            x: "Ramsey freq (Hz)",
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
            // 相位取主值 (−π, π]：不做 unwrap，随 τ 累积的斜坡呈锯齿、跨割线处有竖直落差，换来各条曲线电平可比
            let phase: Vec<f64> = iq.iter().map(|z| z.arg()).collect();
            traces.push(series(x_taus, phase, "Data", DATA_COLOR, false, x_ref, y_ref));
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
                    // 残差与 P1 差若干数量级，画在 P1 面板的右叠加轴（y6）上
                    traces.push(residual_markers(x_taus, residual, false, "x4", "y6"));
                }
                None => {}
            }
        }
        Panel::Spectrum => {
            let (freqs, spectrum_amps) = spectrum(taus, prob);
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
    fit: &Result<&RamseyFit, &RamseyError>,
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
    let settings = layout(&GRID_2X3, &panel_specs(overlay));
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
