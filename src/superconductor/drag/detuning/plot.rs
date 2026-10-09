//! DRAG 载波失谐扫描的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出四面板图（|S21|、相位、IQ 平面、P1）+ 参数线图 + 参数表 + 页脚的 HTML 片段，
//! 可直接插入汇总报表 / iframe / Jupyter。图形由 plotly.js 在浏览器端渲染，本模块只生成
//! `<div>` 与 `Plotly.newPlot` 调用（自带 config 的 `Plotly.newPlot` 片段）；宿主页面需自行加载
//! plotly.js，可用 [`PLOTLY_JS_CDN`] 一行引入。
//!
//! 版式与系数扫描那份（`drag::coeff::plot`）逐格相同，只有两处不一样：横轴是载波失谐，以及
//! **三个与失谐同轴的面板上多一条零失谐参考线**——那是"驱动本来该在的地方"，谷压在它上面说明
//! `f01_working` 已经够准，偏离多少就是这一轮要修掉的残差（IQ 面板的横轴是 I，与失谐无关，
//! 不画）。
//!
//! 整条升阶链一起画：
//!
//! - **四个面板都按阶铺**，颜色就是阶数的编码（[`order_color`]）：同一个阶在四个面板里同名同色，
//!   图例**一阶一条**——同名 trace 归一个 legendgroup，点一次就开关这一阶在全图的所有点与线。
//! - **P1 面板**各阶叠自己的拟合线——低阶铺得开、高阶收得窄，一眼看得出窗口是怎么一步步收到
//!   谷底上的；标定值画一条灰虚线，零失谐画一条灰点线。
//! - **IQ 平面**仍按 P1 上色（|0> 端蓝、|1> 端红），阶数只压在**深浅**上（[`graded`]）。
//! - **图下的参数线图**（[`param_panel`]）：下拉框换谷的四个参数，横轴是阶数 n。
//!
//! 单位一律 SI（Hz）：crate 其余报告（s21 / qspec / ramsey）都是这个口径，不在作图这一层偷偷
//! 换单位（baseline 画的是 MHz，那是它的显示选择）。
//!
//! P1 面板没有残差轴：各阶的残差不在同一个尺度上，一条右轴代表不了谁。

use crate::superconductor::drag::detuning::valley::{Valley, ValleyError, ValleyFit};
use crate::superconductor::{StateCenters, p1};
use crate::utils::heatmap::{
    axis_style, escape_html, figure_font, interactive_config, json_array, title_bar_style,
};
use crate::utils::panels::{
    AxisOpts, CARD_WIDTH, Card, Cell, FIT_COLOR, GRID_2X2, ONE_COLOR, PanelSpec, Titles,
    ZERO_COLOR, axis_refs, block, card, color_samples, curve, dense_grid, layout, projection_axis,
    projection_refs, ref_point, series,
};
use crate::utils::data::{Column, Datum, Payload, Table, fit_row};
use crate::utils::params::{ParamRow, params_table};
use lmfit::{Complex64, ModelParams, ModelResult};
use plotly::common::{DashType, ErrorData, ErrorType, Line, Marker, Mode, Position};
use plotly::layout::{Axis, AxisType, Layout, Margin, Shape, ShapeLayer, ShapeLine, ShapeType};
use plotly::{Plot, Scatter, Trace};

/// plotly.js 的 CDN 引入标签；与 s21 / qspec / rabi / ramsey 报告用的是同一个版本，这里转出
/// 以便本报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

/// 标定值那条竖线的颜色与式样，与 rabi 那边 π 幅度那条一致（baseline `CHOSEN_COLOR`）。
const CHOSEN_COLOR: &str = "gray";

/// 零失谐参考线的颜色与式样：同为灰，靠**点线**与标定值那条**虚线**分开（baseline 同此）。
const ZERO_LINE_COLOR: &str = "gray";

/// P1 面板在 `Panel::ALL` 里的序号：标定值与零失谐两条竖线都画在它上面。
const NORM_INDEX: usize = 3;

/// 横轴是失谐的那几块面板：零失谐参考线画在这些格上（不含 IQ 面板）。
const DETUNING_PANELS: [usize; 3] = [0, 1, NORM_INDEX];

/// 本实验的面板：四个实验共用的那四块，只是 P1 那格要装下整条升阶链。
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

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
.qtool-dragdetune{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高） */
.qtool-dragdetune .qtool-plot{width:100%;max-height:85vh}
/* 参数线图：比四面板图扁，图名行与下拉框的样式走 utils::heatmap::title_bar_style */
.qtool-dragdetune .qtool-line{margin-top:14px}
.qtool-dragdetune .qtool-line-plot{max-height:60vh}
/* plotly 自带工具栏固定贴在容器右上角（top:2px），上边距压小后就会压在图上 —— 挪到图名行
   右端（图名行高约 21px + 6px 外边距，工具栏高 19px，居中即 -26px） */
.qtool-dragdetune .qtool-line-plot .modebar{top:-26px}
.qtool-dragdetune .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
</style>"#;

/// 参数线图的下拉框里可画的四项：`(参数名, 显示名)`，顺序与 [`OrderScan::parameters`] 一致。
const VALLEY_PARAMS: [(&str, &str); 4] = [
    ("centre", "centre (Hz)"),
    ("fwhm", "fwhm (Hz)"),
    ("amp", "amp (P1)"),
    ("offset", "offset (P1)"),
];

/// 参数线图的横轴标题。
const PARAM_X_TITLE: &str = "pulse pairs n";

/// 阶数是几何递增的（1、2、4…），log 轴上各阶等距；`dtick` 在 log 轴以 log10 为单位，
/// 取 log10(2) ⇒ 刻度落在 1、2、4、8…（与 s21 vs power 同一条理由）。
const LOG_TICK_STEP: f64 = std::f64::consts::LOG10_2;

/// 参数线图的上边距，px：图名在 HTML 图名行里，plotly 默认的 100px 上边距全是空白。
const PARAM_PLOT_TOP: usize = 10;

// =========================================================================
// 交给报告的那份数据
// =========================================================================

/// 一阶的扫描与拟合：报告画的就是它。
pub struct OrderScan<'a> {
    /// 脉冲对数：这一阶的标号，也是颜色深浅的次序（越深越高阶）
    pub pairs: usize,
    /// 该阶扫过的载波失谐轴 (n,)，单位 Hz
    pub detunings_hz: &'a [f64],
    /// 该阶的平均复数 IQ (n,)
    pub iq: &'a [Complex64],
    /// 该阶的谷拟合结果（每一阶都拟谷，最低阶也不例外）
    pub fit: Result<&'a ValleyFit, &'a ValleyError>,
}

impl<'a> OrderScan<'a> {
    /// 这一阶画在 P1 面板上的点：有拟合时是它实际拟合的那条，否则是原始投影。
    ///
    /// 形参:
    ///     states: 各态标定中心
    ///
    /// 返回值:
    ///     P1 曲线 (n,)
    fn prob(&self, states: &StateCenters) -> Vec<f64> {
        match &self.fit {
            Ok(fit) => fit.p1.clone(),
            Err(_) => p1(self.iq, Some(states)),
        }
    }

    /// 该阶拟合出来的线型；没有拟合时为 None。
    fn model(&self) -> Option<&'a Valley> {
        match &self.fit {
            Ok(fit) => Some(&fit.result.model),
            Err(_) => None,
        }
    }

    /// 这一阶的谷参数（`centre`/`fwhm`/`amp`/`offset`），顺序与 [`VALLEY_PARAMS`] 一致。
    ///
    /// 没拟合出来时为空——调用方按 [`OrderScan::failure`] 取原因。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     四个参数的值与标准误
    fn valley_series(&self) -> Option<[(f64, Option<f64>); 4]> {
        match &self.fit {
            Ok(fit) => {
                let model = &fit.result.model;
                Some([
                    (model.centre, stderr_of(&fit.result, "centre")),
                    (model.fwhm, stderr_of(&fit.result, "fwhm")),
                    (model.amp, stderr_of(&fit.result, "amp")),
                    (model.offset, stderr_of(&fit.result, "offset")),
                ])
            }
            Err(_) => None,
        }
    }

    /// 这一阶的模型参数：`(参数名, 说明, 值, 标准误)`，顺序即线型里的字段序。
    ///
    /// 形参: 无
    ///
    /// 返回值:
    ///     参数表各行，没有拟合时为空
    fn parameters(&self) -> Vec<(&'static str, &'static str, f64, Option<f64>)> {
        match &self.fit {
            Ok(fit) => {
                let model = &fit.result.model;
                vec![
                    (
                        "centre",
                        "centre of the inverted Lorentzian",
                        model.centre,
                        stderr_of(&fit.result, "centre"),
                    ),
                    (
                        "fwhm",
                        "full width at half depth",
                        model.fwhm,
                        stderr_of(&fit.result, "fwhm"),
                    ),
                    (
                        "amp",
                        "depth below the baseline (negative)",
                        model.amp,
                        stderr_of(&fit.result, "amp"),
                    ),
                    (
                        "offset",
                        "baseline away from the valley",
                        model.offset,
                        stderr_of(&fit.result, "offset"),
                    ),
                ]
            }
            Err(_) => Vec::new(),
        }
    }

    /// 拟合失败时的原因；有拟合时为 None。
    fn failure(&self) -> Option<String> {
        match &self.fit {
            Ok(_) => None,
            Err(err) => Some(err.to_string()),
        }
    }
}

/// 某个参数的标准误；缺失或非有限时返回 None。
///
/// 形参:
///     result: 某条拟合结果
///     name: 参数名
///
/// 返回值:
///     标准误；协方差不可用、参数不存在或值非有限时是 None
fn stderr_of<M: ModelParams>(result: &ModelResult<M>, name: &str) -> Option<f64> {
    match result.params.get(name) {
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

// =========================================================================
// 配色
// =========================================================================

/// 三个通道拼成 `#rrggbb`。
fn hex(channels: [u8; 3]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        channels[0], channels[1], channels[2]
    )
}

/// 阶序深浅的**大地色**（浅棕 → 深褐）：`depth` 0 取浅端、1 取深端。
///
/// 三个 DRAG 实验各用一族色系（幅度蓝、系数品红、失谐大地色），同一份报表里把三条链摆在一起
/// 也一眼分得开。浅端不取纯白（白底上看不见），深端留一点余量。颜色按**阶序位置**插出来，不按
/// `pairs` 取模——阶数再多也不会绕回浅色，"越深越高阶"这一维始终读得出来（baseline 那张写死的
/// 三色表就是在这上面翻过车）。
///
/// 形参:
///     depth: 该阶在整条链里的位置，0 最低阶、1 最高阶（只有一阶时取 0）
///
/// 返回值:
///     `#rrggbb`
fn order_color(depth: f64) -> String {
    const LIGHT: [f64; 3] = [0xdc as f64, 0xc4 as f64, 0x9b as f64];
    const DARK: [f64; 3] = [0x4d as f64, 0x33 as f64, 0x19 as f64];
    let mut channels = [0_u8; 3];
    for (slot, (light, dark)) in channels.iter_mut().zip(LIGHT.iter().zip(DARK.iter())) {
        *slot = (light + (dark - light) * depth).round().clamp(0.0, 255.0) as u8;
    }
    hex(channels)
}

/// IQ 平面那套上色的两个色相端点，与 [`ZERO_COLOR`] / [`ONE_COLOR`] 同一对（royalblue / crimson）。
const ZERO_RGB: [f64; 3] = [65.0, 105.0, 225.0];
const ONE_RGB: [f64; 3] = [220.0, 20.0, 60.0];

/// 最低阶混多少白：浅端仍要一眼认得出是蓝是红，所以不拉满。
const GRADE_WHITENESS: f64 = 0.55;

/// 把基准色按阶数调深浅：`depth` 0 最浅（按 [`GRADE_WHITENESS`] 混白）、1 保持原色。
///
/// 只动明度、不动色相——IQ 平面靠色相读 P1，深浅那一维留给阶数。
///
/// 形参:
///     rgb: 基准色的三个通道
///     depth: 该阶在整条链里的位置，0 最低阶、1 最高阶
///
/// 返回值:
///     `#rrggbb`
fn graded(rgb: [f64; 3], depth: f64) -> String {
    let whiteness = (1.0 - depth) * GRADE_WHITENESS;
    let mut channels = [0_u8; 3];
    for (slot, base) in channels.iter_mut().zip(rgb.iter()) {
        *slot = (base + (255.0 - base) * whiteness).round().clamp(0.0, 255.0) as u8;
    }
    hex(channels)
}

// =========================================================================
// 入口
// =========================================================================

/// 把整条升阶链渲染成自包含的 HTML div。
///
/// 形参:
///     orders: 各阶的扫描与拟合，顺序不限（内部按 `pairs` 升序排，颜色跟着排序位置走）
///     chosen: 标定值（P1 面板上那条竖虚线），单位 Hz——由调用方决定取哪一阶的读数
///     states: 各态标定中心；须与各阶拟合用的是同一组，否则 P1 面板上画的点不是被拟合的那些
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-dragdetune">` 片段（图 + 参数线图 + 参数表 + 页脚），并内蕴一份
///     原始数据供下载（见 [`crate::utils::data`]）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn drag_detuning_plot_div(
    orders: &[OrderScan<'_>],
    chosen: f64,
    states: &StateCenters,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    let layers = layers(orders, states);
    let plot = order_plot(&layers, chosen, states);
    let plot_html = crate::utils::data::plot_script(&plot, div_id);

    // 图下另起一块：谷参数 vs 阶数的线图（四面板图不动）。下拉框的样式与 s21 vs power /
    // qspec vs Z 那两张线图同一套，都在 utils::heatmap::title_bar_style 里。
    let tail = format!(
        "{}{}",
        param_panel(&format!("{div_id}-param"), &layers),
        order_table(&layers)
    );
    let body = format!(
        "{}{}",
        block(div_id, "qtool-plot", (CARD_WIDTH, GRID_2X2.height), &plot_html),
        &tail,
    );
    let payload = payload(&layers, chosen, div_id);
    card(Card {
        class: "qtool-dragdetune",
        style: &format!("{STYLE}{}", title_bar_style()),
        div_id,
        body,
        payload: Some(&payload),
        frame,
    })
}

/// 报告载荷：逐阶原始数据（失谐轴、复 IQ、P1、模型）各占一张 `row_<i>` 子表；谷拟合逐阶一行
/// 进 `fits`（`row` 列指向该阶子表，`pairs` 是它的阶数）；标定值进 `params`。
///
/// 行序即报告里的阶序（按 `pairs` 升序）。
///
/// 形参:
///     layers: 逐阶作图素材（已按 `pairs` 升序）
///     chosen: 标定值，单位 Hz
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（逐阶一张 `row_<i>`，随后 `fits` 与 `params`）
fn payload(layers: &[Layer<'_>], chosen: f64, div_id: &str) -> Payload {
    let mut payload = Payload::new(div_id);
    let mut fits: Vec<(Vec<Column>, Vec<Datum>)> = Vec::new();
    // 逐阶子表攒着，等汇总表都落定再一起追加：CSV 只把第一张表当数据行，汇总表要在前
    let mut series: Vec<Table> = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let order = layer.order;
        let model = order.model();
        let curve = match &model {
            Some(model) => Some(model.at(order.detunings_hz)),
            None => None,
        };
        let mut data = Table::new(
            &format!("row_{index}"),
            vec![
                Column::real("detuning", "Hz"),
                Column::complex("iq", "a.u."),
                Column::real("p1", "1"),
                Column::real("model", "1"),
            ],
        );
        for point in 0..order.detunings_hz.len() {
            data.push(&[
                Datum::Real(order.detunings_hz[point]),
                Datum::Complex(order.iq[point].re, order.iq[point].im),
                Datum::Real(layer.prob[point]),
                match &curve {
                    Some(values) => Datum::Real(values[point]),
                    None => Datum::Missing,
                },
            ]);
        }
        series.push(data);
        match &order.fit {
            Ok(fit) => {
                fits.push(fit_row(
                    &fit.result.params,
                    &PARAM_UNITS,
                    &[
                        (Column::reference("row", "1"), Datum::Real(index as f64)),
                        (Column::real("pairs", "1"), Datum::Real(order.pairs as f64)),
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
            payload.table(table);
        }
        None => {}
    }
    let mut params = Table::new("params", vec![Column::real("chosen", "Hz")]);
    params.push(&[Datum::Real(chosen)]);
    payload.table(params);
    for table in series {
        payload.table(table);
    }
    payload
}

/// 谷模型各参数的单位：失谐轴是 Hz，谷心与半宽跟着 Hz。
const PARAM_UNITS: [(&str, &str); 4] = [
    ("centre", "Hz"),
    ("fwhm", "Hz"),
    ("amp", "1"),
    ("offset", "1"),
];

/// 逐阶的作图素材：一阶一条，颜色由阶序位置推出。
struct Layer<'a> {
    /// 该阶的扫描与拟合
    order: &'a OrderScan<'a>,
    /// 该阶画在 P1 面板上的点
    prob: Vec<f64>,
    /// 该阶在整条链里的位置：0 最低阶、1 最高阶；深浅都按它取
    depth: f64,
}

/// 按 `pairs` 升序排好、配好颜色与 P1 曲线。
///
/// 形参:
///     orders: 各阶的扫描与拟合
///     states: 各态标定中心
///
/// 返回值:
///     逐阶素材，次序即颜色由浅到深的次序
fn layers<'a>(orders: &'a [OrderScan<'a>], states: &StateCenters) -> Vec<Layer<'a>> {
    let mut sorted: Vec<&OrderScan> = orders.iter().collect();
    sorted.sort_by_key(|order| order.pairs);
    let count = sorted.len();
    sorted
        .into_iter()
        .enumerate()
        .map(|(index, order)| {
            let depth = match count > 1 {
                true => index as f64 / (count - 1) as f64,
                false => 0.0,
            };
            Layer {
                order,
                prob: order.prob(states),
                depth,
            }
        })
        .collect()
}

// =========================================================================
// 图
// =========================================================================

/// 某面板的全部 trace。
///
/// 形参:
///     panel: 画哪一块
///     x_ref: 本面板的 x 轴名（由 `layout` 按格位给出，见 `axis_refs`）
///     y_ref: 本面板的 y 轴名
///     layers: 逐阶素材，已按阶数升序
///     states: 各态标定中心
///
/// 返回值:
///     该面板的 trace 列表，按绘制顺序
fn panel_traces(
    panel: Panel,
    x_ref: &str,
    y_ref: &str,
    layers: &[Layer<'_>],
    states: &StateCenters,
) -> Vec<Box<dyn Trace>> {
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    match panel {
        Panel::Magnitude => {
            // 图例只在这一格出面：一阶一条。同名 trace 在四个面板里共用一个 legendgroup，点一次
            // 就开关这一阶在全图里的所有点与线（所以其余三格都 show_legend = false）
            for layer in layers {
                let magnitude: Vec<f64> = layer.order.iq.iter().map(|z| z.norm()).collect();
                let name = format!("N={}", layer.order.pairs);
                traces.push(series(
                    layer.order.detunings_hz.to_vec(),
                    magnitude,
                    name.as_str(),
                    order_color(layer.depth).as_str(),
                    true,
                    x_ref,
                    y_ref,
                ));
            }
        }
        Panel::Phase => {
            for layer in layers {
                // 相位取主值 (−π, π]：不做 unwrap，跨割线处会有竖直落差，换来各条曲线电平可比
                let phase: Vec<f64> = layer.order.iq.iter().map(|z| z.arg()).collect();
                let name = format!("N={}", layer.order.pairs);
                traces.push(series(
                    layer.order.detunings_hz.to_vec(),
                    phase,
                    name.as_str(),
                    order_color(layer.depth).as_str(),
                    false,
                    x_ref,
                    y_ref,
                ));
            }
        }
        Panel::Iq => {
            // 这一格的颜色有两个维度：色相读 P1（|0> 端蓝、|1> 端红），深浅读阶数。投影几何另画
            // ——各态的标定中心各阶共用同一组，所以投影轴只有一条
            for layer in layers {
                let name = format!("N={}", layer.order.pairs);
                traces.push(color_samples(
                    layer.order.iq.iter().map(|z| z.re).collect(),
                    layer.order.iq.iter().map(|z| z.im).collect(),
                    &layer.prob,
                    name.as_str(),
                    graded(ZERO_RGB, layer.depth),
                    graded(ONE_RGB, layer.depth),
                    x_ref,
                    y_ref,
                ));
            }
            match layers.first() {
                Some(layer) => {
                    let (ref_zero, ref_one) =
                        projection_refs(layer.order.iq, Some(states), &layer.prob);
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
                None => {}
            }
        }
        Panel::Norm => {
            for layer in layers {
                let name = format!("N={}", layer.order.pairs);
                traces.push(series(
                    layer.order.detunings_hz.to_vec(),
                    layer.prob.clone(),
                    name.as_str(),
                    order_color(layer.depth).as_str(),
                    false,
                    x_ref,
                    y_ref,
                ));
                match layer.order.model() {
                    Some(model) => match dense_grid(layer.order.detunings_hz) {
                        Some(dense) => {
                            let values = model.at(&dense);
                            traces.push(curve(
                                dense,
                                values,
                                name.as_str(),
                                order_color(layer.depth).as_str(),
                                2.0,
                                false,
                                x_ref,
                                y_ref,
                            ));
                        }
                        None => {}
                    },
                    None => {}
                }
            }
        }
    }
    traces
}

/// 四面板图：|S21| / 相位 / IQ 平面 / P1（整条升阶链）。
///
/// 形参:
///     layers: 逐阶素材
///     chosen: 标定值
///     states: 各态标定中心
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
fn order_plot(layers: &[Layer<'_>], chosen: f64, states: &StateCenters) -> Plot {
    let mut plot = Plot::new();
    for (index, panel) in Panel::ALL.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        for trace in panel_traces(*panel, x_ref.as_str(), y_ref.as_str(), layers, states) {
            plot.add_trace(trace);
        }
    }
    // 标定值一条虚线 + 三个与失谐同轴的面板各一条零失谐点线
    let mut shapes = vec![chosen_line(chosen)];
    for index in DETUNING_PANELS {
        shapes.push(zero_line(index));
    }
    let settings = layout(&GRID_2X2, &panel_specs()).shapes(shapes);
    plot.set_layout(settings);
    plot.set_configuration(interactive_config());
    plot
}

/// 本实验各面板的规格：标题 + 只有格位推不出来的轴选项。
///
/// 值域全交给 plotly 自适应（P1 那格一图多阶，钉死任一条曲线都不合适）；没有残差轴，理由见
/// 模块文档；只有 IQ 平面要等比例锁定，否则投影轴会被拉成任意斜率。
fn panel_specs() -> Vec<PanelSpec> {
    let opts = [
        AxisOpts::PLAIN,
        AxisOpts::PLAIN,
        AxisOpts {
            equal_aspect: true,
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

/// 某个面板的标题表（图名与两个轴标题各只出现一次）。
///
/// 横轴是扫描量——载波失谐，单位 SI（Hz），三个与它同轴的面板写法一致；IQ 面板是 I/Q。
fn titles(panel: Panel) -> Titles {
    match panel {
        Panel::Magnitude => Titles {
            x: "Detuning (Hz)",
            y: "|S21|",
            name: "Magnitude",
        },
        Panel::Phase => Titles {
            x: "Detuning (Hz)",
            y: "Phase (rad)",
            name: "Phase",
        },
        Panel::Iq => Titles {
            x: "I",
            y: "Q",
            name: "IQ plane",
        },
        Panel::Norm => Titles {
            x: "Detuning (Hz)",
            y: "P1",
            name: "P1 (all orders)",
        },
    }
}

/// 标定值的竖虚线：它是整条升阶链真正要产出的量，标在 P1 面板上（与数据同轴）。
fn chosen_line(chosen: f64) -> Shape {
    let (x_ref, y_ref) = axis_refs(NORM_INDEX);
    Shape::new()
        .shape_type(ShapeType::Line)
        .layer(ShapeLayer::Below)
        .x_ref(x_ref.as_str())
        .y_ref(&format!("{y_ref} domain"))
        .x0(chosen)
        .x1(chosen)
        .y0(0.0)
        .y1(1.0)
        .line(
            ShapeLine::new()
                .color(CHOSEN_COLOR)
                .width(1.0)
                .dash(DashType::Dash),
        )
}

/// 零失谐参考线：驱动本来该在的地方。谷压在它上面说明 `f01_working` 已经够准，偏离多少就是这一
/// 轮要修掉的残差（baseline 同此，画成灰点线与标定值那条虚线分开）。
///
/// 形参:
///     panel_index: 画在哪一块面板上（横轴必须是失谐，IQ 面板不画）
///
/// 返回值:
///     x = 0 的竖线
fn zero_line(panel_index: usize) -> Shape {
    let (x_ref, y_ref) = axis_refs(panel_index);
    Shape::new()
        .shape_type(ShapeType::Line)
        .layer(ShapeLayer::Below)
        .x_ref(x_ref.as_str())
        .y_ref(&format!("{y_ref} domain"))
        .x0(0.0)
        .x1(0.0)
        .y0(0.0)
        .y1(1.0)
        .line(
            ShapeLine::new()
                .color(ZERO_LINE_COLOR)
                .width(1.0)
                .dash(DashType::Dot),
        )
}

// =========================================================================
// 参数线图（谷参数 vs 阶数）
// =========================================================================

/// 一条参数曲线：逐阶的拟合值与 stderr。
struct ParamSeries {
    /// 参数名，同时是下拉框的 value
    name: &'static str,
    /// 下拉框显示名（带单位），同时用作 y 轴标题
    label: &'static str,
    /// 逐阶的值；某一阶没拟合出来时那一格是 NaN（曲线断开）
    value: Vec<f64>,
    /// 逐阶的标准误；拿不到时是 NaN
    stderr: Vec<f64>,
}

/// 逐阶收集谷的四个参数。
///
/// 形参:
///     layers: 逐阶素材，已按阶数升序
///
/// 返回值:
///     四条曲线，顺序即 [`VALLEY_PARAMS`]
fn param_series(layers: &[Layer<'_>]) -> Vec<ParamSeries> {
    let mut series: Vec<ParamSeries> = VALLEY_PARAMS
        .iter()
        .map(|(name, label)| ParamSeries {
            name,
            label,
            value: Vec::with_capacity(layers.len()),
            stderr: Vec::with_capacity(layers.len()),
        })
        .collect();
    for layer in layers {
        match layer.order.valley_series() {
            Some(items) => {
                for (item, (value, stderr)) in series.iter_mut().zip(items) {
                    item.value.push(value);
                    item.stderr.push(match stderr {
                        Some(error) => error,
                        None => f64::NAN,
                    });
                }
            }
            // 这一阶没拟合出来：四条曲线在这一格都断掉（轴位置保留，看得出缺的是哪一阶）
            None => {
                for item in series.iter_mut() {
                    item.value.push(f64::NAN);
                    item.stderr.push(f64::NAN);
                }
            }
        }
    }
    series
}

/// 谷参数 vs 阶数的线图面板：图名行里一个下拉框换参数，换的时候只做 `Plotly.restyle`/
/// `relayout`，四条曲线的数据出图时就一并写进页面（与 s21 vs power / qspec vs Z 同一套写法）。
///
/// 形参:
///     div_id: 本面板的 div id（与主图的不同）
///     layers: 逐阶素材，已按阶数升序
///
/// 返回值:
///     自包含的线图面板（图名行 + 图 + 切换脚本 + 缩放注册）；没有阶时返回空串
fn param_panel(div_id: &str, layers: &[Layer<'_>]) -> String {
    if layers.is_empty() {
        return String::new();
    }
    let orders: Vec<f64> = layers
        .iter()
        .map(|layer| layer.order.pairs as f64)
        .collect();
    let series = param_series(layers);
    let first = match series.first() {
        Some(item) => item,
        None => return String::new(),
    };

    let scatter = Scatter::new(orders, first.value.clone())
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(FIT_COLOR).width(1.5))
        .marker(Marker::new().color(FIT_COLOR).size(6))
        .error_y(
            ErrorData::new(ErrorType::Data)
                .array(first.stderr.clone())
                .visible(true)
                .thickness(1.0)
                .width(0)
                .color(FIT_COLOR),
        )
        .show_legend(false)
        .x_axis("x")
        .y_axis("y");
    let mut plot = Plot::new();
    let trace: Box<dyn Trace> = scatter;
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .font(figure_font())
            .margin(Margin::new().top(PARAM_PLOT_TOP))
            .x_axis(axis_style(
                Axis::new()
                    .title(PARAM_X_TITLE)
                    .type_(AxisType::Log)
                    .dtick(LOG_TICK_STEP)
                    .tick_format(".3g"),
            ))
            .y_axis(axis_style(Axis::new().title(first.label))),
    );
    plot.set_configuration(interactive_config());
    let plot_html = crate::utils::data::plot_script(&plot, div_id);

    let options: Vec<String> = series
        .iter()
        .map(|item| format!("<option value=\"{}\">{}</option>", item.name, item.label))
        .collect();
    // 图名行 = 下拉框 + 固定的 " vs order n"：换参数时整行跟着变
    let title_bar = format!(
        "<select id=\"{div_id}-select\">{}</select><span> vs order n</span>",
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
    // 那样"centre (Hz)"和"vs order n"之间会空出一大截，看着不像一个标题。
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
    format!(
        "<div class=\"qtool-line\" id=\"{div_id}\">\
         <div class=\"qtool-line-title\">{title_bar}</div>\
         {}{script}</div>\n",
        block(div_id, "qtool-line-plot", (18.0, 5.0), &plot_html)
    )
}

// =========================================================================
// 参数表
// =========================================================================

/// 参数表：**逐阶列出它的全部模型参数**（名字前缀 `N={pairs}`），各带标准误。
///
/// 没拟合出来的阶那一行写 `—`，原因汇总在表下的提示里（照其余报告的口径：求解器出了状况就该
/// 在页面上看得见，不能只在日志里）。
///
/// 形参:
///     layers: 逐阶素材，已按阶数升序
///
/// 返回值:
///     自包含的表 + 尾注片段
fn order_table(layers: &[Layer<'_>]) -> String {
    // 先把每行的四格都做成自持有的，再让 ParamRow 借过去（ParamRow 的 name 是借用）
    let cells: Vec<(String, &'static str, String, String)> = layers
        .iter()
        .flat_map(|layer| {
            let parameters = layer.order.parameters();
            match parameters.is_empty() {
                true => vec![(
                    format!("N={}", layer.order.pairs),
                    "no fit — this order is drawn as points only",
                    "—".to_string(),
                    "—".to_string(),
                )],
                false => parameters
                    .into_iter()
                    .map(|(name, description, value, stderr)| {
                        (
                            format!("N={} {name}", layer.order.pairs),
                            description,
                            format!("{value:.5}"),
                            match stderr {
                                Some(error) => match error.is_finite() {
                                    true => format!("{error:.1e}"),
                                    false => "—".to_string(),
                                },
                                None => "—".to_string(),
                            },
                        )
                    })
                    .collect(),
            }
        })
        .collect();
    let rows: Vec<ParamRow<'_>> = cells
        .iter()
        .map(|(name, description, value, stderr)| ParamRow {
            name: name.as_str(),
            description,
            value: value.clone(),
            stderr: stderr.clone(),
        })
        .collect();

    let failures: Vec<String> = layers
        .iter()
        .filter_map(|layer| match layer.order.failure() {
            Some(error) => Some(format!(
                "N={}: {}",
                layer.order.pairs,
                escape_html(error.as_str())
            )),
            None => None,
        })
        .collect();
    let note = match failures.is_empty() {
        true => None,
        false => Some(format!(
            "no fit for {} — those orders are drawn as points only",
            failures.join("; ")
        )),
    };
    params_table(&rows, note.as_deref())
}
