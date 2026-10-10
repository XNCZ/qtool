//! IQ 概率实验报告的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 两张图并排：左边是**各态 IQ 云**（散点 + 三条最深密度区域 + 中心 + 判别轴），右边是
//! **判别面板**（两态沿连线投到一维后的两个频数直方图 + 最优阈值）。下面是参数表：每态的中心、
//! 三条面积、展宽与信噪比，以及两两之间的阈值、错分率与 AUC。
//!
//! 那三条区域**不是拟合出来的椭圆**：它们与参数表里的面积出自同一个密度网格
//! （[`crate::utils::density::DensityGrid`]）——网格里点数不低于阈值的格子就是区域本身，
//! 所以画出来的轮廓与报出去的面积不可能是两回事。香蕉、月牙、双峰这些形状也因此能如实
//! 呈现：椭圆会把它压成一个形状，凸包会把凹陷填平，密度区域两者都不会。
//!
//! 深浅是叠出来的：99% 先画、95% 次之、68% 最后，三层同色半透明叠加，内圈自然更深。

use crate::superconductor::iq::iq::{IqStats, LEVELS, StateStats, pair_projection};
use crate::utils::data::{Column, Datum, Payload, Table};
use crate::utils::panels::{CARD_WIDTH, Card, block, card};
use crate::superconductor::iq::{IqError, iq_stats, scatter};
use crate::utils::heatmap::{axis_style, escape_html, figure_font, interactive_config, json_array};
use crate::utils::params::{ParamRow, params_table};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{
    DashType, Fill, HoverInfo, Line, LineShape, Marker, MarkerSymbol, Mode, Position,
};
use plotly::color::Rgba;
use plotly::layout::{
    Axis, BarMode, ItemSizing, Layout, Legend, Margin, Shape, ShapeLayer, ShapeLine, ShapeType,
};
use plotly::{Bar, Plot, Scatter};

/// plotly.js 的 CDN 引入标签；与其它报告用的是同一个版本。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

/// 两个面板在纸面上的横向区间。
///
/// 云图要**方的**：等比例坐标只保证两轴同尺度，面板本身还得是正方形才不会把 IQ 平面画成
/// 扁的。domain 是相对**绘图区**（容器扣掉边距）算的，所以得连着 [`MARGIN`] 一起定：
/// 容器宽高比 1000:480、边距 70/15/12/55 之下，绘图区（plotly 的 `_size`）实测约 814×413 px，
/// 比例 0.507——宽度被图例吃掉一截（`autoexpand` 把 right 边距从 15 撑到约 118），所以不能按
/// 容器算。容器宽度浮动时这个比例会晃，方得差几个百分点；要绝对方只能挪到客户端算。
const CLOUD_DOMAIN: [f64; 2] = [0.0, 0.507];
const DISC_DOMAIN: [f64; 2] = [0.60, 1.0];

/// 固定的绘图区边距（px）：把绘图区尺寸从容器尺寸里"算得出来"，domain 才能按正方形定。
///
/// 上边距 24 是给 plotly 自带工具栏留的：它固定贴在容器右上角（`top:2px`、高 19px），压到
/// 12px 就会骑在绘图区的上边框上。qspec / s21 那两张线图是把工具栏挪进图名行，这里挪不了
/// ——图名条的右端被 `Separation <select>` 占着，工具栏宽 272px 塞不进去，所以改成给它一条
/// 真正的空带（[`PLOT_HEIGHT`] 同步加了 12px，绘图区尺寸因此不变）。
const MARGIN: (usize, usize, usize, usize) = (70, 15, 24, 55);

/// 两个面板共用的纵向区间。
const Y_DOMAIN: [f64; 2] = [0.0, 1.0];

/// 单发点的透明度与尺寸。
const POINT_ALPHA: f64 = 0.3;
const POINT_SIZE: usize = 3;

/// 判别面板两个直方图的柱体透明度：叠放时靠它看出重叠区——两团分得开就是两块各自的色，
/// 分不开时中间那段是混色，这正是"读得准不准"最直观的一眼。
const BAR_ALPHA: f64 = 0.6;

/// 超过这个点数就抽稀着画：单发云动辄几万点，浏览器端铺全部点没有额外信息。
const MAX_POINTS: usize = 6000;

/// 判别面板的直方条数。
const BINS_1D: usize = 40;

/// 每一层密度区域的填充透明度。
///
/// 三层同色叠加（99% 铺底、95% 次之、68% 最后），内圈因此自然更深——像等高线。单层取 0.12
/// 是"叠完仍然淡"的量：最里层叠了三层，等效约 0.34。
const FILL_ALPHA: f64 = 0.12;

/// 画布高度（px，按设计宽度 1000 计）。
///
/// 卡片宽度对齐 [`GRID_2X3`]（与 rabi 报告同宽），高宽比 1000:492 定下画布；两块面板的
/// domain 就是照这个画布量出来的（见 [`CLOUD_DOMAIN`]）。比 480 多出来的 12px 全部给了
/// [`MARGIN`] 的上边距（放工具栏），所以**绘图区尺寸没变**，domain 不用重标。
const PLOT_HEIGHT: f64 = 492.0;

/// 尺寸相关的 CSS：卡片宽度取 [`GRID_2X3`] 的设计宽度（与 rabi / s21 / qspec 的报告并排时
/// 边界对齐），并定下画布高宽比。CSS 读不到 Rust 常量，所以这里插值生成。
/// 各态的主色（|0> 蓝、|1> 红沿用 baseline IQ 云图的既有配色，其余依次补色）。
///
/// 存 RGB 分量而不是色名：`rgb(…)` 字符串能直接喂给 plotly 的任意颜色位。
fn state_rgb(state: usize) -> (u8, u8, u8) {
    match state {
        0 => (65, 105, 225),   // royalblue
        1 => (220, 20, 60),    // crimson
        2 => (46, 139, 87),    // seagreen
        _ => (255, 140, 0),    // darkorange
    }
}

/// 各态的实心色（点、线、文字用）。
fn state_color(state: usize) -> String {
    let (r, g, b) = state_rgb(state);
    format!("rgb({r},{g},{b})")
}

/// 判别面板柱体的颜色：同态同色，只加一层透明度——两块柱靠它看出重叠区。
fn bar_color(state: usize) -> String {
    let (r, g, b) = state_rgb(state);
    format!("rgba({r},{g},{b},{BAR_ALPHA})")
}

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）。
const STYLE: &str = r#"<style>
.qtool-iq{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
.qtool-iq .qtool-iq-bar{display:flex;align-items:baseline;font-size:14px;color:#1f2328}
.qtool-iq .qtool-iq-bar-pad{flex:0 0 7%}
.qtool-iq .qtool-iq-bar-left{flex:0 0 41%;text-align:center}
.qtool-iq .qtool-iq-bar-right{flex:1;text-align:center}
/* 下拉框与 qspec vs Z / s21 vs power 的图名行同一套外观（见 utils::heatmap::title_bar_style）：
   平时无边框无形，悬停或聚焦才显形 */
.qtool-iq .qtool-iq-bar select{font:inherit;color:inherit;background:transparent;border:1px solid transparent;border-radius:6px;margin-left:2px;padding:0 2px;cursor:pointer;appearance:none;-webkit-appearance:none;-moz-appearance:none}
.qtool-iq .qtool-iq-bar select:hover,.qtool-iq .qtool-iq-bar select:focus{border-color:#cbd5e1;background:#ffffff}
.qtool-iq .qtool-error{margin-top:6px;padding:8px 10px;border-radius:6px;background:#fef2f2;color:#b91c1c;font-size:13px}
</style>"#;

/// 把一批各态单发 IQ 云渲染成自包含的 HTML div。
///
/// 形参:
///     iqs: 各态的单发 IQ 云；`iqs[i]` 是第 i 个态，索引即态编号
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-iq">` 片段（图 + 参数表 + 页脚），并内蕴一份原始数据供下载
///     （见 [`crate::utils::data`]）；统计失败时只给错误条（没有量可内蕴，载荷跟着缺）。
///     宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn iq_plot_div(iqs: &[Vec<Complex64>], div_id: &str, frame: Option<&str>) -> String {
    // 统计失败时没有量可内蕴，载荷跟着缺（卡片照出，只给错误条）
    let mut data: Option<Payload> = None;
    let body = match iq_stats(iqs) {
        Ok(stats) => {
            let series = pair_series(iqs, &stats);
            let plot = iq_plot(iqs, &stats);
            let plot_html = crate::utils::data::plot_script(&plot, div_id);
            data = Some(payload(iqs, &stats, div_id));
            format!(
                "{}{}{}{}{}",
                pair_bar(div_id, &series),
                block(div_id, "qtool-plot", (CARD_WIDTH, PLOT_HEIGHT), &plot_html),
                pair_script(div_id, &series),
                legend_script(div_id, iqs.len()),
                params_body(&stats),
            )
        }
        Err(error) => error_body(&error),
    };
    card(Card {
        class: "qtool-iq",
        style: STYLE,
        div_id,
        body,
        payload: data.as_ref(),
        frame,
    })
}

/// 报告载荷：每态的单发 IQ **全量**各占一张以行号为名的子表（i 即态编号）；中心、逐级密度
/// 区域的面积与半径、每对态的判别统计各一张汇总表。
///
/// 这张报告没有拟合：中心、面积、阈值的算法都在 [`crate::superconductor::iq`]，载荷给的是
/// 它算出来的量与喂进去的单发点本身。
///
/// 形参:
///     iqs: 各态的单发 IQ 云（`iqs[i]` 是第 i 个态，索引即态编号）
///     stats: 统计结果
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（`states`、`density`、`pairs` 在前，随后逐态一张以行号为名的子表）
fn payload(iqs: &[Vec<Complex64>], stats: &IqStats, div_id: &str) -> Payload {
    let mut report = Payload::new(div_id);
    let mut states = Table::new(
        "states",
        vec![
            Column::real("state", "1"),
            Column::reference("row", "1"),
            Column::complex("center", "a.u."),
        ],
    );
    let mut density = Table::new(
        "density",
        vec![
            Column::real("state", "1"),
            Column::real("level", "1"),
            Column::real("area", "a.u.^2"),
            Column::real("radius", "a.u."),
        ],
    );
    for (index, state) in stats.states().iter().enumerate() {
        states.push(&[
            Datum::Real(index as f64),
            Datum::Real(index as f64),
            Datum::Complex(state.center.re, state.center.im),
        ]);
        for (slot, level) in LEVELS.iter().enumerate() {
            density.push(&[
                Datum::Real(index as f64),
                Datum::Real(*level),
                Datum::Real(state.areas[slot]),
                Datum::Real(state.radii[slot]),
            ]);
        }
    }
    let mut pairs = Table::new(
        "pairs",
        vec![
            Column::real("p", "1"),
            Column::real("q", "1"),
            Column::real("separation", "a.u."),
            Column::complex("threshold", "a.u."),
            Column::real("error_rate", "1"),
            Column::real("auc", "1"),
            Column::real("snr", "1"),
        ],
    );
    for pair in stats.pairs() {
        pairs.push(&[
            Datum::Real(pair.state_p as f64),
            Datum::Real(pair.state_q as f64),
            Datum::Real(pair.separation),
            Datum::Complex(pair.threshold.re, pair.threshold.im),
            Datum::Real(pair.error_rate),
            Datum::Real(pair.auc),
            Datum::Real(pair.snr),
        ]);
    }
    report.table(states);
    report.table(density);
    report.table(pairs);
    for (index, cloud) in iqs.iter().enumerate() {
        let mut table = Table::new(&format!("{index}"), vec![Column::complex("iq", "a.u.")]);
        for shot in cloud {
            table.push(&[Datum::Complex(shot.re, shot.im)]);
        }
        report.table(table);
    }
    report
}

/// 图对象：左云图 + 右判别面板。
///
/// 形参:
///     iqs: 各态的单发 IQ 云
///     stats: 统计结果
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
pub fn iq_plot(iqs: &[Vec<Complex64>], stats: &IqStats) -> Plot {
    let mut plot = Plot::new();
    for trace in cloud_traces(iqs, stats) {
        plot.add_trace(trace);
    }
    let series = pair_series(iqs, stats);
    for trace in discriminant_traces(&series) {
        plot.add_trace(trace);
    }
    let [x_low, x_high] = cloud_range(stats);
    // 判别面板的横轴标题带态对名（刻度文字是连线上的复数坐标，由浏览器端按视野重算）；
    // 范围也照第一对钉住——不钉的话初画是 plotly 自适应的 min/max，两端的柱贴着边框，
    // 而切换态对时走的是带留边的范围，同一个面板两种边距。无对可比时才退回通用标题、交给自适应
    let x_title = match series.first() {
        Some(item) => item.x_title(),
        None => String::from("IQ coordinate"),
    };
    let mut x2_axis = Axis::new()
        .domain(&DISC_DOMAIN)
        .anchor("y2")
        .title(x_title.as_str());
    match series.first() {
        Some(item) => x2_axis = x2_axis.range(vec![item.range[0], item.range[1]]),
        None => {}
    }
    let mut layout = Layout::new()
        .font(figure_font())
        .show_legend(true)
        // 图例符号一律按固定大小画：散点是 3px 的小点，跟着 trace 走的话图例里就只剩一粒灰
        .legend(Legend::new().item_sizing(ItemSizing::Constant))
                .margin(
            Margin::new()
                .left(MARGIN.0)
                .right(MARGIN.1)
                .top(MARGIN.2)
                .bottom(MARGIN.3),
        )
        .x_axis(axis_style(
            Axis::new()
                .domain(&CLOUD_DOMAIN)
                .anchor("y")
                .title("I")
                .range(vec![x_low, x_high]),
        ))
        .y_axis(axis_style(
            Axis::new()
                .domain(&Y_DOMAIN)
                .anchor("x")
                .title("Q")
                // 等比例：云的形状本身有意义，轴比例不等会把椭圆/香蕉拉成另一个形状
                .scale_anchor("x"),
        ))
        .x_axis2(axis_style(x2_axis))
        .y_axis2(axis_style(
            Axis::new()
                .domain(&Y_DOMAIN)
                .anchor("x2")
                .title("shots (frequency per bin)"),
        ))
        // 两个直方图叠放：同一套 bin 上比两个分布，并排会各占半格宽、看着更碎
        .bar_mode(BarMode::Overlay);
    layout = add_threshold_line(layout, &series);
    plot.set_layout(layout);
    plot.set_configuration(interactive_config());
    plot
}

/// 云图的固定坐标范围：各态 99% 区域的外接框再放两成。
///
/// 钉死范围是为了"点开散点"时不跳视口——散点默认收起，若放任自适应，用户点开的那一刻
/// 视口会跳到能容纳全部单发点（逃逸点可能远在几个 99% 之外），一跳就回不到看形状的构图。
fn cloud_range(stats: &IqStats) -> [f64; 2] {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for state in stats.states() {
        let reach = 2.0 * state.radii[2];
        low = low.min(state.center.re - reach);
        high = high.max(state.center.re + reach);
    }
    match low <= high {
        true => [low, high],
        false => [-1.0, 1.0],
    }
}

/// 云图面板：各态散点、三条密度区域、中心与判别轴。
fn cloud_traces(iqs: &[Vec<Complex64>], stats: &IqStats) -> Vec<Box<dyn Trace>> {
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    for (index, cloud) in iqs.iter().enumerate() {
        let (all_x, all_y) = scatter(cloud);
        let step = match all_x.len() > MAX_POINTS {
            true => all_x.len() / MAX_POINTS + 1,
            false => 1,
        };
        let trace: Box<Scatter<f64, f64>> = Scatter::new(
            all_x.iter().step_by(step).copied().collect(),
            all_y.iter().step_by(step).copied().collect(),
        )
        .name("shots")
        .legend_group("shots")
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .color(state_color(index))
                .size(POINT_SIZE)
                .opacity(POINT_ALPHA),
        )
        .show_legend(index == 0)
        // 逐点看不出更多东西，默认收起，点图例再展开
        .visible(plotly::common::Visible::LegendOnly)
        .x_axis("x")
        .y_axis("y");
        traces.push(trace);
    }
    for (index, state) in stats.states().iter().enumerate() {
        for trace in level_traces(state, index) {
            traces.push(trace);
        }
    }
    for (index, state) in stats.states().iter().enumerate() {
        traces.push(center_point(state.center, index));
    }
    traces
}

/// 一个态的三条最深密度区域：**照多边形画**。
///
/// 多边形来自 [`crate::utils::density::Region`]，与参数表里的面积是同一份坐标——面积积的
/// 就是这些环，画的就是这些环。不经过 plotly 的等值线（它的 constraint 填充语义与直觉相反，
/// 实测 `>=` 会把网格铺满，只能靠 `<=` 反着用），几何全在我们自己手里。
///
/// 由外向内画（99% → 68%），内层叠在外层上，深浅自己就出来了。
fn level_traces(state: &StateStats, index: usize) -> Vec<Box<dyn Trace>> {
    let (r, g, b) = state_rgb(index);
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    for slot in (0..LEVELS.len()).rev() {
        let region = &state.regions[slot];
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (exterior, holes) in region.polygons() {
            // 外环与洞拼成一条路径：环之间用 NaN 断开。洞要真的空出来，否则环形云的
            // 中心会被填实，而报出去的面积里那个洞是减掉了的——画与报就对不上了
            for ring in std::iter::once(exterior).chain(holes.iter()) {
                for (px, py) in ring {
                    x.push(*px);
                    y.push(*py);
                }
                let (first_x, first_y) = match ring.first() {
                    Some(point) => *point,
                    None => continue,
                };
                x.push(first_x);
                y.push(first_y);
                x.push(f64::NAN);
                y.push(f64::NAN);
            }
        }
        if x.is_empty() {
            continue;
        }
        // 图例按**层级**分组（而不是按态）：点一次 "68%" 就能把所有态的 68% 一起开关。
        // 一组只让第一块面板的那条挂名，其余留 `show_legend(false)`——它们同属一组，
        // 点组名会一起翻转
        let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
            .name(&format!("{}%", (LEVELS[slot] * 100.0).round()))
            .legend_group(&format!("level-{}", (LEVELS[slot] * 100.0).round()))
            .show_legend(index == 0)
            .hover_info(HoverInfo::Skip)
            .mode(Mode::Lines)
            .fill(Fill::ToSelf)
            .fill_color(Rgba::new(r, g, b, FILL_ALPHA))
            // 平滑交给 plotly：`shape: spline` + 最大 smoothing。折线本身是 marching squares
            // 出来的，格子尺度上带锯齿；在**这里**平滑而不是先把密度场抹得更狠，是因为抹平
            // 各向同性，带宽一大薄环这类细结构就糊了——拐角磨圆则不动结构。
            .line(
                Line::new()
                    .color(state_color(index))
                    .width(1.0)
                    .shape(LineShape::Spline)
                    .smoothing(1.3),
            )
            .x_axis("x")
            .y_axis("y");
        traces.push(trace);
    }
    traces
}

/// 中心标记（× 号 + 态编号）。
fn center_point(center: Complex64, index: usize) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(vec![center.re], vec![center.im])
        .name("centers")
        .legend_group("centers")
        .mode(Mode::MarkersText)
        .text_array(vec![format!("|{index}>")])
        .text_position(Position::TopRight)
        .marker(
            Marker::new()
                .color(state_color(index))
                .size(12)
                .symbol(MarkerSymbol::X),
        )
        .show_legend(index == 0)
        .x_axis("x")
        .y_axis("y");
    trace
}

/// 一个态对在判别面板上要画的东西：两条直方图 + 阈值 + 横轴范围。
///
/// 下拉框切换态对时，浏览器端拿的就是这份（Rust 侧算好，切的时候不回后端）。
struct PairSeries {
    /// 这一对的两个态编号（用于配色与图例）
    state_p: usize,
    state_q: usize,
    /// 两条直方图的 bin 中心（**距两中心中点的实际距离**，IQ 单位）
    centers: Vec<f64>,
    /// 各 bin 的频率（该 bin 的单发数 ÷ 该态的**总**单发数）：两态单发数不同也可比
    freq_p: Vec<f64>,
    freq_q: Vec<f64>,
    /// 判别边界的位置，换算成"距两中心中点的实际距离"
    threshold: f64,
    /// 横轴范围：这一对自己数据的范围（min/max）左右各放它一成，min/max 不贴着边框
    range: [f64; 2],
    /// 两中心中点（连线上的复数坐标以它为原点）与连线方向的单位复数：横轴刻度文字靠这两个换算
    midpoint: Complex64,
    unit: Complex64,
}

impl PairSeries {
    /// 这一对的图例名（两条柱各一条图例）。
    fn labels(&self) -> (String, String) {
        (
            format!("|{}>", self.state_p),
            format!("|{}>", self.state_q),
        )
    }

    /// 横轴标题：横轴读数要讲清"在哪条线上"——换了一对态，这条线就是另一条。
    fn x_title(&self) -> String {
        format!(
            "IQ coordinate on the |{}>–|{}> axis",
            self.state_p, self.state_q
        )
    }
}

/// 把每个态对都算成一份 [`PairSeries`]。
///
/// [`pair_projection`] 给的是 **P1 坐标**（两中心分别落在 0 与 1），这里乘回这一对的两中心间距、
/// 再平移到**两中心的中点**：横轴于是是实际距离（IQ 数据单位），两团对称落在 ∓d/2，判别边界也
/// 落在它真正的位置上。换算只发生在画图层——[`PairStats`] 的口径不动。
///
/// 判别边界在对外的 [`PairStats`] 里是 IQ 平面上的复数（交点），这里用同一对态的归一化变换
/// 反解出它的 P1，再换算成"距中点"的实际距离。
///
/// 画的是直方图，横轴的值只能是实数（距离），所以"连线上的复数"落在**刻度文字**上：刻度位置
/// 由浏览器端按当前视野算（缩放后跟着变），文字给出该位置在 IQ 平面上的复数坐标——换算只需要
/// 中点与单位方向这两个量，随每对一起交给 [`pair_script`]。
fn pair_series(iqs: &[Vec<Complex64>], stats: &IqStats) -> Vec<PairSeries> {
    let mut series = Vec::with_capacity(stats.pairs().len());
    for pair in stats.pairs() {
        let (p_scores, q_scores, threshold) =
            pair_projection(iqs, stats, pair.state_p, pair.state_q);
        let bins = bin_edges(&[&p_scores, &q_scores], BINS_1D);
        let (centers, freq_p) = histogram_1d(&p_scores, &bins);
        let (_, freq_q) = histogram_1d(&q_scores, &bins);
        // P1 → 实际距离是仿射变换，bin 归属不变，所以只需换算 bin 中心与两端
        let distance = |value: f64| (value - 0.5) * pair.separation;
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for value in p_scores.iter().chain(q_scores.iter()) {
            match value.is_finite() {
                true => {
                    low = low.min(*value);
                    high = high.max(*value);
                }
                false => {}
            }
        }
        let (low, high) = (distance(low), distance(high));
        let pad = 0.1 * (high - low);
        // 中点与单位方向就地算：PairStats 只存"不可再推"的量（中心在 states 里、间距在 separation）
        let p_center = stats.states()[pair.state_p].center;
        let q_center = stats.states()[pair.state_q].center;
        series.push(PairSeries {
            state_p: pair.state_p,
            state_q: pair.state_q,
            centers: centers.iter().map(|value| distance(*value)).collect(),
            freq_p,
            freq_q,
            // 交点是复数，用同一对态的变换反解出它的 P1，再换成距中点的实际距离
            threshold: distance(pair.transform.apply(threshold).re),
            range: [low - pad, high + pad],
            midpoint: 0.5 * (p_center + q_center),
            unit: (q_center - p_center) / pair.separation,
        });
    }
    series
}

/// 判别面板：两态沿连线投影后的两个**频数直方图**（共用同一套 bin，才可比）。
///
/// 画的是**第一对**，其余的对由下拉框切换（切换只需 restyle 这两条柱的数据，见
/// [`pair_script`]）。多态时把所有对同时画出来会挤成一团。
fn discriminant_traces(series: &[PairSeries]) -> Vec<Box<dyn Trace>> {
    let mut traces: Vec<Box<dyn Trace>> = Vec::new();
    let first = match series.first() {
        Some(item) => item,
        None => return traces,
    };
    for (values, index) in [(&first.freq_p, first.state_p), (&first.freq_q, first.state_q)] {
        let trace: Box<Bar<f64, f64>> = Bar::new(first.centers.clone(), values.clone())
            .name(&format!("|{index}>"))
            .legend_group("pair")
            .marker(Marker::new().color(bar_color(index)))
            .show_legend(true)
            .x_axis("x2")
            .y_axis("y2");
        traces.push(trace);
    }
    traces
}

/// 两个分布共用的 bin 边界（覆盖两者的并集范围）。
fn bin_edges(groups: &[&[f64]], bins: usize) -> Vec<f64> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for group in groups {
        for value in group.iter() {
            match value.is_finite() {
                true => {
                    low = low.min(*value);
                    high = high.max(*value);
                }
                false => {}
            }
        }
    }
    match low <= high && high > low {
        true => (0..=bins)
            .map(|k| low + (high - low) * k as f64 / bins as f64)
            .collect(),
        false => Vec::new(),
    }
}

/// 一维直方图的 (bin 中心, 频率)：频率 = 该 bin 的单发数 ÷ **该态自己的总单发数**。
///
/// 按各态自己的总数归一，两态单发数不同的实验里两条柱才可比（柱高读作"这个态有多大比例落在
/// 这一段"）；代价是柱高不再反映绝对单发数。
fn histogram_1d(values: &[f64], edges: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let count = edges.len();
    if count < 2 {
        return (Vec::new(), Vec::new());
    }
    let bins = count - 1;
    let mut counts = vec![0.0; bins];
    let mut used = 0.0;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        let slot = ((value - edges[0]) / (edges[bins] - edges[0]) * bins as f64).floor();
        let slot = match slot < 0.0 {
            true => 0.0,
            false => match slot >= bins as f64 {
                true => bins as f64 - 1.0,
                false => slot,
            },
        } as usize;
        counts[slot] += 1.0;
        used += 1.0;
    }
    let centers = (0..bins)
        .map(|k| 0.5 * (edges[k] + edges[k + 1]))
        .collect();
    let scale = match used > 0.0 {
        true => used,
        false => 1.0,
    };
    (centers, counts.iter().map(|value| value / scale).collect())
}

/// 阈值竖线：画在判别面板上（x2），高度占满面板。
///
/// 画在柱体**之上**（`ShapeLayer::Above`）：判别阈值是要一眼看到的量，压在柱子底下就看不着了。
fn add_threshold_line(layout: Layout, series: &[PairSeries]) -> Layout {
    match series.first() {
        Some(pair) => layout.shapes(vec![
            Shape::new()
                .shape_type(ShapeType::Line)
                .layer(ShapeLayer::Above)
                .x_ref("x2")
                .y_ref("y2 domain")
                .x0(pair.threshold)
                .x1(pair.threshold)
                .y0(0.0)
                .y1(1.0)
                .line(ShapeLine::new().color("gray").width(1.0).dash(DashType::Dash)),
        ]),
        None => layout,
    }
}

/// 报告外壳：根 div + 样式 + 标题条 + 图 + 缩放注册 + 脚本 + 参数表 + 页脚。
///
/// 图上的标题条：左边写云图的面板名，右边是"判别面板 + 选哪一对"的下拉框。
///
/// 下拉框只能放在 HTML 里（plotly 的注解里放不下 `<select>`），所以判别面板的图名从图内
/// 注解挪到了这里——与 s21 vs power / qspec vs Z 的线图同一做法。
fn pair_bar(div_id: &str, series: &[PairSeries]) -> String {
    let options: Vec<String> = series
        .iter()
        .enumerate()
        .map(|(index, item)| {
            format!(
                "<option value=\"{index}\">|{}>–|{}></option>",
                item.state_p, item.state_q
            )
        })
        .collect();
    format!(
        "<div class=\"qtool-iq-bar\"><span class=\"qtool-iq-bar-pad\"></span>         <span class=\"qtool-iq-bar-left\">IQ clouds</span>         <span class=\"qtool-iq-bar-right\">Separation          <select id=\"{div_id}-pair\">{}</select></span></div>",
        options.join("")
    )
}

/// 切换态对的脚本：换的是判别面板那两条柱的数据、颜色、图例名，以及阈值竖线、横轴范围、
/// 刻度与标题。
///
/// 所有态对的数据在出图时就一并写进页面（几个态对 × 几十个 bin，体积可忽略），切换因此
/// 不必回后端；与 qspec vs Z 的参数下拉框同一套写法。**颜色与刻度也必须一起换**——柱的颜色是
/// 态的专属色，刻度是那条线上的复数坐标，换了对却不换这两样，页面上就会画出"|2> 是红的、
/// 刻度还是 |0>–|1> 那条线的坐标"这种错。
fn pair_script(div_id: &str, series: &[PairSeries]) -> String {
    let data: Vec<String> = series
        .iter()
        .map(|item| {
            let (label_a, label_b) = item.labels();
            // m = 中点（刻度文字的复数原点），u = 连线方向的单位复数：浏览器端拿这两个把
            // 横轴的实参数换算成 IQ 平面上的复数坐标
            format!(
                "{{\"labelA\":\"{label_a}\",\"labelB\":\"{label_b}\",\"colorA\":\"{}\",\"colorB\":\"{}\",\"x\":{},\"a\":{},\"b\":{},\"thr\":{:e},\"lo\":{:e},\"hi\":{:e},\"m\":{},\"u\":{},\"xTitle\":\"{}\"}}",
                bar_color(item.state_p),
                bar_color(item.state_q),
                json_array(&item.centers),
                json_array(&item.freq_p),
                json_array(&item.freq_q),
                item.threshold,
                item.range[0],
                item.range[1],
                json_array(&[item.midpoint.re, item.midpoint.im]),
                json_array(&[item.unit.re, item.unit.im]),
                item.x_title(),
            )
        })
        .collect();
    format!(
        r#"<script>
(function () {{
  var select = document.getElementById("{div_id}-pair");
  var gd = document.getElementById("{div_id}-plot");
  var PAIRS = [{}];
  if (!select || !gd || !window.Plotly) {{ return; }}
  // 下拉框宽度按当前选项实测文字宽度来定 —— 原生 select 会撑到最宽的那个选项，
  // 选中窄的那一对时"态对"与后面的空白之间会空出一大截
  var ruler = document.createElement("canvas").getContext("2d");
  var fitWidth = function () {{
    var picked = select.options[select.selectedIndex];
    if (!picked || !ruler) {{ return; }}
    ruler.font = getComputedStyle(select).font;
    select.style.width = Math.ceil(ruler.measureText(picked.text).width + 12) + "px";
  }};
  fitWidth();
  var current = PAIRS[select.value];
  // 横轴的值是"距中点的距离"（实数），刻度文字要写**这条连线上的复数坐标**：plotly 的自动
  // 刻度不会做这个换算，于是自己按当前视野重算——缩放、双击复位、切态对都会改 x2 的范围，
  // 范围一变就重算，刻度因此跟着视野走，不再是钉死的几个数。
  var complexAt = function (p, decimals) {{
    var re = current.m[0] + p * current.u[0];
    var im = current.m[1] + p * current.u[1];
    var half = 0.5 * Math.pow(10, -decimals);
    var trim = function (v) {{ return (Math.abs(v) < half ? 0 : v).toFixed(decimals); }};
    return trim(re) + (im < 0 ? "" : "+") + trim(im) + "i";
  }};
  var drawTicks = function () {{
    if (!current) {{ return; }}
    var axis = gd._fullLayout && gd._fullLayout.xaxis2;
    if (!axis || !axis.range) {{ return; }}
    var span = axis.range[1] - axis.range[0];
    if (!(span > 0)) {{ return; }}
    // 刻度步长按 1/2/5 × 10^k 取（视野里落两三根，标签带复数、比普通刻度长，要留得松些）；
    // 小数位数跟着步长走，缩放到底也读得出差别
    var rough = span / 4;
    var power = Math.pow(10, Math.floor(Math.log10(rough)));
    var step = [1, 2, 5, 10].find(function (m) {{ return rough <= m * power; }}) * power;
    var decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)) + 2));
    var values = [];
    var texts = [];
    var last = axis.range[1] + step * 1e-6;
    for (var p = Math.ceil(axis.range[0] / step) * step; p <= last; p += step) {{
      values.push(p);
      texts.push(complexAt(p, decimals));
    }}
    Plotly.relayout(gd, {{ "xaxis2.tickmode": "array", "xaxis2.tickvals": values, "xaxis2.ticktext": texts }});
  }};
  var apply = function () {{
    var item = PAIRS[select.value];
    if (!item) {{ return; }}
    current = item;
    // 判别面板的两条柱是最后加进去的两条
    var last = gd.data.length - 1;
    Plotly.restyle(gd, {{
      "x": [item.x, item.x],
      "y": [item.a, item.b],
      "name": [item.labelA, item.labelB],
      "marker.color": [item.colorA, item.colorB],
    }}, [last - 1, last]);
    // 范围一改就会触发下面的重算刻度，不必在这里再调一次
    Plotly.relayout(gd, {{
      "xaxis2.range": [item.lo, item.hi],
      "xaxis2.title.text": item.xTitle,
      "shapes[0].x0": item.thr,
      "shapes[0].x1": item.thr,
    }});
    fitWidth();
  }};
  // 只认范围那几个键：自己写刻度的那次 relayout 只带 tick*，不会再触发自己
  gd.on("plotly_relayout", function (event) {{
    // "重置坐标轴"退回的是 plotly 自适应的 min/max（两端的柱又贴着边框）：按本对的带边范围
    // 钉回去，于是复位之后看到的与初画是同一个视野；这一次 relayout 带着 range，刻度在下一轮重算
    if (current && "xaxis2.autorange" in event) {{
      Plotly.relayout(gd, {{ "xaxis2.range": [current.lo, current.hi] }});
      return;
    }}
    var watched = ["xaxis2.range", "xaxis2.range[0]", "xaxis2.range[1]", "xaxis2.autorange"];
    if (watched.some(function (key) {{ return key in event; }})) {{ drawTicks(); }}
  }});
  drawTicks();
  select.onchange = apply;
}})();
</script>"#,
        data.join(",")
    )
}

/// 图例里那几条**一次管所有态**的条目（shots / 68% / 95% / 99% / 100% / centers）取的是组内
/// 第一条 trace 的颜色，于是看着像是在只管 |0>。浏览器端把它们刷成**各态颜色的渐变块**
/// （彩虹）——一眼看出管的是全部态；`|i>` 那几条是单态的，保持原色不动。
///
/// 为什么不用 plotly 自己的：scatter / bar 的图例色块只取颜色数组的第一个（只有 pie 分片），
/// 渐变只能自己往 SVG 上画。画在 `plotly_afterplot` 之后——重画会把图例的节点连同 `defs`
/// 一起重建，颜色得每轮再刷一遍。
fn legend_script(div_id: &str, states: usize) -> String {
    let colors: Vec<String> = (0..states)
        .map(|index| {
            let (r, g, b) = state_rgb(index);
            format!("\"rgb({r},{g},{b})\"")
        })
        .collect();
    format!(
        r#"<script>
(function () {{
  var gd = document.getElementById("{div_id}-plot");
  if (!gd || !window.Plotly) {{ return; }}
  var COLORS = [{}];
  var GROUPS = ["shots", "68%", "95%", "99%", "100%", "centers"];
  var ID = "{div_id}-legend-rainbow";
  var SVG = "http://www.w3.org/2000/svg";
  var paint = function () {{
    var svg = gd.querySelector("svg.main-svg");
    if (!svg) {{ return; }}
    var defs = svg.querySelector("defs");
    if (!defs) {{
      defs = document.createElementNS(SVG, "defs");
      svg.insertBefore(defs, svg.firstChild);
    }}
    var old = document.getElementById(ID);
    if (old) {{ old.remove(); }}
    var gradient = document.createElementNS(SVG, "linearGradient");
    gradient.setAttribute("id", ID);
    gradient.setAttribute("x1", "0%");
    gradient.setAttribute("x2", "100%");
    COLORS.forEach(function (color, index) {{
      var offset = COLORS.length > 1 ? (100 * index) / (COLORS.length - 1) : 0;
      var stop = document.createElementNS(SVG, "stop");
      stop.setAttribute("offset", offset + "%");
      stop.setAttribute("stop-color", color);
      gradient.appendChild(stop);
    }});
    defs.appendChild(gradient);
    var rainbow = "url(#" + ID + ")";
    Array.from(gd.querySelectorAll(".legend g.traces")).forEach(function (row) {{
      if (GROUPS.indexOf(row.querySelector(".legendtext").textContent) < 0) {{ return; }}
      // 密度区域那几条是"色块 + 上面压一条线"：那条线是根零高的线，objectBoundingBox 的渐变
      // 遇到零尺寸包围盒不渲染（SVG 规则），索性把它藏掉、只留色块。颜色也只能写 style——
      // plotly 把颜色写在内联 style 上，`fill` 这种表现属性会被它压住
      Array.from(row.querySelectorAll("path")).forEach(function (path) {{
        if (getComputedStyle(path).stroke !== "none") {{
          path.style.strokeOpacity = 0;
          return;
        }}
        path.style.fill = rainbow;
        // 图例是"键"不是预览：散点的 0.3、区域块的 0.12 半透明在这里只会让颜色看不清
        path.style.fillOpacity = 1;
        path.style.opacity = 1;
      }});
    }});
  }};
  gd.on("plotly_afterplot", paint);
  paint();
}})();
</script>"#,
        colors.join(",")
    )
}

/// 统计失败时的正文：只有一条错误提示。
fn error_body(error: &IqError) -> String {
    format!(
        "<div class=\"qtool-error\">IQ analysis failed: {}</div>",
        escape_html(&error.to_string())
    )
}

/// 参数表：**一行一个参数**——名字写在 Parameter 列，Description 只放解释，不塞名字。
///
/// 表头四列是 `Parameter | Description | Value | stderr`，与各拟合报告同一版式：那边 Parameter
/// 是 `a_pi`/`freq`/`amp`，这边就是 `|0> spread` 这样的名字。一行塞两个量（"spread / sigma"）
/// 会把名字挤进 Description，读的人先得自己拆开才知道哪个数是哪个。
fn params_body(stats: &IqStats) -> String {
    // 先把 (名字, 解释, 值) 攒齐再装配：`ParamRow` 的名字是借用，边推边借会撞上可变借用
    let mut items: Vec<(usize, &'static str, String)> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut push = |name: String, description: &'static str, value: String| {
        labels.push(name);
        items.push((labels.len() - 1, description, value));
    };
    for (index, state) in stats.states().iter().enumerate() {
        push(
            format!("|{index}> center"),
            "mean of the single-shot IQ, a.u.",
            // 直接按复数写：中心本来就是个复数，拆成 (I, Q) 反而多一层翻译
            format!("{:.4}", state.center),
        );
        for slot in 0..LEVELS.len() {
            push(
                format!("|{index}> area {}%", (LEVELS[slot] * 100.0).round()),
                "area holding this share of the shots, densest-first, a.u.²",
                format!("{:.4e}", state.areas[slot]),
            );
        }
        for slot in 0..LEVELS.len() {
            push(
                format!("|{index}> r {}%", (LEVELS[slot] * 100.0).round()),
                "radius of a circle with the same area, a.u.",
                format!("{:.4}", state.radii[slot]),
            );
        }
    }
    for pair in stats.pairs() {
        let tag = format!("|{}>–|{}>", pair.state_p, pair.state_q);
        push(
            format!("{tag} separation"),
            "distance between the two centers, a.u.",
            format!("{:.5}", pair.separation),
        );
        push(
            format!("{tag} SNR"),
            "separation / √(σ_p∥² + σ_q∥²)",
            format!("{:.3}", pair.snr),
        );
        // 阈值对外就是 IQ 平面上的复数：判别边界与两中心连线的交点（与图上刻度文字同一套坐标）
        push(
            format!("{tag} threshold"),
            "cut point on the line joining the two centres",
            format!("{:.4}", pair.threshold),
        );
        push(
            format!("{tag} error rate"),
            "share of shots misclassified at that threshold",
            format!("{:.4}", pair.error_rate),
        );
        push(
            format!("{tag} AUC"),
            "Mann-Whitney statistic, P(second state > first state)",
            format!("{:.5}", pair.auc),
        );
    }
    let rows: Vec<ParamRow<'_>> = items
        .iter()
        .map(|(label, description, value)| ParamRow {
            name: &labels[*label],
            description,
            value: value.clone(),
            stderr: "—".to_string(),
        })
        .collect();
    params_table(&rows, None)
}
