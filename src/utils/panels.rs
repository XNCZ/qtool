//! 多面板报告的公共零件（feature = "plot"，crate 内部件）。
//!
//! s21 / qspec / rabi 的报告版式是同一张：网格切片、每格一个面板、右下角那块叠模型曲线与
//! 残差。各实验不同的只有"每个面板画什么"（各自的 `panel_traces`）、标题文字与参数表、
//! 以及**网格切法**（s21 / qspec 是 2×2，rabi 多一行整宽放频谱）——那些留在各自的
//! `*_plot` 里；本模块放的是逐字相同的那批：网格几何、数值助手、trace 零件、报告外壳。
//!
//! 网格切法是**数据**而不是分支：行列域列在 [`Grid`] 里，面板落在哪一格由 [`Cell`] 说明，
//! "跨列的那一行"只是跨度取 2，不是一类特例；面板集合也是各实验自己的事（每块只回答"我占
//! 哪一格"），本模块不认识任何一块具体面板。
//!
//! 版面由 [`layout`] 从"格位 + 标题 + 轴选项"生成：域、轴名、轴槽位、标题位置全由格位推出，
//! 各实验不必再手写 `x_axis2`/`y_axis3` 那一串。真正推不出来的只有 [`AxisOpts`] 里那几项。

use crate::superconductor::StateCenters;
use crate::utils::heatmap::{
    TITLE_COLOR, TITLE_SIZE, axis_style, escape_html, figure_font,
};
use lmfit::Complex64;
use plotly::Trace;
use plotly::common::{
    Anchor, AxisSide, ColorScale, ColorScaleElement, DashType, ErrorData, ErrorType, Font,
    HoverInfo, Line, Marker, MarkerSymbol, Mode, Position,
};
use plotly::layout::{Annotation, Axis, Layout, Legend};
use plotly::Scatter;

// =========================================================================
// 面板几何
// =========================================================================

/// 面板在网格里占的矩形，行自上而下、列自左而右，均 0 基。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) row_span: usize,
    pub(crate) col_span: usize,
}

impl Cell {
    /// 占一格的矩形。
    ///
    /// 形参:
    ///     row: 行（自上而下，0 基）
    ///     col: 列（自左而右，0 基）
    ///
    /// 返回值:
    ///     1×1 的矩形
    pub(crate) const fn at(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            row_span: 1,
            col_span: 1,
        }
    }

    /// 拉宽成横跨若干列 / 若干行（整宽那一行即 `col_span` 取满）。
    ///
    /// 形参:
    ///     row_span: 跨几行
    ///     col_span: 跨几列
    ///
    /// 返回值:
    ///     放大后的矩形
    pub(crate) const fn span(self, row_span: usize, col_span: usize) -> Self {
        Self {
            row_span,
            col_span,
            ..self
        }
    }
}

/// 报告网格：纸面按行、列切开，每个面板占其中一块矩形。
///
/// 切法是数据：2 列 × 2 行与 2 列 × 3 行只是行列数不同，没有"最后一行特殊"这类分支。
pub(crate) struct Grid {
    /// 各列的 x 域，自左而右
    pub(crate) xs: &'static [[f64; 2]],
    /// 各行的 y 域，自上而下
    pub(crate) ys: &'static [[f64; 2]],
    /// 设计高度（px）：与 [`CARD_WIDTH`] 一起定下主图块的高宽比
    pub(crate) height: f64,
}

/// 2×2 网格的行列域：左右列各 40%、上下排各 40% 上下，其余留给轴标题。
const X_LEFT: [f64; 2] = [0.0, 0.40];
const X_RIGHT: [f64; 2] = [0.62, 1.0];
const Y_TOP: [f64; 2] = [0.58, 1.0];
const Y_BOTTOM: [f64; 2] = [0.0, 0.40];

/// 2×3 网格的行域：三行平分原高度，行间距与 2×2 同量级。
const X_COL0: [f64; 2] = [0.0, 0.28];
const X_COL1: [f64; 2] = [0.36, 0.64];
const X_COL2: [f64; 2] = [0.72, 1.0];

const Y_SINGLE: [f64; 2] = [0.0, 1.0];

/// 2×3 版式的行间距（纸面分数）：与 2×2 版式渲染出来的是同一条白带——那版式的纸面
/// 矮（同宽下 562px），留白 0.18 × 562 ≈ 101px；本版式的纸面高（874px），故取
/// 101 / 874 ≈ 0.116。这段白要放下"上一行的 x 轴标题"与"下一行的面板标题"两段文字，
/// 与版式无关，所以按像素而不是按比例对齐。
const GAP_2X3: f64 = 0.116;

/// 2×3 版式的行高：三段平分"整幅减去两条行间距"。
const ROW_2X3: f64 = (1.0 - 2.0 * GAP_2X3) / 3.0;

const Y_ROW0: [f64; 2] = [1.0 - ROW_2X3, 1.0];
const Y_ROW1: [f64; 2] = [Y_ROW0[0] - GAP_2X3 - ROW_2X3, Y_ROW0[0] - GAP_2X3];
const Y_ROW2: [f64; 2] = [0.0, ROW_2X3];

/// 3 列 × 1 行的版式（bloch）：一排放三个方正的投影格，不需要第二行。
///
/// 单格宽度取 0.28（2 列版式的 0.40 收窄），设计高度相应压低，单格的宽高比与 2×2 版式大致持平。
pub(crate) const GRID_1X3: Grid = Grid {
    xs: &[X_COL0, X_COL1, X_COL2],
    ys: &[Y_SINGLE],
    height: 420.0,
};

/// 2 列 × 2 行的版式（s21 / qspec）。
pub(crate) const GRID_2X2: Grid = Grid {
    xs: &[X_LEFT, X_RIGHT],
    ys: &[Y_TOP, Y_BOTTOM],
    height: 760.0,
};

/// 2 列 × 3 行的版式（rabi）：第三行整宽，放频谱。
///
/// 多一行则每格变矮，设计高度相应抬高，单格的宽高比与 2×2 版式大致持平。
pub(crate) const GRID_2X3: Grid = Grid {
    xs: &[X_LEFT, X_RIGHT],
    ys: &[Y_ROW0, Y_ROW1, Y_ROW2],
    height: 1080.0,
};

impl Grid {
    /// 一块矩形在纸面上的坐标域：列取所在区间的并集、行取所在区间的并集。
    ///
    /// 形参:
    ///     row: 起始行（自上而下，0 基）
    ///     col: 起始列（自左而右，0 基）
    ///     row_span: 跨几行
    ///     col_span: 跨几列
    ///
    /// 返回值:
    ///     (x 域, y 域)，各为 `[下, 上]`；跨度皆为 1 时就是那一格本身
    pub(crate) fn domain(
        &self,
        row: usize,
        col: usize,
        row_span: usize,
        col_span: usize,
    ) -> ([f64; 2], [f64; 2]) {
        (
            [self.xs[col][0], self.xs[col + col_span - 1][1]],
            [self.ys[row + row_span - 1][0], self.ys[row][1]],
        )
    }

    /// 某个面板占的坐标域。
    ///
    /// 形参:
    ///     cell: 面板在其网格里的矩形
    ///
    /// 返回值:
    ///     (x 域, y 域)
    pub(crate) fn cell(&self, cell: Cell) -> ([f64; 2], [f64; 2]) {
        self.domain(cell.row, cell.col, cell.row_span, cell.col_span)
    }
}

/// 第 `index` 个轴的名字：plotly 从 `x`/`y` 起编，第二个才带数字。
///
/// 形参:
///     prefix: `'x'` 或 `'y'`
///     index: 轴序号（0 基，与面板的绘制次序一致）
///
/// 返回值:
///     plotly 的轴名，如 `x` / `x2` / `x7`
pub(crate) fn axis_name(prefix: char, index: usize) -> String {
    match index {
        0 => prefix.to_string(),
        n => format!("{prefix}{}", n + 1),
    }
}

/// 第 `index` 个面板的 (x, y) 轴引用，与 `Layout` 的 `x_axis*`/`y_axis*` 绑定。
///
/// 面板按绘制次序（各实验的 `ALL`）占轴槽位，第一块是 `x`/`y`——加一块面板不必再去数轴号。
///
/// 形参:
///     index: 面板序号（0 基）
///
/// 返回值:
///     (x 轴名, y 轴名)
pub(crate) fn axis_refs(index: usize) -> (String, String) {
    (axis_name('x', index), axis_name('y', index))
}

/// 一个面板的标题：x 取列中点、y 取排顶边（`yanchor=Bottom` ⇒ 字落在面板上方），
/// 字号/颜色对齐 plotly layout title 的默认外观（见 [`TITLE_SIZE`]）。
///
/// 面板的坐标域由它自己在 `grid` 里的矩形算出（与画 trace 时用的是同一个 [`Grid::cell`]），
/// 调用方只给图名——省掉一处"域传错了但看不出来"的机会。
///
/// 形参:
///     grid: 所在网格
///     cell: 哪个面板
///     name: 图名（写区域不写量，如 "Magnitude"）
///
/// 返回值:
///     paper 坐标上的注释对象
fn panel_title(grid: &Grid, cell: Cell, name: &str) -> Annotation {
    let (x_domain, y_domain) = grid.cell(cell);
    Annotation::new()
        .text(name)
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

// =========================================================================
// 版面
// =========================================================================

/// 一个面板的三条标题：x 轴与 y 轴写"量"，图名写"区域"。
pub(crate) struct Titles {
    pub(crate) x: &'static str,
    pub(crate) y: &'static str,
    pub(crate) name: &'static str,
}

/// 面板的残差右叠加轴：叠在宿主面板上、只画残差。
///
/// 槽位、`overlaying`/`anchor` 的目标都由"挂在第几块面板"推出来，这里只剩推不出的两项。
pub(crate) struct ResidualAxis {
    /// 轴标题（量纲不同则分开，如相位残差是 `Residual (rad)`）
    pub(crate) title: &'static str,
    /// 轴范围；None 交给 plotly 自适应
    pub(crate) range: Option<[f64; 2]>,
}

/// 一块面板的**轴选项**：只有格位推不出来的那几项。
///
/// 域、轴名、轴槽位、标题位置、图例与字体都在 [`layout`] 里按格位算，不是选项；这里留下的
/// 每一项都是"数据说了不算、得由拟合结果或物理含义定"的：固定值域、等比例锁定、残差轴。
pub(crate) struct AxisOpts {
    /// x 轴固定范围；None 交给 plotly 自适应（只有占位面板需要钉死）
    pub(crate) x_range: Option<[f64; 2]>,
    /// y 轴固定范围；None 交给 plotly 自适应
    pub(crate) y_range: Option<[f64; 2]>,
    /// y 轴按 x 轴等比例（`scaleanchor`）：IQ 平面这类"几何形状本身有意义"的面板才要，
    /// 否则投影轴会被拉成任意斜率
    pub(crate) equal_aspect: bool,
    /// 本格再挂一条残差右叠加轴
    pub(crate) residual: Option<ResidualAxis>,
}

impl AxisOpts {
    /// 全用默认（自适应值域、不等比例、无残差轴）。
    pub(crate) const PLAIN: AxisOpts = AxisOpts {
        x_range: None,
        y_range: None,
        equal_aspect: false,
        residual: None,
    };
}

/// 一块面板交给 [`layout`] 的全部信息：占哪一格、四条标题、轴选项。
pub(crate) struct PanelSpec {
    pub(crate) cell: Cell,
    pub(crate) titles: Titles,
    pub(crate) opts: AxisOpts,
}

/// 轴槽位分派表的上限：plotly.rs 把 `xaxis`…`xaxis8` 写成了独立字段，只能一张表对上号。
///
/// 这是本模块唯一的硬上限，**面板与残差叠加轴共用同一批槽位**（叠加轴排在面板之后），
/// 所以 8 个槽位大致等于"6 块面板 + 2 条残差轴"。再多就得先给 plotly.rs 的 `Layout` 补字段。
const MAX_AXES: usize = 8;

/// 槽位分派：第 n 对 (x, y) 轴写到 `Layout` 的哪两个字段上。
///
/// 表长即 [`MAX_AXES`]；越界索引是"面板排布超出 plotly.rs 的字段数"，属静态数据的编程错误。
const SET_PANEL: [fn(Layout, Axis, Axis) -> Layout; MAX_AXES] = [
    |layout, x, y| layout.x_axis(x).y_axis(y),
    |layout, x, y| layout.x_axis2(x).y_axis2(y),
    |layout, x, y| layout.x_axis3(x).y_axis3(y),
    |layout, x, y| layout.x_axis4(x).y_axis4(y),
    |layout, x, y| layout.x_axis5(x).y_axis5(y),
    |layout, x, y| layout.x_axis6(x).y_axis6(y),
    |layout, x, y| layout.x_axis7(x).y_axis7(y),
    |layout, x, y| layout.x_axis8(x).y_axis8(y),
];

/// 残差叠加轴只占一条 y 轴（`x` 仍归宿主面板），所以另给一张只写 y 的表。
const SET_OVERLAY: [fn(Layout, Axis) -> Layout; MAX_AXES] = [
    |layout, y| layout.y_axis(y),
    |layout, y| layout.y_axis2(y),
    |layout, y| layout.y_axis3(y),
    |layout, y| layout.y_axis4(y),
    |layout, y| layout.y_axis5(y),
    |layout, y| layout.y_axis6(y),
    |layout, y| layout.y_axis7(y),
    |layout, y| layout.y_axis8(y),
];

/// 按网格与每格的规格生成版面。
///
/// 域、轴名、轴槽位、面板标题的位置全由格位算出，各实验不必再手写一遍 `x_axis2`/`y_axis2`…
/// 残差叠加轴排在所有面板之后，槽位按挂载次序依次取——`overlaying` 与 `anchor` 都指向宿主
/// 面板的轴，标题与范围由 [`ResidualAxis`] 给。
///
/// 形参:
///     grid: 网格（定域与设计尺寸）
///     panels: 各面板的规格，**次序即绘制次序**（也是轴槽位的次序）
///
/// 返回值:
///     plotly 的 Layout（未设 annotation 之外的交互项；调用方自行 `set_layout`）
pub(crate) fn layout(grid: &Grid, panels: &[PanelSpec]) -> Layout {
    let mut result = Layout::new()
        .font(figure_font())
        .show_legend(true)
        // 面板标题在纸面坐标上：列中点 / 排顶边
        .annotations(
            panels
                .iter()
                .map(|spec| panel_title(grid, spec.cell, spec.titles.name))
                .collect(),
        )
        // 图例放图组右侧顶部、上边距里（yanchor=Bottom 让图例往上长，而不是默认的
        // 从 paper y=1 往下长——那样会压住相位面板右上角的刻度）
        .legend(
            Legend::new()
                .x(1.0)
                .x_anchor(Anchor::Left)
                .y(1.0)
                .y_anchor(Anchor::Bottom),
        );
    // 残差叠加轴紧跟在面板后面占槽位，故每挂一条就往后挪一位
    let mut overlays = panels.len();
    for (index, spec) in panels.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        let (x_domain, y_domain) = grid.cell(spec.cell);
        let x_axis = at_range(
            axis_style(
                Axis::new()
                    .domain(&x_domain)
                    .anchor(y_ref.as_str())
                    .title(spec.titles.x),
            ),
            spec.opts.x_range,
        );
        let mut y_axis = at_range(
            axis_style(
                Axis::new()
                    .domain(&y_domain)
                    .anchor(x_ref.as_str())
                    .title(spec.titles.y),
            ),
            spec.opts.y_range,
        );
        match spec.opts.equal_aspect {
            true => y_axis = y_axis.scale_anchor(x_ref.as_str()),
            false => {}
        }
        result = SET_PANEL[index](result, x_axis, y_axis);
        match &spec.opts.residual {
            Some(residual) => {
                let overlay = at_range(
                    axis_style(
                        Axis::new()
                            .overlaying(y_ref.as_str())
                            .side(AxisSide::Right)
                            .anchor(x_ref.as_str())
                            .title(residual.title)
                            // 叠加轴不画网格：否则与数据轴网格叠成双层虚线
                            .show_grid(false),
                    ),
                    residual.range,
                );
                result = SET_OVERLAY[overlays](result, overlay);
                overlays += 1;
            }
            None => {}
        }
    }
    result
}

/// 卡片设计宽度（px）：全仓唯一的那个宽度——卡片容器用它设上限，主图块的比例也按它算。
pub(crate) const CARD_WIDTH: f64 = 1000.0;

/// 一张图块：图 div（带自己的类名与内联比例）+ 该块的缩放注册。
///
/// **全仓的图 div 与缩放注册只出这一处**：报告只决定块放在版式的什么位置、配不配标题栏
/// （标题栏是版式，留在报告自己的 `body` 里）。
///
/// 形参:
///     name: 块名；图 div 的 id 是 `{name}-plot`（缩放注册按同一约定认图）
///     class: 图 div 的类名（`qtool-plot` / `qtool-2d-plot` / `qtool-line-plot` …）
///     ratio: 设计宽高比 (宽, 高)，写成内联 `aspect-ratio`
///     html: 图 div 的内容（plotly 的初始化片段）
///
/// 返回值:
///     自包含的图块片段
pub(crate) fn block(name: &str, class: &str, ratio: (f64, f64), html: &str) -> String {
    format!(
        "<div class=\"{class}\" id=\"{id}-plot\" style=\"width:100%;aspect-ratio:{w}/{h}\">{html}</div>{script}",
        id = escape_html(name),
        w = ratio.0,
        h = ratio.1,
        script = crate::utils::resize::register_script(name),
    )
}

/// 一份报告：一张卡片（结构见 [`card`]）。
pub(crate) struct Card<'a> {
    /// 根类名：报告自己的样式都挂在这个类下
    pub(crate) class: &'a str,
    /// 该报告自己的 `<style>`
    pub(crate) style: &'a str,
    /// 根容器的 id（一页多卡时由调用方保证唯一）
    pub(crate) div_id: &'a str,
    /// 图区到尾部之间的全部内容，按绘制顺序拼好（各块、脚本、参数表）
    pub(crate) body: String,
    /// 内蕴数据（见 [`crate::utils::data`]）：`Some` 时插入导出脚本与载荷 JSON，
    /// **排在 body 之前**——图那段 `Plotly.newPlot` 会直接取 `window.qtoolExport.buttons`
    /// 挂 modebar 按钮
    pub(crate) payload: Option<&'a crate::utils::data::Payload>,
    /// 卡片标题：`Some` 给描边卡片（标题骑在上边线上），`None` 给裸容器（只有宽度上限）
    pub(crate) frame: Option<&'a str>,
}

/// 页脚样式：页脚由 [`card`] 统一输出，规则自然也只此一份——各报告不再各写一遍
/// （写漏了页脚就退回浏览器默认的 16px 黑字，页脚与卡片差得越远越难发现）。
const FOOTER_STYLE: &str =
    "<style>.qtool-footer{margin-top:6px;text-align:right;font-size:11px;color:#9ca3af}</style>";

/// 报告卡片：容器 + 样式 + 内蕴数据 + 内容 + 页脚；外框（描边与标题）只是容器的样式。
///
/// 宽度上限挂在**容器**上而不是内容上——所以框必然贴着内容，不需要给框另设尺寸。
///
/// 形参:
///     spec: 见 [`Card`]
///
/// 返回值:
///     自包含的报告片段（已套外框）
pub(crate) fn card(spec: Card<'_>) -> String {
    let mut html = String::new();
    html.push_str(&format!(
        "<div class=\"{}\" id=\"{}\">",
        spec.class,
        escape_html(spec.div_id)
    ));
    html.push_str(spec.style);
    html.push_str(FOOTER_STYLE);
    match spec.payload {
        Some(data) => {
            html.push_str(crate::utils::data::EXPORT_SCRIPT);
            html.push_str(&data.script());
        }
        None => {}
    }
    html.push_str(&spec.body);
    // 页脚两行：生成时刻 + 版本。有载荷时用载荷里那一份（与下载数据里的是同一个值）
    html.push_str(&format!(
        "<div class=\"qtool-footer\">{}<br>Powered by qtool v{}</div>",
        crate::utils::data::footer_stamp(spec.payload),
        env!("CARGO_PKG_VERSION")
    ));
    html.push_str("</div>\n");
    crate::utils::bubble::card_box(&html, spec.frame, CARD_WIDTH)
}

// =========================================================================
// 数值助手
// =========================================================================

/// 一组序列的取值范围 [min, max]；全为空或无有限值时返回 None。
pub(crate) fn value_bounds(lists: &[&[f64]]) -> Option<[f64; 2]> {
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
pub(crate) const AXIS_PAD: f64 = 0.05;

/// 数据轴范围：值域 ±[`AXIS_PAD`]；退化（span ≤ 0）时交回 plotly 自适应。
pub(crate) fn pad(bounds: Option<[f64; 2]>) -> Option<[f64; 2]> {
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
pub(crate) const RESIDUAL_ZERO_AT: f64 = 0.12;

/// 残差右轴范围：与数据轴同 span（同一 scale），整体下移使残差 0 落在面板
/// [`RESIDUAL_ZERO_AT`] 高度处。
pub(crate) fn residual_axis_range(data_range: Option<[f64; 2]>) -> Option<[f64; 2]> {
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
pub(crate) fn at_range(axis: Axis, range: Option<[f64; 2]>) -> Axis {
    match range {
        Some([low, high]) => axis.range(vec![low, high]),
        None => axis,
    }
}

/// 密集拟合曲线的采样倍率：点数 = 数据点数 × 该值。
pub(crate) const DENSE_FACTOR: usize = 10;

/// 密集网格（[`DENSE_FACTOR`]× 数据点数，闭区间）；点数不足以插值时返回 None。
pub(crate) fn dense_grid(freqs_hz: &[f64]) -> Option<Vec<f64>> {
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

// =========================================================================
// 配色
// =========================================================================

/// 数据点与拟合曲线的颜色：紫=数据、红=模型、金=残差。
pub(crate) const DATA_COLOR: &str = "#663399";
pub(crate) const FIT_COLOR: &str = "#d62728";
pub(crate) const RESIDUAL_COLOR: &str = "#eab308";

/// 两态着色（沿用 baseline `iq_norm.py` 里的那对既有配色）：P1 = 0 端蓝、1 端红。
pub(crate) const ZERO_COLOR: &str = "royalblue";
pub(crate) const ONE_COLOR: &str = "crimson";

/// 投影轴的颜色：灰色——红色在 IQ 面板里已经专指 |1> 端，再用会把两套语义混在一起。
pub(crate) const AXIS_COLOR: &str = "gray";

// =========================================================================
// trace 零件
// =========================================================================

/// 带虚线的数据点：折线看趋势，点看采样位置（与 baseline 的 `lines+markers` 一致）。
pub(crate) fn series(
    x: Vec<f64>,
    y: Vec<f64>,
    name: &str,
    color: impl Into<String>,
    show_legend: bool,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    // 收成 String 再分给 line 与 marker：plotly 的 `Color` 要求 `'static`，借用形态过不去
    let color = color.into();
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        // 同名 trace 归入同一 legendgroup：点一次图例即可开关多个面板里的同类曲线
        .legend_group(name)
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(color.clone()).dash(DashType::Dash))
        .marker(Marker::new().color(color).size(7))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 纯数据点，可带逐点误差棒。
pub(crate) fn markers(
    x: Vec<f64>,
    y: Vec<f64>,
    error_y: Option<Vec<f64>>,
    name: &str,
    color: impl Into<String>,
    size: usize,
    show_legend: bool,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    // 收成 String 再分给 marker 与误差棒：plotly 的 `Color` 要求 `'static`，借用形态过不去
    let color = color.into();
    let mut trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        // 同名 trace 归入同一 legendgroup：点一次图例即可同时开关四个面板里的同类曲线
        .legend_group(name)
        .mode(Mode::Markers)
        .marker(Marker::new().color(color.clone()).size(size))
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

/// 曲线（拟合模型）。
pub(crate) fn curve(
    x: Vec<f64>,
    y: Vec<f64>,
    name: &str,
    color: impl Into<String>,
    width: f64,
    show_legend: bool,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        .legend_group(name)
        .mode(Mode::Lines)
        .line(Line::new().color(color.into()).width(width))
        .show_legend(show_legend)
        .hover_info(HoverInfo::Skip)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 残差标记：金色小点、画在右轴（`y_ref`）上。
pub(crate) fn residual_markers(
    x: Vec<f64>,
    y: Vec<f64>,
    show_legend: bool,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name("Residual")
        .legend_group("Residual")
        .mode(Mode::Markers)
        .marker(Marker::new().color(RESIDUAL_COLOR).size(5))
        .show_legend(show_legend)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 按 P1 着色的样本点：0 端取 `zero`、1 端取 `one`（单条扫描给默认的那对蓝红），状态沿投影轴
/// 的过渡一眼可读。
///
/// 端点由调用方给，是为了让"多阶同图"那类图能把阶数压进**深浅**里：同一个色相、不同明度，
/// 色相仍然只读 P1。
///
/// 形参:
///     x: I 分量 (n,)
///     y: Q 分量 (n,)
///     values: 逐点的 P1 (n,)，取值定在 [0, 1] 上（`cmin`/`cmax`）
///     name: trace 名，同时是 legendgroup——同名的点一次图例一起开关（本 trace 自己不占图例）
///     zero: P1 = 0 端的颜色
///     one: P1 = 1 端的颜色
///     x_ref: 本面板的 x 轴名
///     y_ref: 本面板的 y 轴名
///
/// 返回值:
///     按 P1 着色的样本点 trace（不显示色标、不占图例，但归入 `name` 那个 legendgroup）
pub(crate) fn color_samples(
    x: Vec<f64>,
    y: Vec<f64>,
    values: &[f64],
    name: &str,
    zero: impl Into<String>,
    one: impl Into<String>,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    let scale = vec![
        ColorScaleElement(0.0, zero.into()),
        ColorScaleElement(1.0, one.into()),
    ];
    let trace: Box<Scatter<f64, f64>> = Scatter::new(x, y)
        .name(name)
        // 同名 trace 归入同一 legendgroup：点一次图例即可同时开关四个面板里的同类曲线
        .legend_group(name)
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

/// 投影轴：跨两个参考点的灰色虚线。
pub(crate) fn projection_axis(
    ref_zero: Complex64,
    ref_one: Complex64,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(
        vec![ref_zero.re, ref_one.re],
        vec![ref_zero.im, ref_one.im],
    )
    .mode(Mode::Lines)
    .line(Line::new().color(AXIS_COLOR).width(2.0).dash(DashType::Dash))
    .show_legend(false)
    .hover_info(HoverInfo::Skip)
    .x_axis(x_ref)
    .y_axis(y_ref);
    trace
}

/// 一个态参考点：`×` 号加标签。
pub(crate) fn ref_point(
    point: Complex64,
    label: &str,
    color: &'static str,
    position: Position,
    x_ref: &str,
    y_ref: &str,
) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(vec![point.re], vec![point.im])
        .mode(Mode::MarkersText)
        .text_array(vec![label.to_string()])
        .text_position(position)
        .marker(Marker::new().color(color).size(12).symbol(MarkerSymbol::X))
        .show_legend(false)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// [`crate::superconductor::p1`] 在该数据上实际使用的两个参考点（baseline `iq_norm.projection_refs`）。
///
/// 给定各态标定中心时，参考点即前两态的云中心；未标定时取 P1 最小与最大的那两个样本——
/// 归一化正是把 0 与 1 钉在它们上。两点连成的线段即投影轴（IQ 面板画的就是这条线）。
/// 曲线已定向，所以谁最小谁就是 |0> 端，不必再去问投影轴的朝向。
///
/// 形参:
///     iq: 复数 IQ 数组 (n,)
///     states: 各态标定中心；None 表示未标定
///     prob: 已定向的 P1 曲线 (n,)，与 `iq` 等长；未标定时用它找两个端点
///
/// 返回值:
///     (|0> 端参考点, |1> 端参考点)；`iq` 为空时两点皆为 NaN
pub(crate) fn projection_refs(
    iq: &[Complex64],
    states: Option<&StateCenters>,
    prob: &[f64],
) -> (Complex64, Complex64) {
    match states {
        Some(centers) => match centers.as_slice() {
            [zero, one, ..] => (*zero, *one),
            // 标定不足两态：与 [`crate::superconductor::p1`] 一致，参考点无定义
            [] | [_] => (
                Complex64::new(f64::NAN, f64::NAN),
                Complex64::new(f64::NAN, f64::NAN),
            ),
        },
        None => {
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
