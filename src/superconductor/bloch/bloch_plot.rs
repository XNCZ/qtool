//! 布洛赫球轨迹的自包含 HTML div 渲染（feature = "plot"）。
//!
//! 一次调用产出上下两块：
//!
//! - **上排**（`GRID_1X3`）：YZ / XZ / XY 三个平面投影，各一格；每格一条轨迹，衬一个单位圆做
//!   参照（纯态的边界），等比例锁定。含 Z 的两格标出 `|0>` 在顶、`|1>` 在底；XY 格里两态都落到
//!   原点，没有可标的位置，不标。
//! - **下排**：可交互的三维球——拖拽旋转（plotly 的 orbit 模式，手感同 CAD / 谷歌地球）、滚轮
//!   缩放、工具栏里可复位视角。赤道 + 四条经线 + 三根坐标轴，**不画球面**（理由见
//!   [`frame_traces`]：实体面会把球内的点整个挡在拾取之外）。
//!
//! 两块是**两张独立的 plotly 图**，不是一张图的四个格：plotly 的 3D `scene` 有独立于 2D 子图的
//! 纸面域，而本 crate 用的 plotly 0.14.1 里 `LayoutScene` 的 `domain` 字段是**注释掉的**
//! （`layout/scene.rs:482`），没法把球框进某个格子——不设 domain 时 scene 铺满整张纸，会盖住
//! 三个面板。球自成一图时，满纸正是它该占的那块矩形，那个字段根本不需要。
//!
//! 实验侧的东西——相位解缠、差分剥相、随扫描量走的拟合——都不在这里（见 [`super::bloch`]）。

use crate::superconductor::bloch::BlochVector;
use crate::utils::data::{Column, Datum, Payload, Table};
use crate::utils::heatmap::{axis_style, figure_font, interactive_config};
use crate::utils::panels::{
    CARD_WIDTH, Card, block, card,
    AxisOpts, Cell, GRID_1X3, ONE_COLOR, PanelSpec, Titles, ZERO_COLOR, axis_refs, color_samples,
    curve, layout, };
use plotly::common::{Anchor, ColorScale, ColorScaleElement, Font, HoverInfo, Line, Marker, Mode};
use plotly::layout::{
    Annotation, AspectMode, Axis, Camera, DragMode3D, Layout, LayoutScene, Margin, ProjectionType,
};
use plotly::{Plot, Scatter, Scatter3D, Trace};

/// plotly.js 的 CDN 引入标签；与 s21 / qspec / rabi / ramsey 报告用的是同一个版本，这里转出
/// 以便本报告自成一体（宿主页面只需要放一次）。
pub use crate::superconductor::qspec::qspec_plot::PLOTLY_JS_CDN;

/// 单位圆参照线的颜色：灰细线，压在图底，不抢轨迹。
const GUIDE_COLOR: &str = "#c7c7c7";

/// 经纬线、坐标轴与标注的颜色（与其它面板的参考线同一族的灰）。
const FRAME_COLOR: &str = "#9ca3af";

/// 球上的经线（方位角，度）：`φ` 与 `φ+180°` 是同一个大圆，所以四根就铺满一圈。
const MERIDIANS_DEG: [f64; 4] = [0.0, 45.0, 90.0, 135.0];

/// 轨迹连线的颜色：中性灰。点按 P1 上红下蓝（[`color_samples`]），线只负责把点串起来；比球上
/// 的经纬线深一档，免得跟参照线混在一起。
const LINE_COLOR: &str = "#6b7280";

/// 悬停高亮的颜色与大小：空心圈，衬在轨迹点外面，与红蓝都拉开。
const HIGHLIGHT_COLOR: &str = "#d97706";
const HIGHLIGHT_SIZE: usize = 14;

/// 高亮 trace 的名字：页面里的脚本按这个名字认领它，不写死下标。
const HIGHLIGHT_NAME: &str = "Highlight";

/// 球上轨迹 trace 的名字（脚本要从它身上取点数与坐标）。
const TRAJECTORY_NAME: &str = "Trajectory";

/// 上排那张图的底边距：plotly 默认的 80px 里有一大半是空的，压小后仍由各轴的 `auto_margin`
/// 兜住轴标题。
const PANELS_BOTTOM_MARGIN: usize = 20;

/// 球那块的边距（px）：3D 场景把数据立方体按包围盒塞进画布，**轴的刻度与标题画在立方体之外**，
/// 四边不给就会被画布边缘裁掉——底部最凶（前下方的顶点，X 与 Y 两根轴线在那儿相交）。
const SPHERE_MARGIN: (usize, usize, usize, usize) = (20, 20, 10, 20);

/// 单位圆与经纬线的采样点数（闭合回路，首尾重合）。
const CIRCLE_POINTS: usize = 97;

/// 投影块的三格：每格画哪两个分量。
#[derive(Clone, Copy)]
enum Plane {
    Yz,
    Xz,
    Xy,
}

impl Plane {
    const ALL: [Plane; 3] = [Plane::Yz, Plane::Xz, Plane::Xy];

    /// 本格在网格里的格位（轴名与域都由它推出来，不在这里写）。
    fn cell(self) -> Cell {
        match self {
            Self::Yz => Cell::at(0, 0),
            Self::Xz => Cell::at(0, 1),
            Self::Xy => Cell::at(0, 2),
        }
    }

    /// 本格画的 `(横轴分量, 纵轴分量)`，分量按 X/Y/Z 编号（0/1/2），即 [`components`] 的下标。
    fn axes(self) -> (usize, usize) {
        match self {
            Self::Yz => (1, 2),
            Self::Xz => (0, 2),
            Self::Xy => (0, 1),
        }
    }

    /// 本格的两轴标题与图名。
    fn titles(self) -> Titles {
        match self {
            Self::Yz => Titles {
                x: "⟨Y⟩",
                y: "⟨Z⟩",
                name: "YZ",
            },
            Self::Xz => Titles {
                x: "⟨X⟩",
                y: "⟨Z⟩",
                name: "XZ",
            },
            Self::Xy => Titles {
                x: "⟨X⟩",
                y: "⟨Y⟩",
                name: "XY",
            },
        }
    }
}

/// 每个 div 自带的内联样式（类名统一 `qtool-` 前缀，避免污染宿主页面）；上排的尺寸那条
/// 比例由各自的图块（[`block`]）给：上排是设计网格的比例，球是 5:4。
const STYLE: &str = r#"<style>
.qtool-bloch{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
/* 高度跟着宽度走（比例即网格的设计宽高） */
.qtool-bloch .qtool-plot{width:100%;max-height:85vh}
/* 球那块：与上排之间只留一道缝；比例取到接近正方——3D 场景按画布的高与宽里较小的一边放大，
   块越高球越大（高度另有 78vh 封顶，免得小窗口里把整页撑开） */
.qtool-bloch .qtool-sphere{margin-top:6px}
.qtool-bloch .qtool-sphere-plot{width:100%;max-height:78vh}
</style>"#;

/// 把一条布洛赫轨迹渲染成自包含的 HTML div。
///
/// 形参:
///     vector: 布洛赫向量（三基分量 + 扫描轴），见 [`crate::superconductor::bloch`]
///     x_title: 扫描量的名字，只用在球上轨迹点的悬停标签里（如 `"π amplitude"`）
///     div_id: 外层 div 的 HTML id（一页多图时由调用方保证唯一）
///     frame: 可选外框：`Some(title)` 套上卡片框（`title` 非空时骑在上边线上），`None` 裸图
///
/// 返回值:
///     自包含的 `<div class="qtool-bloch">` 片段（上排三格 + 三维球 + 页脚），并内蕴一份原始
///     数据供下载（见 [`crate::utils::data`]）；宿主页面需自行加载 plotly.js（见 [`PLOTLY_JS_CDN`]）
pub fn bloch_plot_div(
    vector: &BlochVector,
    x_title: &str,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    // 点的颜色由激发概率定：|0> 端蓝、|1> 端红（三个投影格与球共用同一份）
    let prob = state_probability(vector);
    let planes = plane_plot(vector, &prob);
    let planes_html = crate::utils::data::plot_script(&planes, div_id);
    // 球自成一图：块名取 `{div_id}-sphere` ⇒ 图 div 是 `{div_id}-sphere-plot`，缩放按同一约定注册
    let sphere = sphere_plot(vector, &prob, x_title);
    let sphere_name = format!("{div_id}-sphere");
    let sphere_html = crate::utils::data::plot_script(&sphere, sphere_name.as_str());
    let body = format!(
        "{}<div class=\"qtool-sphere\">{}</div>{}",
        block(div_id, "qtool-plot", (CARD_WIDTH, GRID_1X3.height), &planes_html),
        block(sphere_name.as_str(), "qtool-sphere-plot", (5.0, 4.0), &sphere_html),
        highlight_script(div_id),
    );

    let report_data = payload(vector, div_id);
    card(Card {
        class: "qtool-bloch",
        style: STYLE,
        div_id,
        body,
        payload: Some(&report_data),
        frame,
    })
}

/// 报告载荷：轨迹本身——扫描轴与三个基上的泡利期望，逐点一行。
///
/// 这张报告没有拟合（相位解缠、剥相都在 [`crate::superconductor::bloch`] 里做完），所以只有这一张表；
/// 点的着色标量 P1 是 `(1 − z)/2`，从列里一行就能推出来，不另存。
///
/// 形参:
///     vector: 布洛赫向量（三基分量与扫描轴等长）
///     div_id: 报告名（内蕴数据的 `name`，也是下载文件基名）
///
/// 返回值:
///     载荷（一张 `data` 表）
fn payload(vector: &BlochVector, div_id: &str) -> Payload {
    let mut data = Table::new(
        "data",
        vec![
            Column::real("axis", "a.u."),
            Column::real("x", "1"),
            Column::real("y", "1"),
            Column::real("z", "1"),
        ],
    );
    for point in 0..vector.axis.len() {
        data.push(&[
            Datum::Real(vector.axis[point]),
            Datum::Real(vector.x[point]),
            Datum::Real(vector.y[point]),
            Datum::Real(vector.z[point]),
        ]);
    }
    let mut report = Payload::new(div_id);
    report.table(data);
    report
}

/// 三个分量按 X/Y/Z 排好，供面板按下标取用。
///
/// 形参:
///     vector: 布洛赫向量
///
/// 返回值:
///     `[⟨X⟩, ⟨Y⟩, ⟨Z⟩]`
fn components(vector: &BlochVector) -> [&[f64]; 3] {
    [&vector.x, &vector.y, &vector.z]
}

/// 各点的激发概率 `P1 = (1 − ⟨Z⟩)/2`：着色的标量——0 是 `|0>`、1 是 `|1>`，所以点越红离 `|1>`
/// 越近、越蓝离 `|0>` 越近（[`color_samples`] 的两端色）。
///
/// 形参:
///     vector: 布洛赫向量
///
/// 返回值:
///     与扫描轴等长的 P1；投影退化时是 NaN，plotly 对 NaN 的点不填色
fn state_probability(vector: &BlochVector) -> Vec<f64> {
    vector.z.iter().map(|z| (1.0 - z) / 2.0).collect()
}

/// 上排那三个投影格：逐格画参照圆与轨迹。
///
/// 值域交给 plotly 自适应（`equal_aspect` 会连带把两根轴锁成同一个单位长度，圆因此不会被拉成
/// 椭圆）：三个分量可能越出 ±1（标定路径的线性投影不裁剪），钉死值域就得反过来裁数据。
///
/// 形参:
///     vector: 布洛赫向量
///     prob: 各点的激发概率，决定点的颜色
///
/// 返回值:
///     plotly 图对象（layout 与 config 已设好）
fn plane_plot(vector: &BlochVector, prob: &[f64]) -> Plot {
    let mut plot = Plot::new();
    for (index, plane) in Plane::ALL.iter().enumerate() {
        let (x_ref, y_ref) = axis_refs(index);
        for trace in plane_traces(*plane, vector, prob, index == 0, &x_ref, &y_ref) {
            plot.add_trace(trace);
        }
    }
    // 面板标题（图名）由 `layout` 生成，两个极点标注另外追加
    let mut settings = layout(&GRID_1X3, &panel_specs())
        .margin(Margin::new().bottom(PANELS_BOTTOM_MARGIN));
    for (index, plane) in Plane::ALL.iter().enumerate() {
        for label in pole_labels(*plane, index) {
            settings.add_annotation(label);
        }
    }
    plot.set_layout(settings);
    plot.set_configuration(interactive_config());
    plot
}

/// 某一格的全部 trace：参照圆在下，轨迹的连线与着色的点在上，高亮圈最后。
///
/// 形参:
///     plane: 画哪一格
///     vector: 布洛赫向量
///     prob: 各点的激发概率，决定点的颜色
///     show_legend: 是否给轨迹挂图例（同名 trace 归一个 legendgroup，挂一次即可全图开关）
///     x_ref: 本格的 x 轴名（由 `layout` 按格位给出，见 `axis_refs`）
///     y_ref: 本格的 y 轴名
///
/// 返回值:
///     该格的 trace 列表，按绘制顺序
fn plane_traces(
    plane: Plane,
    vector: &BlochVector,
    prob: &[f64],
    show_legend: bool,
    x_ref: &str,
    y_ref: &str,
) -> Vec<Box<dyn Trace>> {
    let (ix, iy) = plane.axes();
    let parts = components(vector);
    let (circle_x, circle_y) = unit_circle();
    vec![
        curve(
            circle_x,
            circle_y,
            "Unit circle",
            GUIDE_COLOR,
            1.0,
            false,
            x_ref,
            y_ref,
        ),
        // 连线与点拆成两条：`color_samples` 只画点，图例挂在线上（同名同组，点一次两条一起开关）
        curve(
            parts[ix].to_vec(),
            parts[iy].to_vec(),
            TRAJECTORY_NAME,
            LINE_COLOR,
            1.5,
            show_legend,
            x_ref,
            y_ref,
        ),
        color_samples(
            parts[ix].to_vec(),
            parts[iy].to_vec(),
            prob,
            TRAJECTORY_NAME,
            ZERO_COLOR,
            ONE_COLOR,
            x_ref,
            y_ref,
        ),
        highlight(x_ref, y_ref),
    ]
}

/// 本格的高亮 trace：一个空心圈，初始是空的——球上划过某一点时由页面里的脚本把它挪过去
/// （见 [`highlight_script`]）。
///
/// 形参:
///     x_ref: 本格的 x 轴名
///     y_ref: 本格的 y 轴名
///
/// 返回值:
///     高亮 trace（不进图例、不响应悬停，免得跟轨迹的点抢事件）
fn highlight(x_ref: &str, y_ref: &str) -> Box<dyn Trace> {
    let trace: Box<Scatter<f64, f64>> = Scatter::new(Vec::<f64>::new(), Vec::<f64>::new())
        .name(HIGHLIGHT_NAME)
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .size(HIGHLIGHT_SIZE)
                .color("rgba(0,0,0,0)")
                .line(Line::new().color(HIGHLIGHT_COLOR).width(2.0)),
        )
        .hover_info(HoverInfo::Skip)
        .show_legend(false)
        .x_axis(x_ref)
        .y_axis(y_ref);
    trace
}

/// 单位圆：半径 1 的等值线，即纯态的边界（两个投影格与球上的赤道、子午线都用它）。
///
/// 形参: 无
///
/// 返回值:
///     (x, y)，各 [`CIRCLE_POINTS`] 点，首尾重合
fn unit_circle() -> (Vec<f64>, Vec<f64>) {
    (0..CIRCLE_POINTS)
        .map(|step| {
            let angle = 2.0 * std::f64::consts::PI * step as f64 / (CIRCLE_POINTS - 1) as f64;
            (angle.cos(), angle.sin())
        })
        .unzip()
}

/// 某个投影格的 `|0>` / `|1>` 标注：Z 轴顶端是 `|0>`、底端是 `|1>`（横轴都在 0 处）。
///
/// 形参:
///     plane: 哪一格
///     index: 该格的绘制序号（定轴名）
///
/// 返回值:
///     标注列表；不含 Z 的那一格（XY）返回空——两态在那里都落到原点，没有可标的位置
fn pole_labels(plane: Plane, index: usize) -> Vec<Annotation> {
    match plane {
        Plane::Yz | Plane::Xz => {
            let (x_ref, y_ref) = axis_refs(index);
            [(1.0, "|0>", Anchor::Bottom), (-1.0, "|1>", Anchor::Top)]
                .into_iter()
                .map(|(y, text, anchor)| {
                    Annotation::new()
                        .text(text)
                        .x(0.0)
                        .y(y)
                        .x_ref(x_ref.as_str())
                        .y_ref(y_ref.as_str())
                        .x_anchor(Anchor::Center)
                        .y_anchor(anchor)
                        .show_arrow(false)
                        .font(Font::new().size(11).color(FRAME_COLOR))
                })
                .collect()
        }
        Plane::Xy => Vec::new(),
    }
}

/// 悬停联动的脚本：球上划过某一点时，把上排三个投影格里的对应点一起点亮。
///
/// 两块是**两张独立的 plotly 图**（理由见模块文档），plotly 自己没有跨图联动的机制，这里用页面
/// 里的一小段 JS 接上：球上 `plotly_hover` 给出点序号，按它 `Plotly.restyle` 三个格里的高亮
/// trace；移开再清空。三格画的是哪两个分量写在 `PLANES` 里（顺序与 [`Plane::ALL`] 一致），
/// 高亮 trace 则以 [`HIGHLIGHT_NAME`] 认领——两边都不写死下标。
///
/// 形参:
///     div_id: 外层 div 的 id（两块图的 id 由它派生）
///
/// 返回值:
///     一段 `<script>`；宿主页面需已加载 plotly.js
fn highlight_script(div_id: &str) -> String {
    format!(
        r#"<script>
(function () {{
  var sphere = document.getElementById("{div_id}-sphere-plot");
  var panels = document.getElementById("{div_id}-plot");
  if (!sphere || !panels || !window.Plotly) {{ return; }}
  // 每格画的是哪两个分量（0/1/2 即 X/Y/Z）：YZ、XZ、XY
  var PLANES = [[1, 2], [0, 2], [0, 1]];
  var CLEAR = {{ "x": [[], [], []], "y": [[], [], []] }};
  var picked = null;
  function targets() {{
    if (!picked) {{
      picked = panels.data
        .map(function (trace, index) {{ return trace.name === "{HIGHLIGHT_NAME}" ? index : -1; }})
        .filter(function (index) {{ return index >= 0; }});
    }}
    return picked.length === PLANES.length ? picked : null;
  }}
  function highlight(number) {{
    var indices = targets();
    var source = sphere.data.filter(function (item) {{ return item.name === "{TRAJECTORY_NAME}"; }})[0];
    if (!indices || !source) {{ return; }}
    var parts = [source.x[number], source.y[number], source.z[number]];
    Plotly.restyle(panels, {{
      "x": PLANES.map(function (pair) {{ return [parts[pair[0]]]; }}),
      "y": PLANES.map(function (pair) {{ return [parts[pair[1]]]; }})
    }}, indices);
  }}
  sphere.on("plotly_hover", function (event) {{
    var point = event.points && event.points[0];
    if (point) {{ highlight(point.pointNumber); }}
  }});
  sphere.on("plotly_unhover", function () {{
    var indices = targets();
    if (indices) {{ Plotly.restyle(panels, CLEAR, indices); }}
  }});
}})();
</script>"#
    )
}

/// 三个投影格的规格：等比例锁定（否则圆会被拉成任意椭圆），值域自适应。
///
/// 形参: 无
///
/// 返回值:
///     各格规格，次序即绘制次序
fn panel_specs() -> Vec<PanelSpec> {
    Plane::ALL
        .iter()
        .map(|plane| PanelSpec {
            cell: plane.cell(),
            titles: plane.titles(),
            opts: AxisOpts {
                equal_aspect: true,
                ..AxisOpts::PLAIN
            },
        })
        .collect()
}

/// 下排的三维球：半透明球面 + 经纬与坐标轴 + 轨迹。
///
/// 形参:
///     vector: 布洛赫向量
///     prob: 各点的激发概率，决定点的颜色
///     x_title: 扫描量的名字，用在轨迹点的悬停标签里
///
/// 返回值:
///     plotly 图对象（scene、layout 与 config 已设好）
fn sphere_plot(vector: &BlochVector, prob: &[f64], x_title: &str) -> Plot {
    let mut plot = Plot::new();
    for trace in frame_traces() {
        plot.add_trace(trace);
    }
    plot.add_trace(trajectory(vector, prob, x_title));
    // aspect_mode=Data 让三根轴同比例，球才是球；drag_mode=Orbit 是自由旋转（CAD 手感），
    // 也是 plotly 的默认值，这里写明意图：滚轮缩放、拖拽旋转、工具栏可复位视角
    //
    // 相机取**正交投影**：gl-plot3d 是按包围盒把立方体塞进画布的，透视投影下离相机最近的那个
    // 底角会被放大、探出画布底边——X 与 Y 两根轴线在那儿相交，于是那个交点整个看不见
    // （断在画布边缘，两根线根本没接到一起）。正交投影下近远同尺寸，塞进去就是塞进去。
    let scene = LayoutScene::new()
        .x_axis(axis_style(Axis::new().title("⟨X⟩")))
        .y_axis(axis_style(Axis::new().title("⟨Y⟩")))
        .z_axis(axis_style(Axis::new().title("⟨Z⟩")))
        .aspect_mode(AspectMode::Data)
        .drag_mode(DragMode3D::Orbit)
        .camera(Camera::new().projection(ProjectionType::Orthographic.into()));
    // 上排的图例已经把轨迹标过了，这块不再重复挂一个
    let (left, right, top, bottom) = SPHERE_MARGIN;
    plot.set_layout(
        Layout::new()
            .font(figure_font())
            .scene(scene)
            .show_legend(false)
            .margin(
                Margin::new()
                    .left(left)
                    .right(right)
                    .top(top)
                    .bottom(bottom),
            ),
    );
    plot.set_configuration(interactive_config());
    plot
}

/// 球上的参照：赤道 + 四条经线 + 三根过原点的坐标轴。
///
/// **不画球面**：3D 的悬停按深度拾取，任何实体面（哪怕半透明）都会把球内的轨迹点整个挡在
/// 拾取之外——轨迹本来就整个在球里，那样等于全都 hover 不到。线只在自己经过的那几像素上遮，
/// 球内的点照常可悬停。线本身不进图例、不响应悬停。
///
/// 三根轴各自从 −1 走到 +1，两端就是 `|0>` / `|1>` 两极，不另画极点标记。
///
/// 形参: 无
///
/// 返回值:
///     参照线 trace 列表
fn frame_traces() -> Vec<Box<dyn Trace>> {
    let mut traces = vec![equator()];
    traces.extend(
        MERIDIANS_DEG
            .iter()
            .map(|degrees| meridian(degrees.to_radians())),
    );
    for index in 0..3 {
        let mut ends = [[0.0; 2]; 3];
        ends[index] = [-1.0, 1.0];
        traces.push(line3d(
            ends[0].to_vec(),
            ends[1].to_vec(),
            ends[2].to_vec(),
        ));
    }
    traces
}

/// 赤道：`x–y` 平面上的单位圆。
///
/// 形参: 无
///
/// 返回值:
///     赤道 trace
fn equator() -> Box<dyn Trace> {
    let (cos, sin) = unit_circle();
    line3d(cos, sin, vec![0.0; CIRCLE_POINTS])
}

/// 一条经线：过两极的大圆，方位角 `φ` 与 `φ+π` 是同一条。
///
/// 形参:
///     azimuth: 方位角 (rad)，0 在 `x–z` 平面
///
/// 返回值:
///     经线 trace
fn meridian(azimuth: f64) -> Box<dyn Trace> {
    let (cos, sin) = unit_circle();
    let (sin_azimuth, cos_azimuth) = azimuth.sin_cos();
    line3d(
        cos.iter().map(|value| value * cos_azimuth).collect(),
        cos.iter().map(|value| value * sin_azimuth).collect(),
        sin,
    )
}

/// 一条 3D 参照折线：细灰实线、不进图例、不响应悬停。
///
/// 形参:
///     x: 顶点 x
///     y: 顶点 y
///     z: 顶点 z
///
/// 返回值:
///     折线 trace
fn line3d(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Box<dyn Trace> {
    Scatter3D::new(x, y, z)
        .mode(Mode::Lines)
        .line(Line::new().color(FRAME_COLOR).width(1.0))
        .hover_info(HoverInfo::Skip)
        .show_legend(false)
}

/// 轨迹：一条 3D 折线，点的颜色按 P1 走（`|0>` 端蓝、`|1>` 端红），悬停报出该点的三个分量与
/// 扫描值。
///
/// 三分量由 `hovertemplate` 里的 `%{x}`/`%{y}`/`%{z}` 现取（这条 trace 的三个坐标轴就是
/// `⟨X⟩`/`⟨Y⟩`/`⟨Z⟩`），扫描值走 `text`——它是唯一没法从坐标里读出来的量。
///
/// 形参:
///     vector: 布洛赫向量
///     prob: 各点的激发概率，决定点的颜色
///     x_title: 扫描量的名字，用在悬停模板里
///
/// 返回值:
///     轨迹 trace
fn trajectory(vector: &BlochVector, prob: &[f64], x_title: &str) -> Box<dyn Trace> {
    let sweep: Vec<String> = vector
        .axis
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect();
    // 与 `color_samples` 同一套映射：0 端蓝、1 端红，两端固定，噪声越界的点被夹在端点上
    let scale = vec![
        ColorScaleElement(0.0, ZERO_COLOR.to_string()),
        ColorScaleElement(1.0, ONE_COLOR.to_string()),
    ];
    Scatter3D::new(vector.x.clone(), vector.y.clone(), vector.z.clone())
        .name(TRAJECTORY_NAME)
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(LINE_COLOR).width(2.0))
        .marker(
            Marker::new()
                .color_array(prob.to_vec())
                .color_scale(ColorScale::Vector(scale))
                .cmin(0.0)
                .cmax(1.0)
                .show_scale(false)
                .size(5),
        )
        .text_array(sweep)
        .hover_template(format!(
            "⟨X⟩ %{{x:.4f}}<br>⟨Y⟩ %{{y:.4f}}<br>⟨Z⟩ %{{z:.4f}}<br>{x_title} = %{{text}}<extra></extra>"
        ))
        .show_legend(true)
}
