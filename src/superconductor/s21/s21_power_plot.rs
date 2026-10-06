//! S21 功率扫描的二维报告（feature = "plot"）：两张并排热图 + 行面板交互。
//!
//! - 热图一：颜色 = |S21|；热图二：颜色 = unwrap + detrend 后的相位（与
//!   [`s21_plot`](crate::superconductor::s21::s21_plot) 的相位面板同口径）；
//! - 两张热图都由 [`crate::utils::heatmap::heatmap`] 生成（点击取
//!   该点的 row / col 到右、下边线图，带 continuous error bar）；
//! - **hover 到某一行** → 跟随光标的对话气泡显示该功率的 S21 四面板（首次使用时才渲染）；
//! - **double click** → 把气泡**原地**钉成固定浮层（位置不再跟走、移开鼠标也不消失；
//!   可多个、按行去重、允许重叠），每个浮层带 × 关闭；
//! - 产物是自包含 div，宿主页面需提供 plotly.js（见
//!   [`s21_plot::PLOTLY_JS_CDN`](crate::superconductor::s21::s21_plot::PLOTLY_JS_CDN)）。

use crate::superconductor::s21::{Complex64, S21Error, S21Model};
use crate::superconductor::s21::s21_plot::{DESIGN_WIDTH, s21_fit_plot_div};
use crate::utils::bubble::{Bubble, CARD_STYLE, bubble, card_title};
use crate::utils::heatmap::{Grid2d, Palette, heatmap};
use crate::utils::{detrend, unwrap_phase};
use lmfit::ComplexResult;

/// 一条频率线：泵幅 + 数据 + 拟合结果（`Err` 表示该线拟合失败）。
pub struct PowerLine<'a> {
    pub power: f64,
    pub iq: &'a [Complex64],
    pub sigma: Option<&'a [f64]>,
    pub fit: Result<&'a ComplexResult<S21Model>, &'a S21Error>,
}

/// 渲染功率扫描报告 div。
///
/// 形参:
///     freqs_hz: 公共读出频率轴 (n,)，Hz
///     lines: 逐条频率线（按功率递增），每条含泵幅/数据/σ/拟合结果
///     div_id: 外层 div 的 HTML id（须是合法 id）
///
/// 返回值:
///     自包含的 `<div class="qtool-power">` 片段（两张热图 + 跟随光标的行面板气泡
///     + 可钉住的浮层）。行标签由 `PowerLine::power` 派生。
pub fn s21_power_plot_div(freqs_hz: &[f64], lines: &[PowerLine<'_>], div_id: &str) -> String {
    // 频率轴直接用 SI 基本单位 Hz（与数据、与 s21_plot 的面板一致）
    let powers: Vec<f64> = lines.iter().map(|line| line.power).collect();

    let mut amp = Vec::with_capacity(lines.len());
    let mut amp_err = Vec::with_capacity(lines.len());
    let mut phase = Vec::with_capacity(lines.len());
    let mut phase_err = Vec::with_capacity(lines.len());
    for line in lines {
        let magnitudes: Vec<f64> = line.iq.iter().map(|z| z.norm()).collect();
        let raw_phase: Vec<f64> = line.iq.iter().map(|z| z.arg()).collect();
        let (detrended, _, _) = detrend(freqs_hz, &unwrap_phase(&raw_phase));
        amp.push(magnitudes.clone());
        phase.push(detrended);
        match line.sigma {
            Some(sigma) => {
                amp_err.push(sigma.to_vec());
                phase_err.push(
                    sigma
                        .iter()
                        .zip(magnitudes.iter())
                        .map(|(value, magnitude)| match *magnitude > 0.0 {
                            true => value / magnitude,
                            false => f64::NAN,
                        })
                        .collect(),
                );
            }
            None => {
                amp_err.push(vec![f64::NAN; freqs_hz.len()]);
                phase_err.push(vec![f64::NAN; freqs_hz.len()]);
            }
        }
    }

    let amp_map = heatmap(
        &Grid2d {
            x: freqs_hz,
            y: &powers,
            z: &amp,
            z_err: Some(&amp_err),
            x_title: "freq (Hz)",
            y_title: "power",
            value_title: "|S21|",
            palette: Palette::Turbo,
        },
        &format!("{div_id}-amp"),
    );
    let phase_map = heatmap(
        &Grid2d {
            x: freqs_hz,
            y: &powers,
            z: &phase,
            z_err: Some(&phase_err),
            x_title: "freq (Hz)",
            y_title: "power",
            value_title: "phase (rad)",
            palette: Palette::RdBu,
        },
        &format!("{div_id}-phase"),
    );

    // 每个功率行的面板（[`s21_fit_plot_div`] 的四面板图 + 参数表，含自身样式）嵌在
    // <template> 里惰性实例化；`{div_id}-panel{row}` 是这一行的 id 前缀，前端挂载时
    // 会整体改写成该实例专属的 id（同一行可能同时挂在气泡与浮层上，id 不能重复）。
    let mut templates = String::new();
    for (row, line) in lines.iter().enumerate() {
        let panel_base = format!("{div_id}-panel{row}");
        let panel_html = s21_fit_plot_div(freqs_hz, line.iq, line.sigma, line.fit, &panel_base);
        templates.push_str(&format!(
            "<template id=\"{div_id}-tpl-{row}\"><div class=\"qtool-panel\">{panel_html}</div></template>\n"
        ));
    }

    // 行标签由泵幅派生：气泡/浮层的标题栏、hover 提示都用它
    let labels: Vec<String> = powers
        .iter()
        .map(|power| format!("power {power:.4}"))
        .collect();
    let title = card_title("s21 vs power");
    let bubble_html = bubble(&Bubble {
        div_id,
        triggers: &[format!("{div_id}-amp-plot"), format!("{div_id}-phase-plot")],
        row_values: &powers,
        labels: &labels,
        panel_width: DESIGN_WIDTH,
        scale: SCALE,
    });

    format!(
        r##"<div class="qtool-power" id="{div_id}">
{STYLE}
<style>{CARD_STYLE}</style>
<div class="qtool-card qtool-power-maps">{title}{amp_map}{phase_map}</div>
{bubble_html}
{templates}</div>
"##
    )
}

/// 面板在气泡/浮层里的缩放比例。
const SCALE: f64 = 0.62;

const STYLE: &str = r#"<style>
/* 不设 max-width：报告宽度跟着宿主容器走，宽屏就铺满（图由 ResizeObserver 跟尺寸重排） */
.qtool-power{font-family:system-ui,'Segoe UI',sans-serif;color:#1f2328}
.qtool-power .qtool-power-maps{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
/* 每张图至少 420px：容器放不下两张时自动换行、各占一行（窄屏不再互相压扁） */
.qtool-power .qtool-power-maps>.qtool-2d{flex:1 1 420px;min-width:0}
</style>"#;
