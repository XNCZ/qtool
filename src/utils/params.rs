//! 拟合参数表（名称 / 说明 / 值 / 标准误四列），各报告共用。
//!
//! 只负责排版与样式：四个字段由调用方给（值已经格式化好），本模块不读拟合结果、不猜单位
//! 与有效位数，于是任何模型都能用它列表。表自带 `<style>`，调用方在页面里放一次即可
//! （重复放置只是多几行同样的 CSS）。

use crate::utils::heatmap::escape_html;

/// 参数表的一行。
pub(crate) struct ParamRow<'a> {
    /// 参数名（等宽字体那一列）
    pub(crate) name: &'a str,
    /// 一句话说明
    pub(crate) description: &'a str,
    /// 参数值，已格式化
    pub(crate) value: String,
    /// 标准误，已格式化；协方差不可用时调用方给占位符（如 `—`）
    pub(crate) stderr: String,
}

/// 表格样式：列宽与对齐——说明列吃掉剩余宽度，数值列右对齐且不换行。
const PARAMS_STYLE: &str = "<style>\
.qtool-params{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}\
.qtool-params th,.qtool-params td{border-bottom:1px solid #e5e7eb;padding:2px 8px;text-align:right;white-space:nowrap}\
.qtool-params th:first-child,.qtool-params td:first-child{text-align:left;font-family:ui-monospace,Consolas,monospace}\
.qtool-params th:nth-child(2),.qtool-params td:nth-child(2){text-align:left;width:100%;white-space:normal;color:#4b5563}\
.qtool-params th{color:#6b7280;font-weight:500}\
.qtool-note{margin-top:4px;font-size:12px;color:#b45309}\
</style>";

/// 参数表 + 可选尾注。
///
/// 形参:
///     rows: 各行（顺序即渲染顺序）
///     note: 表下的提示行（求解器未收敛、取向被翻转等），None 表示不画
///
/// 返回值:
///     自包含的 `<style>` + `<table class="qtool-params">`（含表头）+ 尾注片段
pub(crate) fn params_table(rows: &[ParamRow<'_>], note: Option<&str>) -> String {
    let mut html = String::from(PARAMS_STYLE);
    html.push_str(
        "<table class=\"qtool-params\"><thead><tr><th>Parameter</th><th>Description</th><th>Value</th><th>stderr</th></tr></thead><tbody>",
    );
    for row in rows {
        html.push_str(&format!(
            "<tr><td>{}</td><td class=\"qtool-desc\">{}</td><td>{}</td><td>{}</td></tr>",
            row.name, row.description, row.value, row.stderr
        ));
    }
    html.push_str("</tbody></table>");
    match note {
        Some(text) => html.push_str(&format!(
            "<div class=\"qtool-note\">⚠ {}</div>",
            escape_html(text)
        )),
        None => {}
    }
    html
}
