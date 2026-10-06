//! 跟随光标的对话气泡 + 可钉住的固定浮层（feature = "plot"，crate 内部件）。
//!
//! 宿主约定（`div_id` 是本模块的命名根）：
//! - 每行面板放在 `<template id="{div_id}-tpl-{row}">` 里，内容用
//!   `<div class="qtool-panel">` 包一层；
//! - 面板内嵌脚本里出现的 id 前缀 `{div_id}-panel{row}`，挂载时会被整体改写成该实例
//!   专属的 id —— 同一行可能同时挂在气泡与浮层上，id 不能重复；
//! - 把 [`bubble`] 的产物放在热图之后、模板之前：它的脚本要等热图（触发源）建好才跑。
//!
//! 另外这里导出 [`CARD_STYLE`]：那套"漫画卡片"外壳（描边/圆角/硬阴影）宿主也可以拿去
//! 当图组的外框用。
//!
//! 交互：hover 到热图（curveNumber 0）→ 气泡贴着光标显示该行面板；双击 → 在气泡当下
//! 的位置钉成固定浮层（可多个、按行去重、允许重叠，各自带 × 关闭）；光标离开二维图
//! 区域即收起气泡。浮层里挂的是独立的 plotly 实例，所以图例、缩放都能用。

use crate::utils::heatmap::json_strings;

/// 气泡配置。
pub(crate) struct Bubble<'a> {
    /// 宿主 div 的 id，同时也是模板/面板 id 的命名根。
    pub(crate) div_id: &'a str,
    /// 触发 hover / 双击的 plotly 图 div 的 id（布局稳定后会一并 resize 一次）。
    pub(crate) triggers: &'a [String],
    /// 每行在热图 y 轴上的取值（命中判定用）。
    pub(crate) row_values: &'a [f64],
    /// 每行的标题（气泡与浮层标题栏）。
    pub(crate) labels: &'a [String],
    /// 行面板的原始宽度（缩放前），px。
    pub(crate) panel_width: f64,
    /// 气泡/浮层里面板的缩放比例。
    pub(crate) scale: f64,
}

/// 生成气泡 + 浮层所需的 HTML（markup + style + script）。
pub(crate) fn bubble(spec: &Bubble<'_>) -> String {
    let div_id = spec.div_id;
    let json_rows = crate::utils::heatmap::json_array(spec.row_values);
    let json_labels = json_strings(spec.labels);
    let json_triggers = json_strings(spec.triggers);
    let panel_width = spec.panel_width;
    let scale = spec.scale;
    // 面板的布局宽度必须与 JS 里的 PANEL_W 是同一个值（气泡按它原样布局、再整体缩放），
    // 所以由这里插值生成；CSS 里不再写第二份。
    let panel_css = format!(".qtool-src,.qtool-panel{{width:{panel_width}px}}");

    format!(
        r##"<style>{CARD_STYLE}</style>{STYLE}<style>{panel_css}</style>
<div class="qtool-card qtool-bubble" id="{div_id}-bubble">
<div class="qtool-bubble-head"><span class="qtool-card-label" id="{div_id}-bubble-title"></span><span class="qtool-card-tools"><button type="button" class="qtool-card-btn" id="{div_id}-bubble-max" title="maximize"><svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></button><button type="button" class="qtool-card-btn qtool-card-btn-close" id="{div_id}-bubble-close" title="close">×</button></span></div>
<div class="qtool-bubble-body" id="{div_id}-bubble-body"></div></div>
<script>
(function () {{
  var DIV = "{div_id}";
  var root = document.getElementById(DIV);
  if (!root) {{ return; }}
  var bubble = document.getElementById(DIV + "-bubble");
  var bubbleTitle = document.getElementById(DIV + "-bubble-title");
  var bubbleBody = document.getElementById(DIV + "-bubble-body");
  var TRIGGERS = {json_triggers};
  var Y = {json_rows};
  var LABELS = {json_labels};
  var sources = {{}};
  var pinned = {{}};
  var bubbleRow = -1, bubbleW = 0, bubbleH = 0, liveSeq = 0;
  var SCALE = {scale}, PANEL_W = {panel_width};
  // 交互参数：改手感只动这里
  var MAX_FIT = 0.9;         // 最大化后卡片占视口的比例（宽高两方向取更紧的那个）
  var CURSOR_DX = 16;        // 气泡相对光标的水平偏移，px
  var CURSOR_ANCHOR = 0.3;   // 气泡纵向定位：光标落在气泡自身高度的这个比例处
  var EDGE_GAP = 8;          // 气泡/卡片离视口边缘的最小留白，px
  var DOUBLE_CLICK_MS = 400; // 双击（钉住）的判定窗口，ms

  var rowFromEvent = function (ev) {{
    var points = (ev && ev.points) || [];
    if (!points.length) {{ return -1; }}
    var y = points[0].y;
    var best = -1, bestDist = Infinity;
    for (var i = 0; i < Y.length; i++) {{
      var d = Math.abs(Y[i] - y);
      if (d < bestDist) {{ bestDist = d; best = i; }}
    }}
    return best;
  }};

  var sourceFor = function (row) {{
    if (sources[row]) {{ return sources[row]; }}
    var template = document.getElementById(DIV + "-tpl-" + row);
    if (!template) {{ return null; }}
    var holder = document.createElement("div");
    holder.className = "qtool-src";
    root.appendChild(holder);
    holder.innerHTML = template.innerHTML;
    var inert = holder.querySelectorAll("script");
    for (var i = 0; i < inert.length; i++) {{
      var script = document.createElement("script");
      script.textContent = inert[i].textContent;
      holder.appendChild(script);
    }}
    sources[row] = holder.querySelector(".qtool-panel");
    return sources[row];
  }};

  var newInner = function (parent) {{
    var inner = document.createElement("div");
    inner.className = "qtool-scaled";
    inner.style.width = PANEL_W + "px";
    parent.appendChild(inner);
    return inner;
  }};

  // scale 不改变布局尺寸，所以先量 inner 的自然高度，再按比例把占位尺寸写回容器。
  // 当前比例记在 body 的 __qtoolScale 上（最大化/还原都改它）。
  var scaleInto = function (container, inner, scale) {{
    var ratio = scale || SCALE;
    container.style.width = Math.round(PANEL_W * ratio) + "px";
    container.style.height = Math.round(inner.offsetHeight * ratio) + "px";
    inner.style.transform = "scale(" + ratio + ")";
    container.__qtoolScale = ratio;
  }};

  // 最大化：放大到"能塞进视口"的最大比例（允许超过 1 放得更大），并把卡片挪到视口中央；
  // 再点一次还原成原来的比例与位置。放大用的是 transform，plotly 的 SVG 是矢量，不会糊。
  var toggleMax = function (card) {{
    var body = card.querySelector(".qtool-bubble-body, .qtool-pin-body");
    var inner = body && body.querySelector(".qtool-scaled");
    if (!body || !inner) {{ return; }}
    var btn = card.querySelector(".qtool-card-btn-max");
    if (card.__qtoolMax) {{
      card.__qtoolMax = false;
      scaleInto(body, inner, card.__qtoolPreScale);
      card.style.left = card.__qtoolPreLeft;
      card.style.top = card.__qtoolPreTop;
      if (btn) {{ btn.classList.remove("is-max"); }}
      return;
    }}
    card.__qtoolPreScale = body.__qtoolScale || SCALE;
    card.__qtoolPreLeft = card.style.left;
    card.__qtoolPreTop = card.style.top;
    // 放大到视口的 MAX_FIT（两个方向取更紧的那个），按卡片*当前*尺寸等比换算 ——
    // 标题行、内边距、边框都算在内，不会出现面板塞下了、卡片却超出屏幕。
    // 参照物用视口而不是图组卡：图组是"宽而矮"的图，MAX_FIT×它的高度比卡片本来就小，
    // 拿它当上限会让"最大化"变成缩小。
    var fit = Math.min(
      MAX_FIT * window.innerWidth / card.offsetWidth,
      MAX_FIT * window.innerHeight / card.offsetHeight
    );
    card.__qtoolMax = true;
    scaleInto(body, inner, Math.max(SCALE, (body.__qtoolScale || SCALE) * fit));
    card.style.left = Math.max(EDGE_GAP, Math.round((window.innerWidth - card.offsetWidth) / 2)) + "px";
    card.style.top = Math.max(EDGE_GAP, Math.round((window.innerHeight - card.offsetHeight) / 2)) + "px";
    if (btn) {{ btn.classList.add("is-max"); }}
  }};

  var makeMaxButton = function (card) {{
    var button = document.createElement("button");
    button.type = "button";
    button.className = "qtool-card-btn qtool-card-btn-max";
    button.title = "maximize";
    button.innerHTML =
      '<svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">' +
      '<rect x="1.5" y="1.5" width="9" height="9" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>';
    button.onclick = function (ev) {{
      if (ev) {{ ev.stopPropagation(); }}
      toggleMax(card);
    }};
    return button;
  }};

  // 气泡用克隆（快，够看）：克隆出来的只是静态 SVG，plotly 不认
  var mountScaled = function (container, source) {{
    container.innerHTML = "";
    var inner = newInner(container);
    inner.appendChild(source.cloneNode(true));
    scaleInto(container, inner);
  }};

  // 浮层要能交互（图例、缩放、悬停）：从 <template> 重新起一份 plotly 实例，并把
  // 该行的 id 前缀换成这份实例专属的 —— 同一行可能同时挂在气泡和浮层上，id 不能重复
  var mountLiveScaled = function (container, row) {{
    var template = document.getElementById(DIV + "-tpl-" + row);
    if (!template) {{ return; }}
    container.innerHTML = "";
    // 必须先把节点挂进文档再执行脚本，plotly 才量得到容器尺寸
    var inner = newInner(container);
    var base = DIV + "-panel" + row;
    var uid = base + "-live" + (liveSeq += 1);
    inner.innerHTML = template.innerHTML.split(base).join(uid);
    var inert = inner.querySelectorAll("script");
    for (var i = 0; i < inert.length; i++) {{
      var script = document.createElement("script");
      script.textContent = inert[i].textContent;
      inner.appendChild(script);
    }}
    scaleInto(container, inner);
  }};

  // 跟随鼠标的气泡：位置贴着光标，靠边时自动翻到另一侧
  var placeBubble = function (mouse) {{
    if (!mouse) {{ return; }}
    var x = mouse.clientX, y = mouse.clientY;
    var left = x + CURSOR_DX;
    if (left + bubbleW > window.innerWidth - EDGE_GAP) {{
      left = x - CURSOR_DX - bubbleW;
    }}
    if (left < EDGE_GAP) {{ left = EDGE_GAP; }}
    var top = y - Math.round(bubbleH * CURSOR_ANCHOR);
    if (top < EDGE_GAP) {{ top = EDGE_GAP; }}
    if (top + bubbleH > window.innerHeight - EDGE_GAP) {{ top = window.innerHeight - EDGE_GAP - bubbleH; }}
    if (top < EDGE_GAP) {{ top = EDGE_GAP; }}
    bubble.style.left = left + "px";
    bubble.style.top = top + "px";
  }};

  var showBubble = function (row, mouse) {{
    if (row < 0) {{ return; }}
    var source = sourceFor(row);
    if (!source) {{ return; }}
    // 先显示再量尺寸：display:none 时 offsetHeight 恒为 0
    bubble.style.display = "block";
    if (row !== bubbleRow) {{
      bubbleRow = row;
      bubbleTitle.textContent = LABELS[row] || ("line " + row);
      mountScaled(bubbleBody, source);
      bubbleW = bubble.offsetWidth;
      bubbleH = bubble.offsetHeight;
    }}
    placeBubble(mouse);
  }};

  var hideBubble = function () {{
    bubble.style.display = "none";
    bubbleRow = -1;
  }};

  // 光标一离开数据区就收起气泡（延时 0）—— 气泡只是预览，交互都在钉住的浮层上，
  // 所以不需要给它留点击窗口。
  var scheduleHide = hideBubble;

  // 钉住：在气泡当下的位置放一个同款固定浮层（可交互的独立 plotly 实例），
  // 位置照搬气泡，看起来就是同一个气泡被冻住了
  var pin = function (row) {{
    if (row < 0 || pinned[row]) {{ return; }}
    var card = document.createElement("div");
    card.className = "qtool-card qtool-pin";
    card.style.left = bubble.style.left;
    card.style.top = bubble.style.top;
    var head = document.createElement("div");
    head.className = "qtool-pin-head";
    var label = document.createElement("span");
    label.textContent = LABELS[row] || ("line " + row);
    var tools = document.createElement("span");
    tools.className = "qtool-card-tools";
    tools.appendChild(makeMaxButton(card));
    var close = document.createElement("button");
    close.type = "button";
    close.className = "qtool-card-btn qtool-card-btn-close";
    close.textContent = "×";
    close.onclick = function () {{
      card.remove();
      delete pinned[row];
    }};
    tools.appendChild(close);
    head.appendChild(label);
    head.appendChild(tools);
    var body = document.createElement("div");
    body.className = "qtool-pin-body";
    card.appendChild(head);
    card.appendChild(body);
    root.appendChild(card);
    mountLiveScaled(body, row);
    pinned[row] = card;
    // 气泡让位：否则它（z-index 更高）正好盖住刚钉出来的浮层，看不出钉没钉上
    hideBubble();
  }};

  // 气泡工具栏里的"最大化"：先把这一行钉住（拿到一份可交互的实例），再放大它
  var bubbleMax = document.getElementById(DIV + "-bubble-max");
  if (bubbleMax) {{
    bubbleMax.onclick = function (ev) {{
      if (ev) {{ ev.stopPropagation(); }}
      var row = bubbleRow;
      if (row < 0) {{ return; }}
      pin(row);
      if (pinned[row]) {{ toggleMax(pinned[row]); }}
    }};
  }}

  // 气泡工具栏里的"×"：收掉气泡；如果这一行已经钉住，顺手取消钉住
  var bubbleClose = document.getElementById(DIV + "-bubble-close");
  if (bubbleClose) {{
    bubbleClose.onclick = function (ev) {{
      if (ev) {{ ev.stopPropagation(); }}
      var row = bubbleRow;
      if (row >= 0 && pinned[row]) {{ pinned[row].remove(); delete pinned[row]; }}
      hideBubble();
    }};
  }}

  bubble.addEventListener("mouseleave", scheduleHide);

  var lastClick = {{ row: -1, time: 0 }};
  TRIGGERS.forEach(function (id) {{
    var gd = document.getElementById(id);
    if (!gd || !gd.on) {{ return; }}
    // 只有命中最底层那条热图（curveNumber 0）才出气泡：轴区、边线图、colorbar
    // 上要么不派发 plotly_hover，要么命中曲线 1/2，都归为"离开二维图区域"。
    gd.on("plotly_hover", function (ev) {{
      var points = (ev && ev.points) || [];
      if (!points.length || points[0].curveNumber !== 0) {{
        scheduleHide();
        return;
      }}
      showBubble(rowFromEvent(ev), ev ? ev.event : null);
    }});
    // 光标离开数据点（含在同一张图内移到轴区/边线图上）即收起
    gd.on("plotly_unhover", scheduleHide);
    // plotly_hover 只在命中的数据点变化时才派发，光标在同一格内移动不会触发，
    // 这里补一个 mousemove 让气泡真正贴着光标走（尺寸已缓存，不做逐帧测量）。
    gd.addEventListener("mousemove", function (ev) {{
      if (bubbleRow >= 0) {{ placeBubble(ev); }}
    }});
    gd.addEventListener("mouseleave", scheduleHide);
    // plotly_dblclick 实测不派发（被 autoscale 动作吞掉），改从 click 事件识别双击：
    // 同一行、DOUBLE_CLICK_MS 内两次 click 即 pin 该行。
    gd.on("plotly_click", function (ev) {{
      var row = rowFromEvent(ev);
      var now = Date.now();
      if (row >= 0 && row === lastClick.row && now - lastClick.time < DOUBLE_CLICK_MS) {{
        pin(row);
        lastClick = {{ row: -1, time: 0 }};
        return;
      }}
      lastClick = {{ row: row, time: now }};
    }});
  }});

}})();
</script>
"##
    )
}

/// 共享的"漫画卡片"外壳（气泡、浮层、以及宿主的图组外框都用它），含可选标题。
///
/// 单独成一条规则、由各处自己放进 `<style>` 里，这样宿主不需要复制样式。
/// `position:relative` 是给标题用的定位上下文（气泡/浮层各自再用两类的选择器
/// 覆盖成 `fixed`）。
pub(crate) const CARD_STYLE: &str = "\
.qtool-card{position:relative;background:#ffffff;border:2.5px solid #0f172a;border-radius:12px;padding:10px;box-shadow:4px 4px 0 0 #0f172a}\
/* 可选标题：骑在上边线上（纵向中心与边线中心重合），白底盖掉被它压住的那段边线 */\
.qtool-card-title{position:absolute;left:14px;top:-1.25px;transform:translateY(-50%);background:#ffffff;padding:0 8px;font-weight:700;font-size:16px;color:#1e293b}";

/// 可选的卡片标题（没有就不输出，卡片上边线保持完整）。
pub(crate) fn card_title(text: &str) -> String {
    format!(
        "<div class=\"qtool-card-title\">{}</div>",
        crate::utils::heatmap::escape_html(text)
    )
}

const STYLE: &str = r#"<style>
.qtool-card.qtool-bubble{position:fixed;z-index:60;display:none;pointer-events:none;left:0;top:0}
.qtool-bubble-head,.qtool-pin-head{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:700;font-size:13px;color:#1e293b;border-bottom:2px solid #e2e8f0;padding-bottom:4px;margin-bottom:8px}
.qtool-bubble-body,.qtool-pin-body{overflow:hidden}
.qtool-scaled{transform-origin:top left}
/* 双击钉住的浮层：与气泡同一套外壳，停在原地、允许重叠；只有它可点（气泡是
   pointer-events:none），所以 × 能按到 */
.qtool-card.qtool-pin{position:fixed;z-index:55;pointer-events:auto}
/* 卡片工具栏：贴着标题栏右端的一组小按钮（最大化 / 关闭），风格照 Windows 的窗口按钮 */
.qtool-card-tools{display:flex;align-items:center;gap:2px;flex:none}
.qtool-card-btn{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;padding:0;border:none;border-radius:5px;background:transparent;color:#64748b;font-size:15px;line-height:1;cursor:pointer}
.qtool-card-btn:hover{background:#f1f5f9;color:#0f172a}
.qtool-card-btn.qtool-card-btn-close:hover{background:#fee2e2;color:#b91c1c}
/* 气泡整体不吃鼠标（否则会挡住热图的 hover），只有工具栏例外 */
.qtool-card.qtool-bubble .qtool-card-tools{pointer-events:auto}
.qtool-src{position:absolute;left:-20000px;top:0;visibility:hidden}
</style>"#;
