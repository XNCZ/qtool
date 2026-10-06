//! 全页共享的尺寸观察器（feature = "plot"，crate 内部件）。
//!
//! 一份报告可能有几十张 plotly 图（够铺满屏幕的只有一两张），尺寸变化必须由**同一个**
//! 观察器统一驱动，所以这里定三条规矩：
//!
//! 1. **仓库里只允许本模块 `new ResizeObserver`**；别处再建就会退化成"每张图各挂一个"，
//!    这正是我们关掉 plotly 自带 `responsive` 的原因；
//! 2. 片段自带的内联脚本是**幂等**的（`window.__qtoolResize || (…)`）：第一个片段创建、
//!    其余复用其引用 —— 所以同一页不管挂多少个片段（含将来新加的模块）都只有一个观察器，
//!    只有 iframe 那种独立 document 才会各有一个（这是对的）；
//! 3. 只有**视口内**的图才真的 `Plotly.Plots.resize`，离屏的先标脏、等滚进来再补，避免
//!    拖一下窗口就把几十张图全重排；已从文档摘掉的节点（气泡换行丢弃的克隆、关掉的浮层）
//!    在 flush 时 prune 掉，不让观察器一直攥着它们。
//!
//! 用法：片段在 plot div 之后插入 [`register_script`] 的产物即可。

/// 把自己的 plot div 注册进全局观察器（`{div_id}-plot`）。
pub(crate) fn register_script(div_id: &str) -> String {
    format!(
        "<script>\n{BOOTSTRAP}\nwindow.__qtoolResize.add(document.getElementById(\"{div_id}-plot\"));\n</script>"
    )
}

/// 幂等的 bootstrap：只在页面里第一个片段执行时真正建观察器。
const BOOTSTRAP: &str = r#"(function () {
  if (window.__qtoolResize) { return; }
  var targets = [];
  var queued = false;
  var flush = function () {
    queued = false;
    if (!window.Plotly) { return; }
    var limit = window.innerHeight + 200;
    for (var i = targets.length - 1; i >= 0; i--) {
      var t = targets[i];
      if (!t.el.isConnected) { observer.unobserve(t.el); targets.splice(i, 1); continue; }
      if (!t.dirty) { continue; }
      var box = t.el.getBoundingClientRect();
      if (box.width < 1 || box.height < 1 || box.bottom < -200 || box.top > limit) { continue; }
      t.dirty = false;
      Plotly.Plots.resize(t.el);
    }
  };
  var schedule = function () {
    if (queued) { return; }
    queued = true;
    requestAnimationFrame(flush);
  };
  var observer = new ResizeObserver(function (entries) {
    for (var i = 0; i < entries.length; i++) {
      for (var j = 0; j < targets.length; j++) {
        if (targets[j].el === entries[i].target) { targets[j].dirty = true; break; }
      }
    }
    schedule();
  });
  window.addEventListener("scroll", schedule, { passive: true });
  window.__qtoolResize = {
    add: function (el) {
      if (!el) { return; }
      targets.push({ el: el, dirty: true });
      observer.observe(el);
    }
  };
})();"#;
