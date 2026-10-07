"""qtool —— 超导量子比特标定分析（Rust 内核 + plotly 报告）的 Python 绑定。

模块路径与 Rust 侧一一对应：`qtool.t1`、`qtool.qspec`、`qtool.drag.coeff`……
每个实验模块给三样东西：`fit`（抛错）、`fit_batch`（失败项给异常实例）、`plot`（返回 `Html`）。

    import qtool
    fit = qtool.t1.fit(taus, iq, states=states)
    qtool.t1.plot(taus, iq, states=states, fit=fit)     # notebook 单元格最后一行 → 直接出图

出图不需要用户挂任何脚本：`Html` 自带 plotly.js 的 CDN 标签（那份整页跑在 srcdoc iframe 里），
落盘用 `plt.persist(Path("reports"))`。
"""

from __future__ import annotations

import html as _html
import sys as _sys
from pathlib import Path as _Path

from . import _qtool
from ._qtool import (  # noqa: F401  （顶层词汇原样转出）
    QtoolError,
    StateCenters,
    direction_rs as direction,
    p1_rs as p1,
    p1_sigma_rs as p1_sigma,
    plotly_js_cdn,
)
from ._qtool import (  # noqa: F401
    bloch,
    drag,
    iq,
    qspec,
    rabi,
    ramsey,
    s21,
    t1,
    t2_echo,
)

__all__ = [
    "Html",
    "QtoolError",
    "StateCenters",
    "bloch",
    "direction",
    "drag",
    "iq",
    "p1",
    "p1_sigma",
    "plotly_js_cdn",
    "qspec",
    "rabi",
    "ramsey",
    "s21",
    "t1",
    "t2_echo",
    "__version__",
]

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

    __version__ = _version("qtool-py")
except _PackageNotFoundError:  # 源码树里直接 import 时（没装过 wheel）
    __version__ = "0.0.0+unknown"

#: iframe 的兜底高度 (px)：页面里的脚本量完自己的高度会写回外层，这个数只在不给量时兜底。
_FALLBACK_HEIGHT = 420

#: plotly.js 的引入标签，URL 的唯一来源是 Rust 侧的 `PLOTLY_JS_CDN`。
_CDN = plotly_js_cdn()

#: 报告整页的模板：`persist` 落盘的与 `_repr_html_` 塞进 iframe 的是**同一份**。
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{cdn}
</head>
<body style="margin:16px;background:#ffffff">
{fragment}{fit_js}
</body>
</html>
"""

#: 量高并写回外层 iframe；独立页面里 `frameElement` 拿不到，直接空转。
_FIT_JS = """<script>
(function () {
  var fe = window.frameElement;
  if (!fe) { return; }
  function fit() {
    var height = document.documentElement.scrollHeight;
    if (Math.abs(height - (fe.dataset.h || 0)) < 2) { return; }
    fe.dataset.h = height;
    fe.style.height = height + "px";
  }
  window.addEventListener("load", fit);
  if (window.ResizeObserver) { new ResizeObserver(fit).observe(document.body); }
  setTimeout(fit, 300);
})();
</script>
"""


class Html(str):
    """一段自包含的报告 HTML（`str` 子类）。

    - **notebook**：把它放在单元格最后一行就出图（`_repr_html_` 把整页装进 srcdoc iframe）。
    - **脚本**：它就是个字符串，`open(...).write(...)`、切片、正则照旧。
    - **落盘**：`persist(Path("reports/q0"))` → `reports/q0/<显示名>.html`（目录不存在会建）。
    """

    def __new__(cls, fragment: str, name: str = "plot") -> "Html":
        instance = super().__new__(cls, fragment)
        instance._name = name
        return instance

    @property
    def name(self) -> str:
        """显示名（也是 `persist` 的文件名主干与页面 `<title>`）。"""
        return self._name

    def page(self, title: str | None = None) -> str:
        """整页 HTML：补上 doctype/head/CDN 标签；`persist` 与 notebook 共用它。"""
        return _PAGE.format(
            title=_html.escape(title or self._name),
            cdn=_CDN,
            fragment=str(self),
            fit_js=_FIT_JS,
        )

    def persist(self, path: Path | str, title: str | None = None) -> Path:
        """把整页写进 `path` 目录下的 `<name>.html`，返回写出的文件路径。"""
        target = _Path(path).expanduser() / f"{self._name}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.page(title), encoding="utf-8")
        return target

    def _repr_html_(self) -> str:
        return (
            f'<iframe srcdoc="{_html.escape(self.page(), quote=True)}" '
            f'style="width:100%;height:{_FALLBACK_HEIGHT}px;border:0"></iframe>'
        )


def _wrap_plot(function, name: str):
    """把 native 的 `plot`（返回裸 `str`）包成返回 `Html`，并就手把显示名钉上。"""

    def plot(*args, **kwargs) -> Html:
        return Html(function(*args, **kwargs), name=name)

    plot.__doc__ = function.__doc__
    plot.__name__ = "plot"
    return plot


def _bind(module, function: str, name: str) -> None:
    setattr(module, function, _wrap_plot(getattr(module, function), name))


# 把 native 子模块登记进 sys.modules，`import qtool.t1` 这种写法才成立（只 import qtool 也行）
for _module in (s21, qspec, rabi, iq, ramsey, t1, t2_echo, bloch, drag):
    _sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module

# 每个实验的绘图入口：native 给裸 HTML，这里补上"显示名"（persist 的文件名 / 页面标题）
_bind(s21, "plot", "s21")
_bind(s21, "power_plot", "s21_power")
_bind(qspec, "plot", "qspec")
_bind(qspec, "z_plot", "qspec_z")
_bind(rabi, "plot", "rabi")
_bind(iq, "plot", "iq")
_bind(ramsey, "plot", "ramsey")
_bind(t1, "plot", "t1")
_bind(t2_echo, "plot", "t2_echo")
_bind(drag.amplitude, "plot", "drag_amplitude")
_bind(drag.coeff, "plot", "drag_coeff")
_bind(drag.detuning, "plot", "drag_detuning")
_bind(bloch, "plot", "bloch")

# `from __future__ import annotations` 会在模块里绑一个同名变量，抹掉它 dir(qtool) 才干净
del annotations
