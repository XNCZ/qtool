"""冒烟测试：拟合能出数、绘图是自包含的整页、批量失败项是可判的异常实例。

只覆盖"接口接得上"这一层；拟合本身的正确性由 Rust 侧的测试与 demo 负责。
"""

from __future__ import annotations

import numpy as np
import qtool

#: 真值：T1 = 40 µs，两态中心任意取一对。
T1_TRUE = 40e-6
G0 = 0.30 + 0.10j
G1 = 0.42 + 0.30j


def synthetic_t1() -> tuple[np.ndarray, np.ndarray]:
    taus = np.linspace(0.0, 200e-6, 41)
    prob = 0.02 + 0.95 * np.exp(-taus / T1_TRUE)
    return taus, np.asarray(G0 + (G1 - G0) * prob, dtype=complex)


def test_fit_returns_model_and_arrays() -> None:
    taus, iq = synthetic_t1()
    states = qtool.StateCenters([G0, G1])
    fit = qtool.t1.fit(taus, iq, states=states)

    assert abs(fit.model.t1 - T1_TRUE) < 2e-6
    assert isinstance(fit.p1, np.ndarray) and fit.p1.shape == taus.shape
    assert set(dict(fit.params)) == {"offset", "amplitude", "t1"}
    value, stderr = dict(fit.params)["t1"]
    assert value == fit.model.t1 and stderr is not None


def test_plot_is_self_contained_page(tmp_path) -> None:
    taus, iq = synthetic_t1()
    states = qtool.StateCenters([G0, G1])
    fit = qtool.t1.fit(taus, iq, states=states)

    page = qtool.t1.plot(taus, iq, states=states, fit=fit)
    # 是 str，但带显示名与两个消费入口
    assert isinstance(page, str) and page.name == "t1"
    assert "Plotly.newPlot" in page

    written = page.persist(tmp_path)
    assert written == tmp_path / "t1.html"
    text = written.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "plotly-3.0.1.min.js" in text  # CDN 标签就在整页里，用户不用自己挂
    assert "<iframe srcdoc=" in page._repr_html_()


def test_batch_failure_comes_back_as_exception_instance() -> None:
    outcomes = qtool.t1.fit_batch([0.0, 1e-6], [[G0, G1]])
    assert isinstance(outcomes[0], qtool.t1.T1Error)
    assert "at least" in str(outcomes[0])
    # 失败项喂回 plot：原样抛出，而不是画一张空卡
    try:
        qtool.t1.plot([0.0, 1e-6], [G0, G1], fit=outcomes[0])
    except qtool.t1.T1Error:
        pass
    else:  # pragma: no cover
        raise AssertionError("a failed fit must be re-raised by plot")


def test_states_accept_plain_lists() -> None:
    taus, iq = synthetic_t1()
    fit = qtool.t1.fit(list(taus), list(iq), states=qtool.StateCenters([G0, G1]))
    assert abs(fit.model.t1 - T1_TRUE) < 2e-6
