"""各实验模块的接线测试：拟合出数、绘图是自包含整页、逐行输入类收得下拟合结果。

数据一律**无噪声的合成扫描**（真值已知，容差按线型宽度给），只验"接口接得上、参数回得来"；
拟合本身的统计性质由 Rust 侧的测试与 demo 负责。
"""

from __future__ import annotations

import numpy as np
import qtool

#: 两态标定中心：任意夹角的一对（连线不平行于任一坐标轴）。
G0 = 0.30 + 0.10j
G1 = 0.42 + 0.30j


def states() -> qtool.StateCenters:
    return qtool.StateCenters([G0, G1])


def on_line(prob: np.ndarray) -> np.ndarray:
    """把 P1 落到 |0>–|1> 连线上（无噪声）。"""
    return np.asarray(G0 + (G1 - G0) * prob, dtype=complex)


# ---------------------------------------------------------------- qspec


def test_qspec_fit_and_plot(tmp_path) -> None:
    fq_true, fwhm_true = 5.0e9, 2.4e6
    freqs = np.linspace(fq_true - 3e6, fq_true + 3e6, 51)
    truth = qtool.qspec.Lorentz(fq_true, fwhm_true, 1.0, 0.05)
    iq = on_line(truth.at(freqs))

    fit = qtool.qspec.fit(freqs, iq, states=states())
    assert abs(fit.model.fq - fq_true) < 1e3
    assert abs(fit.model.fwhm - fwhm_true) < 1e3
    assert set(dict(fit.params)) >= {"fq", "fwhm", "amp", "offset"}

    page = qtool.qspec.plot(freqs, iq, states=states(), fit=fit)
    assert isinstance(page, str) and page.name == "qspec"
    written = page.persist(tmp_path)
    assert written == tmp_path / "qspec.html"
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_qspec_z_plot_lines(tmp_path) -> None:
    """逐 Z 的行自带偏置；没拟合的行（`fit=None`）也画得出来。"""
    freqs = np.linspace(4.96e9, 5.04e9, 41)
    truth = qtool.qspec.Lorentz(5.0e9, 2.4e6, 1.0, 0.05)
    fitted_iq = on_line(truth.at(freqs))
    fit = qtool.qspec.fit(freqs, fitted_iq, states=states())

    lines = [
        qtool.qspec.QspecZLine(-0.01, list(fitted_iq), fit),
        qtool.qspec.QspecZLine(0.01, list(fitted_iq)),
    ]
    page = qtool.qspec.z_plot(freqs, lines, states=states())
    assert page.name == "qspec_z"
    assert page.persist(tmp_path) == tmp_path / "qspec_z.html"


# ---------------------------------------------------------------- s21


def s21_truth() -> qtool.s21.S21Model:
    """demo 里那组已知参数（`s21_demo.rs`）。"""
    return qtool.s21.S21Model(
        6.8982e9, 8.0e3, -1.4e4, 6.02, 0.1, -2.4e-7, 4.4e9, 4.4e9, 10521.0, -3.0e6, 8.2e6
    )


def test_s21_fit_and_plot(tmp_path) -> None:
    truth = s21_truth()
    freqs = np.linspace(truth.fr - 5e6, truth.fr + 5e6, 51)
    iq = truth.at(freqs)

    fit = qtool.s21.fit(freqs, iq)
    assert abs(fit.model.fr - truth.fr) < 1e2
    assert abs(fit.model.ql - truth.ql) / abs(truth.ql) < 1e-3
    assert set(dict(fit.params)) >= {"fr", "ql", "qc"}

    page = qtool.s21.plot(freqs, iq, fit=fit)
    assert page.name == "s21"
    written = page.persist(tmp_path)
    assert written == tmp_path / "s21.html"
    assert "Plotly.newPlot" in written.read_text(encoding="utf-8")


def test_s21_power_plot_lines(tmp_path) -> None:
    """功率扫描的每一行自带泵幅；没拟合的行也画得出来。"""
    truth = s21_truth()
    freqs = np.linspace(truth.fr - 5e6, truth.fr + 5e6, 51)
    iq = truth.at(freqs)
    fit = qtool.s21.fit(freqs, iq)

    lines = [
        qtool.s21.PowerLine(0.01, list(iq), fit=fit),
        qtool.s21.PowerLine(0.05, list(iq)),
    ]
    page = qtool.s21.power_plot(freqs, lines)
    assert page.name == "s21_power"
    assert page.persist(tmp_path) == tmp_path / "s21_power.html"


# ---------------------------------------------------------------- iq


def clouds() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [
        np.asarray(c + 0.05 * (rng.normal(size=2000) + 1j * rng.normal(size=2000)))
        for c in (-0.5 + 0.0j, 0.5 + 0.0j)
    ]


def test_iq_stats_and_plot(tmp_path) -> None:
    iqs = clouds()
    stats = qtool.iq.stats(iqs)

    assert len(stats.states) == 2
    assert abs(stats.states[0].center.real - (-0.5)) < 0.01
    assert stats.states[0].radii.shape == (4,)
    assert len(stats.pairs) == 1
    pair = stats.pairs[0]
    assert (pair.state_p, pair.state_q) == (0, 1)
    assert pair.separation > 0.95 and pair.auc > 0.99
    # 各态中心能直接喂给别的实验
    assert len(stats.centers) == 2

    page = qtool.iq.plot(iqs)
    assert page.name == "iq"
    assert page.persist(tmp_path) == tmp_path / "iq.html"


# ---------------------------------------------------------------- bloch


def test_bloch_vector_and_plot(tmp_path) -> None:
    amps = np.linspace(0.0, 3.0, 61)
    x_true = np.sin(np.pi * amps) * 0.5
    y_true = np.zeros_like(amps)
    z_true = np.cos(np.pi * amps)
    # 三基各自把泡利期望值搬成 P1 = (1 − ⟨P⟩)/2
    iq = [on_line((1.0 - component) / 2.0) for component in (x_true, y_true, z_true)]

    trajectory = qtool.bloch.Trajectory(amps, iq[0], iq[1], iq[2])
    vector = qtool.bloch.bloch_vector(trajectory, states=states())

    assert np.allclose(vector.x, x_true, atol=1e-9)
    assert np.allclose(vector.y, y_true, atol=1e-9)
    assert np.allclose(vector.z, z_true, atol=1e-9)
    assert np.allclose(vector.radius(), np.hypot(x_true, z_true), atol=1e-9)

    page = qtool.bloch.plot(vector, "π amplitude")
    assert page.name == "bloch"
    assert page.persist(tmp_path) == tmp_path / "bloch.html"
