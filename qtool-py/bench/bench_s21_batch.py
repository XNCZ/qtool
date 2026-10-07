"""为什么拟合住在 Rust 里：同一批 S21 问题的分档对照。

问题集与 Rust 侧 `examples/s21_batch_bench.rs` **逐位一致**：同一个 LCG、同一种子、
同一个 `line_params(index)`、同一条频率网格、同一个噪声尺度（4e-3）。Python 侧的
numpy 模型在跑之前先与 Rust 内核的 `S21Model.at` 对拍，确认两边解的是同一批问题；
手写的解析雅可比也先与中心差分对拍（4.6e-06），才拿去喂 `curve_fit`。

输出分两档，**不要跨档比**：

  【同任务对照】同一起点、同一份数据、只拟合一次 —— 与 Rust bench 的同名行逐项可比
  （Rust 侧 `S21Model::fit` 走同一个 lmfit 内核，用的是 crate 自带的解析雅可比）。这一档里：
    1. `scipy.optimize.curve_fit` 串行，两种雅可比：手写解析的（**同一条起跑线**）与
       scipy 默认的有限差分（同一批问题的默认解法）。
    2. 同一个串行循环铺到 `multiprocessing` 池（全核，两种切块）—— 绕开 GIL 的代价。
    3. p0 换成"估计器量级"的抖动 —— 起点好坏对这个问题的价格。

  【全流程对照】`qtool.s21.fit_batch` —— Rust 内核（rayon，全核），每条线做的是
  **初值估计 + 最多 8 个候选各拟合一次**；scipy 侧没有与 `estimate()` 对应的入口，
  所以这一档只有一行，它的公平性由同任务对照决定：同样起点下内核的快慢 ×
  每条线的拟合次数。

**p0 策略（直接影响 scipy 的数字，必须明确）**：Rust 侧的初值来自 `estimate()`
（圆拟合定 θ + 去趋势定 τ/φ + 判据掩码 + 8 个候选取残差最小者），移植过来约 90 行，
不"短"。因此同任务对照的基准行**直接用生成数据的真值参数作 p0**——即假定初值估计
免费且完美。这是对 scipy 最有利的假设：p0 已落在最优解上，迭代次数最少。

运行：uv run python bench/bench_s21_batch.py [--lines 3000]

依赖：numpy、scipy、qtool（multiprocessing 是标准库）。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import time
from functools import partial

import numpy as np
import qtool
import scipy
from scipy.optimize import curve_fit

#: u64 环绕算术的掩码：Python 整数无位宽，LCG 靠它模拟 Rust 的 wrapping_*。
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


class Lcg:
    """Rust 侧 `Lcg` 的逐位移植（同一个乘子/增量，输出 `[-0.5, 0.5)`）。"""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_unit(self) -> float:
        self.state = (self.state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & MASK64
        return (self.state >> 33) / float(1 << 31) - 0.5


def line_params(index: int) -> np.ndarray:
    """第 index 条线的 11 个真值参数（`JAC_NAMES` 序），抖动方式与 Rust 一致。

    LCG 的四个抽样按同一顺序求值，Python 的列表字面量自左向右求值，与 Rust 数组
    字面量一致，故两边的随机数流逐位相同。
    """
    lcg = Lcg(index + 1)
    return np.array(
        [
            6.8982e9 + lcg.next_unit() * 6e6,       # fr ±3 MHz
            8.0e3 * (1.0 + lcg.next_unit() * 0.5),  # ql ±25%
            -1.4e4 * (1.0 + lcg.next_unit() * 0.6),  # qc ±30%（保持负号）
            6.02 + lcg.next_unit() * 0.8,           # theta ±0.4 rad
            0.1,
            -2.4e-7,
            4.4e9,
            4.4e9,
            10521.0,
            -3.0e6,
            8.2e6,
        ]
    )


def freqs_grid() -> np.ndarray:
    """Rust 侧的 51 点频率网格。"""
    i = np.arange(51, dtype=np.float64)
    return 6.8982e9 - 5e6 + 10e6 * i / 50.0


def _solve_detuning_y(y0: np.ndarray, ap: float) -> np.ndarray:
    """Duffing 隐式方程 `y = y0 + ap/(1+4y²)` 的解析解（Cardano/Viète）。

    等价三次方程 `4y³ − 4y0·y² + y − (y0+ap) = 0`；双稳态区有三个实根，物理分支取
    最接近 y0 的那支——与 Rust 侧 `solve_detuning_y`（`roots::find_roots_cubic`）
    用同一个解析解、同一条选根规则，只是这里对整条频率网格向量化。
    """
    a2 = -y0
    b = 0.25
    c = -0.25 * (y0 + ap)
    p = b - a2 * a2 / 3.0
    q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * b / 3.0 + c
    shift = -a2 / 3.0  # y = t + shift 把二次项消掉

    with np.errstate(divide="ignore", invalid="ignore"):
        disc = 0.25 * q * q + (p / 3.0) ** 3
        # 单实根支：t = ∛(−q/2 + √Δ) + ∛(−q/2 − √Δ)
        root_disc = np.sqrt(np.maximum(disc, 0.0))
        t_single = np.cbrt(-0.5 * q + root_disc) + np.cbrt(-0.5 * q - root_disc)
        # 三实根支（disc < 0 蕴含 p < 0）：t_k = 2√(−p/3)·cos((φ − 2πk)/3)
        amp = 2.0 * np.sqrt(np.maximum(-p / 3.0, 0.0))
        cos_arg = 3.0 * q / (2.0 * p) * np.sqrt(np.maximum(-3.0 / p, 0.0))
        phi = np.arccos(np.clip(cos_arg, -1.0, 1.0)) / 3.0
        roots = [amp * np.cos(phi - 2.0 * np.pi * k / 3.0) + shift for k in (0, 1, 2)]

    nearest = roots[0]
    for candidate in roots[1:]:
        nearest = np.where(np.abs(candidate - y0) < np.abs(nearest - y0), candidate, nearest)
    return np.where(disc < 0.0, nearest, t_single + shift)


def model(freqs: np.ndarray, p: np.ndarray) -> np.ndarray:
    """`src/superconductor/s21/s21.rs` 的 `model_at` 向量化移植。

    S21 = zc + (A·cos(2πfτ) − j·B·sin(2πfτ))·e^{−jφ}·res，res = 1 − (Ql/Qc·e^{jθ})/((1+2j·y)·cosθ)
    """
    fr, ql, qc, theta, ap, tau, a, b, phi, zc_re, zc_im = p
    y = _solve_detuning_y(ql * (freqs - fr) / fr, ap)

    cos_t = np.cos(theta)
    kappa = (ql / qc / cos_t) * (np.cos(theta) + 1j * np.sin(theta))
    notch = 1.0 - kappa / (1.0 + 2j * y)

    arg = 2.0 * np.pi * freqs * tau
    background = (a * np.cos(arg) - 1j * b * np.sin(arg)) * (np.cos(-phi) + 1j * np.sin(-phi))
    return (zc_re + 1j * zc_im) + background * notch


#: 中心差分定步长用的参数量级（与 Rust 侧 `jacobian_matches_central_difference` 同一张表）。
JAC_SCALE = np.array([1e9, 1e4, 1e4, 1.0, 1.0, 1e-7, 1e9, 1e9, 1.0, 1e9, 1e9])


def jacobian(freqs: np.ndarray, p: np.ndarray) -> np.ndarray:
    """`src/superconductor/s21/s21.rs` 的 `jacobian` 向量化移植，返回 (n, 11) 复数偏导。

    与 `model` 同一套中间量：隐函数定理给 ∂y/∂p（`d_impl = 1 + 8·ap·y/(1+4y²)²`），
    notch 与背景各自求导，`zc` 两列恒为 1 与 j。这一步是 scipy 能和我们站在同一起跑线
    的前提——两边都用解析雅可比，比的才是"解释器 + numpy 调用"这件外衣。
    """
    fr, ql, qc, theta, ap, tau, a, b, phi, zc_re, zc_im = p
    y = _solve_detuning_y(ql * (freqs - fr) / fr, ap)
    quad = 1.0 + 4.0 * y * y
    d_impl = 1.0 + 8.0 * ap * y / (quad * quad)
    u = 1.0 + 2j * y
    cos_t = np.cos(theta)
    kappa = (ql / qc / cos_t) * (np.cos(theta) + 1j * np.sin(theta))
    res = 1.0 - kappa / u

    arg = 2.0 * np.pi * freqs * tau
    cos_tau, sin_tau = np.cos(arg), np.sin(arg)
    e = np.cos(-phi) + 1j * np.sin(-phi)
    bg = (a * cos_tau - 1j * b * sin_tau) * e

    dy_dfr = -ql * freqs / (fr * fr * d_impl)
    dy_dql = (freqs - fr) / (fr * d_impl)
    dy_dap = 1.0 / (quad * d_impl)

    u2 = u * u
    w = 2.0 * np.pi * freqs
    ones = np.ones_like(freqs, dtype=np.complex128)
    return np.stack(
        [
            bg * (1j * 2.0 * kappa * dy_dfr / u2),
            bg * (-(kappa / ql) / u + 1j * 2.0 * kappa * dy_dql / u2),
            bg * ((kappa / qc) / u),
            bg * (-(1j * ql / (qc * cos_t * cos_t)) / u),
            bg * (1j * 2.0 * kappa * dy_dap / u2),
            (a * (-w * sin_tau) - 1j * b * (w * cos_tau)) * e * res,
            cos_tau * e * res,
            -1j * sin_tau * e * res,
            -1j * bg * res,
            ones,
            1j * ones,
        ],
        axis=1,
    )


def jacobian_error(freqs: np.ndarray, p: np.ndarray) -> float:
    """解析偏导 vs 中心差分的最大相对误差（按列归一），口径同 Rust 侧单测。

    分母取**整列的最大模**，而不是每个元素各自的模：网格上恰好有 `2πfτ = −3312π`
    的点（f = 6.9 GHz），那里 `sin = 0`、真偏导恒等于 0，逐元素相对误差会被 0/0
    放大成 1，量不出任何东西。步长与 Rust 单测同取 1e-7（各列误差随步长呈
    截断↔舍入的 V 形，1e-7 落在谷底附近）。
    """
    ana = jacobian(freqs, p)
    worst = 0.0
    for k in range(p.size):
        h = max(abs(p[k]), JAC_SCALE[k]) * 1e-7
        up, down = p.copy(), p.copy()
        up[k] += h
        down[k] -= h
        num = (model(freqs, up) - model(freqs, down)) / (2.0 * h)
        worst = max(worst, float(np.max(np.abs(ana[:, k] - num))) / float(np.max(np.abs(num))))
    return worst


def build_lines(freqs: np.ndarray, n_lines: int) -> list[np.ndarray]:
    """逐线生成数据：真值参数 + 独立噪声，种子与抽取顺序与 Rust 一致。"""
    lines = []
    for index in range(n_lines):
        lcg = Lcg(0x9E37_79B9_7F4A_7C15 ^ (index << 17))
        noise = np.empty(freqs.size, dtype=np.complex128)
        for k in range(freqs.size):
            # Rust 侧 Complex64::new(re, im) 的实参自左向右求值：先实部后虚部
            noise[k] = complex(4e-3 * lcg.next_unit(), 4e-3 * lcg.next_unit())
        lines.append(model(freqs, line_params(index)) + noise)
    return lines


# ---------------------------------------------------------------- scipy 这一路


def _stacked(z: np.ndarray) -> np.ndarray:
    """复数残差按实虚部堆叠成实向量——与 lmfit 处理复数数据的方式同构。"""
    return np.concatenate((z.real, z.imag))


def _curve(freqs: np.ndarray, *params: float) -> np.ndarray:
    return _stacked(model(freqs, np.asarray(params)))


def _curve_jac(freqs: np.ndarray, *params: float) -> np.ndarray:
    """(102, 11) 实雅可比，与 `_stacked` 同序：先 51 个实部，再 51 个虚部。"""
    d = jacobian(freqs, np.asarray(params))
    return np.concatenate((d.real, d.imag), axis=0)


def fit_one(
    freqs: np.ndarray, data: np.ndarray, p0: np.ndarray, analytic: bool = False
) -> tuple[float, float, int] | None:
    """一次 scipy 拟合，返回 `(fr, ql, nfev)`；不收敛返回 None。

    `analytic=True` 把解析雅可比交给 `curve_fit`——和 Rust 内核同一起跑线。
    """
    try:
        popt, _, info, _, _ = curve_fit(
            _curve,
            freqs,
            _stacked(data),
            p0=p0.tolist(),
            jac=_curve_jac if analytic else None,
            full_output=True,
        )
    except (RuntimeError, ValueError):
        return None
    return float(popt[0]), float(popt[1]), int(info["nfev"])


def _fit_chunk(
    chunk: tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]], analytic: bool = False
) -> tuple[int, int, int]:
    """multiprocessing 的工作单元：一串 `(p0, data)`，返回 (成功数, 失败数, nfev 合计)。"""
    freqs, items = chunk
    outcomes = [fit_one(freqs, data, p0, analytic) for p0, data in items]
    ok = [outcome for outcome in outcomes if outcome is not None]
    return len(ok), len(outcomes) - len(ok), sum(outcome[2] for outcome in ok)


def scipy_serial(
    freqs: np.ndarray,
    lines: list[np.ndarray],
    p0_all: list[np.ndarray],
    analytic: bool = False,
) -> tuple[float, list]:
    """路径 1：串行 curve_fit，返回 (秒, 每条的 (fr, ql, nfev)|None)。"""
    start = time.perf_counter()
    results = [fit_one(freqs, data, p0, analytic) for data, p0 in zip(lines, p0_all)]
    return time.perf_counter() - start, results


def scipy_parallel(
    freqs: np.ndarray,
    lines: list[np.ndarray],
    p0_all: list[np.ndarray],
    workers: int,
    chunks_per_worker: int = 4,
    analytic: bool = False,
) -> tuple[float, int, int, int, float]:
    """路径 2：同一个串行循环铺到进程池（GIL 的绕行方案）。

    返回 `(map 秒, 成功, 失败, nfev 合计, 建池秒)`。**计时只覆盖 `pool.map`**：进程池
    的冷启动在 Windows 上是 spawn，每个子进程要重新 import numpy/scipy，那是另一笔
    账，单独作为第 5 个返回值报出来，不混进吞吐里。

    `chunks_per_worker` 越大块越小：块小负载均衡好、但 IPC/pickle 摊不薄；块大反过来。
    两个极端都测，才知道进程池最好能到哪。
    """
    chunk_count = max(1, workers * chunks_per_worker)
    chunk_size = max(1, len(lines) // chunk_count + 1)
    chunks: list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = []
    for start in range(0, len(lines), chunk_size):
        stop = start + chunk_size
        chunks.append((freqs, list(zip(p0_all[start:stop], lines[start:stop]))))

    pool_start = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        pool_seconds = time.perf_counter() - pool_start
        start = time.perf_counter()
        outcomes = pool.map(partial(_fit_chunk, analytic=analytic), chunks)
        seconds = time.perf_counter() - start
    return (
        seconds,
        sum(ok for ok, _, _ in outcomes),
        sum(failed for _, failed, _ in outcomes),
        sum(nfev for _, _, nfev in outcomes),
        pool_seconds,
    )


# ---------------------------------------------------------------- 输出


def _pad(text: str, width: int) -> str:
    """中文按两列宽算，表格才对齐。"""
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def _per_call(body, repeats: int) -> float:
    """单次调用的平均耗时（µs）：先热身一次再计时。

    量的是**裸开销**——一次 51 点模型扫描、一次 11 列雅可比要花掉多少解释器时间。
    表格里"剩下的差距是解释器"那句话的原始数据就是它（Rust 侧的同名数字由 bench 直接打印）。
    """
    body()
    start = time.perf_counter()
    for _ in range(repeats):
        body()
    return (time.perf_counter() - start) / repeats * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description="S21 批量拟合：Python 三路 vs Rust 内核")
    parser.add_argument("--lines", type=int, default=3000, help="问题条数（默认 3000，与 Rust bench 一致）")
    args = parser.parse_args()
    n_lines = args.lines
    workers = os.cpu_count() or 1
    script_start = time.perf_counter()

    print(f"机器  : {platform.platform()} | {platform.machine()} | 逻辑核 {workers}")
    print(
        f"版本  : Python {platform.python_version()} | numpy {np.__version__}"
        f" | scipy {scipy.__version__} | qtool {qtool.__version__}"
    )

    freqs = freqs_grid()
    build_start = time.perf_counter()
    lines = build_lines(freqs, n_lines)
    p0_all = [line_params(index) for index in range(n_lines)]
    build_seconds = time.perf_counter() - build_start
    print(f"问题数 = {n_lines} 条 × {freqs.size} 频点 | 造数据 {build_seconds:.2f} s")

    # 端口自检：numpy 模型 vs Rust 内核的 model_at；数据由前者生成，一致才谈得上同一批问题
    checks = min(3, n_lines)
    worst_abs = 0.0
    scale = 0.0
    for index in range(checks):
        truth = p0_all[index]
        reference = qtool.s21.S21Model(*truth.tolist()).at(freqs)
        worst_abs = max(worst_abs, float(np.max(np.abs(model(freqs, truth) - reference))))
        scale = max(scale, float(np.max(np.abs(reference))))
    print(
        f"模型对拍（前 {checks} 条，numpy vs Rust model_at）：最大绝对偏差 {worst_abs:.3e}"
        f"（相对 {worst_abs / scale:.3e}）"
    )
    # 解析雅可比自检：与中心差分逐列比对，口径同 Rust 侧 `jacobian_matches_central_difference`
    jac_worst = max(
        jacobian_error(freqs, p0_all[index]) for index in range(checks)
    )
    print(f"雅可比自检（numpy 解析 vs 中心差分）：最大相对误差 {jac_worst:.3e}")
    model_us = _per_call(lambda: model(freqs, p0_all[0]), 20000)
    jac_us = _per_call(lambda: jacobian(freqs, p0_all[0]), 5000)
    print(f"裸开销（numpy）：模型扫描 {model_us:.1f} µs/次，解析雅可比 {jac_us:.1f} µs/次")

    serial_seconds, serial_results = scipy_serial(freqs, lines, p0_all)
    jac_seconds, jac_results = scipy_serial(freqs, lines, p0_all, analytic=True)
    parallel_seconds, parallel_ok, parallel_failed, parallel_nfev, parallel_pool = scipy_parallel(
        freqs, lines, p0_all, workers, analytic=True
    )
    # 进程池的另一种切法：每核一整块。块大 IPC 摊得薄，但负载均衡差——两行一起看，
    # 才知道"绕过 GIL"这条路最好能到多少。
    coarse_seconds, coarse_ok, coarse_failed, coarse_nfev, coarse_pool = scipy_parallel(
        freqs, lines, p0_all, workers, chunks_per_worker=1, analytic=True
    )

    rust_start = time.perf_counter()
    rust_results = qtool.s21.fit_batch(freqs, lines)
    rust_seconds = time.perf_counter() - rust_start
    rust_failed = sum(1 for outcome in rust_results if isinstance(outcome, BaseException))

    serial_failed = sum(1 for outcome in serial_results if outcome is None)

    # 一致性对照：同一条线，scipy 与 Rust 拟合出的 fr/ql 应当落在同一处
    compared = min(checks, n_lines)
    fr_rel = 0.0
    ql_rel = 0.0
    print("\n一致性对照（scipy 串行 vs Rust 内核，同一批数据）")
    for index in range(compared):
        scipy_outcome = serial_results[index]
        rust_outcome = rust_results[index]
        if scipy_outcome is None or isinstance(rust_outcome, BaseException):
            print(f"  第 {index} 条：一侧失败，跳过")
            continue
        fr_s, ql_s, _ = scipy_outcome
        fr_r = rust_outcome.model.fr
        ql_r = rust_outcome.model.ql
        fr_delta = abs(fr_s - fr_r) / abs(fr_r)
        ql_delta = abs(ql_s - ql_r) / abs(ql_r)
        fr_rel = max(fr_rel, fr_delta)
        ql_rel = max(ql_rel, ql_delta)
        print(
            f"  第 {index} 条：Δfr/fr = {fr_delta:.3e}  Δql/ql = {ql_delta:.3e}"
            f"  (fr {fr_s:.6e} vs {fr_r:.6e})"
        )
    print(f"  最大相对差：Δfr/fr = {fr_rel:.3e}，Δql/ql = {ql_rel:.3e}")

    serial_nfev = sum(outcome[2] for outcome in serial_results if outcome is not None)

    # 初值灵敏度对照：同一条量级的抖动（fr 偏 1 MHz ≈ 1.2 个线宽，ql/qc 偏 25~30%，
    # θ 偏 0.3 rad）——即"初值估计器真的在工作、但只给到量级"时的样子。前两行的 p0
    # 是真值，等于白拿了 Rust 侧 estimate() 的活；这里量出那份活值多少。
    control_n = min(300, n_lines)
    control_p0 = []
    for p0 in p0_all[:control_n]:
        jittered = p0.copy()
        jittered[0] += 1.0e6
        jittered[1] *= 0.75
        jittered[2] *= 1.3
        jittered[3] += 0.3
        control_p0.append(jittered)
    control_seconds, control_results = scipy_serial(freqs, lines[:control_n], control_p0)
    control_ok = [outcome for outcome in control_results if outcome is not None]
    control_failed = control_n - len(control_ok)
    control_nfev = sum(outcome[2] for outcome in control_ok)
    control_jac_seconds, control_jac_results = scipy_serial(
        freqs, lines[:control_n], control_p0, analytic=True
    )
    control_jac_ok = [outcome for outcome in control_jac_results if outcome is not None]
    control_jac_failed = control_n - len(control_jac_ok)
    control_jac_nfev = sum(outcome[2] for outcome in control_jac_ok)

    jac_failed = sum(1 for outcome in jac_results if outcome is None)
    jac_nfev = sum(outcome[2] for outcome in jac_results if outcome is not None)

    print("\n【同任务对照 · p0 = 真值】同一起点、同一份数据、只拟合一次（与 Rust bench 同名行逐项可比）")
    print(f"{'路径':<42}{'总耗时':>12}{'ms/条':>12}{'对默认':>9}{'平均 nfev':>11}{'失败':>8}")
    for name, seconds, count, failed, nfev in (
        ("scipy curve_fit 串行（差分，scipy 默认）", serial_seconds, n_lines, serial_failed, serial_nfev),
        ("scipy curve_fit 串行（解析雅可比）", jac_seconds, n_lines, jac_failed, jac_nfev),
        (
            f"scipy curve_fit 进程池 {workers}（每核 4 块，解析）",
            parallel_seconds,
            n_lines,
            parallel_failed,
            parallel_nfev,
        ),
        (
            f"scipy curve_fit 进程池 {workers}（每核 1 块，解析）",
            coarse_seconds,
            n_lines,
            coarse_failed,
            coarse_nfev,
        ),
    ):
        print(
            f"{_pad(name, 42)}{seconds:>11.2f}s{seconds / count * 1e3:>12.3f}"
            f"{serial_seconds / n_lines / (seconds / count):>8.1f}×"
            f"{nfev / max(count - failed, 1):>11.0f}{failed:>8}"
        )

    print(f"\n【同任务对照 · p0 = 抖动】同上，但起点换成估计器量级（前 {control_n} 条）")
    print(f"{'路径':<42}{'总耗时':>12}{'ms/条':>12}{'对差分':>9}{'平均 nfev':>11}{'失败':>8}")
    for name, seconds, failed, nfev in (
        ("scipy curve_fit 串行（差分，scipy 默认）", control_seconds, control_failed, control_nfev),
        ("scipy curve_fit 串行（解析雅可比）", control_jac_seconds, control_jac_failed, control_jac_nfev),
    ):
        print(
            f"{_pad(name, 42)}{seconds:>11.2f}s{seconds / control_n * 1e3:>12.3f}"
            f"{control_seconds / seconds:>8.1f}×"
            f"{nfev / max(control_n - failed, 1):>11.0f}{failed:>8}"
        )

    print("\n【全流程对照】qtool 一路：每条线做初值估计 + 最多 8 个候选各拟合一次")
    print(f"{'路径':<42}{'总耗时':>12}{'ms/条':>12}{'失败':>8}")
    print(
        f"{_pad('qtool.s21.fit_batch（Rust/rayon，全流程）', 42)}"
        f"{rust_seconds:>11.2f}s{rust_seconds / n_lines * 1e3:>12.3f}{rust_failed:>8}"
    )
    print(
        f"\n注：同任务对照里的 scipy 每条线只拟合 1 次；全流程对照里的 qtool 每条线拟合"
        f"最多 8 次且带初值估计，不要跨这两档比。scipy 没有与 estimate() 对应的入口，"
        f"\n所以全流程这一档只有 qtool 一行；它的公平性由同任务对照决定——"
        f"同样起点下内核的快慢，乘上每条线的拟合次数就是全流程的差距。"
        f"\n解析雅可比是手写的 11 列偏导（与 Rust 侧 `jacobian` 同一份数学，逐列中心差分自检见上）；"
        f"拿它和 qtool 比才是同一条起跑线，差分那一行是 scipy 的默认设置。"
        f"\n进程池两行的计时只含 pool.map；建池（Windows spawn，子进程要重 import numpy/scipy）另计："
        f"每核 4 块 {parallel_pool:.2f} s、每核 1 块 {coarse_pool:.2f} s——这一笔要么摊在更长的工作上，要么就白花。"
        f"\n脚本总墙钟 = {time.perf_counter() - script_start:.1f} s（含造数据、进程池启动与对照）"
    )


if __name__ == "__main__":
    main()
