//! qtool 的 Python 扩展模块（pyo3）。
//!
//! 只包**公开 API**：入口函数、模型、拟合结果、错误。`pub(crate)` 的实现细节、绘图几何辅助、
//! qpt 那一块都不在这里。模块路径照抄 Rust（`qtool.t1`、`qtool.drag.coeff`…），函数名去掉
//! 模块前缀（`qtool::superconductor::t1::t1_fit` → `qtool.t1.fit`）。
//!
//! 数组边界按 numpy 走：大数组（IQ、曲线）进出是 `ndarray`，参数与标量是原生类型。

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::{StateCenters, direction, p1, p1_sigma};

pub mod bloch;
pub mod drag;
pub mod iq;
pub mod qspec;
pub mod rabi;
pub mod ramsey;
pub mod s21;
pub mod t1;
pub mod t2_echo;

pyo3::create_exception!(_qtool, QtoolError, pyo3::exceptions::PyException);

/// 各态标定中心（`qtool::superconductor::StateCenters`）。
#[pyclass(name = "StateCenters", from_py_object)]
#[derive(Clone)]
pub struct PyStateCenters {
    inner: StateCenters,
}

#[pymethods]
impl PyStateCenters {
    #[new]
    fn new(centers: Vec<lmfit::Complex64>) -> Self {
        Self {
            inner: StateCenters::new(centers),
        }
    }

    /// 各态中心的切片，索引即态编号。
    fn centers(&self) -> Vec<lmfit::Complex64> {
        self.inner.as_slice().to_vec()
    }

    fn __len__(&self) -> usize {
        self.inner.as_slice().len()
    }

    fn __repr__(&self) -> String {
        format!("StateCenters({:?})", self.inner.as_slice())
    }
}

/// IQ → 激发概率。
#[pyfunction]
#[pyo3(signature = (iq, states=None))]
fn p1_rs<'py>(
    py: Python<'py>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
) -> Bound<'py, PyArray1<f64>> {
    p1(&iq, states.as_ref().map(|item| &item.inner)).into_pyarray(py)
}

/// IQ 域的逐点 σ → P1 域的 σ。
#[pyfunction]
#[pyo3(signature = (iq, states, sigma))]
fn p1_sigma_rs<'py>(
    py: Python<'py>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Vec<f64>,
) -> Bound<'py, PyArray1<f64>> {
    p1_sigma(&iq, states.as_ref().map(|item| &item.inner), &sigma).into_pyarray(py)
}

/// 复平面点云的重心与单位主轴方向。
#[pyfunction]
fn direction_rs(iq: Vec<lmfit::Complex64>) -> (lmfit::Complex64, lmfit::Complex64) {
    direction(&iq)
}

/// 报告页要自己挂的那行 plotly.js 标签（URL 的唯一来源是 Rust 侧常量）。
#[pyfunction]
fn plotly_js_cdn() -> &'static str {
    qtool::superconductor::s21::s21_plot::PLOTLY_JS_CDN
}

/// 批量入口的 `states`：单个 `StateCenters`（广播到每一行）或与行数等长的序列。
///
/// 形参:
///     states: Python 侧传进来的对象；None 表示全部未标定
///     lines: 批量的行数
///
/// 返回值:
///     逐线的一串各态标定中心；None 给空表。既不是单个也不是序列、或序列份数对不上行数时
///     抛 [`QtoolError`]
pub(crate) fn per_line_states(
    states: Option<&Bound<'_, PyAny>>,
    lines: usize,
) -> PyResult<Vec<PyStateCenters>> {
    let object = match states {
        Some(object) => object,
        None => return Ok(Vec::new()),
    };
    match object.extract::<PyStateCenters>() {
        Ok(single) => Ok(vec![single; lines]),
        Err(_not_single) => {
            let list: Vec<PyStateCenters> = object.extract().map_err(|_invalid| {
                QtoolError::new_err("states must be a StateCenters or a sequence of StateCenters")
            })?;
            match list.len() == lines {
                true => Ok(list),
                false => Err(QtoolError::new_err(format!(
                    "states has {} entries but the batch has {lines} lines",
                    list.len()
                ))),
            }
        }
    }
}

/// 批量入口的 `sigmas`：单个一维数组（广播到每一行）或与行数等长的序列。
///
/// 形参:
///     sigmas: Python 侧传进来的对象；None 表示全部不加权
///     lines: 批量的行数
///
/// 返回值:
///     逐线的一串逐点不确定度；None 给空表。既不是单个数组也不是序列、或序列份数对不上行数
///     时抛 [`QtoolError`]
pub(crate) fn per_line_sigmas(
    sigmas: Option<&Bound<'_, PyAny>>,
    lines: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let object = match sigmas {
        Some(object) => object,
        None => return Ok(Vec::new()),
    };
    match object.extract::<Vec<f64>>() {
        Ok(single) => Ok(vec![single; lines]),
        Err(_not_single) => {
            let list: Vec<Vec<f64>> = object.extract().map_err(|_invalid| {
                QtoolError::new_err("sigmas must be an array or a sequence of arrays")
            })?;
            match list.len() == lines {
                true => Ok(list),
                false => Err(QtoolError::new_err(format!(
                    "sigmas has {} entries but the batch has {lines} lines",
                    list.len()
                ))),
            }
        }
    }
}

/// 逐线的 σ 数组转成 Rust 批量入口的形状；空表给 `None`（全部不加权）。
///
/// 形参:
///     sigmas: [`per_line_sigmas`] 归一出来的逐线不确定度
///
/// 返回值:
///     `Some(逐线切片)`；`sigmas` 为空表时给 None
pub(crate) fn batch_sigmas_arg(sigmas: &[Vec<f64>]) -> Option<&[Vec<f64>]> {
    match sigmas.is_empty() {
        true => None,
        false => Some(sigmas),
    }
}

/// 逐线的各态标定中心转成逐线引用（Rust 侧批量入口的形状）；空表给 `None`（逐条自估）。
///
/// 形参:
///     states: [`per_line_states`] 归一出来的逐线中心
///
/// 返回值:
///     `Some(逐线引用)`；`states` 为空表时给 None
pub(crate) fn batch_states_arg(states: &[PyStateCenters]) -> Option<Vec<&StateCenters>> {
    match states.is_empty() {
        true => None,
        false => Some(states.iter().map(|item| &item.inner).collect()),
    }
}

/// native 模块 `qtool._qtool`：顶层放共用词汇，各实验按子模块挂上去。
#[pymodule]
fn _qtool(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("QtoolError", py.get_type::<QtoolError>())?;
    m.add_class::<PyStateCenters>()?;
    m.add_function(wrap_pyfunction!(p1_rs, m)?)?;
    m.add_function(wrap_pyfunction!(p1_sigma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(direction_rs, m)?)?;
    m.add_function(wrap_pyfunction!(plotly_js_cdn, m)?)?;

    let submodule = PyModule::new(py, "t1")?;
    t1::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "t2_echo")?;
    t2_echo::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "ramsey")?;
    ramsey::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "qspec")?;
    qspec::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "rabi")?;
    rabi::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "s21")?;
    s21::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "iq")?;
    iq::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "bloch")?;
    bloch::register(py, &submodule)?;
    m.add_submodule(&submodule)?;

    let submodule = PyModule::new(py, "drag")?;
    let nested = PyModule::new(py, "amplitude")?;
    drag::amplitude::register(py, &nested)?;
    submodule.add_submodule(&nested)?;
    let nested = PyModule::new(py, "coeff")?;
    drag::coeff::register(py, &nested)?;
    submodule.add_submodule(&nested)?;
    let nested = PyModule::new(py, "detuning")?;
    drag::detuning::register(py, &nested)?;
    submodule.add_submodule(&nested)?;
    m.add_submodule(&submodule)?;
    Ok(())
}
