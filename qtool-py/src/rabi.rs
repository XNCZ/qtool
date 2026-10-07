//! Rabi 幅度扫描：`qtool::superconductor::rabi` 的绑定。
//!
//! 对外四个名字：模型 `Cos`、结果 `RabiFit`、错误 `RabiError`，以及 `fit` / `fit_batch` / `plot`。

use crate::{PyStateCenters, QtoolError};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::rabi::rabi_amp_plot::rabi_amp_plot_div;
use qtool::superconductor::rabi::{
    Cos, RabiError as CoreRabiError, RabiFit as CoreRabiFit, rabi_amp_fit, rabi_amp_fit_batch,
};

pyo3::create_exception!(_qtool, RabiError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreRabiError) -> PyErr {
    RabiError::new_err(error.to_string())
}

/// Rabi 余弦线型的参数。
///
/// 线型没有自由常数项：零驱动处的 |0> 态把起点钉在 0，这既是物理锚点，也是取向判据的来源。
#[pyclass(name = "Cos", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCos {
    pub inner: Cos,
}

#[pymethods]
impl PyCos {
    #[new]
    fn new(freq: f64, amp: f64) -> Self {
        Self {
            inner: Cos::new(freq, amp),
        }
    }

    /// 振荡频率，单位是幅度的倒数
    #[getter]
    fn freq(&self) -> f64 {
        self.inner.freq
    }

    /// 半幅（峰值的一半）
    #[getter]
    fn amp(&self) -> f64 {
        self.inner.amp
    }

    /// π 脉冲幅（派生量：`1/(2·freq)`）
    #[getter]
    fn a_pi(&self) -> f64 {
        self.inner.a_pi
    }

    /// 在给定幅度网格上求值。
    fn at<'py>(&self, py: Python<'py>, amps: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&amps).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Cos(freq={}, amp={}, a_pi={})",
            self.inner.freq, self.inner.amp, self.inner.a_pi
        )
    }
}

/// 单条幅度扫描的拟合结果。
#[pyclass(name = "RabiFit", skip_from_py_object)]
pub struct PyRabiFit {
    pub inner: CoreRabiFit,
}

#[pymethods]
impl PyRabiFit {
    #[getter]
    fn model(&self) -> PyCos {
        PyCos {
            inner: self.inner.result.model,
        }
    }

    /// 实际参与拟合的那条 P1（始终朝对的那条）。
    #[getter]
    fn p1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.p1.clone().into_pyarray(py)
    }

    /// 参数名 → `(值, 标准误)`；协方差不可用时标准误是 None。
    #[getter]
    fn params(&self) -> Vec<(String, (f64, Option<f64>))> {
        self.inner
            .result
            .params
            .iter()
            .map(|parameter| {
                (
                    parameter.name.clone(),
                    (
                        parameter.value,
                        parameter.stderr.filter(|value| value.is_finite()),
                    ),
                )
            })
            .collect()
    }

    #[getter]
    fn chisqr(&self) -> f64 {
        self.inner.result.chisqr
    }

    #[getter]
    fn redchi(&self) -> f64 {
        self.inner.result.redchi
    }

    #[getter]
    fn nfev(&self) -> usize {
        self.inner.result.nfev
    }

    #[getter]
    fn success(&self) -> bool {
        self.inner.result.success
    }

    fn __repr__(&self) -> String {
        format!(
            "RabiFit(a_pi={:.6e}, freq={:.6e}, redchi={:.3})",
            self.inner.result.model.a_pi, self.inner.result.model.freq, self.inner.result.redchi
        )
    }
}

/// 拟合一条 Rabi 幅度扫描。
#[pyfunction]
#[pyo3(signature = (amps, iq, states=None, sigma=None))]
fn fit(
    amps: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyRabiFit> {
    let fitted = rabi_amp_fit(
        &amps,
        &iq,
        states.as_ref().map(|item| &item.inner),
        sigma.as_deref(),
    )
    .map_err(|error| to_py_error(&error))?;
    Ok(PyRabiFit { inner: fitted })
}

/// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）——`isinstance` 判得出来、
/// `str()` 有消息、还能直接喂回 [`plot`]（后者会把它原样抛出）。
#[pyfunction]
#[pyo3(signature = (amps, iqs, states=None, sigmas=None))]
fn fit_batch(
    py: Python<'_>,
    amps: Vec<f64>,
    iqs: Vec<Vec<lmfit::Complex64>>,
    states: Option<PyStateCenters>,
    sigmas: Option<Vec<Vec<f64>>>,
) -> Vec<Py<PyAny>> {
    let states = states.as_ref().map(|item| &item.inner);
    rabi_amp_fit_batch(&amps, &iqs, states, sigmas.as_deref())
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyRabiFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => RabiError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect()
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`RabiFit`（叠拟合曲线）、`None`（只画数据）、或一条 `RabiError` 实例——
/// 后者会被**原样抛出**。
#[pyfunction]
#[pyo3(signature = (amps, iq, states=None, sigma=None, fit=None, div_id="rabi", frame=None))]
fn plot(
    amps: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
    fit: Option<&Bound<'_, PyAny>>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let states = states.as_ref().map(|item| &item.inner);
    let fitted = match fit {
        Some(value) => match value.extract::<PyRef<'_, PyRabiFit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a RabiFit, a RabiError instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => rabi_amp_plot_div(
            &amps,
            &iq,
            states,
            sigma.as_deref(),
            Ok(result),
            div_id,
            frame,
        ),
        None => rabi_amp_plot_div(
            &amps,
            &iq,
            states,
            sigma.as_deref(),
            Err(&CoreRabiError::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RabiError", py.get_type::<RabiError>())?;
    module.add_class::<PyCos>()?;
    module.add_class::<PyRabiFit>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
