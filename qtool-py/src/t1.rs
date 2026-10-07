//! T1（能量弛豫）：`qtool::superconductor::t1` 的绑定。
//!
//! 对外四个名字：模型 `Decay`、结果 `T1Fit`、错误 `T1Error`，以及 `fit` / `fit_batch` / `plot`。

use crate::{PyStateCenters, QtoolError};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::t1::t1_plot::t1_plot_div;
use qtool::superconductor::t1::{
    Decay, T1Error as CoreT1Error, T1Fit as CoreT1Fit, t1_fit, t1_fit_batch,
};

pyo3::create_exception!(_qtool, T1Error, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreT1Error) -> PyErr {
    T1Error::new_err(error.to_string())
}

/// 三参数指数模型：`offset + amplitude·exp(−τ/t1)`。
#[pyclass(name = "Decay", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDecay {
    pub inner: Decay,
}

#[pymethods]
impl PyDecay {
    #[new]
    fn new(offset: f64, amplitude: f64, t1: f64) -> Self {
        Self {
            inner: Decay {
                offset,
                amplitude,
                t1,
            },
        }
    }

    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }

    #[getter]
    fn amplitude(&self) -> f64 {
        self.inner.amplitude
    }

    #[getter]
    fn t1(&self) -> f64 {
        self.inner.t1
    }

    /// 在给定延时网格上求值。
    fn at<'py>(&self, py: Python<'py>, taus: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&taus).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Decay(offset={}, amplitude={}, t1={})",
            self.inner.offset, self.inner.amplitude, self.inner.t1
        )
    }
}

/// 一条 T1 扫描的拟合结果。
#[pyclass(name = "T1Fit", skip_from_py_object)]
pub struct PyT1Fit {
    pub inner: CoreT1Fit,
}

#[pymethods]
impl PyT1Fit {
    #[getter]
    fn model(&self) -> PyDecay {
        PyDecay {
            inner: self.inner.result.model.clone(),
        }
    }

    /// 实际参与拟合的那条 P1。
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
            "T1Fit(t1={:.6e}, redchi={:.3})",
            self.inner.result.model.t1, self.inner.result.redchi
        )
    }
}

/// 拟合一条 T1 弛豫扫描。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None))]
fn fit(
    taus: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyT1Fit> {
    let fitted = t1_fit(
        &taus,
        &iq,
        states.as_ref().map(|item| &item.inner),
        sigma.as_deref(),
    )
    .map_err(|error| to_py_error(&error))?;
    Ok(PyT1Fit { inner: fitted })
}

/// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）——`isinstance` 判得出来、
/// `str()` 有消息、还能直接喂回 [`plot`]（后者会把它原样抛出）。
#[pyfunction]
#[pyo3(signature = (taus, iqs, states=None, sigmas=None))]
fn fit_batch(
    py: Python<'_>,
    taus: Vec<f64>,
    iqs: Vec<Vec<lmfit::Complex64>>,
    states: Option<PyStateCenters>,
    sigmas: Option<Vec<Vec<f64>>>,
) -> Vec<Py<PyAny>> {
    let states = states.as_ref().map(|item| &item.inner);
    t1_fit_batch(&taus, &iqs, states, sigmas.as_deref())
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyT1Fit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => T1Error::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect()
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`T1Fit`（叠拟合曲线）、`None`（只画数据）、或一条 `T1Error` 实例——
/// 后者会被**原样抛出**（失败的拟合画出来只会是一张空的错误卡，直接抛更符合 Python 的预期）。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None, fit=None, div_id="t1", frame=None))]
fn plot(
    taus: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
    fit: Option<&Bound<'_, PyAny>>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let states = states.as_ref().map(|item| &item.inner);
    let fitted = match fit {
        Some(value) => match value.extract::<PyRef<'_, PyT1Fit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a T1Fit, a T1Error instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => t1_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Ok(result),
            div_id,
            frame,
        ),
        None => t1_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Err(&CoreT1Error::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("T1Error", py.get_type::<T1Error>())?;
    module.add_class::<PyDecay>()?;
    module.add_class::<PyT1Fit>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
