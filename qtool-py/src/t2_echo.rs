//! T2 echo（回波退相位）：`qtool::superconductor::t2_echo` 的绑定。
//!
//! 对外四个名字：模型 `EchoDecay`、结果 `T2EchoFit`、错误 `T2EchoError`，以及 `fit` / `fit_batch` / `plot`。

use crate::{
    PyStateCenters, QtoolError, batch_sigmas_arg, batch_states_arg, per_line_sigmas,
    per_line_states,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::t2_echo::t2_echo_plot::t2_echo_plot_div;
use qtool::superconductor::t2_echo::{
    EchoDecay, T2EchoError as CoreT2EchoError, T2EchoFit as CoreT2EchoFit, t2_echo_fit,
    t2_echo_fit_batch,
};

pyo3::create_exception!(_qtool, T2EchoError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreT2EchoError) -> PyErr {
    T2EchoError::new_err(error.to_string())
}

/// 带本底的指数衰减线型。
///
/// 三个参数全自由（只钉时间常数的下界），曲线朝上还是朝下由 `amplitude` 的符号承担。
#[pyclass(name = "EchoDecay", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEchoDecay {
    pub inner: EchoDecay,
}

#[pymethods]
impl PyEchoDecay {
    #[new]
    fn new(offset: f64, amplitude: f64, t2_echo: f64) -> Self {
        Self {
            inner: EchoDecay {
                offset,
                amplitude,
                t2_echo,
            },
        }
    }

    /// 长延时下的本底
    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }

    /// 衰减项的幅度（带符号）
    #[getter]
    fn amplitude(&self) -> f64 {
        self.inner.amplitude
    }

    /// 回波退相位时间 (s)
    #[getter]
    fn t2_echo(&self) -> f64 {
        self.inner.t2_echo
    }

    /// 在给定延时网格上求值。
    fn at<'py>(&self, py: Python<'py>, taus: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&taus).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "EchoDecay(offset={}, amplitude={}, t2_echo={})",
            self.inner.offset, self.inner.amplitude, self.inner.t2_echo
        )
    }
}

/// 单条回波扫描的拟合结果。
#[pyclass(name = "T2EchoFit", skip_from_py_object)]
pub struct PyT2EchoFit {
    pub inner: CoreT2EchoFit,
}

#[pymethods]
impl PyT2EchoFit {
    #[getter]
    fn model(&self) -> PyEchoDecay {
        PyEchoDecay {
            inner: self.inner.result.model,
        }
    }

    /// 实际参与拟合的那条 P1；朝向原样交付，不做翻转。
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
            "T2EchoFit(t2_echo={:.6e}, redchi={:.3})",
            self.inner.result.model.t2_echo, self.inner.result.redchi
        )
    }
}

/// 拟合一条回波扫描。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None))]
fn fit(
    taus: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyT2EchoFit> {
    let fitted = t2_echo_fit(
        &taus,
        &iq,
        states.as_ref().map(|item| &item.inner),
        sigma.as_deref(),
    )
    .map_err(|error| to_py_error(&error))?;
    Ok(PyT2EchoFit { inner: fitted })
}

/// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）——`isinstance` 判得出来、
/// `str()` 有消息、还能直接喂回 [`plot`]（后者会把它原样抛出）。
#[pyfunction]
#[pyo3(signature = (taus, iqs, states=None, sigmas=None))]
fn fit_batch(
    py: Python<'_>,
    taus: Vec<f64>,
    iqs: Vec<Vec<lmfit::Complex64>>,
    states: Option<&Bound<'_, PyAny>>,
    sigmas: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let lines = iqs.len();
    let centers = per_line_states(states, lines)?;
    let sigmas = per_line_sigmas(sigmas, lines)?;
    let outcome = t2_echo_fit_batch(
        &taus,
        &iqs,
        batch_states_arg(&centers).as_deref(),
        batch_sigmas_arg(&sigmas),
    );
    Ok(outcome
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyT2EchoFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => T2EchoError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect())
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`T2EchoFit`（叠拟合曲线）、`None`（只画数据）、或一条 `T2EchoError` 实例——
/// 后者会被**原样抛出**。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None, fit=None, div_id="t2-echo", frame=None))]
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
        Some(value) => match value.extract::<PyRef<'_, PyT2EchoFit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a T2EchoFit, a T2EchoError instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => t2_echo_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Ok(result),
            div_id,
            frame,
        ),
        None => t2_echo_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Err(&CoreT2EchoError::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("T2EchoError", py.get_type::<T2EchoError>())?;
    module.add_class::<PyEchoDecay>()?;
    module.add_class::<PyT2EchoFit>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
