//! T2*（Ramsey）：`qtool::superconductor::ramsey` 的绑定。
//!
//! 对外四个名字：模型 `CosDamp`、结果 `RamseyFit`、错误 `RamseyError`，以及 `fit` / `fit_batch` / `plot`。

use crate::{PyStateCenters, QtoolError};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::ramsey::ramsey_plot::ramsey_plot_div;
use qtool::superconductor::ramsey::{
    CosDamp, RamseyError as CoreRamseyError, RamseyFit as CoreRamseyFit, ramsey_fit,
    ramsey_fit_batch,
};

pyo3::create_exception!(_qtool, RamseyError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreRamseyError) -> PyErr {
    RamseyError::new_err(error.to_string())
}

/// 阻尼余弦线型的参数：常数项 + 指数包络的余弦。
///
/// 与 rabi 的 `Cos` 不同，这里**五个参数全自由**：起点电平由 `offset` 承担，相位与衰减也都要拟。
#[pyclass(name = "CosDamp", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCosDamp {
    pub inner: CosDamp,
}

#[pymethods]
impl PyCosDamp {
    #[new]
    fn new(offset: f64, amplitude: f64, frequency: f64, phase: f64, decay: f64) -> Self {
        Self {
            inner: CosDamp {
                offset,
                amplitude,
                frequency,
                phase,
                decay,
            },
        }
    }

    /// 常数项：条纹衰减完之后剩下的电平
    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }

    /// 振荡项的峰值幅度（带符号）
    #[getter]
    fn amplitude(&self) -> f64 {
        self.inner.amplitude
    }

    /// 条纹频率 (Hz)
    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    /// 条纹相位 (rad)
    #[getter]
    fn phase(&self) -> f64 {
        self.inner.phase
    }

    /// 包络的衰减时间常数 (s)，**即 T2\***
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// 在给定延时网格上求值。
    fn at<'py>(&self, py: Python<'py>, taus: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&taus).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CosDamp(offset={}, amplitude={}, frequency={}, phase={}, decay={})",
            self.inner.offset,
            self.inner.amplitude,
            self.inner.frequency,
            self.inner.phase,
            self.inner.decay
        )
    }
}

/// 单条延时扫描的拟合结果。
#[pyclass(name = "RamseyFit", skip_from_py_object)]
pub struct PyRamseyFit {
    pub inner: CoreRamseyFit,
}

#[pymethods]
impl PyRamseyFit {
    #[getter]
    fn model(&self) -> PyCosDamp {
        PyCosDamp {
            inner: self.inner.result.model,
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
            "RamseyFit(decay={:.6e}, frequency={:.6e}, redchi={:.3})",
            self.inner.result.model.decay, self.inner.result.model.frequency, self.inner.result.redchi
        )
    }
}

/// 拟合一条 Ramsey 延时扫描。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None))]
fn fit(
    taus: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyRamseyFit> {
    let fitted = ramsey_fit(
        &taus,
        &iq,
        states.as_ref().map(|item| &item.inner),
        sigma.as_deref(),
    )
    .map_err(|error| to_py_error(&error))?;
    Ok(PyRamseyFit { inner: fitted })
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
    ramsey_fit_batch(&taus, &iqs, states, sigmas.as_deref())
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyRamseyFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => RamseyError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect()
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`RamseyFit`（叠拟合曲线）、`None`（只画数据）、或一条 `RamseyError` 实例——
/// 后者会被**原样抛出**。
#[pyfunction]
#[pyo3(signature = (taus, iq, states=None, sigma=None, fit=None, div_id="ramsey", frame=None))]
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
        Some(value) => match value.extract::<PyRef<'_, PyRamseyFit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a RamseyFit, a RamseyError instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => ramsey_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Ok(result),
            div_id,
            frame,
        ),
        None => ramsey_plot_div(
            &taus,
            &iq,
            states,
            sigma.as_deref(),
            Err(&CoreRamseyError::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RamseyError", py.get_type::<RamseyError>())?;
    module.add_class::<PyCosDamp>()?;
    module.add_class::<PyRamseyFit>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
