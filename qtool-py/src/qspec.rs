//! 比特谱（qspec）：`qtool::superconductor::qspec` 的绑定。
//!
//! 对外名字：模型 `Lorentz`、结果 `QspecFit`、Z 扫描的逐行输入 `QspecZLine`、错误 `QspecError`，
//! 以及 `fit` / `fit_batch` / `plot` / `z_plot` 四个函数。

use crate::{PyStateCenters, QtoolError};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::qspec::qspec_plot::qspec_fit_plot_div;
use qtool::superconductor::qspec::qspec_z_plot::{QspecZLine, qspec_z_plot_div};
use qtool::superconductor::qspec::{
    Lorentz, QspecError as CoreQspecError, QspecFit as CoreQspecFit, qspec_fit, qspec_fit_batch,
};

pyo3::create_exception!(_qtool, QspecError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreQspecError) -> PyErr {
    QspecError::new_err(error.to_string())
}

/// 洛伦兹线型的参数（SI 单位）。
///
/// 线型只以 `fwhm²` 出现，正负给出同一条曲线；初值候选恒取正宽度，故拟合结果通常为正。
#[pyclass(name = "Lorentz", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLorentz {
    pub inner: Lorentz,
}

#[pymethods]
impl PyLorentz {
    #[new]
    fn new(fq: f64, fwhm: f64, amp: f64, offset: f64) -> Self {
        Self {
            inner: Lorentz {
                fq,
                fwhm,
                amp,
                offset,
            },
        }
    }

    /// 峰中心（比特频率），Hz
    #[getter]
    fn fq(&self) -> f64 {
        self.inner.fq
    }

    /// 半高全宽，Hz
    #[getter]
    fn fwhm(&self) -> f64 {
        self.inner.fwhm
    }

    /// 峰高（峰值相对基线）
    #[getter]
    fn amp(&self) -> f64 {
        self.inner.amp
    }

    /// 远离共振处的基线电平
    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }

    /// 在给定频率网格上求值。
    fn at<'py>(&self, py: Python<'py>, freqs: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&freqs).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Lorentz(fq={}, fwhm={}, amp={}, offset={})",
            self.inner.fq, self.inner.fwhm, self.inner.amp, self.inner.offset
        )
    }
}

/// qspec 单条谱线的拟合结果。
#[pyclass(name = "QspecFit", from_py_object)]
#[derive(Clone)]
pub struct PyQspecFit {
    pub inner: CoreQspecFit,
}

#[pymethods]
impl PyQspecFit {
    #[getter]
    fn model(&self) -> PyLorentz {
        PyLorentz {
            inner: self.inner.result.model,
        }
    }

    /// 实际参与拟合的那条 P1；朝向原样交付，自估路径下可能是谷（`amp` 为负）。
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
            "QspecFit(fq={:.6e}, fwhm={:.6e}, redchi={:.3})",
            self.inner.result.model.fq, self.inner.result.model.fwhm, self.inner.result.redchi
        )
    }
}

/// 一条 Z 偏置处的 qspec 谱线：偏置 + 数据 + 该处的洛伦兹拟合。
///
/// `fit=None` 表示这一行没拟合（报告里的该行退化成只用数据算出来的 P1），与 Rust 侧 `Err` 同一个出口。
/// `freqs=None` 表示该行与其余行共用 `z_plot` 的公共轴；动窗扫描给每行自己的轴。
#[pyclass(name = "QspecZLine", from_py_object)]
#[derive(Clone)]
pub struct PyQspecZLine {
    z: f64,
    iq: Vec<lmfit::Complex64>,
    fit: Option<CoreQspecFit>,
    freqs: Option<Vec<f64>>,
}

#[pymethods]
impl PyQspecZLine {
    #[new]
    #[pyo3(signature = (z, iq, fit=None, freqs=None))]
    fn new(
        z: f64,
        iq: Vec<lmfit::Complex64>,
        fit: Option<PyQspecFit>,
        freqs: Option<Vec<f64>>,
    ) -> Self {
        Self {
            z,
            iq,
            fit: fit.map(|value| value.inner),
            freqs,
        }
    }

    /// Z 线偏置，V
    #[getter]
    fn z(&self) -> f64 {
        self.z
    }

    fn __repr__(&self) -> String {
        format!(
            "QspecZLine(z={}, points={}, fitted={})",
            self.z,
            self.iq.len(),
            self.fit.is_some()
        )
    }
}

/// 拟合一条比特谱线。
#[pyfunction]
#[pyo3(signature = (freqs, iq, states=None, sigma=None))]
fn fit(
    freqs: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyQspecFit> {
    let fitted = qspec_fit(
        &freqs,
        &iq,
        states.as_ref().map(|item| &item.inner),
        sigma.as_deref(),
    )
    .map_err(|error| to_py_error(&error))?;
    Ok(PyQspecFit { inner: fitted })
}

/// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）——`isinstance` 判得出来、
/// `str()` 有消息、还能直接喂回 [`plot`]（后者会把它原样抛出）。
#[pyfunction]
#[pyo3(signature = (freqs, iqs, states=None, sigmas=None))]
fn fit_batch(
    py: Python<'_>,
    freqs: Vec<f64>,
    iqs: Vec<Vec<lmfit::Complex64>>,
    states: Option<PyStateCenters>,
    sigmas: Option<Vec<Vec<f64>>>,
) -> Vec<Py<PyAny>> {
    let states = states.as_ref().map(|item| &item.inner);
    qspec_fit_batch(&freqs, &iqs, states, sigmas.as_deref())
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyQspecFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => QspecError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect()
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`QspecFit`（叠拟合曲线）、`None`（只画数据）、或一条 `QspecError` 实例——
/// 后者会被**原样抛出**。
#[pyfunction]
#[pyo3(signature = (freqs, iq, states=None, sigma=None, fit=None, div_id="qspec", frame=None))]
fn plot(
    freqs: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    states: Option<PyStateCenters>,
    sigma: Option<Vec<f64>>,
    fit: Option<&Bound<'_, PyAny>>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let states = states.as_ref().map(|item| &item.inner);
    let fitted = match fit {
        Some(value) => match value.extract::<PyRef<'_, PyQspecFit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a QspecFit, a QspecError instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => qspec_fit_plot_div(
            &freqs,
            &iq,
            states,
            sigma.as_deref(),
            Ok(result),
            div_id,
            frame,
        ),
        None => qspec_fit_plot_div(
            &freqs,
            &iq,
            states,
            sigma.as_deref(),
            Err(&CoreQspecError::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 渲染「P1 vs Z」报告 div：逐行给 `QspecZLine`，每行自带偏置与该行的拟合。
#[pyfunction]
#[pyo3(signature = (freqs, lines, states=None, div_id="qspec-z", frame=None))]
fn z_plot(
    freqs: Vec<f64>,
    lines: Vec<PyQspecZLine>,
    states: Option<PyStateCenters>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let states = states.as_ref().map(|item| &item.inner);
    let borrowed: Vec<QspecZLine<'_>> = lines
        .iter()
        .map(|line| QspecZLine {
            z: line.z,
            iq: &line.iq,
            fit: match &line.fit {
                Some(result) => Ok(result),
                None => Err(&CoreQspecError::EmptyData),
            },
            freqs: match &line.freqs {
                Some(axis) => Some(axis.as_slice()),
                None => None,
            },
        })
        .collect();
    Ok(qspec_z_plot_div(&freqs, &borrowed, states, div_id, frame))
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("QspecError", py.get_type::<QspecError>())?;
    module.add_class::<PyLorentz>()?;
    module.add_class::<PyQspecFit>()?;
    module.add_class::<PyQspecZLine>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    module.add_function(wrap_pyfunction!(z_plot, module)?)?;
    Ok(())
}
