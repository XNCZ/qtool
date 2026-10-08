//! 比特谱（qspec）：`qtool::superconductor::qspec` 的绑定。
//!
//! 对外名字：模型 `Lorentz`、结果 `QspecFit`、Z 扫描的逐行输入 `QspecZLine`、错误 `QspecError`，
//! 以及 `fit` / `fit_batch` / `plot` / `z_plot` 四个函数。

use crate::{
    PyStateCenters, QtoolError, batch_sigmas_arg, batch_states_arg, per_line_sigmas,
    per_line_states,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::qspec::qspec_plot::qspec_fit_plot_div;
use qtool::superconductor::qspec::qspec_z_plot::{QspecZLine, qspec_z_plot_div};
use qtool::superconductor::qspec::flux::{
    Flux as CoreFlux, FluxFit as CoreFluxFit, FluxLine as CoreFluxLine, flux_fit, flux_fit_batch,
};
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

/// 通量调谐线型（f01 vs Z）的参数（SI 单位）。
///
/// 前三个是拟合参数，`eta` 与 `asymmetry` 是版图给定的结构常数（固定不拟合，随结果带回）。
#[pyclass(name = "Flux", skip_from_py_object)]
#[derive(Clone)]
pub struct PyFlux {
    pub inner: CoreFlux,
}

#[pymethods]
impl PyFlux {
    /// 甜点处的频率，Hz
    #[getter]
    fn f_max(&self) -> f64 {
        self.inner.f_max
    }

    /// 甜点对应的偏置，V
    #[getter]
    fn z_offset(&self) -> f64 {
        self.inner.z_offset
    }

    /// 一个磁通量子对应的偏置跨度，V（可正可负）
    #[getter]
    fn z_period(&self) -> f64 {
        self.inner.z_period
    }

    /// 充电能 Ec/h，Hz（结构常数）
    #[getter]
    fn eta(&self) -> f64 {
        self.inner.eta
    }

    /// 两结不对称度（结构常数）
    #[getter]
    fn asymmetry(&self) -> f64 {
        self.inner.asymmetry
    }

    /// 在给定偏置网格上求线型值。
    fn at<'py>(&self, py: Python<'py>, zs: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.at(&zs).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Flux(f_max={}, z_offset={}, z_period={}, eta={}, asymmetry={})",
            self.inner.f_max,
            self.inner.z_offset,
            self.inner.z_period,
            self.inner.eta,
            self.inner.asymmetry
        )
    }
}

/// 通量调谐（f01 vs Z）的拟合结果。
#[pyclass(name = "FluxFit", from_py_object)]
#[derive(Clone)]
pub struct PyFluxFit {
    pub inner: CoreFluxFit,
}

#[pymethods]
impl PyFluxFit {
    #[getter]
    fn model(&self) -> PyFlux {
        PyFlux {
            inner: self.inner.result.model,
        }
    }

    /// 参数名 → `(值, 标准误)`；结构常数与协方差不可用的参数，标准误是 None。
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

    /// 反解：把频率调到 `f` 需要多大的 Z 偏置，取离参照点 `near` 最近的那个解。
    ///
    /// `near=None` 表示以甜点为准；目标频率超出这条曲线可达的范围时返回 None。
    #[pyo3(signature = (f, near=None))]
    fn tune_to(&self, f: f64, near: Option<f64>) -> Option<f64> {
        self.inner.tune_to(f, near)
    }

    #[getter]
    fn redchi(&self) -> f64 {
        self.inner.result.redchi
    }

    fn __repr__(&self) -> String {
        format!(
            "FluxFit(f_max={:.6e}, z_offset={}, z_period={}, redchi={:.3})",
            self.inner.result.model.f_max,
            self.inner.result.model.z_offset,
            self.inner.result.model.z_period,
            self.inner.result.redchi
        )
    }
}

/// 把逐偏置的峰位 `(z, f01)` 拟成通量调谐线型（SQUID 模型）。
#[pyfunction]
#[pyo3(signature = (zs, peaks, eta, asymmetry=0.0))]
fn fit_flux(
    zs: Vec<f64>,
    peaks: Vec<f64>,
    eta: f64,
    asymmetry: f64,
) -> PyResult<PyFluxFit> {
    match flux_fit(&zs, &peaks, eta, asymmetry) {
        Ok(fit) => Ok(PyFluxFit { inner: fit }),
        Err(error) => Err(QspecError::new_err(error.to_string())),
    }
}

/// 一条通量调谐拟合的输入：偏置轴 + 逐偏置峰位 + 两个结构常数。
#[pyclass(name = "FluxLine", from_py_object)]
#[derive(Clone)]
pub struct PyFluxLine {
    zs: Vec<f64>,
    peaks: Vec<f64>,
    eta: f64,
    asymmetry: f64,
}

#[pymethods]
impl PyFluxLine {
    #[new]
    #[pyo3(signature = (zs, peaks, eta, asymmetry=0.0))]
    fn new(zs: Vec<f64>, peaks: Vec<f64>, eta: f64, asymmetry: f64) -> Self {
        Self {
            zs,
            peaks,
            eta,
            asymmetry,
        }
    }

    /// Z 偏置轴，V
    #[getter]
    fn zs(&self) -> Vec<f64> {
        self.zs.clone()
    }

    /// 逐偏置的峰位，Hz
    #[getter]
    fn peaks(&self) -> Vec<f64> {
        self.peaks.clone()
    }

    /// 充电能 Ec/h，Hz（结构常数）
    #[getter]
    fn eta(&self) -> f64 {
        self.eta
    }

    /// 两结不对称度（结构常数）
    #[getter]
    fn asymmetry(&self) -> f64 {
        self.asymmetry
    }

    fn __repr__(&self) -> String {
        format!(
            "FluxLine(points={}, eta={}, asymmetry={})",
            self.zs.len(),
            self.eta,
            self.asymmetry
        )
    }
}

/// 批量通量调谐拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）。
#[pyfunction]
fn fit_flux_batch(py: Python<'_>, lines: Vec<PyFluxLine>) -> PyResult<Vec<Py<PyAny>>> {
    let borrowed: Vec<CoreFluxLine<'_>> = lines
        .iter()
        .map(|line| CoreFluxLine {
            zs: &line.zs,
            peaks: &line.peaks,
            eta: line.eta,
            asymmetry: line.asymmetry,
        })
        .collect();
    Ok(flux_fit_batch(&borrowed)
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyFluxFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => QspecError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect())
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
    states: Option<&Bound<'_, PyAny>>,
    sigmas: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let lines = iqs.len();
    let centers = per_line_states(states, lines)?;
    let sigmas = per_line_sigmas(sigmas, lines)?;
    let outcome = qspec_fit_batch(
        &freqs,
        &iqs,
        batch_states_arg(&centers).as_deref(),
        batch_sigmas_arg(&sigmas),
    );
    Ok(outcome
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyQspecFit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => QspecError::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect())
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
///
/// `flux` 给 [`fit_flux`] 的结果时，报告最下方多一张通量调谐面板（峰位点 + 拟合曲线 + 参数表）；
/// `None` 表示不做通量拟合。
#[pyfunction]
#[pyo3(signature = (freqs, lines, states=None, div_id="qspec-z", frame=None, flux=None))]
fn z_plot(
    freqs: Vec<f64>,
    lines: Vec<PyQspecZLine>,
    states: Option<PyStateCenters>,
    div_id: &str,
    frame: Option<&str>,
    flux: Option<PyFluxFit>,
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
    let flux_arg = match &flux {
        Some(item) => Some(Ok(&item.inner)),
        None => None,
    };
    Ok(qspec_z_plot_div(
        &freqs, &borrowed, states, div_id, frame, flux_arg,
    ))
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
    module.add_class::<PyFlux>()?;
    module.add_class::<PyFluxFit>()?;
    module.add_class::<PyFluxLine>()?;
    module.add_function(wrap_pyfunction!(fit_flux, module)?)?;
    module.add_function(wrap_pyfunction!(fit_flux_batch, module)?)?;
    Ok(())
}
