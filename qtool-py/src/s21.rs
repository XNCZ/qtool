//! S21 谐振腔：`qtool::superconductor::s21` 的绑定。
//!
//! 对外名字：模型 `S21Model`、结果 `S21Fit`、功率扫描的逐行输入 `PowerLine`、错误 `S21Error`，
//! 以及 `fit` / `fit_batch` / `plot` / `power_plot` 四个函数。
//!
//! 这一档**不吃标定中心**：拟合的是原始复数 IQ（`Result<ComplexResult<S21Model>, S21Error>`），
//! 与其余实验的 P1 路径不同。

use crate::QtoolError;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::s21::s21_plot::s21_fit_plot_div;
use qtool::superconductor::s21::s21_power_plot::{PowerLine, s21_power_plot_div};
use qtool::superconductor::s21::{
    S21Error as CoreS21Error, S21Model, model_at, s21_fit, s21_fit_batch,
};

pyo3::create_exception!(_qtool, S21Error, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreS21Error) -> PyErr {
    S21Error::new_err(error.to_string())
}

/// S21 模型的全部参数（SI 单位）。
///
/// 前 11 个字段是拟合参数；`qi` 与 `kappa_ex` 是派生量，值由前 11 个定出、不参与拟合。
#[pyclass(name = "S21Model", skip_from_py_object)]
#[derive(Clone)]
pub struct PyS21Model {
    pub inner: S21Model,
}

#[pymethods]
impl PyS21Model {
    #[new]
    fn new(
        fr: f64,
        ql: f64,
        qc: f64,
        theta: f64,
        ap: f64,
        tau: f64,
        a: f64,
        b: f64,
        phi: f64,
        zc_re: f64,
        zc_im: f64,
    ) -> Self {
        Self {
            inner: S21Model::from_fit_array(&[
                fr, ql, qc, theta, ap, tau, a, b, phi, zc_re, zc_im,
            ]),
        }
    }

    /// 谐振频率，Hz
    #[getter]
    fn fr(&self) -> f64 {
        self.inner.fr
    }

    /// 有载品质因数
    #[getter]
    fn ql(&self) -> f64 {
        self.inner.ql
    }

    /// 耦合品质因数
    #[getter]
    fn qc(&self) -> f64 {
        self.inner.qc
    }

    /// notch 相位，rad
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Duffing 非线性参数
    #[getter]
    fn ap(&self) -> f64 {
        self.inner.ap
    }

    /// 背景延迟，s
    #[getter]
    fn tau(&self) -> f64 {
        self.inner.tau
    }

    /// cos 支背景幅度
    #[getter]
    fn a(&self) -> f64 {
        self.inner.a
    }

    /// sin 支背景幅度
    #[getter]
    fn b(&self) -> f64 {
        self.inner.b
    }

    /// 背景相位，rad
    #[getter]
    fn phi(&self) -> f64 {
        self.inner.phi
    }

    /// 常数复偏置实部
    #[getter]
    fn zc_re(&self) -> f64 {
        self.inner.zc_re
    }

    /// 常数复偏置虚部
    #[getter]
    fn zc_im(&self) -> f64 {
        self.inner.zc_im
    }

    /// 内部品质因数（派生量）
    #[getter]
    fn qi(&self) -> f64 {
        self.inner.qi
    }

    /// 外部耦合率（派生量），Hz
    #[getter]
    fn kappa_ex(&self) -> f64 {
        self.inner.kappa_ex
    }

    /// 在给定频率网格上求模型值（复数）。
    fn at<'py>(&self, py: Python<'py>, freqs: Vec<f64>) -> Bound<'py, PyArray1<lmfit::Complex64>> {
        let values: Vec<lmfit::Complex64> = freqs
            .iter()
            .map(|freq| model_at(*freq, &self.inner))
            .collect();
        values.into_pyarray(py)
    }

    /// 11 个拟合参数，按 `JAC_NAMES` 序。
    fn to_array(&self) -> Vec<f64> {
        self.inner.to_array().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "S21Model(fr={:.6e}, ql={}, qc={}, qi={})",
            self.inner.fr, self.inner.ql, self.inner.qc, self.inner.qi
        )
    }
}

/// 一条 S21 扫描的拟合结果。
#[pyclass(name = "S21Fit", from_py_object)]
#[derive(Clone)]
pub struct PyS21Fit {
    pub inner: lmfit::ComplexResult<S21Model>,
}

#[pymethods]
impl PyS21Fit {
    #[getter]
    fn model(&self) -> PyS21Model {
        PyS21Model {
            inner: self.inner.model,
        }
    }

    /// 参数名 → `(值, 标准误)`；协方差不可用时标准误是 None。
    #[getter]
    fn params(&self) -> Vec<(String, (f64, Option<f64>))> {
        self.inner
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
        self.inner.chisqr
    }

    #[getter]
    fn redchi(&self) -> f64 {
        self.inner.redchi
    }

    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    fn __repr__(&self) -> String {
        format!(
            "S21Fit(fr={:.6e}, ql={}, redchi={:.3})",
            self.inner.model.fr, self.inner.model.ql, self.inner.redchi
        )
    }
}

/// 一条频率线：泵幅 + 数据 + 拟合结果。
///
/// `fit=None` 表示该线没拟合（报告里该行只有数据）。
#[pyclass(name = "PowerLine", from_py_object)]
#[derive(Clone)]
pub struct PyPowerLine {
    power: f64,
    iq: Vec<lmfit::Complex64>,
    sigma: Option<Vec<f64>>,
    fit: Option<PyS21Fit>,
}

#[pymethods]
impl PyPowerLine {
    #[new]
    #[pyo3(signature = (power, iq, sigma=None, fit=None))]
    fn new(
        power: f64,
        iq: Vec<lmfit::Complex64>,
        sigma: Option<Vec<f64>>,
        fit: Option<PyS21Fit>,
    ) -> Self {
        Self {
            power,
            iq,
            sigma,
            fit,
        }
    }

    /// 读出泵幅
    #[getter]
    fn power(&self) -> f64 {
        self.power
    }

    fn __repr__(&self) -> String {
        format!(
            "PowerLine(power={}, points={}, fitted={})",
            self.power,
            self.iq.len(),
            self.fit.is_some()
        )
    }
}

/// 拟合一条 S21 频率扫描。
#[pyfunction]
#[pyo3(signature = (freqs, iq, sigma=None))]
fn fit(
    freqs: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    sigma: Option<Vec<f64>>,
) -> PyResult<PyS21Fit> {
    let fitted = s21_fit(&freqs, &iq, sigma.as_deref()).map_err(|error| to_py_error(&error))?;
    Ok(PyS21Fit { inner: fitted })
}

/// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）——`isinstance` 判得出来、
/// `str()` 有消息、还能直接喂回 [`plot`]（后者会把它原样抛出）。
#[pyfunction]
#[pyo3(signature = (freqs, iqs, sigmas=None))]
fn fit_batch(
    py: Python<'_>,
    freqs: Vec<f64>,
    iqs: Vec<Vec<lmfit::Complex64>>,
    sigmas: Option<Vec<Vec<f64>>>,
) -> Vec<Py<PyAny>> {
    s21_fit_batch(&freqs, &iqs, sigmas.as_deref())
        .into_iter()
        .map(|outcome| match outcome {
            Ok(fitted) => Py::new(py, PyS21Fit { inner: fitted })
                .map(|value| value.into_any())
                .unwrap_or_else(|error| error.into_value(py).into_any()),
            Err(error) => S21Error::new_err(error.to_string())
                .into_value(py)
                .into_any(),
        })
        .collect()
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `fit` 收三种东西：`S21Fit`（叠拟合曲线）、`None`（只画数据）、或一条 `S21Error` 实例——
/// 后者会被**原样抛出**。
#[pyfunction]
#[pyo3(signature = (freqs, iq, sigma=None, fit=None, div_id="s21", frame=None))]
fn plot(
    freqs: Vec<f64>,
    iq: Vec<lmfit::Complex64>,
    sigma: Option<Vec<f64>>,
    fit: Option<&Bound<'_, PyAny>>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let fitted = match fit {
        Some(value) => match value.extract::<PyRef<'_, PyS21Fit>>() {
            Ok(fitted) => Some(fitted.inner.clone()),
            // 传进来的是一条异常实例（`fit_batch` 的失败项）：原样抛出
            Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                true => return Err(PyErr::from_value(value.clone())),
                false => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "fit must be a S21Fit, a S21Error instance or None",
                    ));
                }
            },
        },
        None => None,
    };
    Ok(match &fitted {
        Some(result) => s21_fit_plot_div(&freqs, &iq, sigma.as_deref(), Ok(result), div_id, frame),
        None => s21_fit_plot_div(
            &freqs,
            &iq,
            sigma.as_deref(),
            Err(&CoreS21Error::EmptyData),
            div_id,
            frame,
        ),
    })
}

/// 渲染功率扫描报告 div：逐行给 `PowerLine`，每行自带泵幅与该行的拟合。
#[pyfunction]
#[pyo3(signature = (freqs, lines, div_id="s21-power", frame=None))]
fn power_plot(
    freqs: Vec<f64>,
    lines: Vec<PyPowerLine>,
    div_id: &str,
    frame: Option<&str>,
) -> PyResult<String> {
    let borrowed: Vec<PowerLine<'_>> = lines
        .iter()
        .map(|line| PowerLine {
            power: line.power,
            iq: &line.iq,
            sigma: line.sigma.as_deref(),
            fit: match &line.fit {
                Some(result) => Ok(&result.inner),
                None => Err(&CoreS21Error::EmptyData),
            },
        })
        .collect();
    Ok(s21_power_plot_div(&freqs, &borrowed, div_id, frame))
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("S21Error", py.get_type::<S21Error>())?;
    module.add_class::<PyS21Model>()?;
    module.add_class::<PyS21Fit>()?;
    module.add_class::<PyPowerLine>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    module.add_function(wrap_pyfunction!(power_plot, module)?)?;
    Ok(())
}
