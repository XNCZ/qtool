//! 布洛赫层析：`qtool::superconductor::bloch` 的绑定。
//!
//! 对外名字：三基输入 `Trajectory`、结果 `BlochVector`、错误 `BlochError`，以及 `bloch_vector` / `plot`。
//! 没有 `fit`：这一档做的是把三基读出投影成激发概率再反解成泡利期望值，`bloch_vector` 就是入口。

use crate::{PyStateCenters, QtoolError};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::bloch::bloch_plot::bloch_plot_div;
use qtool::superconductor::bloch::{
    BlochError as CoreBlochError, BlochVector, Trajectory, bloch_vector,
};

pyo3::create_exception!(_qtool, BlochError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreBlochError) -> PyErr {
    BlochError::new_err(error.to_string())
}

/// 一条布洛赫轨迹的原始输入：扫描轴 + 三基 IQ。
///
/// 三个基各一条序列，长度须与 `axis` 一致。**哪个基是哪条由参数名定**，不靠位置。
#[pyclass(name = "Trajectory", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTrajectory {
    pub axis: Vec<f64>,
    /// X 基（前置 H）的读出 IQ
    pub x: Vec<lmfit::Complex64>,
    /// Y 基（前置 `Rx(−π/2)`）的读出 IQ
    pub y: Vec<lmfit::Complex64>,
    /// Z 基（直测）的读出 IQ
    pub z: Vec<lmfit::Complex64>,
}

#[pymethods]
impl PyTrajectory {
    #[new]
    fn new(
        axis: Vec<f64>,
        x: Vec<lmfit::Complex64>,
        y: Vec<lmfit::Complex64>,
        z: Vec<lmfit::Complex64>,
    ) -> Self {
        Self { axis, x, y, z }
    }

    /// 扫描轴的拷贝。
    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.axis.clone().into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.axis.len()
    }

    fn __repr__(&self) -> String {
        format!("Trajectory(points={})", self.axis.len())
    }
}

/// 布洛赫向量 `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)(扫描)`；扫描轴原样带回。
#[pyclass(name = "BlochVector", skip_from_py_object)]
pub struct PyBlochVector {
    pub inner: BlochVector,
}

#[pymethods]
impl PyBlochVector {
    /// 扫描轴
    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.axis.clone().into_pyarray(py)
    }

    /// X 基上的泡利期望值
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.x.clone().into_pyarray(py)
    }

    /// Y 基上的泡利期望值
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.y.clone().into_pyarray(py)
    }

    /// Z 基上的泡利期望值
    #[getter]
    fn z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.z.clone().into_pyarray(py)
    }

    /// 径向长度 `R = √(X²+Y²+Z²)`：1 在球面上（纯态），小于 1 在球里。
    fn radius<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.radius().into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.axis.len()
    }

    fn __repr__(&self) -> String {
        format!("BlochVector(points={})", self.inner.axis.len())
    }
}

/// 反解一条三基扫描：由 `trajectory` 与（可选的）各态标定中心得到布洛赫向量。
#[pyfunction]
#[pyo3(name = "bloch_vector", signature = (trajectory, states=None))]
fn bloch_vector_py(
    trajectory: PyRef<'_, PyTrajectory>,
    states: Option<PyStateCenters>,
) -> PyResult<PyBlochVector> {
    let scan = Trajectory {
        axis: &trajectory.axis,
        x: &trajectory.x,
        y: &trajectory.y,
        z: &trajectory.z,
    };
    let vector = bloch_vector(&scan, states.as_ref().map(|item| &item.inner))
        .map_err(|error| to_py_error(&error))?;
    Ok(PyBlochVector { inner: vector })
}

/// 渲染报告 div（返回 HTML 片段）。
///
/// `x_title` 是扫描量的名字，只用在球上轨迹点的悬停标签里（如 `"π amplitude"`）。
#[pyfunction]
#[pyo3(signature = (vector, x_title, div_id="bloch", frame=None))]
fn plot(
    vector: PyRef<'_, PyBlochVector>,
    x_title: &str,
    div_id: &str,
    frame: Option<&str>,
) -> String {
    bloch_plot_div(&vector.inner, x_title, div_id, frame)
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("BlochError", py.get_type::<BlochError>())?;
    module.add_class::<PyTrajectory>()?;
    module.add_class::<PyBlochVector>()?;
    module.add_function(wrap_pyfunction!(bloch_vector_py, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
