//! IQ 概率实验：`qtool::superconductor::iq` 的绑定。
//!
//! 这一档没有拟合入口（[`fit`] 的角色由 `iq_stats` 担着），对外名字是统计结果 `IqStats`、
//! 逐态统计 `StateStats`、两两可分性 `PairStats`、错误 `IqError`，以及 `stats` / `plot`。

use crate::QtoolError;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use qtool::superconductor::iq::iq_plot::iq_plot_div;
use qtool::superconductor::iq::{
    IqError as CoreIqError, IqStats as CoreIqStats, PairStats as CorePairStats,
    StateStats as CoreStateStats, iq_stats,
};

pyo3::create_exception!(_qtool, IqError, QtoolError);

/// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
fn to_py_error(error: &CoreIqError) -> PyErr {
    IqError::new_err(error.to_string())
}

/// 一个态的统计：中心与弥散（三条最深密度区域的面积）。
#[pyclass(name = "StateStats", skip_from_py_object)]
pub struct PyStateStats {
    pub inner: CoreStateStats,
}

#[pymethods]
impl PyStateStats {
    /// 云中心（所有单发点的均值）
    #[getter]
    fn center(&self) -> lmfit::Complex64 {
        self.inner.center
    }

    /// 最深 68 / 95 / 99 / 100% 区域的面积，数据单位²
    #[getter]
    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.areas.to_vec().into_pyarray(py)
    }

    /// 各条等密度线的"等效半径" √(A/π)，数据单位
    #[getter]
    fn radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.radii.to_vec().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "StateStats(center={}, r68={}, r95={}, r99={})",
            self.inner.center, self.inner.radii[0], self.inner.radii[1], self.inner.radii[2]
        )
    }
}

/// 一对态之间的可分性。
#[pyclass(name = "PairStats", skip_from_py_object)]
pub struct PyPairStats {
    pub inner: CorePairStats,
}

#[pymethods]
impl PyPairStats {
    /// p 态编号（`state_p < state_q`）
    #[getter]
    fn state_p(&self) -> usize {
        self.inner.state_p
    }

    /// q 态编号
    #[getter]
    fn state_q(&self) -> usize {
        self.inner.state_q
    }

    /// 中心间距，数据单位
    #[getter]
    fn separation(&self) -> f64 {
        self.inner.separation
    }

    /// 最优判别阈值（IQ 平面上的复数）
    #[getter]
    fn threshold(&self) -> lmfit::Complex64 {
        self.inner.threshold
    }

    /// 该阈值处的错分率
    #[getter]
    fn error_rate(&self) -> f64 {
        self.inner.error_rate
    }

    /// 无阈值的可分性度量，0.5 完全重叠、1 完全分开
    #[getter]
    fn auc(&self) -> f64 {
        self.inner.auc
    }

    /// baseline 口径的信噪比
    #[getter]
    fn snr(&self) -> f64 {
        self.inner.snr
    }

    fn __repr__(&self) -> String {
        format!(
            "PairStats({}-{}: separation={:.4}, error_rate={:.4}, auc={:.5}, snr={:.3})",
            self.inner.state_p,
            self.inner.state_q,
            self.inner.separation,
            self.inner.error_rate,
            self.inner.auc,
            self.inner.snr
        )
    }
}

/// 一次 IQ 概率实验的全部统计结果。
#[pyclass(name = "IqStats", skip_from_py_object)]
pub struct PyIqStats {
    pub inner: CoreIqStats,
}

#[pymethods]
impl PyIqStats {
    /// 各态统计，索引即态编号
    #[getter]
    fn states(&self) -> Vec<PyStateStats> {
        self.inner
            .states()
            .iter()
            .map(|state| PyStateStats {
                inner: state.clone(),
            })
            .collect()
    }

    /// 两两可分性，按 (0,1)、(0,2)、… 排列
    #[getter]
    fn pairs(&self) -> Vec<PyPairStats> {
        self.inner
            .pairs()
            .iter()
            .map(|pair| PyPairStats { inner: *pair })
            .collect()
    }

    /// 各态中心，可直接喂给其它实验的 `states`。
    #[getter]
    fn centers(&self) -> crate::PyStateCenters {
        crate::PyStateCenters::new(self.inner.centers().as_slice().to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "IqStats(states={}, pairs={})",
            self.inner.states().len(),
            self.inner.pairs().len()
        )
    }
}

/// 由各态的单发 IQ 云计算中心、弥散与可分性。
#[pyfunction]
fn stats(iqs: Vec<Vec<lmfit::Complex64>>) -> PyResult<PyIqStats> {
    let analyzed = iq_stats(&iqs).map_err(|error| to_py_error(&error))?;
    Ok(PyIqStats { inner: analyzed })
}

/// 渲染报告 div（返回 HTML 片段）：每态一团点云 + 密度等值线。
#[pyfunction]
#[pyo3(signature = (iqs, div_id="iq", frame=None))]
fn plot(iqs: Vec<Vec<lmfit::Complex64>>, div_id: &str, frame: Option<&str>) -> String {
    iq_plot_div(&iqs, div_id, frame)
}

/// 把本模块的公共项挂到给定模块上。
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("IqError", py.get_type::<IqError>())?;
    module.add_class::<PyStateStats>()?;
    module.add_class::<PyPairStats>()?;
    module.add_class::<PyIqStats>()?;
    module.add_function(wrap_pyfunction!(stats, module)?)?;
    module.add_function(wrap_pyfunction!(plot, module)?)?;
    Ok(())
}
