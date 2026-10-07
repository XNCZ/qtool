//! DRAG 标定（幅度 / 系数 / 载波失谐）：`qtool::superconductor::drag` 的绑定。
//!
//! 三个子模块各自独立：`qtool.drag.amplitude`、`qtool.drag.coeff`、`qtool.drag.detuning`。
//! 三档**都必须**吃各态标定中心（Rust 侧是 `&StateCenters`，不是 `Option`），这是与其它实验
//! 唯一的接口差别。

pub mod amplitude {
    //! DRAG 幅度扫描：`qtool::superconductor::drag::amplitude` 的绑定。
    //!
    //! 两种线型各一套拟合：最低阶（`pairs = 1`）是余弦 `CosWave`，升阶是谷 `Valley`。
    //! 这里的 `fit` / `fit_batch` 指最低阶那一套，升阶那一套叫 `valley_fit` / `valley_fit_batch`。

    use crate::{PyStateCenters, QtoolError};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::prelude::*;
    use qtool::superconductor::drag::amplitude::plot::{
        OrderFit, OrderScan, drag_amplitude_plot_div,
    };
    use qtool::superconductor::drag::amplitude::{
        CosWave, FactorNError as CoreFactorNError, FactorNFit as CoreFactorNFit,
        FactorOneError as CoreFactorOneError, FactorOneFit as CoreFactorOneFit, Valley,
        factor_n_fit, factor_n_fit_batch, factor_one_fit, factor_one_fit_batch,
    };

    pyo3::create_exception!(_qtool, FactorOneError, QtoolError);
    pyo3::create_exception!(_qtool, FactorNError, QtoolError);

    /// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
    fn to_py_error(error: &CoreFactorOneError) -> PyErr {
        FactorOneError::new_err(error.to_string())
    }

    /// 升阶那一套的错误转换。
    fn to_py_error_n(error: &CoreFactorNError) -> PyErr {
        FactorNError::new_err(error.to_string())
    }

    /// `2·pairs` 发同号脉冲下 P1 随幅度的线型：零驱动处锚在 |0> 上的余弦。
    ///
    /// 与 rabi 的 `Cos` 是同一个形状，差别只在派生量 `a_pi` 的式子（这里是 `1/freq`）。
    #[pyclass(name = "CosWave", skip_from_py_object)]
    #[derive(Clone)]
    pub struct PyCosWave {
        pub inner: CosWave,
    }

    #[pymethods]
    impl PyCosWave {
        #[new]
        fn new(freq: f64, amp: f64) -> Self {
            Self {
                inner: CosWave::new(freq, amp),
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

        /// π 幅度（派生量：`1/freq`）
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
                "CosWave(freq={}, amp={}, a_pi={})",
                self.inner.freq, self.inner.amp, self.inner.a_pi
            )
        }
    }

    /// 谷的线型：倒置的洛伦兹，谷心、半高全宽、深度、基线四个参数全自由。
    #[pyclass(name = "Valley", skip_from_py_object)]
    #[derive(Clone)]
    pub struct PyValley {
        pub inner: Valley,
    }

    #[pymethods]
    impl PyValley {
        #[new]
        fn new(centre: f64, fwhm: f64, amp: f64, offset: f64) -> Self {
            Self {
                inner: Valley {
                    centre,
                    fwhm,
                    amp,
                    offset,
                },
            }
        }

        /// 谷心：扫描轴上 P1 取极小的位置
        #[getter]
        fn centre(&self) -> f64 {
            self.inner.centre
        }

        /// 半高全宽
        #[getter]
        fn fwhm(&self) -> f64 {
            self.inner.fwhm
        }

        /// 深度（带符号）：谷取负
        #[getter]
        fn amp(&self) -> f64 {
            self.inner.amp
        }

        /// 远处（谷外）的基线电平
        #[getter]
        fn offset(&self) -> f64 {
            self.inner.offset
        }

        /// 在给定幅度网格上求值。
        fn at<'py>(&self, py: Python<'py>, amps: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
            self.inner.at(&amps).into_pyarray(py)
        }

        fn __repr__(&self) -> String {
            format!(
                "Valley(centre={}, fwhm={}, amp={}, offset={})",
                self.inner.centre, self.inner.fwhm, self.inner.amp, self.inner.offset
            )
        }
    }

    /// 单条最低阶幅度扫描的拟合结果。
    #[pyclass(name = "FactorOneFit", from_py_object)]
    #[derive(Clone)]
    pub struct PyFactorOneFit {
        pub inner: CoreFactorOneFit,
    }

    #[pymethods]
    impl PyFactorOneFit {
        #[getter]
        fn model(&self) -> PyCosWave {
            PyCosWave {
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
            params_of(&self.inner.result.params)
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
                "FactorOneFit(a_pi={:.6e}, redchi={:.3})",
                self.inner.result.model.a_pi, self.inner.result.redchi
            )
        }
    }

    /// 单条升阶扫描的拟合结果。
    #[pyclass(name = "FactorNFit", from_py_object)]
    #[derive(Clone)]
    pub struct PyFactorNFit {
        pub inner: CoreFactorNFit,
    }

    #[pymethods]
    impl PyFactorNFit {
        #[getter]
        fn model(&self) -> PyValley {
            PyValley {
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
            params_of(&self.inner.result.params)
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
                "FactorNFit(centre={:.6e}, fwhm={:.6e}, redchi={:.3})",
                self.inner.result.model.centre, self.inner.result.model.fwhm, self.inner.result.redchi
            )
        }
    }

    /// 参数表：`params` 那三个 getter 共用的取法。
    fn params_of(params: &lmfit::Parameters) -> Vec<(String, (f64, Option<f64>))> {
        params
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

    /// 某阶拟合出来的线型：由传进来的拟合对象的类型定。
    #[derive(Clone)]
    enum OwnedOrderFit {
        Cosine(Result<CoreFactorOneFit, CoreFactorOneError>),
        Valley(Result<CoreFactorNFit, CoreFactorNError>),
    }

    /// 没给拟合结果时按阶序挑线型：最低阶是余弦，升阶是谷（与 Rust 侧建链的约定一致）。
    fn absent_fit(pairs: usize) -> OwnedOrderFit {
        match pairs {
            1 => OwnedOrderFit::Cosine(Err(CoreFactorOneError::EmptyData)),
            _ => OwnedOrderFit::Valley(Err(CoreFactorNError::EmptyData)),
        }
    }

    /// 一阶的扫描与拟合：报告画的就是它。
    ///
    /// `fit` 收 `FactorOneFit`（最低阶）、`FactorNFit`（升阶）、一条异常实例（`fit_batch` 的失败项）
    /// 或 `None`；后两者在有阶序时按阶序补一个「没拟合」的占位。
    #[pyclass(name = "OrderScan", from_py_object)]
    #[derive(Clone)]
    pub struct PyOrderScan {
        pairs: usize,
        amps: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        fit: OwnedOrderFit,
    }

    #[pymethods]
    impl PyOrderScan {
        #[new]
        #[pyo3(signature = (pairs, amps, iq, fit=None))]
        fn new(
            pairs: usize,
            amps: Vec<f64>,
            iq: Vec<lmfit::Complex64>,
            fit: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let owned = match fit {
                Some(value) => match value.extract::<PyRef<'_, PyFactorOneFit>>() {
                    Ok(one) => OwnedOrderFit::Cosine(Ok(one.inner.clone())),
                    Err(_) => match value.extract::<PyRef<'_, PyFactorNFit>>() {
                        Ok(n) => OwnedOrderFit::Valley(Ok(n.inner.clone())),
                        Err(_) => match value
                            .is_instance_of::<pyo3::exceptions::PyBaseException>()
                        {
                            true => absent_fit(pairs),
                            false => {
                                return Err(pyo3::exceptions::PyTypeError::new_err(
                                    "fit must be a FactorOneFit, a FactorNFit, an exception instance or None",
                                ));
                            }
                        },
                    },
                },
                None => absent_fit(pairs),
            };
            Ok(Self {
                pairs,
                amps,
                iq,
                fit: owned,
            })
        }

        /// 脉冲对数：这一阶的标号，也是颜色深浅的次序
        #[getter]
        fn pairs(&self) -> usize {
            self.pairs
        }

        fn __repr__(&self) -> String {
            format!(
                "OrderScan(pairs={}, points={})",
                self.pairs,
                self.amps.len()
            )
        }
    }

    /// 拟合一条最低阶（`pairs = 1`）幅度扫描。
    #[pyfunction]
    #[pyo3(signature = (amps, iq, states, sigma=None))]
    fn fit(
        amps: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        states: PyStateCenters,
        sigma: Option<Vec<f64>>,
    ) -> PyResult<PyFactorOneFit> {
        let fitted = factor_one_fit(&amps, &iq, &states.inner, sigma.as_deref())
            .map_err(|error| to_py_error(&error))?;
        Ok(PyFactorOneFit { inner: fitted })
    }

    /// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）。
    #[pyfunction]
    #[pyo3(signature = (amps, iqs, states, sigmas=None))]
    fn fit_batch(
        py: Python<'_>,
        amps: Vec<f64>,
        iqs: Vec<Vec<lmfit::Complex64>>,
        states: PyStateCenters,
        sigmas: Option<Vec<Vec<f64>>>,
    ) -> Vec<Py<PyAny>> {
        factor_one_fit_batch(&amps, &iqs, &states.inner, sigmas.as_deref())
            .into_iter()
            .map(|outcome| match outcome {
                Ok(fitted) => Py::new(py, PyFactorOneFit { inner: fitted })
                    .map(|value| value.into_any())
                    .unwrap_or_else(|error| error.into_value(py).into_any()),
                Err(error) => FactorOneError::new_err(error.to_string())
                    .into_value(py)
                    .into_any(),
            })
            .collect()
    }

    /// 拟合一条升阶（`pairs ≥ 2`）幅度扫描：谷的洛伦兹。
    #[pyfunction]
    #[pyo3(signature = (amps, iq, states, sigma=None))]
    fn valley_fit(
        amps: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        states: PyStateCenters,
        sigma: Option<Vec<f64>>,
    ) -> PyResult<PyFactorNFit> {
        let fitted = factor_n_fit(&amps, &iq, &states.inner, sigma.as_deref())
            .map_err(|error| to_py_error_n(&error))?;
        Ok(PyFactorNFit { inner: fitted })
    }

    /// 批量拟合升阶：逐条给出结果，失败的那条给**异常类实例**（不抛）。
    #[pyfunction]
    #[pyo3(signature = (amps, iqs, states, sigmas=None))]
    fn valley_fit_batch(
        py: Python<'_>,
        amps: Vec<f64>,
        iqs: Vec<Vec<lmfit::Complex64>>,
        states: PyStateCenters,
        sigmas: Option<Vec<Vec<f64>>>,
    ) -> Vec<Py<PyAny>> {
        factor_n_fit_batch(&amps, &iqs, &states.inner, sigmas.as_deref())
            .into_iter()
            .map(|outcome| match outcome {
                Ok(fitted) => Py::new(py, PyFactorNFit { inner: fitted })
                    .map(|value| value.into_any())
                    .unwrap_or_else(|error| error.into_value(py).into_any()),
                Err(error) => FactorNError::new_err(error.to_string())
                    .into_value(py)
                    .into_any(),
            })
            .collect()
    }

    /// 升阶扫描窗的半宽 `a_pi / (2·pairs)`：围着上一阶的谷位取窗时用它。
    #[pyfunction]
    fn window_half_width(a_pi: f64, pairs: usize) -> f64 {
        qtool::superconductor::drag::amplitude::window_half_width(a_pi, pairs)
    }

    /// 渲染整条升阶链的报告 div（返回 HTML 片段）。
    #[pyfunction]
    #[pyo3(signature = (orders, chosen, states, div_id="drag-amplitude", frame=None))]
    fn plot(
        orders: Vec<PyOrderScan>,
        chosen: f64,
        states: PyStateCenters,
        div_id: &str,
        frame: Option<&str>,
    ) -> String {
        let borrowed: Vec<OrderScan<'_>> = orders
            .iter()
            .map(|scan| OrderScan {
                pairs: scan.pairs,
                amps: &scan.amps,
                iq: &scan.iq,
                fit: match &scan.fit {
                    OwnedOrderFit::Cosine(one) => OrderFit::Cosine(one.as_ref()),
                    OwnedOrderFit::Valley(n) => OrderFit::Valley(n.as_ref()),
                },
            })
            .collect();
        drag_amplitude_plot_div(&borrowed, chosen, &states.inner, div_id, frame)
    }

    /// 把本子模块的公共项挂到给定模块上。
    pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("FactorOneError", py.get_type::<FactorOneError>())?;
        module.add("FactorNError", py.get_type::<FactorNError>())?;
        module.add_class::<PyCosWave>()?;
        module.add_class::<PyValley>()?;
        module.add_class::<PyFactorOneFit>()?;
        module.add_class::<PyFactorNFit>()?;
        module.add_class::<PyOrderScan>()?;
        module.add_function(wrap_pyfunction!(fit, module)?)?;
        module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
        module.add_function(wrap_pyfunction!(valley_fit, module)?)?;
        module.add_function(wrap_pyfunction!(valley_fit_batch, module)?)?;
        module.add_function(wrap_pyfunction!(window_half_width, module)?)?;
        module.add_function(wrap_pyfunction!(plot, module)?)?;
        Ok(())
    }
}

pub mod coeff {
    //! DRAG 系数扫描：`qtool::superconductor::drag::coeff` 的绑定。
    //!
    //! 每一阶都拟谷（系数从零起、曲线不是 Rabi 余弦，没有周期可给窗口定尺度）。

    use crate::{PyStateCenters, QtoolError};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::prelude::*;
    use qtool::superconductor::drag::coeff::plot::{OrderScan, drag_coeff_plot_div};
    use qtool::superconductor::drag::coeff::{
        Valley, ValleyError as CoreValleyError, ValleyFit as CoreValleyFit, valley_fit,
        valley_fit_batch,
    };

    pyo3::create_exception!(_qtool, ValleyError, QtoolError);

    /// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
    fn to_py_error(error: &CoreValleyError) -> PyErr {
        ValleyError::new_err(error.to_string())
    }

    /// 谷的线型：倒置的洛伦兹，谷心、半高全宽、深度、基线四个参数全自由。
    #[pyclass(name = "Valley", skip_from_py_object)]
    #[derive(Clone)]
    pub struct PyValley {
        pub inner: Valley,
    }

    #[pymethods]
    impl PyValley {
        #[new]
        fn new(centre: f64, fwhm: f64, amp: f64, offset: f64) -> Self {
            Self {
                inner: Valley {
                    centre,
                    fwhm,
                    amp,
                    offset,
                },
            }
        }

        /// 谷心：DRAG 系数轴上 P1 取极小的位置
        #[getter]
        fn centre(&self) -> f64 {
            self.inner.centre
        }

        /// 半高全宽
        #[getter]
        fn fwhm(&self) -> f64 {
            self.inner.fwhm
        }

        /// 深度（带符号）：谷取负
        #[getter]
        fn amp(&self) -> f64 {
            self.inner.amp
        }

        /// 远处（谷外）的基线电平
        #[getter]
        fn offset(&self) -> f64 {
            self.inner.offset
        }

        /// 在给定系数网格上求值。
        fn at<'py>(&self, py: Python<'py>, coeffs: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
            self.inner.at(&coeffs).into_pyarray(py)
        }

        fn __repr__(&self) -> String {
            format!(
                "Valley(centre={}, fwhm={}, amp={}, offset={})",
                self.inner.centre, self.inner.fwhm, self.inner.amp, self.inner.offset
            )
        }
    }

    /// 单条系数扫描的拟合结果。
    #[pyclass(name = "ValleyFit", from_py_object)]
    #[derive(Clone)]
    pub struct PyValleyFit {
        pub inner: CoreValleyFit,
    }

    #[pymethods]
    impl PyValleyFit {
        #[getter]
        fn model(&self) -> PyValley {
            PyValley {
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
                "ValleyFit(centre={:.6e}, fwhm={:.6e}, redchi={:.3})",
                self.inner.result.model.centre, self.inner.result.model.fwhm, self.inner.result.redchi
            )
        }
    }

    /// 一阶的扫描与拟合：报告画的就是它。
    ///
    /// `fit` 收 `ValleyFit`、一条异常实例（`fit_batch` 的失败项）或 `None`；后两者按「该阶没拟合」
    /// 处理（报告里那一阶只有数据）。
    #[pyclass(name = "OrderScan", from_py_object)]
    #[derive(Clone)]
    pub struct PyOrderScan {
        pairs: usize,
        coeffs: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        fit: Option<CoreValleyFit>,
    }

    #[pymethods]
    impl PyOrderScan {
        #[new]
        #[pyo3(signature = (pairs, coeffs, iq, fit=None))]
        fn new(
            pairs: usize,
            coeffs: Vec<f64>,
            iq: Vec<lmfit::Complex64>,
            fit: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let owned = match fit {
                Some(value) => match value.extract::<PyRef<'_, PyValleyFit>>() {
                    Ok(fitted) => Some(fitted.inner.clone()),
                    Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                        true => None,
                        false => {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "fit must be a ValleyFit, a ValleyError instance or None",
                            ));
                        }
                    },
                },
                None => None,
            };
            Ok(Self {
                pairs,
                coeffs,
                iq,
                fit: owned,
            })
        }

        /// 脉冲对数：这一阶的标号
        #[getter]
        fn pairs(&self) -> usize {
            self.pairs
        }

        fn __repr__(&self) -> String {
            format!(
                "OrderScan(pairs={}, points={}, fitted={})",
                self.pairs,
                self.coeffs.len(),
                self.fit.is_some()
            )
        }
    }

    /// 拟合一条 DRAG 系数扫描。
    #[pyfunction]
    #[pyo3(signature = (coeffs, iq, states, sigma=None))]
    fn fit(
        coeffs: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        states: PyStateCenters,
        sigma: Option<Vec<f64>>,
    ) -> PyResult<PyValleyFit> {
        let fitted = valley_fit(&coeffs, &iq, &states.inner, sigma.as_deref())
            .map_err(|error| to_py_error(&error))?;
        Ok(PyValleyFit { inner: fitted })
    }

    /// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）。
    #[pyfunction]
    #[pyo3(signature = (coeffs, iqs, states, sigmas=None))]
    fn fit_batch(
        py: Python<'_>,
        coeffs: Vec<f64>,
        iqs: Vec<Vec<lmfit::Complex64>>,
        states: PyStateCenters,
        sigmas: Option<Vec<Vec<f64>>>,
    ) -> Vec<Py<PyAny>> {
        valley_fit_batch(&coeffs, &iqs, &states.inner, sigmas.as_deref())
            .into_iter()
            .map(|outcome| match outcome {
                Ok(fitted) => Py::new(py, PyValleyFit { inner: fitted })
                    .map(|value| value.into_any())
                    .unwrap_or_else(|error| error.into_value(py).into_any()),
                Err(error) => ValleyError::new_err(error.to_string())
                    .into_value(py)
                    .into_any(),
            })
            .collect()
    }

    /// 渲染整条升阶链的报告 div（返回 HTML 片段）。
    #[pyfunction]
    #[pyo3(signature = (orders, chosen, states, div_id="drag-coeff", frame=None))]
    fn plot(
        orders: Vec<PyOrderScan>,
        chosen: f64,
        states: PyStateCenters,
        div_id: &str,
        frame: Option<&str>,
    ) -> String {
        let borrowed: Vec<OrderScan<'_>> = orders
            .iter()
            .map(|scan| OrderScan {
                pairs: scan.pairs,
                coeffs: &scan.coeffs,
                iq: &scan.iq,
                fit: match &scan.fit {
                    Some(fitted) => Ok(fitted),
                    None => Err(&CoreValleyError::EmptyData),
                },
            })
            .collect();
        drag_coeff_plot_div(&borrowed, chosen, &states.inner, div_id, frame)
    }

    /// 把本子模块的公共项挂到给定模块上。
    pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("ValleyError", py.get_type::<ValleyError>())?;
        module.add_class::<PyValley>()?;
        module.add_class::<PyValleyFit>()?;
        module.add_class::<PyOrderScan>()?;
        module.add_function(wrap_pyfunction!(fit, module)?)?;
        module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
        module.add_function(wrap_pyfunction!(plot, module)?)?;
        Ok(())
    }
}

pub mod detuning {
    //! DRAG 载波失谐扫描：`qtool::superconductor::drag::detuning` 的绑定。
    //!
    //! 每一阶都拟谷，谷心就是这一轮要修掉的**载波残差**（Hz）。

    use crate::{PyStateCenters, QtoolError};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::prelude::*;
    use qtool::superconductor::drag::detuning::plot::{OrderScan, drag_detuning_plot_div};
    use qtool::superconductor::drag::detuning::{
        Valley, ValleyError as CoreValleyError, ValleyFit as CoreValleyFit, valley_fit,
        valley_fit_batch,
    };

    pyo3::create_exception!(_qtool, ValleyError, QtoolError);

    /// 核心错误 → Python 异常（错误类型来自别的 crate，orphan rule 不允许 `impl From`，只能逐处转）。
    fn to_py_error(error: &CoreValleyError) -> PyErr {
        ValleyError::new_err(error.to_string())
    }

    /// 谷的线型：倒置的洛伦兹，谷心、半高全宽、深度、基线四个参数全自由。
    #[pyclass(name = "Valley", skip_from_py_object)]
    #[derive(Clone)]
    pub struct PyValley {
        pub inner: Valley,
    }

    #[pymethods]
    impl PyValley {
        #[new]
        fn new(centre: f64, fwhm: f64, amp: f64, offset: f64) -> Self {
            Self {
                inner: Valley {
                    centre,
                    fwhm,
                    amp,
                    offset,
                },
            }
        }

        /// 谷心：这一轮要修掉的载波残差 (Hz)，理想值 0
        #[getter]
        fn centre(&self) -> f64 {
            self.inner.centre
        }

        /// 半高全宽 (Hz)
        #[getter]
        fn fwhm(&self) -> f64 {
            self.inner.fwhm
        }

        /// 深度（带符号）：谷取负
        #[getter]
        fn amp(&self) -> f64 {
            self.inner.amp
        }

        /// 远处（谷外）的基线电平
        #[getter]
        fn offset(&self) -> f64 {
            self.inner.offset
        }

        /// 在给定失谐网格上求值。
        fn at<'py>(&self, py: Python<'py>, detunings: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
            self.inner.at(&detunings).into_pyarray(py)
        }

        fn __repr__(&self) -> String {
            format!(
                "Valley(centre={}, fwhm={}, amp={}, offset={})",
                self.inner.centre, self.inner.fwhm, self.inner.amp, self.inner.offset
            )
        }
    }

    /// 单条失谐扫描的拟合结果。
    #[pyclass(name = "ValleyFit", from_py_object)]
    #[derive(Clone)]
    pub struct PyValleyFit {
        pub inner: CoreValleyFit,
    }

    #[pymethods]
    impl PyValleyFit {
        #[getter]
        fn model(&self) -> PyValley {
            PyValley {
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
                "ValleyFit(centre={:.6e}, fwhm={:.6e}, redchi={:.3})",
                self.inner.result.model.centre, self.inner.result.model.fwhm, self.inner.result.redchi
            )
        }
    }

    /// 一阶的扫描与拟合：报告画的就是它。
    ///
    /// `fit` 收 `ValleyFit`、一条异常实例（`fit_batch` 的失败项）或 `None`；后两者按「该阶没拟合」
    /// 处理（报告里那一阶只有数据）。
    #[pyclass(name = "OrderScan", from_py_object)]
    #[derive(Clone)]
    pub struct PyOrderScan {
        pairs: usize,
        detunings_hz: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        fit: Option<CoreValleyFit>,
    }

    #[pymethods]
    impl PyOrderScan {
        #[new]
        #[pyo3(signature = (pairs, detunings_hz, iq, fit=None))]
        fn new(
            pairs: usize,
            detunings_hz: Vec<f64>,
            iq: Vec<lmfit::Complex64>,
            fit: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let owned = match fit {
                Some(value) => match value.extract::<PyRef<'_, PyValleyFit>>() {
                    Ok(fitted) => Some(fitted.inner.clone()),
                    Err(_) => match value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                        true => None,
                        false => {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "fit must be a ValleyFit, a ValleyError instance or None",
                            ));
                        }
                    },
                },
                None => None,
            };
            Ok(Self {
                pairs,
                detunings_hz,
                iq,
                fit: owned,
            })
        }

        /// 脉冲对数：这一阶的标号
        #[getter]
        fn pairs(&self) -> usize {
            self.pairs
        }

        fn __repr__(&self) -> String {
            format!(
                "OrderScan(pairs={}, points={}, fitted={})",
                self.pairs,
                self.detunings_hz.len(),
                self.fit.is_some()
            )
        }
    }

    /// 拟合一条载波失谐扫描。
    #[pyfunction]
    #[pyo3(signature = (detunings_hz, iq, states, sigma=None))]
    fn fit(
        detunings_hz: Vec<f64>,
        iq: Vec<lmfit::Complex64>,
        states: PyStateCenters,
        sigma: Option<Vec<f64>>,
    ) -> PyResult<PyValleyFit> {
        let fitted = valley_fit(&detunings_hz, &iq, &states.inner, sigma.as_deref())
            .map_err(|error| to_py_error(&error))?;
        Ok(PyValleyFit { inner: fitted })
    }

    /// 批量拟合：逐条给出结果，失败的那条给**异常类实例**（不抛）。
    #[pyfunction]
    #[pyo3(signature = (detunings_hz, iqs, states, sigmas=None))]
    fn fit_batch(
        py: Python<'_>,
        detunings_hz: Vec<f64>,
        iqs: Vec<Vec<lmfit::Complex64>>,
        states: PyStateCenters,
        sigmas: Option<Vec<Vec<f64>>>,
    ) -> Vec<Py<PyAny>> {
        valley_fit_batch(&detunings_hz, &iqs, &states.inner, sigmas.as_deref())
            .into_iter()
            .map(|outcome| match outcome {
                Ok(fitted) => Py::new(py, PyValleyFit { inner: fitted })
                    .map(|value| value.into_any())
                    .unwrap_or_else(|error| error.into_value(py).into_any()),
                Err(error) => ValleyError::new_err(error.to_string())
                    .into_value(py)
                    .into_any(),
            })
            .collect()
    }

    /// 渲染整条升阶链的报告 div（返回 HTML 片段）。
    #[pyfunction]
    #[pyo3(signature = (orders, chosen, states, div_id="drag-detuning", frame=None))]
    fn plot(
        orders: Vec<PyOrderScan>,
        chosen: f64,
        states: PyStateCenters,
        div_id: &str,
        frame: Option<&str>,
    ) -> String {
        let borrowed: Vec<OrderScan<'_>> = orders
            .iter()
            .map(|scan| OrderScan {
                pairs: scan.pairs,
                detunings_hz: &scan.detunings_hz,
                iq: &scan.iq,
                fit: match &scan.fit {
                    Some(fitted) => Ok(fitted),
                    None => Err(&CoreValleyError::EmptyData),
                },
            })
            .collect();
        drag_detuning_plot_div(&borrowed, chosen, &states.inner, div_id, frame)
    }

    /// 把本子模块的公共项挂到给定模块上。
    pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("ValleyError", py.get_type::<ValleyError>())?;
        module.add_class::<PyValley>()?;
        module.add_class::<PyValleyFit>()?;
        module.add_class::<PyOrderScan>()?;
        module.add_function(wrap_pyfunction!(fit, module)?)?;
        module.add_function(wrap_pyfunction!(fit_batch, module)?)?;
        module.add_function(wrap_pyfunction!(plot, module)?)?;
        Ok(())
    }
}
