//! 以外部 crate 方式使用 qtool 公共模块的冒烟测试。
//!
//! 此处代码位于 `tests/`，会被编译成独立 crate 链接 qtool 库，
//! 从而验证 `use qtool::qpt::*;` 的对外可用性。

use qtool::qpt::*;

#[test]
fn qpt_module_public_api_is_reachable() -> Result<(), QPTError> {
    // 基础类型
    let _ = SingleQubitState::PlusZ;
    let _ = Pauli::X;
    let _ = "Z+".parse::<QuantumState>().unwrap();
    let _ = PauliString::from_index(1, 0);
    let _cfg = QptSolverConfig::default();

    // 数据 → 求解 → 结果 主链路
    let mut ds = QptDataset::new(1);
    ds.add_data("Z+", "Z", 1.0)?
        .add_data("Z-", "Z", 0.0)?
        .add_data("X+", "X", 0.0)?
        .add_data("X-", "X", 1.0)?
        .add_data("Y+", "Y", 1.0)?
        .add_data("Y-", "Y", 0.0)?;

    let result: QptResult = QptSolver::new(1, None).solve(&ds)?;
    let _: &PtmMatrix = &result.ptm;
    let _: &QptDiagnostics = &result.diagnostics;
    let _: &QptSolverMeta = &result.meta;

    // PTM 查询接口：X 门 PTM 应为 diag(1, 1, -1, -1)
    let ptm = &result.ptm;
    assert!((ptm.get("I", "I")? - 1.0).abs() < 1e-3);
    assert!((ptm.get("X", "X")? - 1.0).abs() < 1e-3);
    assert!((ptm.get("Y", "Y")? - (-1.0)).abs() < 1e-3);
    assert!((ptm.get("Z", "Z")? - (-1.0)).abs() < 1e-3);
    Ok(())
}
