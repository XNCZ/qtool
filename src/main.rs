use qtool::qpt::{PtmMatrix, QptDataset, QptSolverConfig};

fn main() {
    // 以库的方式使用 qtool：打印 1 比特单位门 PTM 示例。
    let _cfg = QptSolverConfig::default();
    let _ds = QptDataset::new(1);
    println!("{}", PtmMatrix::identity(1).display_table());
}
