use qtool::qpt::{PtmMatrix, QptDataset, QptSolverConfig};

fn main() {
    // Use qtool as a library: print the 1-qubit identity-gate PTM as a demo.
    let _cfg = QptSolverConfig::default();
    let _ds = QptDataset::new(1);
    println!("{}", PtmMatrix::identity(1).display_table());
}
