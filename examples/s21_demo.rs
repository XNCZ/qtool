//! S21 拟合报告 div 的演示：成功与失败各渲染一个 div，写出可直接打开的 HTML。
//!
//! 运行: cargo run --release --example s21_demo
//! 输出 s21_demo.html（`.gitignore` 已忽略 `*.html`）。页面用 [`PLOTLY_JS_CDN`]
//! 引入 plotly.js，需要能访问 cdn.plot.ly。

use qtool::superconductor::s21::{Complex64, S21Model, model_at, s21_fit};
use qtool::superconductor::s21::s21_plot::{PLOTLY_JS_CDN, s21_fit_plot_div};

/// 与其它 example 一致的确定性均匀噪声源（取值 [-0.5, 0.5)）。
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new() -> Self {
        Self { state: 0 }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5
    }
}

fn main() {
    let truth = S21Model::from_fit_array(&[
        6.8982e9, 8.0e3, -1.4e4, 6.02, 0.1, -2.4e-7, 4.4e9, 4.4e9, 10521.0, -3.0e6, 8.2e6,
    ]);
    let freqs: Vec<f64> = (0..51)
        .map(|i| 6.8982e9 - 5e6 + 10e6 * i as f64 / 50.0)
        .collect();
    // 噪声量级取 |S21| 峰值的 ~0.7%（夹具那套 4e-3 相对 4.4e9 近似无噪声，
    // error bar 会看不见）；σ 与噪声一致，既用于加权拟合也画成 error bar
    let noise = 5e7_f64;
    let mut lcg = Lcg::new();
    let iq: Vec<Complex64> = freqs
        .iter()
        .map(|f| model_at(*f, &truth) + Complex64::new(noise * lcg.next_unit(), noise * lcg.next_unit()))
        .collect();
    let sigma = vec![noise; freqs.len()];

    // 成功分支：正常拟合 → 四面板 + 参数表
    let ok_div = match s21_fit(&freqs, &iq, Some(&sigma)) {
        Ok(result) => {
            println!("success branch: nfev={} chisqr={:.3e}", result.nfev, result.chisqr);
            s21_fit_plot_div(
                &freqs,
                &iq,
                Some(&sigma),
                Ok(&result),
                "demo-ok",
                Some("fit: ok"),
            )
        }
        Err(err) => {
            println!("demo data unexpectedly failed to fit: {err}");
            String::new()
        }
    };

    // 失败分支只报数字、不上卡片：常数谱（无 notch 结构）会让初值估计失效，s21_fit 返回 Err。
    // `*_plot_div` 的 `fit` 参数收 `Result`，真要看那张错误卡自己换成 Err 分支即可。
    let flat: Vec<Complex64> = freqs.iter().map(|_| Complex64::new(1.0, 0.0)).collect();
    match s21_fit(&freqs, &flat, None) {
        Ok(result) => println!("failure branch: constant spectrum unexpectedly fitted, nfev={}", result.nfev),
        Err(err) => println!("failure branch: {err}"),
    }

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool s21 demo</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{ok_div}\n</body>\n</html>\n"
    );
    // 产物统一落在 plt/（生成物集中一处，可整目录 gitignore）
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/s21_demo.html", html) {
        Ok(()) => println!("wrote plt/s21_demo.html"),
        Err(err) => println!("failed to write plt/s21_demo.html: {err}"),
    }
}
