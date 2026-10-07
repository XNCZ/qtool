//! 真实数据的功率扫描报告 → plt/s21_power_demo.html。
//!
//! 数据来自一台机器上实测的 S21 夹具（`data/s21/fit_input.bin`，由 baseline 的导出脚本产出），
//! **那份数据不入库**——跑这个 demo 要自备，路径取第一个命令行参数，缺省为
//! `data/s21/fit_input.bin`。页面把 `s21_power_plot_div` 的 div 包成整页（含 plotly.js CDN）。
//!
//! 运行: cargo run --release --example s21_power_demo [数据文件]

use lmfit::ComplexResult;
use qtool::superconductor::s21::{Complex64, S21Error, S21Model, s21_fit};
use qtool::superconductor::s21::s21_plot::PLOTLY_JS_CDN;
use qtool::superconductor::s21::s21_power_plot::{PowerLine, s21_power_plot_div};

/// 数据文件的缺省位置（相对仓库根）；实测数据不入库，跑之前先自备。
const INPUT: &str = "data/s21/fit_input.bin";
/// 每个谐振器 21 档功率。
const LINES_PER_FILE: usize = 21;

/// 顺序读取一个小端 u32，越界返回 None。
fn next_u32(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    let end = *cursor + 4;
    let chunk = match bytes.get(*cursor..end) {
        Some(slice) => slice,
        None => return None,
    };
    let raw: [u8; 4] = match chunk.try_into() {
        Ok(array) => array,
        Err(_) => return None,
    };
    *cursor = end;
    Some(u32::from_le_bytes(raw))
}

/// 顺序读取一个小端 f64，越界返回 None。
fn next_f64(bytes: &[u8], cursor: &mut usize) -> Option<f64> {
    let end = *cursor + 8;
    let chunk = match bytes.get(*cursor..end) {
        Some(slice) => slice,
        None => return None,
    };
    let raw: [u8; 8] = match chunk.try_into() {
        Ok(array) => array,
        Err(_) => return None,
    };
    *cursor = end;
    Some(f64::from_le_bytes(raw))
}

/// 读取数据文件，返回 (线数, 频点数, 全部 f64 负载)。
fn read_input(path: &str) -> Result<(usize, usize, Vec<f64>), String> {
    let bytes = std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))?;
    let mut cursor = 0_usize;
    let n_lines = match next_u32(&bytes, &mut cursor) {
        Some(value) => value as usize,
        None => return Err("missing n_lines in header".to_string()),
    };
    let n_freq = match next_u32(&bytes, &mut cursor) {
        Some(value) => value as usize,
        None => return Err("missing n_freq in header".to_string()),
    };
    let mut values = Vec::with_capacity(n_lines * 4 * n_freq);
    while cursor < bytes.len() {
        match next_f64(&bytes, &mut cursor) {
            Some(value) => values.push(value),
            None => return Err("payload length is not a multiple of 8".to_string()),
        }
    }
    Ok((n_lines, n_freq, values))
}

/// 第 index 档的读出泵幅：geomspace(0.01, 0.5, 21)。
fn power_of(index: usize) -> f64 {
    0.01 * (0.5_f64 / 0.01).powf(index as f64 / (LINES_PER_FILE - 1) as f64)
}

fn main() {
    // 数据路径：第一个命令行参数优先，缺省用仓库里的相对路径
    let path = match std::env::args().nth(1) {
        Some(given) => given,
        None => INPUT.to_string(),
    };
    let (n_lines, n_freq, values) = match read_input(&path) {
        Ok(parsed) => parsed,
        Err(err) => {
            println!("{err}");
            return;
        }
    };
    // 文件里每 LINES_PER_FILE 条线是一个谐振器的完整功率扫描；只出第一张卡（想看别的谐振器
    // 把下面的上界改成 resonators 即可）
    let resonators = n_lines / LINES_PER_FILE;
    if resonators == 0 {
        println!("input has only {n_lines} lines");
        return;
    }

    let mut divs = String::new();
    for file in 0..1 {
        // 该谐振器的 21 条线（频率轴逐行相同，取第一行）
        let first_base = file * LINES_PER_FILE * 4 * n_freq;
        let freqs_hz = &values[first_base..first_base + n_freq];
        let mut iq_lines: Vec<Vec<Complex64>> = Vec::with_capacity(LINES_PER_FILE);
        let mut sigma_lines: Vec<Vec<f64>> = Vec::with_capacity(LINES_PER_FILE);
        for power in 0..LINES_PER_FILE {
            let base = (file * LINES_PER_FILE + power) * 4 * n_freq;
            let re = &values[base + n_freq..base + 2 * n_freq];
            let im = &values[base + 2 * n_freq..base + 3 * n_freq];
            let sigma = &values[base + 3 * n_freq..base + 4 * n_freq];
            iq_lines.push((0..n_freq).map(|k| Complex64::new(re[k], im[k])).collect());
            sigma_lines.push(sigma.to_vec());
        }

        let fits: Vec<Result<ComplexResult<S21Model>, S21Error>> = (0..LINES_PER_FILE)
            .map(|power| s21_fit(freqs_hz, &iq_lines[power], Some(&sigma_lines[power])))
            .collect();
        for (power, fit) in fits.iter().enumerate() {
            match fit {
                Ok(result) => println!(
                    "res{file} p{power:02}: nfev={} chisqr={:.3e}",
                    result.nfev, result.chisqr
                ),
                Err(err) => println!("res{file} p{power:02}: {err}"),
            }
        }

        let lines: Vec<PowerLine<'_>> = (0..LINES_PER_FILE)
            .map(|power| PowerLine {
                power: power_of(power),
                iq: &iq_lines[power],
                sigma: Some(&sigma_lines[power]),
                fit: fits[power].as_ref(),
            })
            .collect();

        let div_id = format!("power-res{file}");
        let title = format!("s21 vs power — res{file}");
        divs.push_str(&s21_power_plot_div(
            freqs_hz,
            &lines,
            &div_id,
            Some(title.as_str()),
        ));
        divs.push_str("<hr style=\"border:none;border-top:1px solid #e5e7eb;margin:24px 0\">\n");
    }

    let html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>qtool s21 power report</title>\n{PLOTLY_JS_CDN}\n</head>\n\
         <body style=\"margin:24px;background:#ffffff\">\n{divs}</body>\n</html>\n"
    );
    match std::fs::create_dir_all("plt") {
        Ok(()) => {}
        Err(err) => {
            println!("failed to create plt/: {err}");
            return;
        }
    }
    match std::fs::write("plt/s21_power_demo.html", html) {
        Ok(()) => println!("wrote plt/s21_power_demo.html"),
        Err(err) => println!("failed to write plt/s21_power_demo.html: {err}"),
    }
}
