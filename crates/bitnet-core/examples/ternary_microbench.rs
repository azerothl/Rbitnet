fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let r = bitnet_core::kernels::microbench_ternary_ns(n, k, iters);
    println!("{}", r.markdown_row());
    eprintln!(
        "widest_gap={} bit_exact={}",
        r.widest_gap_label, r.bit_exact
    );
    if !r.bit_exact {
        std::process::exit(1);
    }
}
