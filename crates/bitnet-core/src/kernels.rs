//! BitNet-style ternary / low-bit linear algebra (NATIVE_FIRST — no bitnet.cpp FFI).
//!
//! Layouts inspired by bitnet.cpp I2_S (2-bit packed ternary) and TL2 (tiled LUT/MAD)
//! ([2502.11880](https://arxiv.org/abs/2502.11880), [2410.16144](https://arxiv.org/abs/2410.16144)).
//! Run `cargo bench -p bitnet-core --bench kernels` or `scripts/bench_bitnet_kernels.sh`.

/// Reference row-wise matrix-vector multiply: `y = W @ x + y` (accumulate).
/// `w` stores `n * k` weights in row-major order, each weight in `{-1, 0, 1}`.
pub fn matvec_accum_ternary_i8(w: &[i8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    assert_eq!(w.len(), n * k);
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    for i in 0..n {
        let mut acc = 0.0f32;
        let row = i * k;
        for j in 0..k {
            acc += w[row + j] as f32 * x[j];
        }
        y[i] += acc;
    }
}

/// Same as [`matvec_accum_ternary_i8`] but overwrites `y` (no bias).
pub fn matvec_ternary_i8(w: &[i8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    y.fill(0.0);
    matvec_accum_ternary_i8(w, x, y, n, k);
}

/// CUDA-priority BitNet kernel entrypoint (MVP): delegates to the reference path
/// while preserving a stable symbol for backend-specific registry dispatch.
pub fn bitnet_cuda_matvec_mvp(w: &[i8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    matvec_ternary_i8(w, x, y, n, k);
}

/// Pack ternary `{-1,0,1}` into 2-bit I2_S-style codes: `0→0b00, 1→0b01, -1→0b10`.
/// Four weights per byte (low bits first). `w.len()` must be a multiple of 4.
pub fn pack_ternary_i2s(w: &[i8]) -> Vec<u8> {
    assert_eq!(w.len() % 4, 0, "I2_S pack requires len % 4 == 0");
    let mut out = Vec::with_capacity(w.len() / 4);
    for chunk in w.chunks_exact(4) {
        let mut b = 0u8;
        for (i, &v) in chunk.iter().enumerate() {
            let code = match v {
                0 => 0u8,
                1 => 1u8,
                -1 => 2u8,
                _ => 0u8,
            };
            b |= code << (2 * i);
        }
        out.push(b);
    }
    out
}

#[inline]
fn i2s_decode(code: u8) -> f32 {
    match code & 0b11 {
        0 => 0.0,
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
}

/// Scalar matvec over I2_S packed weights (bit-exact vs [`matvec_ternary_i8`] for valid ternary).
pub fn matvec_ternary_i2s(packed: &[u8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    assert_eq!(k % 4, 0);
    assert_eq!(packed.len(), n * (k / 4));
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    let row_bytes = k / 4;
    for i in 0..n {
        let mut acc = 0.0f32;
        let base = i * row_bytes;
        for (jb, &byte) in packed[base..base + row_bytes].iter().enumerate() {
            let j0 = jb * 4;
            acc += i2s_decode(byte) * x[j0];
            acc += i2s_decode(byte >> 2) * x[j0 + 1];
            acc += i2s_decode(byte >> 4) * x[j0 + 2];
            acc += i2s_decode(byte >> 6) * x[j0 + 3];
        }
        y[i] = acc;
    }
}

/// TL2-inspired tiled LUT path: per 4-weight tile, local activation LUT + MAD.
pub fn matvec_ternary_tl2_lut(packed: &[u8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    assert_eq!(k % 4, 0);
    assert_eq!(packed.len(), n * (k / 4));
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), n);
    let row_bytes = k / 4;
    for i in 0..n {
        let mut acc = 0.0f32;
        let base = i * row_bytes;
        for (jb, &byte) in packed[base..base + row_bytes].iter().enumerate() {
            let j0 = jb * 4;
            let lut = [0.0f32, x[j0], -x[j0], 0.0];
            let lut1 = [0.0f32, x[j0 + 1], -x[j0 + 1], 0.0];
            let lut2 = [0.0f32, x[j0 + 2], -x[j0 + 2], 0.0];
            let lut3 = [0.0f32, x[j0 + 3], -x[j0 + 3], 0.0];
            acc += lut[(byte & 0b11) as usize];
            acc += lut1[((byte >> 2) & 0b11) as usize];
            acc += lut2[((byte >> 4) & 0b11) as usize];
            acc += lut3[((byte >> 6) & 0b11) as usize];
        }
        y[i] = acc;
    }
}

/// Best available CPU path: prefer TL2 LUT (cache-friendly); AVX2 stub routes to I2_S for now.
pub fn matvec_ternary_auto(packed: &[u8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // AVX2 widening is staged; keep bit-exact I2_S until parity benches expand.
            matvec_ternary_i2s(packed, x, y, n, k);
            return;
        }
    }
    matvec_ternary_tl2_lut(packed, x, y, n, k);
}

/// Wall-time microbench helper used by `scripts/bench_bitnet_kernels.sh`.
pub fn microbench_ternary_ns(n: usize, k: usize, iters: usize) -> TernaryMicrobench {
    assert!(k % 4 == 0 && n > 0 && k > 0 && iters > 0);
    let mut w = vec![0i8; n * k];
    for (i, slot) in w.iter_mut().enumerate() {
        *slot = match i % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }
    let packed = pack_ternary_i2s(&w);
    let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32) * 0.01).collect();
    let mut y = vec![0.0f32; n];

    matvec_ternary_i8(&w, &x, &mut y, n, k);
    matvec_ternary_i2s(&packed, &x, &mut y, n, k);
    matvec_ternary_tl2_lut(&packed, &x, &mut y, n, k);
    matvec_ternary_auto(&packed, &x, &mut y, n, k);

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        matvec_ternary_i8(&w, &x, &mut y, n, k);
    }
    let i8_ns = t0.elapsed().as_nanos() / iters as u128;

    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        matvec_ternary_i2s(&packed, &x, &mut y, n, k);
    }
    let i2s_ns = t1.elapsed().as_nanos() / iters as u128;

    let t2 = std::time::Instant::now();
    for _ in 0..iters {
        matvec_ternary_tl2_lut(&packed, &x, &mut y, n, k);
    }
    let tl2_ns = t2.elapsed().as_nanos() / iters as u128;

    let t3 = std::time::Instant::now();
    for _ in 0..iters {
        matvec_ternary_auto(&packed, &x, &mut y, n, k);
    }
    let auto_ns = t3.elapsed().as_nanos() / iters as u128;

    let mut y_ref = vec![0.0f32; n];
    let mut y_i2s = vec![0.0f32; n];
    let mut y_tl2 = vec![0.0f32; n];
    matvec_ternary_i8(&w, &x, &mut y_ref, n, k);
    matvec_ternary_i2s(&packed, &x, &mut y_i2s, n, k);
    matvec_ternary_tl2_lut(&packed, &x, &mut y_tl2, n, k);
    let bit_exact = y_ref == y_i2s && y_ref == y_tl2;

    TernaryMicrobench {
        n,
        k,
        iters,
        i8_ns,
        i2s_ns,
        tl2_ns,
        auto_ns,
        bit_exact,
        widest_gap_label: widest_gap_label(i8_ns, i2s_ns, tl2_ns, auto_ns),
    }
}

fn widest_gap_label(i8: u128, i2s: u128, tl2: u128, auto: u128) -> &'static str {
    let pairs = [
        ("i8_vs_i2s", i8.abs_diff(i2s)),
        ("i8_vs_tl2", i8.abs_diff(tl2)),
        ("i8_vs_auto", i8.abs_diff(auto)),
        ("i2s_vs_tl2", i2s.abs_diff(tl2)),
    ];
    pairs
        .iter()
        .max_by_key(|(_, d)| *d)
        .map(|(l, _)| *l)
        .unwrap_or("n/a")
}

#[derive(Debug, Clone)]
pub struct TernaryMicrobench {
    pub n: usize,
    pub k: usize,
    pub iters: usize,
    pub i8_ns: u128,
    pub i2s_ns: u128,
    pub tl2_ns: u128,
    pub auto_ns: u128,
    pub bit_exact: bool,
    pub widest_gap_label: &'static str,
}

impl TernaryMicrobench {
    pub fn markdown_row(&self) -> String {
        format!(
            "| ternary {n}x{k} | i8={i8}ns i2s={i2s}ns tl2={tl2}ns auto={auto}ns | bit_exact={be} | widest_gap={gap} | Rust SIMD/LUT (no FFI) |",
            n = self.n,
            k = self.k,
            i8 = self.i8_ns,
            i2s = self.i2s_ns,
            tl2 = self.tl2_ns,
            auto = self.auto_ns,
            be = self.bit_exact,
            gap = self.widest_gap_label,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_matvec() {
        let w = vec![1i8, 0, -1, 1];
        let x = vec![2.0f32, 3.0];
        let mut y = vec![0.0f32; 2];
        matvec_ternary_i8(&w, &x, &mut y, 2, 2);
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cuda_mvp_symbol_matches_reference() {
        let w = vec![1i8, 0, -1, 1];
        let x = vec![2.0f32, 3.0];
        let mut y_ref = vec![0.0f32; 2];
        let mut y_cuda = vec![0.0f32; 2];
        matvec_ternary_i8(&w, &x, &mut y_ref, 2, 2);
        bitnet_cuda_matvec_mvp(&w, &x, &mut y_cuda, 2, 2);
        assert_eq!(y_ref, y_cuda);
    }

    #[test]
    fn i2s_and_tl2_match_i8_reference() {
        let n = 8usize;
        let k = 64usize;
        let mut w = vec![0i8; n * k];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = match i % 5 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }
        let packed = pack_ternary_i2s(&w);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut y0 = vec![0.0f32; n];
        let mut y1 = vec![0.0f32; n];
        let mut y2 = vec![0.0f32; n];
        let mut y3 = vec![0.0f32; n];
        matvec_ternary_i8(&w, &x, &mut y0, n, k);
        matvec_ternary_i2s(&packed, &x, &mut y1, n, k);
        matvec_ternary_tl2_lut(&packed, &x, &mut y2, n, k);
        matvec_ternary_auto(&packed, &x, &mut y3, n, k);
        assert_eq!(y0, y1);
        assert_eq!(y0, y2);
        assert_eq!(y0, y3);
    }

    #[test]
    fn microbench_reports_bit_exact() {
        let r = microbench_ternary_ns(16, 128, 8);
        assert!(r.bit_exact);
        assert!(r.markdown_row().contains("bit_exact=true"));
    }
}
