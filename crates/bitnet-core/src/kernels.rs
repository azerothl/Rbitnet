//! BitNet-style ternary / low-bit linear algebra (NATIVE_FIRST — no bitnet.cpp FFI).
//!
//! Layouts inspired by bitnet.cpp I2_S (2-bit packed ternary) and TL2 (tiled LUT/MAD)
//! ([2502.11880](https://arxiv.org/abs/2502.11880), [2410.16144](https://arxiv.org/abs/2410.16144)).
//! Run `cargo bench -p bitnet-core --bench kernels` or `scripts/bench_bitnet_kernels.sh`.
//!
//! **Scope note:** these helpers are the research / microbench surface for packed ternary.
//! Production Microsoft b1.58 GGUF inference uses mmap **TQ1_0 / TQ2_0** row dots in
//! [`crate::ggml::quant_dot`] — not this I2_S byte layout. Closing the e2e gap vs bitnet.cpp
//! is primarily a *layout/wiring* problem (fuse TQ GEMV + optional repack), not more scalar
//! I2_S micro-opts alone.

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
            acc = (w[row + j] as f32).mul_add(x[j], acc);
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

/// Branchless I2_S code → ternary magnitude in `{0, ±1}` (invalid `0b11` → 0).
#[inline(always)]
fn i2s_signed(code: u8) -> f32 {
    // 0→0, 1→+1, 2→−1, 3→0 without a match jump.
    const TABLE: [f32; 4] = [0.0, 1.0, -1.0, 0.0];
    TABLE[(code & 0b11) as usize]
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
            acc = i2s_signed(byte).mul_add(x[j0], acc);
            acc = i2s_signed(byte >> 2).mul_add(x[j0 + 1], acc);
            acc = i2s_signed(byte >> 4).mul_add(x[j0 + 2], acc);
            acc = i2s_signed(byte >> 6).mul_add(x[j0 + 3], acc);
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
            // One activation table per lane (bitnet.cpp TL2-style local LUT).
            let x0 = x[j0];
            let x1 = x[j0 + 1];
            let x2 = x[j0 + 2];
            let x3 = x[j0 + 3];
            let lut0 = [0.0f32, x0, -x0, 0.0];
            let lut1 = [0.0f32, x1, -x1, 0.0];
            let lut2 = [0.0f32, x2, -x2, 0.0];
            let lut3 = [0.0f32, x3, -x3, 0.0];
            acc += lut0[(byte & 0b11) as usize];
            acc += lut1[((byte >> 2) & 0b11) as usize];
            acc += lut2[((byte >> 4) & 0b11) as usize];
            acc += lut3[((byte >> 6) & 0b11) as usize];
        }
        y[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matvec_ternary_i2s_avx2(
    packed: &[u8],
    x: &[f32],
    y: &mut [f32],
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    let row_bytes = k / 4;
    for i in 0..n {
        let mut acc_vec = _mm256_setzero_ps();
        let mut acc_tail = 0.0f32;
        let base = i * row_bytes;
        let row = &packed[base..base + row_bytes];
        let mut j = 0usize;
        // 8 packed bytes → 32 weights → four AVX2 FMA lanes of 8.
        while j + 8 <= row_bytes {
            let mut decoded = [0.0f32; 32];
            for t in 0..8 {
                let byte = row[j + t];
                let o = t * 4;
                decoded[o] = i2s_signed(byte);
                decoded[o + 1] = i2s_signed(byte >> 2);
                decoded[o + 2] = i2s_signed(byte >> 4);
                decoded[o + 3] = i2s_signed(byte >> 6);
            }
            let j0 = j * 4;
            for lane in 0..4 {
                let w = _mm256_loadu_ps(decoded.as_ptr().add(lane * 8));
                let xv = _mm256_loadu_ps(x.as_ptr().add(j0 + lane * 8));
                acc_vec = _mm256_fmadd_ps(w, xv, acc_vec);
            }
            j += 8;
        }
        while j < row_bytes {
            let byte = row[j];
            let j0 = j * 4;
            acc_tail = i2s_signed(byte).mul_add(x[j0], acc_tail);
            acc_tail = i2s_signed(byte >> 2).mul_add(x[j0 + 1], acc_tail);
            acc_tail = i2s_signed(byte >> 4).mul_add(x[j0 + 2], acc_tail);
            acc_tail = i2s_signed(byte >> 6).mul_add(x[j0 + 3], acc_tail);
            j += 1;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc_vec);
        // Lane reduction order differs from scalar; auto path is approx-equal, not bit-identical.
        y[i] = lanes.iter().copied().sum::<f32>() + acc_tail;
    }
}

/// Best available CPU path: AVX2+FMA I2_S when present, else TL2 LUT.
pub fn matvec_ternary_auto(packed: &[u8], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature gated by runtime detection above.
            unsafe {
                matvec_ternary_i2s_avx2(packed, x, y, n, k);
            }
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
    let mut y_auto = vec![0.0f32; n];
    matvec_ternary_i8(&w, &x, &mut y_ref, n, k);
    matvec_ternary_i2s(&packed, &x, &mut y_i2s, n, k);
    matvec_ternary_tl2_lut(&packed, &x, &mut y_tl2, n, k);
    matvec_ternary_auto(&packed, &x, &mut y_auto, n, k);
    // Scalar I2_S / TL2 stay bit-exact; AVX2 auto may differ by FMA lane reduction order.
    let bit_exact = y_ref == y_i2s
        && y_ref == y_tl2
        && y_ref
            .iter()
            .zip(y_auto.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs())));

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
            "| ternary {n}x{k} | i8={i8}ns i2s={i2s}ns tl2={tl2}ns auto={auto}ns | bit_exact={be} | widest_gap={gap} | Rust AVX2/FMA auto when available (no FFI) |",
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
        for (a, b) in y0.iter().zip(y3.iter()) {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs())),
                "auto approx mismatch {a} vs {b}"
            );
        }
    }

    #[test]
    fn microbench_reports_bit_exact() {
        let r = microbench_ternary_ns(16, 128, 8);
        assert!(r.bit_exact);
        assert!(r.markdown_row().contains("bit_exact=true"));
    }
}
