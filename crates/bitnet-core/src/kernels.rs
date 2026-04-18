//! BitNet-style ternary / low-bit linear algebra (reference implementations).
//!
//! Production paths will add SIMD (`std::arch`) and packed layouts matching
//! `microsoft/BitNet` preset kernels. Run `cargo bench -p bitnet-core` for baselines.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_matvec() {
        // W = [[1, 0], [-1, 1]]  (2x2)
        let w = vec![1i8, 0, -1, 1];
        let x = vec![2.0f32, 3.0];
        let mut y = vec![0.0f32; 2];
        matvec_ternary_i8(&w, &x, &mut y, 2, 2);
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-6);
    }
}
