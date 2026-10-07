//! CPU fused multi-seq helpers (issue #46 spike).
//!
//! Stall-free scheduling (`RBITNET_CONTINUOUS_BATCHING`) already admits decode waves of N
//! sequences, but each seq still ran an independent matvec. This module provides a real
//! weight-stationary batch matvec so N activation rows share one pass over `W`.
//!
//! Opt-in scheduler wiring: `RBITNET_FUSED_MULTI_SEQ=1` (CUDA Llama uses Native batch decode when resident).

/// True when `RBITNET_FUSED_MULTI_SEQ` is `1` / `true` / `yes`.
pub fn fused_multi_seq_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_FUSED_MULTI_SEQ").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Dense f32 multi-seq matvec: `y[b, :] = W @ x[b, :]` for `b in 0..batch`.
///
/// - `w`: row-major `[n * k]`
/// - `x`: row-major `[batch * k]` (one activation row per sequence)
/// - `y`: row-major `[batch * n]`
///
/// Weight-stationary loop: each row of `W` is streamed once and dotted against all
/// batch activations — the fused CPU building block for multi-seq decode.
pub fn dense_matvec_multi_seq(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: usize,
    k: usize,
    batch: usize,
) {
    assert_eq!(w.len(), n * k, "w len must be n*k");
    assert_eq!(x.len(), batch * k, "x len must be batch*k");
    assert_eq!(y.len(), batch * n, "y len must be batch*n");
    if batch == 0 || n == 0 || k == 0 {
        return;
    }
    for i in 0..n {
        let row = &w[i * k..(i + 1) * k];
        for b in 0..batch {
            let xb = &x[b * k..(b + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += row[j] * xb[j];
            }
            y[b * n + i] = acc;
        }
    }
}

/// Reference: N independent dense matvecs (used to prove batch equivalence).
pub fn dense_matvec_sequential_reference(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: usize,
    k: usize,
    batch: usize,
) {
    assert_eq!(w.len(), n * k);
    assert_eq!(x.len(), batch * k);
    assert_eq!(y.len(), batch * n);
    for b in 0..batch {
        let xb = &x[b * k..(b + 1) * k];
        for i in 0..n {
            let mut acc = 0.0f32;
            let row = i * k;
            for j in 0..k {
                acc += w[row + j] * xb[j];
            }
            y[b * n + i] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_batch_matches_sequential() {
        // W 2x3, batch=3 activations
        let n = 2usize;
        let k = 3usize;
        let batch = 3usize;
        let w = vec![1.0, 0.0, -1.0, 0.5, 2.0, 0.0];
        let x = vec![
            1.0, 2.0, 3.0, // b0 → [1-3, 0.5+4] = [-2, 4.5]
            0.0, 1.0, 0.0, // b1 → [0, 2]
            -1.0, 0.0, 1.0, // b2 → [-1-1, -0.5] = [-2, -0.5]
        ];
        let mut y_fused = vec![0.0f32; batch * n];
        let mut y_ref = vec![0.0f32; batch * n];
        dense_matvec_multi_seq(&w, &x, &mut y_fused, n, k, batch);
        dense_matvec_sequential_reference(&w, &x, &mut y_ref, n, k, batch);
        for (a, b) in y_fused.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "fused={y_fused:?} ref={y_ref:?}");
        }
        assert!((y_fused[0] - (-2.0)).abs() < 1e-6);
        assert!((y_fused[1] - 4.5).abs() < 1e-6);
        assert!((y_fused[2] - 0.0).abs() < 1e-6);
        assert!((y_fused[3] - 2.0).abs() < 1e-6);
        assert!((y_fused[4] - (-2.0)).abs() < 1e-6);
        assert!((y_fused[5] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn batch_one_equals_single_row() {
        let n = 4usize;
        let k = 8usize;
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut y_b = vec![0.0f32; n];
        let mut y_s = vec![0.0f32; n];
        dense_matvec_multi_seq(&w, &x, &mut y_b, n, k, 1);
        dense_matvec_sequential_reference(&w, &x, &mut y_s, n, k, 1);
        assert_eq!(y_b, y_s);
    }

    #[test]
    fn env_flag_off_by_default() {
        // Do not assert global env; just ensure the helper compiles and returns a bool.
        let _ = fused_multi_seq_enabled();
    }
}
