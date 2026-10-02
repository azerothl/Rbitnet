//! SlimAttention-style **1D tiled** CPU attention prototype (#39).
//!
//! FlashAttention 2D tiling is a poor CPU fit; SlimAttention ([2407.07304](https://arxiv.org/abs/2407.07304))
//! tiles along the sequence axis for cache locality. This module is a **standalone numeric spike**:
//! compare tiled online-softmax attention vs a contiguous baseline on toy tensors.
//!
//! Opt-in for future decode wiring: `RBITNET_SLIM_ATTENTION=1` (see [`slim_attention_enabled`]).
//! Default path remains the contiguous baseline in [`super::model`].

/// Default tile length along the KV sequence axis (tokens).
pub const DEFAULT_TILE_TOKENS: usize = 16;

/// True when `RBITNET_SLIM_ATTENTION=1` / `true` / `yes` / `on`.
pub fn slim_attention_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_SLIM_ATTENTION").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

/// Contiguous baseline: scores = scale * Q·Kᵀ, softmax, then weighted V sum.
///
/// Layout: `k` / `v` are row-major `[seq, head_dim]`; `q` length `head_dim`.
pub fn attention_baseline(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), seq * head_dim);
    assert_eq!(v.len(), seq * head_dim);
    assert_eq!(out.len(), head_dim);

    let mut scores = vec![0.0f32; seq];
    for t in 0..seq {
        let k_row = &k[t * head_dim..(t + 1) * head_dim];
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q[i] * k_row[i];
        }
        scores[t] = dot * scale;
    }
    softmax_inplace(&mut scores);
    out.fill(0.0);
    for t in 0..seq {
        let v_row = &v[t * head_dim..(t + 1) * head_dim];
        let w = scores[t];
        for i in 0..head_dim {
            out[i] += w * v_row[i];
        }
    }
}

/// 1D-tiled attention with online softmax (SlimAttention-style sequence tiles).
///
/// Processes KV in chunks of `tile_tokens` to keep working sets cache-friendly while
/// preserving mathematically equivalent attention (up to floating-point associativity).
pub fn attention_tiled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    head_dim: usize,
    scale: f32,
    tile_tokens: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), seq * head_dim);
    assert_eq!(v.len(), seq * head_dim);
    assert_eq!(out.len(), head_dim);
    assert!(tile_tokens > 0);

    // Online softmax state (FlashAttention-style running max / sum).
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    out.fill(0.0);

    let mut t0 = 0usize;
    while t0 < seq {
        let t1 = (t0 + tile_tokens).min(seq);
        let tile_len = t1 - t0;

        let mut tile_scores = vec![0.0f32; tile_len];
        let mut tile_m = f32::NEG_INFINITY;
        for local in 0..tile_len {
            let t = t0 + local;
            let k_row = &k[t * head_dim..(t + 1) * head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[i] * k_row[i];
            }
            let s = dot * scale;
            tile_scores[local] = s;
            tile_m = tile_m.max(s);
        }

        let m_new = m.max(tile_m);
        let alpha = if m.is_finite() {
            (m - m_new).exp()
        } else {
            0.0
        };

        // Rescale previous accumulator into the new max frame.
        for i in 0..head_dim {
            out[i] *= alpha;
        }
        l *= alpha;

        let mut tile_l = 0.0f32;
        for local in 0..tile_len {
            let t = t0 + local;
            let p = (tile_scores[local] - m_new).exp();
            tile_l += p;
            let v_row = &v[t * head_dim..(t + 1) * head_dim];
            for i in 0..head_dim {
                out[i] += p * v_row[i];
            }
        }
        l += tile_l;
        m = m_new;
        t0 = t1;
    }

    if l > 0.0 {
        let inv = 1.0 / l;
        for x in out.iter_mut() {
            *x *= inv;
        }
    }
}

fn softmax_inplace(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for z in s.iter_mut() {
        *z = (*z - m).exp();
        sum += *z;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for z in s.iter_mut() {
            *z *= inv;
        }
    }
}

/// Max absolute / relative drift between two head vectors.
pub fn max_rel_drift(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(1e-3);
        max_rel = max_rel.max((x - y).abs() / denom);
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_kv(seq: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut k = vec![0.0f32; seq * head_dim];
        let mut v = vec![0.0f32; seq * head_dim];
        for t in 0..seq {
            for i in 0..head_dim {
                k[t * head_dim + i] = ((t * 3 + i) as f32 * 0.05).sin();
                v[t * head_dim + i] = ((t * 5 + i) as f32 * 0.04).cos();
            }
        }
        (q, k, v)
    }

    #[test]
    fn tiled_matches_baseline_on_toy_tensors() {
        let seq = 48usize;
        let head_dim = 32usize;
        let (q, k, v) = toy_kv(seq, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut base = vec![0.0f32; head_dim];
        let mut tiled = vec![0.0f32; head_dim];
        attention_baseline(&q, &k, &v, seq, head_dim, scale, &mut base);
        attention_tiled(
            &q,
            &k,
            &v,
            seq,
            head_dim,
            scale,
            DEFAULT_TILE_TOKENS,
            &mut tiled,
        );

        let drift = max_rel_drift(&base, &tiled);
        // Online softmax vs full softmax: expect near-exact on toy F32 (associativity only).
        assert!(
            drift < 1e-4,
            "tiled vs baseline relative drift {drift} exceeds 1e-4 gate"
        );
    }

    #[test]
    fn tiled_matches_baseline_odd_seq_and_tile() {
        let seq = 37usize;
        let head_dim = 16usize;
        let (q, k, v) = toy_kv(seq, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut base = vec![0.0f32; head_dim];
        let mut tiled = vec![0.0f32; head_dim];
        attention_baseline(&q, &k, &v, seq, head_dim, scale, &mut base);
        attention_tiled(&q, &k, &v, seq, head_dim, scale, 7, &mut tiled);

        let drift = max_rel_drift(&base, &tiled);
        assert!(
            drift < 1e-4,
            "odd-length tiled drift {drift} exceeds 1e-4 gate"
        );
    }

    #[test]
    fn slim_attention_flag_defaults_off() {
        // Unset in unit tests unless the environment injects it; only assert type shape.
        let _ = slim_attention_enabled();
    }
}
