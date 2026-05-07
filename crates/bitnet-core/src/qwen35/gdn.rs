//! Gated Delta Net core (`ggml` reference: `ggml_compute_forward_ssm_conv`, `ggml_compute_forward_gated_delta_net`).
//!
//! Single-token recurrence (`n_t = n_seqs = 1`).

/// Depthwise causal conv (`ggml_ssm_conv`) with F32 tensors.
///
/// `window` packs the last `d_conv` taps per inner channel (`window[i0 + i1 * d_conv]`).
/// `kernel` packs weights the same way (`kernel[i0 + i1 * d_conv]`).
pub fn ssm_conv_f32(window: &[f32], kernel: &[f32], d_conv: usize, d_inner: usize) -> Vec<f32> {
    debug_assert_eq!(window.len(), d_conv.saturating_mul(d_inner));
    debug_assert_eq!(kernel.len(), d_conv.saturating_mul(d_inner));
    let mut out = vec![0f32; d_inner];
    for i1 in 0..d_inner {
        let mut sum = 0f32;
        let ncs = d_conv;
        for i0 in 0..d_conv {
            sum += window[i0 + i1 * ncs] * kernel[i0 + i1 * ncs];
        }
        out[i1] = sum;
    }
    out
}

/// Concatenate recurrent tail `hist` `[d_conv - 1][d_inner]` and current strip `strip` `[1][d_inner]`
/// into `window` `[d_conv][d_inner]` ordered oldest → newest tap along `i0`.
pub fn stitch_conv_window_mut(hist: &[f32], strip: &[f32], dst: &mut [f32], d_conv: usize, d_inner: usize) {
    let expect = (d_conv - 1).saturating_mul(d_inner).saturating_add(d_inner.min(d_inner));
    debug_assert_eq!(hist.len().saturating_add(strip.len()), d_conv.saturating_mul(d_inner));
    debug_assert!(dst.len() >= d_conv * d_inner || d_conv == 0);
    if d_conv == 0 {
        return;
    }
    let _ = expect;
    let k1 = d_conv.saturating_sub(1);
    dst[..k1 * d_inner].copy_from_slice(&hist[..k1 * d_inner.max(1)]);
    let base = k1 * d_inner;
    dst[base..base + d_inner].copy_from_slice(strip);
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply `ggml_gated_delta_net` for one timestep; returns attention chunk `[S_v]` and writes new state `[S_v*S_v]`.
///
/// Layout matches GGML contiguous-row storage with transposed conceptual state (`s[row * S_v + col]` corresponds to logical `S[col][row]`).
pub fn gated_delta_net_step(
    state_in: &[f32], // S_v*S_v
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32], // scalar or length S_v (`kda` path)
    beta: f32,
    sv: usize,
    scale_over_sqrt_sv: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut s_out = vec![0f32; sv * sv];
    s_out.copy_from_slice(state_in);
    let kda = gate.len() == sv;
    if kda {
        let mut scratch = vec![0f32; sv];
        for i in 0..sv {
            scratch[i] = (gate[i]).exp();
        }
        for j in 0..sv {
            for i in 0..sv {
                let idx = j * sv + i;
                s_out[idx] *= scratch[i];
            }
        }
    } else {
        let g = (gate.get(0).copied().unwrap_or(0f32)).exp();
        for x in &mut s_out {
            *x *= g;
        }
    }

    let mut delta = vec![0f32; sv];
    for j in 0..sv {
        let mut dot = 0f32;
        let row_off = j * sv;
        for i in 0..sv {
            dot += s_out[row_off + i] * k[i];
        }
        delta[j] = (v[j] - dot) * beta;
    }
    for j in 0..sv {
        let row_off = j * sv;
        for i in 0..sv {
            s_out[row_off + i] += k[i] * delta[j];
        }
    }

    let scale = if scale_over_sqrt_sv {
        1.0 / (sv as f32).sqrt()
    } else {
        1.0
    };

    let mut attn = vec![0f32; sv];
    for j in 0..sv {
        let mut sum = 0f32;
        let row_off = j * sv;
        for i in 0..sv {
            sum += s_out[row_off + i] * q[i];
        }
        attn[j] = sum * scale;
    }
    (attn, s_out)
}

pub(crate) fn l2_normalize_vec(x: &mut [f32], eps: f32) {
    let s: f32 = x.iter().map(|v| v * v).sum::<f32>();
    let inv = 1.0 / (s.max(eps)).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub(crate) fn silu_inplace(x: &mut [f32]) {
    for v in x {
        *v = silu(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssm_conv_smoke_matches_dot() {
        let d_conv = 2;
        let d_inner = 1;
        let window = vec![1.0, 2.0];
        let kernel = vec![3.0, 4.0];
        let out = ssm_conv_f32(&window, &kernel, d_conv, d_inner);
        assert!((out[0] - 11.0).abs() < 1e-4);
        assert_eq!(out.len(), d_inner);
    }

    #[test]
    fn gdn_updates_state() {
        let sv = 2usize;
        let s0 = vec![0f32; sv * sv];
        let q = vec![1.0, 0.0];
        let k = vec![0.0, 1.0];
        let v = vec![3.0, 4.0];
        let (attn, s1) = gated_delta_net_step(&s0, &q, &k, &v, &[0.0], 1.0, sv, false);
        assert!(s1.iter().chain(attn.iter()).all(|z| z.is_finite()));
    }
}
