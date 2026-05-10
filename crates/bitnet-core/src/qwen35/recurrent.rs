//! Recurrent linear-attention block (GDN + depthwise conv), one token at a time.

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

use super::config::Qwen35Config;
use super::gdn::{
    gated_delta_net_step, l2_normalize_vec, silu_inplace, ssm_conv_f32, stitch_conv_window_mut,
};
use super::qmatvec::quant_matmul_vec;

pub struct RecurrentState {
    pub conv_hist: Vec<f32>,
    pub ssm_state: Vec<f32>,
}

/// Output width of `blk.L.attn_gate` for the first recurrent block (GDN value subspace).
pub fn first_recurrent_gate_out_dim(archive: &GgufArchive, cfg: &Qwen35Config) -> Result<usize> {
    for il in 0..cfg.n_layer {
        if cfg.is_recurrent_layer(il) {
            let name = format!("blk.{il}.attn_gate.weight");
            let t = archive.tensor_by_name(&name).ok_or_else(|| {
                BitNetError::Inference(format!("missing `{name}` for recurrent value width"))
            })?;
            if t.dimensions.len() < 2 {
                return Err(BitNetError::Inference(format!("`{name}` expected rank-2")));
            }
            let ne1 = usize::try_from(t.dimensions[1])
                .map_err(|_| BitNetError::Inference("attn_gate ne1".into()))?;
            return Ok(ne1);
        }
    }
    Err(BitNetError::Inference(
        "no recurrent layer in config — cannot infer attn_gate output width".into(),
    ))
}

fn factor_qk_heads(il: usize, key_dim: usize, cfg: &Qwen35Config) -> Result<(usize, usize)> {
    let hk_cfg = cfg.ssm_d_state;
    let nk_cfg = cfg.ssm_n_group;
    if hk_cfg.saturating_mul(nk_cfg) == key_dim {
        return Ok((hk_cfg, nk_cfg));
    }
    if nk_cfg > 0 && key_dim % nk_cfg == 0 {
        return Ok((key_dim / nk_cfg, nk_cfg));
    }
    if hk_cfg > 0 && key_dim % hk_cfg == 0 {
        return Ok((hk_cfg, key_dim / hk_cfg));
    }
    Err(BitNetError::Inference(format!(
        "recurrent layer {il}: key_dim {key_dim} incompatible with ssm state_size={hk_cfg} and group_count={nk_cfg}"
    )))
}

impl RecurrentState {
    pub fn new(d_conv: usize, d_inner: usize, state_dim: usize, num_v_heads: usize) -> Self {
        let hist_len = d_conv.saturating_sub(1).saturating_mul(d_inner);
        let state_elems = state_dim * state_dim * num_v_heads;
        Self {
            conv_hist: vec![0f32; hist_len],
            ssm_state: vec![0f32; state_elems],
        }
    }

    pub fn advance_hist(&mut self, strip: &[f32], d_conv: usize, d_inner: usize) {
        if d_conv <= 1 {
            return;
        }
        let k1 = d_conv - 1;
        let row = d_inner;
        if self.conv_hist.len() != k1 * row {
            return;
        }
        // shift up one row of taps
        if k1 > 1 {
            self.conv_hist.copy_within(row.., 0);
        }
        let base = (k1 - 1) * row;
        self.conv_hist[base..base + row].copy_from_slice(strip);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn recurrent_forward(
    _archive: &GgufArchive,
    cfg: &Qwen35Config,
    st: &mut RecurrentState,
    il: usize,
    x: &[f32],
    wqkv: (&[u8], u32, usize, usize),
    wgate: (&[u8], u32, usize, usize),
    ssm_conv_w: &[f32],
    ssm_conv_d0: usize,
    ssm_conv_d1: usize,
    ssm_beta: (&[u8], u32, usize, usize),
    ssm_alpha: (&[u8], u32, usize, usize),
    ssm_dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_norm: &[f32],
    ssm_out: (&[u8], u32, usize, usize),
) -> Result<Vec<f32>> {
    let d_conv = ssm_conv_d0;
    let d_inner = ssm_conv_d1;
    // `ssm.inner_size` in KV is the mixed QKV width (conv / qkv out), not the value subspace alone.
    // Value width is the output dim of `attn_gate` (GGUF ne1), same as `z` length.
    let value_dim = wgate.3;
    let num_v = cfg.ssm_dt_rank;
    if num_v == 0 {
        return Err(BitNetError::Inference("ssm time_step_rank is zero".into()));
    }
    if value_dim % num_v != 0 {
        return Err(BitNetError::Inference(format!(
            "recurrent layer {il}: attn_gate width {value_dim} not divisible by time_step_rank ({num_v})"
        )));
    }
    let head_v = value_dim / num_v;
    let rest = d_inner.saturating_sub(value_dim);
    if rest % 2 != 0 {
        return Err(BitNetError::Inference(format!(
            "recurrent layer {il}: d_inner {d_inner} minus value_dim {value_dim} must be even (Q+K split)"
        )));
    }
    let key_dim = rest / 2;
    if key_dim.saturating_mul(2).saturating_add(value_dim) != d_inner {
        return Err(BitNetError::Inference(format!(
            "recurrent layer {il}: inner layout d_inner={d_inner}, key_dim={key_dim}, value_dim={value_dim}"
        )));
    }
    let (head_k, num_k) = factor_qk_heads(il, key_dim, cfg)?;

    let qkv_strip = quant_matmul_vec(wqkv.0, wqkv.1, wqkv.2, wqkv.3, x)?;
    if qkv_strip.len() != d_inner {
        return Err(BitNetError::Inference(format!(
            "layer {il}: attn_qkv width {} != d_inner {d_inner}",
            qkv_strip.len()
        )));
    }

    let z = quant_matmul_vec(wgate.0, wgate.1, wgate.2, wgate.3, x)?;
    if z.len() != value_dim {
        return Err(BitNetError::Inference(
            "attn_gate output width mismatch".into(),
        ));
    }

    let beta_logits = quant_matmul_vec(ssm_beta.0, ssm_beta.1, ssm_beta.2, ssm_beta.3, x)?;
    if beta_logits.len() != num_v {
        return Err(BitNetError::Inference("ssm_beta width mismatch".into()));
    }
    let mut beta_v = vec![0f32; num_v];
    for (i, b) in beta_logits.iter().enumerate() {
        beta_v[i] = 1.0 / (1.0 + (-b).exp());
    }

    let alpha = quant_matmul_vec(ssm_alpha.0, ssm_alpha.1, ssm_alpha.2, ssm_alpha.3, x)?;
    if alpha.len() != num_v || ssm_dt_bias.len() != num_v || ssm_a.len() != num_v {
        return Err(BitNetError::Inference(
            "ssm alpha/dt/a shape mismatch".into(),
        ));
    }

    let mut gate = vec![0f32; num_v];
    for h in 0..num_v {
        let t = alpha[h] + ssm_dt_bias[h];
        let sp = softplus_f32(t);
        gate[h] = sp * ssm_a[h];
    }

    let mut window = vec![0f32; d_conv * d_inner];
    if d_conv > 1 {
        let need_hist = (d_conv - 1).saturating_mul(d_inner);
        if st.conv_hist.len() != need_hist {
            return Err(BitNetError::Inference(format!(
                "layer {il}: conv_hist has {} floats; expected {} for d_conv={d_conv}, d_inner={d_inner} \
                 (GGUF `ssm.inner_size` / `ssm.conv_kernel` must match `blk.{il}.ssm_conv1d.weight` shape)",
                st.conv_hist.len(),
                need_hist
            )));
        }
        stitch_conv_window_mut(&st.conv_hist, &qkv_strip, &mut window, d_conv, d_inner);
    } else {
        window[..d_inner].copy_from_slice(&qkv_strip);
    }

    let mut conv_out = ssm_conv_f32(&window, ssm_conv_w, d_conv, d_inner);
    silu_inplace(&mut conv_out);

    let mut q_part = vec![0f32; key_dim];
    let mut k_part = vec![0f32; key_dim];
    let mut v_part = vec![0f32; value_dim];
    q_part.copy_from_slice(&conv_out[0..key_dim]);
    k_part.copy_from_slice(&conv_out[key_dim..key_dim * 2]);
    v_part.copy_from_slice(&conv_out[key_dim * 2..]);

    l2_normalize_vec(&mut q_part, cfg.norm_eps);
    l2_normalize_vec(&mut k_part, cfg.norm_eps);
    l2_normalize_vec(&mut v_part, cfg.norm_eps);

    let mut attn_flat = vec![0f32; value_dim];
    let sv = head_v;
    for h in 0..num_v {
        let q_slice = resize_head_vec(slice_head(&q_part, head_k, num_k, h), sv);
        let k_slice = resize_head_vec(slice_head(&k_part, head_k, num_k, h), sv);
        let v_slice = &v_part[h * sv..(h + 1) * sv];
        let state_off = h * sv * sv;
        let s_in = &st.ssm_state[state_off..state_off + sv * sv];
        let (attn_chunk, s_new) = gated_delta_net_step(
            s_in,
            &q_slice,
            &k_slice,
            v_slice,
            &[gate[h]],
            beta_v[h],
            sv,
            true,
        );
        st.ssm_state[state_off..state_off + sv * sv].copy_from_slice(&s_new);
        attn_flat[h * sv..(h + 1) * sv].copy_from_slice(&attn_chunk);
    }

    let mut normed = vec![0f32; value_dim];
    for h in 0..num_v {
        let a = &attn_flat[h * sv..(h + 1) * sv];
        let zsl = &z[h * sv..(h + 1) * sv];
        let norm_slice: &[f32] = if ssm_norm.len() == sv {
            ssm_norm
        } else if ssm_norm.len() == value_dim {
            &ssm_norm[h * sv..(h + 1) * sv]
        } else {
            return Err(BitNetError::Inference(format!(
                "layer {il}: ssm_norm length {} (expected per-head {sv} or full {value_dim})",
                ssm_norm.len()
            )));
        };
        let r = rmsnorm_small(a, norm_slice, cfg.norm_eps);
        for i in 0..sv {
            let g = zsl[i] / (1.0 + (-zsl[i]).exp());
            normed[h * sv + i] = r[i] * g;
        }
    }

    let ssm_out_in = ssm_out.2;
    let normed_proj = if normed.len() == ssm_out_in {
        normed
    } else if normed.len() < ssm_out_in {
        let mut v = vec![0f32; ssm_out_in];
        v[..normed.len()].copy_from_slice(&normed);
        v
    } else {
        normed[..ssm_out_in].to_vec()
    };
    let y = quant_matmul_vec(ssm_out.0, ssm_out.1, ssm_out.2, ssm_out.3, &normed_proj)?;
    if y.len() != cfg.n_embd {
        return Err(BitNetError::Inference(
            "ssm_out projection width mismatch".into(),
        ));
    }

    st.advance_hist(&qkv_strip, d_conv, d_inner);
    Ok(y)
}

fn softplus_f32(x: f32) -> f32 {
    if x > 35.0 {
        x
    } else if x < -35.0 {
        0.0
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

fn slice_head(buf: &[f32], head_k: usize, num_k: usize, h: usize) -> Vec<f32> {
    let hk = h % num_k;
    let s = hk * head_k;
    buf[s..s + head_k].to_vec()
}

fn resize_head_vec(buf: Vec<f32>, out_len: usize) -> Vec<f32> {
    if buf.len() == out_len {
        return buf;
    }
    if buf.is_empty() {
        return vec![0f32; out_len];
    }
    if buf.len() > out_len {
        let mut v = buf;
        v.truncate(out_len);
        return v;
    }
    let mut out = Vec::with_capacity(out_len);
    while out.len() < out_len {
        for x in buf.iter().copied() {
            out.push(x);
            if out.len() >= out_len {
                break;
            }
        }
    }
    out
}

fn rmsnorm_small(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let sc = 1.0 / (s + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * sc)
        .collect()
}
