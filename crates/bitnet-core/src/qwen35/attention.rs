//! Full multi-head attention (non-recurrent layers), GQA + RoPE (MRoPE approximated as standard RoPE on `rope_dim_pairs`).

use crate::error::{BitNetError, Result};
use super::config::Qwen35Config;
use super::qmatvec::quant_matmul_vec;

#[derive(Clone, Default)]
pub struct AttnKvCache {
    /// Per layer: `k` vectors length `max_seq * n_kv * head_dim`.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AttnKvCache {
    pub fn new(cfg: &Qwen35Config, max_seq: usize) -> Self {
        let stride = cfg.n_head_kv * cfg.head_dim;
        let len = stride * max_seq;
        let k = (0..cfg.n_layer).map(|_| vec![0f32; len]).collect();
        let v = (0..cfg.n_layer).map(|_| vec![0f32; len]).collect();
        Self { k, v }
    }

    pub fn clear(&mut self) {
        for row in &mut self.k {
            row.fill(0f32);
        }
        for row in &mut self.v {
            row.fill(0f32);
        }
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let scale = 1.0 / (s + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi * scale).collect()
}

fn softmax_inplace(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for z in s.iter_mut() {
        *z = (*z - m).exp();
        sum += *z;
    }
    if sum > 0.0 {
        for z in s.iter_mut() {
            *z /= sum;
        }
    }
}

fn rope_inplace_partial(slice: &mut [f32], pos: usize, theta: f32, rot_dims: usize) {
    let h = slice.len().min(rot_dims);
    if h < 2 {
        return;
    }
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * (i as f32) / (h.max(2) as f32));
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[2 * i];
        let x1 = slice[2 * i + 1];
        slice[2 * i] = x0 * c - x1 * s;
        slice[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Qwen3 full attention block (Llama-cpp style gate on attention output branches).
#[allow(clippy::too_many_arguments)]
pub fn block_full_attention(
    cfg: &Qwen35Config,
    kv: &mut AttnKvCache,
    il: usize,
    pos: usize,
    x: &[f32],
    wq_py: &[u8],
    wq_ty: u32,
    wq_ne0: usize,
    wq_ne1: usize,
    wk_py: &[u8],
    wk_ty: u32,
    wk_ne0: usize,
    wk_ne1: usize,
    wv_py: &[u8],
    wv_ty: u32,
    wv_ne0: usize,
    wv_ne1: usize,
    wo_py: &[u8],
    wo_ty: u32,
    wo_ne0: usize,
    wo_ne1: usize,
    attn_q_norm_w: &[f32],
    attn_k_norm_w: &[f32],
) -> Result<Vec<f32>> {
    if x.len() != cfg.n_embd {
        return Err(BitNetError::Inference("attention: bad hidden size".into()));
    }
    let n_rep = cfg.n_head / cfg.n_head_kv;

    let q_full = quant_matmul_vec(wq_py, wq_ty, wq_ne0, wq_ne1, x)?;
    if q_full.len() != cfg.n_head * cfg.head_dim * 2 {
        return Err(BitNetError::Inference("attention: unexpected q projection width".into()));
    }

    let mut q_heads_actual = vec![0f32; cfg.n_head * cfg.head_dim];
    for hid in 0..cfg.n_head {
        let base = hid * (cfg.head_dim * 2);
        let qh = &q_full[base..base + cfg.head_dim];
        let nr = rmsnorm(qh, attn_q_norm_w, cfg.norm_eps);
        q_heads_actual[hid * cfg.head_dim..(hid + 1) * cfg.head_dim].copy_from_slice(&nr);
    }

    let k_lin = quant_matmul_vec(wk_py, wk_ty, wk_ne0, wk_ne1, x)?;
    let v_lin = quant_matmul_vec(wv_py, wv_ty, wv_ne0, wv_ne1, x)?;
    if k_lin.len() != cfg.n_head_kv * cfg.head_dim || v_lin.len() != cfg.n_head_kv * cfg.head_dim {
        return Err(BitNetError::Inference("attention: k/v width mismatch".into()));
    }

    let mut k_heads = vec![0f32; cfg.n_head_kv * cfg.head_dim];
    for h in 0..cfg.n_head_kv {
        let s = h * cfg.head_dim;
        let kn = rmsnorm(&k_lin[s..s + cfg.head_dim], attn_k_norm_w, cfg.norm_eps);
        k_heads[s..s + cfg.head_dim].copy_from_slice(&kn);
    }

    let rot = cfg.rope_dim_pairs.min(cfg.head_dim / 2 * 2);

    for h in 0..cfg.n_head {
        let s = h * cfg.head_dim;
        rope_inplace_partial(
            &mut q_heads_actual[s..s + cfg.head_dim],
            pos,
            cfg.rope_freq_base,
            rot,
        );
    }
    for h in 0..cfg.n_head_kv {
        let s = h * cfg.head_dim;
        rope_inplace_partial(&mut k_heads[s..s + cfg.head_dim], pos, cfg.rope_freq_base, rot);
    }

    let stride = cfg.n_head_kv * cfg.head_dim;
    let off = pos * stride;
    kv.k[il][off..off + stride].copy_from_slice(&k_heads);
    kv.v[il][off..off + stride].copy_from_slice(&v_lin);

    let mut attn_out = vec![0f32; cfg.n_head * cfg.head_dim];

    for qh in 0..cfg.n_head {
        let kv_h = qh / n_rep;
        let q_slice = &q_heads_actual[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
        let mut scores: Vec<f32> = (0..=pos)
            .map(|p| {
                let k_off = p * stride + kv_h * cfg.head_dim;
                let k_slice = &kv.k[il][k_off..k_off + cfg.head_dim];
                let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
                dot * cfg.attn_scale
            })
            .collect();
        softmax_inplace(&mut scores);
        let mut comb = vec![0f32; cfg.head_dim];
        for p in 0..=pos {
            let v_off = p * stride + kv_h * cfg.head_dim;
            let v_slice = &kv.v[il][v_off..v_off + cfg.head_dim];
            let sp = scores[p];
            for i in 0..cfg.head_dim {
                comb[i] += sp * v_slice[i];
            }
        }
        let dst = qh * cfg.head_dim;
        let gate_base = qh * (cfg.head_dim * 2) + cfg.head_dim;
        let gate_slice = &q_full[gate_base..gate_base + cfg.head_dim];
        for i in 0..cfg.head_dim {
            let g = 1.0 / (1.0 + (-gate_slice[i]).exp());
            attn_out[dst + i] = comb[i] * g;
        }
    }

    let y = quant_matmul_vec(wo_py, wo_ty, wo_ne0, wo_ne1, &attn_out)?;
    if y.len() != cfg.n_embd {
        return Err(BitNetError::Inference("attention: wo output width mismatch".into()));
    }
    Ok(y)
}
