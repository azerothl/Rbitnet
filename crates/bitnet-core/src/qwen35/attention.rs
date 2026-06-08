//! Full multi-head attention (non-recurrent layers), GQA + RoPE (MRoPE approximated as standard RoPE on `rope_dim_pairs`).

use super::config::Qwen35Config;
use super::qmatvec::quant_matmul_vec;
use crate::error::{BitNetError, Result};

#[derive(Clone, Default)]
pub struct AttnKvCache {
    mode: KvMode,
    stride: usize,
    kv_head_span: usize,
}

#[derive(Clone, Default)]
enum KvMode {
    #[default]
    Dense,
    DenseBuffers {
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
    },
    Paged {
        page_size_tokens: usize,
        max_pages: usize,
        layers: Vec<PagedLayerKv>,
    },
}

#[derive(Clone, Default)]
struct PagedLayerKv {
    k_pages: Vec<Vec<f32>>,
    v_pages: Vec<Vec<f32>>,
    token_to_page: Vec<usize>,
    token_to_offset: Vec<usize>,
}

impl AttnKvCache {
    pub fn new(cfg: &Qwen35Config, max_seq: usize) -> Self {
        let stride = cfg.n_head_kv * cfg.head_dim;
        let paged_enabled = matches!(
            std::env::var("RBITNET_PAGED_KV").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        if !paged_enabled {
            let len = stride * max_seq;
            let k = (0..cfg.n_layer).map(|_| vec![0f32; len]).collect();
            let v = (0..cfg.n_layer).map(|_| vec![0f32; len]).collect();
            return Self {
                mode: KvMode::DenseBuffers { k, v },
                stride,
                kv_head_span: cfg.head_dim,
            };
        }

        let page_size_tokens = std::env::var("RBITNET_PAGED_KV_PAGE_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(16);
        let max_pages = std::env::var("RBITNET_PAGED_KV_MAX_PAGES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4096);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(PagedLayerKv {
                k_pages: Vec::new(),
                v_pages: Vec::new(),
                token_to_page: vec![0; max_seq],
                token_to_offset: vec![0; max_seq],
            });
        }
        Self {
            mode: KvMode::Paged {
                page_size_tokens,
                max_pages,
                layers,
            },
            stride,
            kv_head_span: cfg.head_dim,
        }
    }

    pub fn clear(&mut self) {
        match &mut self.mode {
            KvMode::Dense => {}
            KvMode::DenseBuffers { k, v } => {
                for row in k {
                    row.fill(0f32);
                }
                for row in v {
                    row.fill(0f32);
                }
            }
            KvMode::Paged { layers, .. } => {
                for layer in layers {
                    layer.k_pages.clear();
                    layer.v_pages.clear();
                    layer.token_to_page.fill(0);
                    layer.token_to_offset.fill(0);
                }
            }
        }
    }

    fn write_token(
        &mut self,
        il: usize,
        pos: usize,
        k_heads: &[f32],
        v_heads: &[f32],
    ) -> Result<()> {
        match &mut self.mode {
            KvMode::Dense => Err(BitNetError::Inference("kv cache is uninitialized".into())),
            KvMode::DenseBuffers { k, v } => {
                let off = pos
                    .checked_mul(self.stride)
                    .ok_or_else(|| BitNetError::Inference("kv dense offset overflow".into()))?;
                let end = off + self.stride;
                if end > k[il].len() || end > v[il].len() {
                    return Err(BitNetError::Inference(
                        "kv dense write out-of-bounds".into(),
                    ));
                }
                k[il][off..end].copy_from_slice(k_heads);
                v[il][off..end].copy_from_slice(v_heads);
                Ok(())
            }
            KvMode::Paged {
                page_size_tokens,
                max_pages,
                layers,
            } => {
                let layer = layers.get_mut(il).ok_or_else(|| {
                    BitNetError::Inference("kv paged layer index out-of-range".into())
                })?;
                let page_idx = pos / *page_size_tokens;
                let page_off_tokens = pos % *page_size_tokens;
                if page_idx >= *max_pages {
                    return Err(BitNetError::Inference("kv paged exceeded max pages".into()));
                }
                while layer.k_pages.len() <= page_idx {
                    layer
                        .k_pages
                        .push(vec![0f32; self.stride * *page_size_tokens]);
                    layer
                        .v_pages
                        .push(vec![0f32; self.stride * *page_size_tokens]);
                }
                let vec_off = page_off_tokens * self.stride;
                layer.k_pages[page_idx][vec_off..vec_off + self.stride].copy_from_slice(k_heads);
                layer.v_pages[page_idx][vec_off..vec_off + self.stride].copy_from_slice(v_heads);
                if pos < layer.token_to_page.len() {
                    layer.token_to_page[pos] = page_idx;
                    layer.token_to_offset[pos] = vec_off;
                }
                Ok(())
            }
        }
    }

    fn read_k_slice<'a>(
        &'a self,
        il: usize,
        token_pos: usize,
        kv_h: usize,
        kv_head_dim: usize,
    ) -> Result<&'a [f32]> {
        let base = kv_h
            .checked_mul(self.kv_head_span)
            .ok_or_else(|| BitNetError::Inference("kv head base overflow".into()))?;
        match &self.mode {
            KvMode::Dense => Err(BitNetError::Inference("kv cache is uninitialized".into())),
            KvMode::DenseBuffers { k, .. } => {
                let off = token_pos
                    .checked_mul(self.stride)
                    .and_then(|x| x.checked_add(base))
                    .ok_or_else(|| BitNetError::Inference("kv dense offset overflow".into()))?;
                let start = off;
                let end = start + kv_head_dim;
                k.get(il)
                    .and_then(|row| row.get(start..end))
                    .ok_or_else(|| BitNetError::Inference("kv dense read OOB".into()))
            }
            KvMode::Paged {
                page_size_tokens,
                layers,
                ..
            } => {
                let layer = layers.get(il).ok_or_else(|| {
                    BitNetError::Inference("kv paged layer index out-of-range".into())
                })?;
                if token_pos >= layer.token_to_page.len() {
                    return Err(BitNetError::Inference("kv paged token out-of-range".into()));
                }
                let page_idx = layer.token_to_page[token_pos];
                let page = layer
                    .k_pages
                    .get(page_idx)
                    .ok_or_else(|| BitNetError::Inference("kv paged page missing".into()))?;
                let offset = (token_pos % *page_size_tokens) * self.stride + base;
                page.get(offset..offset + kv_head_dim)
                    .ok_or_else(|| BitNetError::Inference("kv paged read OOB".into()))
            }
        }
    }

    fn read_v_slice<'a>(
        &'a self,
        il: usize,
        token_pos: usize,
        kv_h: usize,
        kv_head_dim: usize,
    ) -> Result<&'a [f32]> {
        let base = kv_h
            .checked_mul(self.kv_head_span)
            .ok_or_else(|| BitNetError::Inference("kv head base overflow".into()))?;
        match &self.mode {
            KvMode::Dense => Err(BitNetError::Inference("kv cache is uninitialized".into())),
            KvMode::DenseBuffers { v, .. } => {
                let off = token_pos
                    .checked_mul(self.stride)
                    .and_then(|x| x.checked_add(base))
                    .ok_or_else(|| BitNetError::Inference("kv dense offset overflow".into()))?;
                let start = off;
                let end = start + kv_head_dim;
                v.get(il)
                    .and_then(|row| row.get(start..end))
                    .ok_or_else(|| BitNetError::Inference("kv dense read OOB".into()))
            }
            KvMode::Paged {
                page_size_tokens,
                layers,
                ..
            } => {
                let layer = layers.get(il).ok_or_else(|| {
                    BitNetError::Inference("kv paged layer index out-of-range".into())
                })?;
                if token_pos >= layer.token_to_page.len() {
                    return Err(BitNetError::Inference("kv paged token out-of-range".into()));
                }
                let page_idx = layer.token_to_page[token_pos];
                let page = layer
                    .v_pages
                    .get(page_idx)
                    .ok_or_else(|| BitNetError::Inference("kv paged page missing".into()))?;
                let offset = (token_pos % *page_size_tokens) * self.stride + base;
                page.get(offset..offset + kv_head_dim)
                    .ok_or_else(|| BitNetError::Inference("kv paged read OOB".into()))
            }
        }
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let scale = 1.0 / (s + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * scale)
        .collect()
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
    let mut h = rot_dims.min(slice.len());
    if h < 2 {
        return;
    }
    h -= h % 2;
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * (i as f32) / (h as f32));
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[i];
        let x1 = slice[i + half];
        slice[i] = x0 * c - x1 * s;
        slice[i + half] = x0 * s + x1 * c;
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

    let k_lin = quant_matmul_vec(wk_py, wk_ty, wk_ne0, wk_ne1, x)?;
    let v_lin = quant_matmul_vec(wv_py, wv_ty, wv_ne0, wv_ne1, x)?;
    if cfg.n_head_kv == 0 {
        return Err(BitNetError::Inference(
            "attention: n_head_kv is zero".into(),
        ));
    }
    if k_lin.len() % cfg.n_head_kv != 0 || v_lin.len() % cfg.n_head_kv != 0 {
        return Err(BitNetError::Inference(format!(
            "attention: k/v width mismatch (k={}, v={}, n_head_kv={})",
            k_lin.len(),
            v_lin.len(),
            cfg.n_head_kv
        )));
    }
    let k_stride = k_lin.len() / cfg.n_head_kv;
    let v_stride = v_lin.len() / cfg.n_head_kv;
    let kv_head_dim = cfg.head_dim.min(k_stride).min(v_stride);
    if kv_head_dim == 0 {
        return Err(BitNetError::Inference(
            "attention: computed kv head_dim is zero".into(),
        ));
    }

    let q_full = quant_matmul_vec(wq_py, wq_ty, wq_ne0, wq_ne1, x)?;
    if cfg.n_head == 0 || q_full.len() % cfg.n_head != 0 {
        return Err(BitNetError::Inference(format!(
            "attention: q projection width {} not divisible by n_head {}",
            q_full.len(),
            cfg.n_head
        )));
    }
    let q_stride = q_full.len() / cfg.n_head;
    if q_stride < kv_head_dim {
        return Err(BitNetError::Inference(format!(
            "attention: per-head q stride {} smaller than kv head_dim {}",
            q_stride, kv_head_dim
        )));
    }
    let q_has_gate = q_stride >= kv_head_dim * 2;
    let mut q_heads_actual = vec![0f32; cfg.n_head * kv_head_dim];
    for hid in 0..cfg.n_head {
        let base = hid * q_stride;
        let qh = &q_full[base..base + kv_head_dim];
        let nr = rmsnorm(qh, attn_q_norm_w, cfg.norm_eps);
        q_heads_actual[hid * kv_head_dim..(hid + 1) * kv_head_dim].copy_from_slice(&nr);
    }

    let mut k_heads = vec![0f32; cfg.n_head_kv * cfg.head_dim];
    let mut v_heads = vec![0f32; cfg.n_head_kv * cfg.head_dim];
    for h in 0..cfg.n_head_kv {
        let k_src = h * k_stride;
        let v_src = h * v_stride;
        let s = h * cfg.head_dim;
        let kn = rmsnorm(
            &k_lin[k_src..k_src + kv_head_dim],
            attn_k_norm_w,
            cfg.norm_eps,
        );
        k_heads[s..s + kv_head_dim].copy_from_slice(&kn);
        v_heads[s..s + kv_head_dim].copy_from_slice(&v_lin[v_src..v_src + kv_head_dim]);
    }

    let rot = cfg.rope_dim_pairs.min(kv_head_dim / 2 * 2);

    for h in 0..cfg.n_head {
        let s = h * cfg.head_dim;
        rope_inplace_partial(
            &mut q_heads_actual[s..s + kv_head_dim],
            pos,
            cfg.rope_freq_base,
            rot,
        );
    }
    for h in 0..cfg.n_head_kv {
        let s = h * cfg.head_dim;
        rope_inplace_partial(
            &mut k_heads[s..s + kv_head_dim],
            pos,
            cfg.rope_freq_base,
            rot,
        );
    }

    kv.write_token(il, pos, &k_heads, &v_heads)?;

    if cfg.n_head == 0 {
        return Err(BitNetError::Inference("attention: n_head is zero".into()));
    }
    let attn_out_head_dim = (wo_ne0 / cfg.n_head).max(1);
    let mut attn_out = vec![0f32; wo_ne0];

    for qh in 0..cfg.n_head {
        let kv_h = qh / n_rep;
        let q_slice = &q_heads_actual[qh * cfg.head_dim..qh * cfg.head_dim + kv_head_dim];
        let mut scores = Vec::with_capacity(pos + 1);
        for p in 0..=pos {
            let k_slice = kv.read_k_slice(il, p, kv_h, kv_head_dim)?;
            let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot * cfg.attn_scale);
        }
        softmax_inplace(&mut scores);
        let mut comb = vec![0f32; kv_head_dim];
        for p in 0..=pos {
            let v_slice = kv.read_v_slice(il, p, kv_h, kv_head_dim)?;
            let sp = scores[p];
            for i in 0..kv_head_dim {
                comb[i] += sp * v_slice[i];
            }
        }
        let dst = qh * attn_out_head_dim;
        if dst >= attn_out.len() {
            continue;
        }
        let write_dim = kv_head_dim.min(attn_out_head_dim).min(attn_out.len() - dst);
        for i in 0..write_dim {
            let g = if q_has_gate {
                let gate_base = qh * q_stride + kv_head_dim;
                let gv = q_full[gate_base + i];
                1.0 / (1.0 + (-gv).exp())
            } else {
                1.0
            };
            attn_out[dst + i] = comb[i] * g;
        }
    }

    let y = quant_matmul_vec(wo_py, wo_ty, wo_ne0, wo_ne1, &attn_out)?;
    if y.len() != cfg.n_embd {
        return Err(BitNetError::Inference(
            "attention: wo output width mismatch".into(),
        ));
    }
    Ok(y)
}
