//! Llama forward (dequantized F32 weights).

use crate::error::{BitNetError, Result};
use crate::ggml::tensor_to_f32;
use crate::gguf::GgufArchive;

use super::config::LlamaConfig;

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub token_embd: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,
}

pub struct KvCache {
    /// Per layer: flattened `k` / `v` with stride `n_kv * head_dim` per sequence position.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig) -> Self {
        let stride = cfg.n_kv * cfg.head_dim;
        let len = stride * cfg.max_seq;
        let k = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        let v = (0..cfg.n_layer).map(|_| vec![0.0f32; len]).collect();
        Self { k, v }
    }

    pub fn clear(&mut self) {
        for row in &mut self.k {
            row.fill(0.0);
        }
        for row in &mut self.v {
            row.fill(0.0);
        }
    }
}

fn load_tensor(archive: &GgufArchive, name: &str) -> Result<Vec<f32>> {
    let t = archive
        .tensor_by_name(name)
        .ok_or_else(|| BitNetError::Inference(format!("missing tensor {name}")))?;
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

/// `y[out] = sum_i W[i + out * n_embd] * x[i]` — GGUF layout `ne[0]=n_embd`, `ne[1]=out`.
fn matvec_embd_out(w: &[f32], x: &[f32], n_embd: usize, n_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_out];
    for o in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n_embd {
            acc += w[i + o * n_embd] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// `ffn_down`: `ne[0]=n_ff`, `ne[1]=n_embd` — `y[out] = sum_i W[i + out * n_ff] * x[i]`.
fn matvec_ff_embd(w: &[f32], x: &[f32], n_ff: usize, n_embd: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_embd];
    for o in 0..n_embd {
        let mut acc = 0.0f32;
        for i in 0..n_ff {
            acc += w[i + o * n_ff] * x[i];
        }
        y[o] = acc;
    }
    y
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
    let mut sum = 0.0f32;
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

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v / (1.0 + (-v).exp()))
        .collect()
}

fn rope_inplace(slice: &mut [f32], pos: usize, theta: f32) {
    let h = slice.len();
    assert!(h % 2 == 0);
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * (i as f32) / (h as f32));
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[2 * i];
        let x1 = slice[2 * i + 1];
        slice[2 * i] = x0 * c - x1 * s;
        slice[2 * i + 1] = x0 * s + x1 * c;
    }
}

impl LlamaModel {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive)?;
        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = load_tensor(archive, "token_embd.weight")?;
        if token_embd.len() != n_embd * n_vocab {
            return Err(BitNetError::Inference(
                "token_embd.weight element count mismatch".into(),
            ));
        }

        let output_norm = load_tensor(archive, "output_norm.weight")?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = load_tensor(archive, "output.weight")?;
        if output.len() != n_embd * n_vocab {
            return Err(BitNetError::Inference("output.weight shape mismatch".into()));
        }

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = load_tensor(archive, &format!("{p}.attn_norm.weight"))?;
            let wq = load_tensor(archive, &format!("{p}.attn_q.weight"))?;
            let wk = load_tensor(archive, &format!("{p}.attn_k.weight"))?;
            let wv = load_tensor(archive, &format!("{p}.attn_v.weight"))?;
            let wo = load_tensor(archive, &format!("{p}.attn_output.weight"))?;
            let ffn_norm = load_tensor(archive, &format!("{p}.ffn_norm.weight"))?;
            let ffn_gate = load_tensor(archive, &format!("{p}.ffn_gate.weight"))?;
            let ffn_up = load_tensor(archive, &format!("{p}.ffn_up.weight"))?;
            let ffn_down = load_tensor(archive, &format!("{p}.ffn_down.weight"))?;

            if attn_norm.len() != n_embd
                || wq.len() != n_embd * n_embd
                || wk.len() != n_embd * n_embd_kv
                || wv.len() != n_embd * n_embd_kv
                || wo.len() != n_embd * n_embd
                || ffn_norm.len() != n_embd
                || ffn_gate.len() != n_embd * n_ff
                || ffn_up.len() != n_embd * n_ff
                || ffn_down.len() != n_ff * n_embd
            {
                return Err(BitNetError::Inference(format!(
                    "layer {i} weight shape mismatch"
                )));
            }

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// Run one forward step: token embedding + all layers + output matmul. Returns logits `[n_vocab]`.
    pub fn forward(&self, kv: &mut KvCache, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference("sequence position >= max_seq".into()));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let n_embd = cfg.n_embd;
        let mut x: Vec<f32> = (0..n_embd)
            .map(|j| self.token_embd[j + tok * n_embd])
            .collect();

        let n_rep = cfg.n_head / cfg.n_kv;

        for (il, layer) in self.layers.iter().enumerate() {
            let h = rmsnorm(&x, &layer.attn_norm, cfg.norm_eps);
            let q = matvec_embd_out(&layer.wq, &h, n_embd, n_embd);
            let k = matvec_embd_out(&layer.wk, &h, n_embd, cfg.n_kv * cfg.head_dim);
            let v = matvec_embd_out(&layer.wv, &h, n_embd, cfg.n_kv * cfg.head_dim);

            let mut q_heads = q;
            for h in 0..cfg.n_head {
                let s = &mut q_heads[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                rope_inplace(s, pos, cfg.rope_theta);
            }

            let mut k_heads = k;
            for h in 0..cfg.n_kv {
                let s = &mut k_heads[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                rope_inplace(s, pos, cfg.rope_theta);
            }

            let stride = cfg.n_kv * cfg.head_dim;
            let off = pos * stride;
            kv.k[il][off..off + stride].copy_from_slice(&k_heads);
            kv.v[il][off..off + stride].copy_from_slice(&v);

            let mut attn_out = vec![0.0f32; n_embd];
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();

            for qh in 0..cfg.n_head {
                let kv_h = qh / n_rep;
                let q_slice = &q_heads[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                let mut scores = vec![0.0f32; pos + 1];
                for p in 0..=pos {
                    let k_off = p * stride + kv_h * cfg.head_dim;
                    let k_slice = &kv.k[il][k_off..k_off + cfg.head_dim];
                    let s: f32 = q_slice
                        .iter()
                        .zip(k_slice.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        * scale;
                    scores[p] = s;
                }
                softmax_inplace(&mut scores);
                let mut comb = vec![0.0f32; cfg.head_dim];
                for p in 0..=pos {
                    let v_off = p * stride + kv_h * cfg.head_dim;
                    let v_slice = &kv.v[il][v_off..v_off + cfg.head_dim];
                    let sp = scores[p];
                    for i in 0..cfg.head_dim {
                        comb[i] += sp * v_slice[i];
                    }
                }
                let dst = qh * cfg.head_dim;
                attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
            }

            let y = matvec_embd_out(&layer.wo, &attn_out, n_embd, n_embd);
            for i in 0..n_embd {
                x[i] += y[i];
            }

            let h2 = rmsnorm(&x, &layer.ffn_norm, cfg.norm_eps);
            let gate = silu(&matvec_embd_out(&layer.ffn_gate, &h2, n_embd, cfg.n_ff));
            let up = matvec_embd_out(&layer.ffn_up, &h2, n_embd, cfg.n_ff);
            let mut tmp = vec![0.0f32; cfg.n_ff];
            for i in 0..cfg.n_ff {
                tmp[i] = gate[i] * up[i];
            }
            let y2 = matvec_ff_embd(&layer.ffn_down, &tmp, cfg.n_ff, n_embd);
            for i in 0..n_embd {
                x[i] += y2[i];
            }
        }

        let xn = rmsnorm(&x, &self.output_norm, cfg.norm_eps);
        Ok(matvec_embd_out(
            &self.output,
            &xn,
            n_embd,
            cfg.n_vocab,
        ))
    }
}
