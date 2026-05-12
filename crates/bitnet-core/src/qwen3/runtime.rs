//! Reference dense Qwen3 transformer runtime.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::ggml::{embedding_row_mmap, matvec_embd_out_mmap, matvec_ff_mmap, tensor_to_f32};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::timings::PhaseTimings;

use super::config::Qwen3Config;

#[derive(Clone)]
struct LayerTensors {
    attn_norm: GgufTensorInfo,
    q: GgufTensorInfo,
    q_norm: GgufTensorInfo,
    k: GgufTensorInfo,
    k_norm: GgufTensorInfo,
    v: GgufTensorInfo,
    o: GgufTensorInfo,
    ffn_norm: GgufTensorInfo,
    ffn_gate: GgufTensorInfo,
    ffn_up: GgufTensorInfo,
    ffn_down: GgufTensorInfo,
}

pub struct Qwen3Runtime {
    cfg: Qwen3Config,
    archive: Arc<GgufArchive>,
    tokenizer: LoadedPromptTokenizer,
    tok_embd: GgufTensorInfo,
    out_norm: GgufTensorInfo,
    out_head: GgufTensorInfo,
    layers: Vec<LayerTensors>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
}

fn must_tensor(archive: &GgufArchive, name: &str) -> Result<GgufTensorInfo> {
    archive
        .tensor_by_name(name)
        .cloned()
        .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
}

fn tensor_f32_flat(archive: &GgufArchive, t: &GgufTensorInfo) -> Result<Vec<f32>> {
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != w.len() {
        return Err(BitNetError::Inference(format!(
            "rmsnorm shape mismatch: x={} w={}",
            x.len(),
            w.len()
        )));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = 1.0 / (s + eps).sqrt();
    Ok(x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * scale)
        .collect())
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) -> Result<()> {
    if x.len() != w.len() {
        return Err(BitNetError::Inference(format!(
            "head rmsnorm shape mismatch: x={} w={}",
            x.len(),
            w.len()
        )));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = 1.0 / (s + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi *= scale * *wi;
    }
    Ok(())
}

fn rope_inplace(slice: &mut [f32], pos: usize, theta: f32) {
    let h = slice.len();
    debug_assert!(h % 2 == 0);
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / h as f32);
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[2 * i];
        let x1 = slice[2 * i + 1];
        slice[2 * i] = x0 * c - x1 * s;
        slice[2 * i + 1] = x0 * s + x1 * c;
    }
}

fn softmax_inplace(scores: &mut [f32]) {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

fn silu_inplace(v: &mut [f32]) {
    for x in v {
        *x = *x / (1.0 + (-*x).exp());
    }
}

fn matvec_out(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    matvec_embd_out_mmap(archive, tensor, x, ne0, ne1)
}

fn matvec_ff(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    x: &[f32],
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    matvec_ff_mmap(archive, tensor, x, n_ff, n_embd)
}

fn resolve_lm_head(archive: &GgufArchive, tok_embd: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for name in ["output.weight", "lm_head.weight"] {
        if let Some(t) = archive.tensor_by_name(name) {
            return Ok(t.clone());
        }
    }
    Ok(tok_embd.clone())
}

impl Qwen3Runtime {
    pub fn load(archive: Arc<GgufArchive>, tokenizer_path: &Path) -> Result<Self> {
        let cfg = Qwen3Config::from_gguf(archive.as_ref())?;
        let tok_embd = must_tensor(archive.as_ref(), "token_embd.weight")?;
        let out_norm = must_tensor(archive.as_ref(), "output_norm.weight")?;
        let out_head = resolve_lm_head(archive.as_ref(), &tok_embd)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("blk.{il}");
            layers.push(LayerTensors {
                attn_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_norm.weight"))?,
                q: must_tensor(archive.as_ref(), &format!("{p}.attn_q.weight"))?,
                q_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_q_norm.weight"))?,
                k: must_tensor(archive.as_ref(), &format!("{p}.attn_k.weight"))?,
                k_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_k_norm.weight"))?,
                v: must_tensor(archive.as_ref(), &format!("{p}.attn_v.weight"))?,
                o: must_tensor(archive.as_ref(), &format!("{p}.attn_output.weight"))?,
                ffn_norm: must_tensor(archive.as_ref(), &format!("{p}.ffn_norm.weight"))?,
                ffn_gate: must_tensor(archive.as_ref(), &format!("{p}.ffn_gate.weight"))?,
                ffn_up: must_tensor(archive.as_ref(), &format!("{p}.ffn_up.weight"))?,
                ffn_down: must_tensor(archive.as_ref(), &format!("{p}.ffn_down.weight"))?,
            });
        }
        let tokenizer = LoadedPromptTokenizer::from_path(tokenizer_path)?;

        let stride = cfg.n_kv * cfg.head_dim;
        cfg.n_layer
            .checked_mul(cfg.max_seq)
            .and_then(|v| v.checked_mul(stride))
            .ok_or_else(|| BitNetError::Inference("qwen3 KV cache size overflow".into()))?;
        let max_seq = cfg.max_seq;
        let n_layer = cfg.n_layer;
        Ok(Self {
            cfg,
            archive,
            tokenizer,
            tok_embd,
            out_norm,
            out_head,
            layers,
            k_cache: vec![vec![0f32; max_seq * stride]; n_layer],
            v_cache: vec![vec![0f32; max_seq * stride]; n_layer],
        })
    }

    pub fn generate_with_timings(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        for row in &mut self.k_cache {
            row.fill(0.0);
        }
        for row in &mut self.v_cache {
            row.fill(0.0);
        }

        let t_enc = Instant::now();
        let prompt_ids = self.tokenizer.encode_ids(prompt, true)?;
        let encode_ms = t_enc.elapsed().as_millis() as u64;
        if prompt_ids.is_empty() {
            return Ok((
                String::new(),
                PhaseTimings {
                    encode_ms,
                    ..Default::default()
                },
            ));
        }

        let t_pf = Instant::now();
        let mut logits = Vec::new();
        let chunk_sz = std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(128);
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            logits = self.prefill_chunk(chunk, chunk_idx * chunk_sz)?;
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        let eos_id = self.tokenizer.eos_token_id();
        let t_dec = Instant::now();
        let mut generated = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();
        for _ in 0..max_tokens {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next_id = sample_token(&logits, &sampling, &generated, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            generated.push(next_id);
            logits = self.decode_one(next_id, pos)?;
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode_ids(&generated, true)?;

        Ok((
            text,
            PhaseTimings {
                encode_ms,
                prefill_ms,
                decode_ms,
                prompt_tokens: prompt_ids.len() as u32,
                completion_tokens: generated.len() as u32,
            },
        ))
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32], base_pos: usize) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for (idx, &tid) in tokens.iter().enumerate() {
            logits = self.decode_one(tid, base_pos + idx)?;
        }
        Ok(logits)
    }

    pub fn decode_one(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let archive = Arc::clone(&self.archive);
        self.forward_one(token, pos, archive.as_ref())
    }

    fn forward_one(&mut self, token: u32, pos: usize, archive: &GgufArchive) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let mut x = vec![0f32; cfg.n_embd];
        embedding_row_mmap(
            archive,
            &self.tok_embd,
            tok,
            cfg.n_embd,
            cfg.n_vocab,
            &mut x,
        )?;

        let n_q = cfg.n_head * cfg.head_dim;
        let n_kv = cfg.n_kv * cfg.head_dim;
        let n_rep = cfg.n_head / cfg.n_kv;
        let kv_stride = n_kv;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        for il in 0..cfg.n_layer {
            let layer = &self.layers[il];
            let residual = x.clone();
            let attn_norm_w = tensor_f32_flat(archive, &layer.attn_norm)?;
            let h = rmsnorm(&x, &attn_norm_w, cfg.norm_eps)?;

            let mut q = matvec_out(archive, &layer.q, &h, cfg.n_embd, n_q)?;
            let mut k = matvec_out(archive, &layer.k, &h, cfg.n_embd, n_kv)?;
            let v = matvec_out(archive, &layer.v, &h, cfg.n_embd, n_kv)?;
            let q_norm_w = tensor_f32_flat(archive, &layer.q_norm)?;
            let k_norm_w = tensor_f32_flat(archive, &layer.k_norm)?;
            for hidx in 0..cfg.n_head {
                let s = &mut q[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rmsnorm_inplace(s, &q_norm_w, cfg.norm_eps)?;
                rope_inplace(s, pos, cfg.rope_theta);
            }
            for hidx in 0..cfg.n_kv {
                let s = &mut k[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rmsnorm_inplace(s, &k_norm_w, cfg.norm_eps)?;
                rope_inplace(s, pos, cfg.rope_theta);
            }

            let kv_off = pos * kv_stride;
            self.k_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&k);
            self.v_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&v);

            let mut attn_out = vec![0f32; n_q];
            for qh in 0..cfg.n_head {
                let kv_h = qh / n_rep;
                let q_slice = &q[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        let off = p * kv_stride + kv_h * cfg.head_dim;
                        let k_slice = &self.k_cache[il][off..off + cfg.head_dim];
                        q_slice
                            .iter()
                            .zip(k_slice.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                softmax_inplace(&mut scores);
                let mut comb = vec![0f32; cfg.head_dim];
                for p in 0..=pos {
                    let off = p * kv_stride + kv_h * cfg.head_dim;
                    let v_slice = &self.v_cache[il][off..off + cfg.head_dim];
                    let sp = scores[p];
                    for i in 0..cfg.head_dim {
                        comb[i] += sp * v_slice[i];
                    }
                }
                let dst = qh * cfg.head_dim;
                attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
            }

            let y = matvec_out(archive, &layer.o, &attn_out, n_q, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = residual[i] + y[i];
            }

            let ffn_residual = x.clone();
            let ffn_norm_w = tensor_f32_flat(archive, &layer.ffn_norm)?;
            let h2 = rmsnorm(&x, &ffn_norm_w, cfg.norm_eps)?;
            let mut gate = matvec_out(archive, &layer.ffn_gate, &h2, cfg.n_embd, cfg.n_ff)?;
            silu_inplace(&mut gate);
            let up = matvec_out(archive, &layer.ffn_up, &h2, cfg.n_embd, cfg.n_ff)?;
            for i in 0..cfg.n_ff {
                gate[i] *= up[i];
            }
            let y2 = matvec_ff(archive, &layer.ffn_down, &gate, cfg.n_ff, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + y2[i];
            }
        }

        let out_norm_w = tensor_f32_flat(archive, &self.out_norm)?;
        let xn = rmsnorm(&x, &out_norm_w, cfg.norm_eps)?;
        matvec_out(archive, &self.out_head, &xn, cfg.n_embd, cfg.n_vocab)
    }
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn u32_le(w: &mut File, x: u32) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn u64_le(w: &mut File, x: u64) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn s(w: &mut File, v: &str) -> std::io::Result<()> {
        u64_le(w, v.len() as u64)?;
        w.write_all(v.as_bytes())
    }

    fn kv_s(w: &mut File, k: &str, v: &str) -> std::io::Result<()> {
        s(w, k)?;
        u32_le(w, 8)?;
        s(w, v)
    }

    fn kv_u(w: &mut File, k: &str, v: u32) -> std::io::Result<()> {
        s(w, k)?;
        u32_le(w, 4)?;
        u32_le(w, v)
    }

    fn tensor(w: &mut File, name: &str, dims: &[u64]) -> std::io::Result<()> {
        s(w, name)?;
        u32_le(w, dims.len() as u32)?;
        for &d in dims {
            u64_le(w, d)?;
        }
        u32_le(w, 0)?;
        u64_le(w, 0)
    }

    #[test]
    fn qwen3_runtime_reports_missing_q_norm_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen3_missing_q_norm.gguf");
        let mut f = File::create(&path).unwrap();
        let tensor_names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        f.write_all(b"GGUF").unwrap();
        u32_le(&mut f, 3).unwrap();
        u64_le(&mut f, tensor_names.len() as u64).unwrap();
        u64_le(&mut f, 10).unwrap();
        kv_s(&mut f, "general.architecture", "qwen3").unwrap();
        kv_u(&mut f, "qwen3.embedding_length", 16).unwrap();
        kv_u(&mut f, "qwen3.vocab_size", 32).unwrap();
        kv_u(&mut f, "qwen3.block_count", 1).unwrap();
        kv_u(&mut f, "qwen3.attention.head_count", 2).unwrap();
        kv_u(&mut f, "qwen3.attention.head_count_kv", 1).unwrap();
        kv_u(&mut f, "qwen3.feed_forward_length", 24).unwrap();
        kv_u(&mut f, "qwen3.context_length", 128).unwrap();
        kv_u(&mut f, "general.alignment", 32).unwrap();
        kv_u(&mut f, "qwen3.rope.dimension_count", 16).unwrap();
        for name in tensor_names {
            let dims: &[u64] = match name {
                "token_embd.weight" => &[16, 32],
                "output_norm.weight" | "blk.0.attn_norm.weight" | "blk.0.ffn_norm.weight" => &[16],
                "blk.0.attn_k_norm.weight" => &[16],
                "blk.0.attn_q.weight" => &[16, 32],
                "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => &[16, 16],
                "blk.0.attn_output.weight" => &[32, 16],
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => &[16, 24],
                "blk.0.ffn_down.weight" => &[24, 16],
                _ => unreachable!(),
            };
            tensor(&mut f, name, dims).unwrap();
        }
        let pos = f.metadata().unwrap().len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad]).unwrap();
        drop(f);

        let archive = Arc::new(GgufArchive::mmap_path(&path).unwrap());
        let err = match Qwen3Runtime::load(archive, dir.path().join("tokenizer.json").as_path()) {
            Ok(_) => panic!("expected missing q norm before tokenizer loading"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("attn_q_norm"));
    }
}
