//! Dense Spark-X2.5 (`spark2_5`) transformer runtime.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::backend::BackendKind;
use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::timings::PhaseTimings;

use super::config::Spark25Config;
use super::weights::Spark25Weights;

pub struct Spark25Runtime {
    cfg: Spark25Config,
    pub(crate) weights: Spark25Weights,
    pub(super) tokenizer: Arc<LoadedPromptTokenizer>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
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

/// GPT-NeoX style RoPE on the leading `n_rot` dims of a head (Spark partial rotary).
fn rope_partial_inplace(slice: &mut [f32], pos: usize, theta: f32, n_rot: usize) {
    if n_rot == 0 || n_rot > slice.len() || n_rot % 2 != 0 {
        return;
    }
    let half = n_rot / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / n_rot as f32);
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[i];
        let x1 = slice[i + half];
        slice[i] = x0 * c - x1 * s;
        slice[i + half] = x0 * s + x1 * c;
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

/// ggml `GELU` tanh approximation (llama.cpp `LLM_FFN_GELU`).
fn gelu_inplace(v: &mut [f32]) {
    const K: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    for x in v {
        let x3 = *x * *x * *x;
        *x = 0.5 * *x * (1.0 + (K * (*x + 0.044_715 * x3)).tanh());
    }
}

fn sigmoid_inplace(v: &mut [f32]) {
    for x in v {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

impl Spark25Runtime {
    pub(crate) fn context_capacity(&self) -> usize {
        self.cfg.max_seq
    }

    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        if crate::context_native::enabled() {
            return Err(BitNetError::NotImplemented(
                "context tiers do not support Spark-X2.5 yet",
            ));
        }

        let cfg = Spark25Config::from_gguf(archive.as_ref())?;
        let weights = Spark25Weights::load(Arc::clone(&archive), &cfg, backend_kind)?;
        let tokenizer = Arc::new(LoadedPromptTokenizer::from_path_for_gguf(
            tokenizer_path,
            &archive,
        )?);

        let stride = cfg.n_kv * cfg.head_dim;
        cfg.n_layer
            .checked_mul(cfg.max_seq)
            .and_then(|v| v.checked_mul(stride))
            .ok_or_else(|| BitNetError::Inference("spark2_5 KV cache size overflow".into()))?;
        let max_seq = cfg.max_seq;
        let n_layer = cfg.n_layer;
        Ok(Self {
            cfg,
            weights,
            tokenizer,
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
        sampling.validate_structured_output()?;
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
        crate::context_capacity::check_request(prompt_ids.len(), max_tokens, self.cfg.max_seq)?;
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
        let mut finish_reason = crate::timings::GenerationFinishReason::Length;
        for _ in 0..max_tokens {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next_id = sample_token(&logits, &sampling, &generated, &mut rng);
            if Some(next_id) == eos_id {
                finish_reason = crate::timings::GenerationFinishReason::Stop;
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
                finish_reason,
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
        self.forward_one(token, pos)
    }

    fn forward_one(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
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
        self.weights.tok_embd.embed_row(
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
        let qkv_out = n_q + 2 * n_kv;

        for il in 0..cfg.n_layer {
            let layer = &self.weights.layers[il];
            let residual = x.clone();
            let h = rmsnorm(&x, &layer.attn_norm, cfg.norm_eps)?;

            let qkv = layer.qkv.matvec_embd_out(&h, cfg.n_embd, qkv_out)?;
            let mut q = qkv[..n_q].to_vec();
            let mut k = qkv[n_q..n_q + n_kv].to_vec();
            let v = qkv[n_q + n_kv..].to_vec();

            let theta = cfg.rope_theta_for_layer(il);
            let n_rot = cfg.rope_dim_for_layer(il);
            for hidx in 0..cfg.n_head {
                let s = &mut q[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rope_partial_inplace(s, pos, theta, n_rot);
            }
            for hidx in 0..cfg.n_kv {
                let s = &mut k[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rope_partial_inplace(s, pos, theta, n_rot);
            }

            let kv_off = pos * kv_stride;
            self.k_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&k);
            self.v_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&v);

            let key_start = cfg.kv_key_start(il, pos);
            let mut attn_out = vec![0f32; n_q];
            for qh in 0..cfg.n_head {
                let kv_h = qh / n_rep;
                let q_slice = &q[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                let mut scores: Vec<f32> = (key_start..=pos)
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
                for (si, p) in (key_start..=pos).enumerate() {
                    let off = p * kv_stride + kv_h * cfg.head_dim;
                    let v_slice = &self.v_cache[il][off..off + cfg.head_dim];
                    let sp = scores[si];
                    for i in 0..cfg.head_dim {
                        comb[i] += sp * v_slice[i];
                    }
                }
                let dst = qh * cfg.head_dim;
                attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
            }

            // Head-wise sigmoid gate from pre-attn RMSNorm activation (llama.cpp spark2-5).
            let mut gate = layer
                .attn_gate
                .matvec_embd_out(&h, cfg.n_embd, cfg.n_head)?;
            sigmoid_inplace(&mut gate);
            for qh in 0..cfg.n_head {
                let g = gate[qh];
                let dst = qh * cfg.head_dim;
                for i in 0..cfg.head_dim {
                    attn_out[dst + i] *= g;
                }
            }

            let y = layer.o.matvec_embd_out(&attn_out, n_q, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = residual[i] + y[i];
            }

            let ffn_residual = x.clone();
            let h2 = rmsnorm(&x, &layer.ffn_norm, cfg.norm_eps)?;
            let mut gate_ff = layer
                .ffn_gate
                .matvec_embd_out(&h2, cfg.n_embd, cfg.n_ff)?;
            gelu_inplace(&mut gate_ff);
            let up = layer.ffn_up.matvec_embd_out(&h2, cfg.n_embd, cfg.n_ff)?;
            for i in 0..cfg.n_ff {
                gate_ff[i] *= up[i];
            }
            let y2 = layer
                .ffn_down
                .matvec_ff(&gate_ff, cfg.n_ff, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + y2[i];
            }
        }

        let xn = rmsnorm(&x, &self.weights.out_norm, cfg.norm_eps)?;
        self.weights
            .out_head
            .matvec_embd_out(&xn, cfg.n_embd, cfg.n_vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_zero_is_zero() {
        let mut v = [0.0f32];
        gelu_inplace(&mut v);
        assert!((v[0]).abs() < 1e-6);
    }

    #[test]
    fn rope_partial_leaves_tail_untouched() {
        let mut s = [1.0f32, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0];
        let tail = s[4..].to_vec();
        rope_partial_inplace(&mut s, 1, 10_000.0, 4);
        assert_eq!(&s[4..], tail.as_slice());
        // Rotated pair should move.
        assert!((s[0] - 1.0).abs() > 1e-6 || (s[1]).abs() > 1e-6);
    }
}
