//! Mixtral MoE transformer runtime (CPU-first, Llama attention + top-k experts).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::ggml::{embedding_row_mmap, matvec_embd_out_mmap, tensor_to_f32};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::timings::PhaseTimings;

use super::config::MixtralConfig;
use super::moe::moe_forward;

#[derive(Clone)]
struct LayerTensors {
    attn_norm: GgufTensorInfo,
    q: GgufTensorInfo,
    k: GgufTensorInfo,
    v: GgufTensorInfo,
    o: GgufTensorInfo,
    ffn_norm: GgufTensorInfo,
    gate_inp: GgufTensorInfo,
    up_exps: GgufTensorInfo,
    gate_exps: GgufTensorInfo,
    down_exps: GgufTensorInfo,
}

pub struct MixtralRuntime {
    cfg: MixtralConfig,
    archive: Arc<GgufArchive>,
    pub(super) tokenizer: Arc<LoadedPromptTokenizer>,
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

fn rope_inplace(slice: &mut [f32], pos: usize, theta: f32) {
    let h = slice.len();
    debug_assert!(h % 2 == 0);
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / h as f32);
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

fn matvec_out(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    matvec_embd_out_mmap(archive, tensor, x, ne0, ne1)
}

fn resolve_lm_head(archive: &GgufArchive, tok_embd: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for name in ["output.weight", "lm_head.weight"] {
        if let Some(t) = archive.tensor_by_name(name) {
            return Ok(t.clone());
        }
    }
    Ok(tok_embd.clone())
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

impl MixtralRuntime {
    pub(crate) fn context_capacity(&self) -> usize {
        self.cfg.max_seq
    }

    pub fn load(archive: Arc<GgufArchive>, tokenizer_path: &Path) -> Result<Self> {
        let cfg = MixtralConfig::from_gguf(archive.as_ref())?;
        let tok_embd = must_tensor(archive.as_ref(), "token_embd.weight")?;
        let out_norm = must_tensor(archive.as_ref(), "output_norm.weight")?;
        let out_head = resolve_lm_head(archive.as_ref(), &tok_embd)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("blk.{il}");
            layers.push(LayerTensors {
                attn_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_norm.weight"))?,
                q: must_tensor(archive.as_ref(), &format!("{p}.attn_q.weight"))?,
                k: must_tensor(archive.as_ref(), &format!("{p}.attn_k.weight"))?,
                v: must_tensor(archive.as_ref(), &format!("{p}.attn_v.weight"))?,
                o: must_tensor(archive.as_ref(), &format!("{p}.attn_output.weight"))?,
                ffn_norm: must_tensor(archive.as_ref(), &format!("{p}.ffn_norm.weight"))?,
                gate_inp: must_tensor(archive.as_ref(), &format!("{p}.ffn_gate_inp.weight"))?,
                up_exps: must_tensor(archive.as_ref(), &format!("{p}.ffn_up_exps.weight"))?,
                gate_exps: must_tensor(archive.as_ref(), &format!("{p}.ffn_gate_exps.weight"))?,
                down_exps: must_tensor(archive.as_ref(), &format!("{p}.ffn_down_exps.weight"))?,
            });
        }
        let tokenizer = Arc::new(LoadedPromptTokenizer::from_path_for_gguf(
            tokenizer_path,
            &archive,
        )?);

        let stride = cfg.n_kv * cfg.head_dim;
        cfg.n_layer
            .checked_mul(cfg.max_seq)
            .and_then(|v| v.checked_mul(stride))
            .ok_or_else(|| BitNetError::Inference("mixtral KV cache size overflow".into()))?;
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
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.decode_one(tid, pos)?;
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

    pub fn greedy_next_token_id_after_prompt(&mut self, prompt: &str) -> Result<u32> {
        for row in &mut self.k_cache {
            row.fill(0.0);
        }
        for row in &mut self.v_cache {
            row.fill(0.0);
        }
        let prompt_ids = self.tokenizer.encode_ids(prompt, true)?;
        if prompt_ids.is_empty() {
            return Err(BitNetError::Inference(
                "greedy_next_token: empty prompt encoding".into(),
            ));
        }
        let mut logits = Vec::new();
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.decode_one(tid, pos)?;
        }
        let mut rng = seeded_rng(Some(0));
        Ok(sample_token(
            &logits,
            &SamplingOptions {
                temperature: 0.0,
                top_p: None,
                seed: Some(0),
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                structured_json: false,
            },
            &[],
            &mut rng,
        ))
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
            for hidx in 0..cfg.n_head {
                let s = &mut q[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rope_inplace(s, pos, cfg.rope_theta);
            }
            for hidx in 0..cfg.n_kv {
                let s = &mut k[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
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
            let moe_delta = moe_forward(
                archive,
                cfg,
                &h2,
                &layer.gate_inp,
                &layer.up_exps,
                &layer.gate_exps,
                &layer.down_exps,
            )?;
            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + moe_delta[i];
            }
        }

        let out_norm_w = tensor_f32_flat(archive, &self.out_norm)?;
        let xn = rmsnorm(&x, &out_norm_w, cfg.norm_eps)?;
        matvec_out(archive, &self.out_head, &xn, cfg.n_embd, cfg.n_vocab)
    }
}
