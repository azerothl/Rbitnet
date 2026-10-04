//! One-token Transformer stack for hybrid Qwen3 MoE GGUF checkpoints.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::backend::BackendKind;
use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::ggml::tensor_to_f32;
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::stream::emit_text_delta;
use crate::timings::PhaseTimings;

use super::attention::{block_full_attention, AttnKvCache};
use super::config::Qwen35Config;
use super::moe::{moe_forward, shared_expert_forward};
use super::qmatvec::token_embedding_row;
use super::recurrent::{first_recurrent_gate_out_dim, recurrent_forward, RecurrentState};
use crate::native::prefix::{self, PrefixStore};
use crate::native::weights::Weights;

struct SavedPrefix {
    attention: Option<super::attention::AttnPrefix>,
    device_attention: Vec<Option<crate::native::qwen_full::SavedAttention>>,
    recurrent: Vec<RecurrentState>,
    device: Vec<Option<crate::native::qwen_recurrent::SavedRecurrent>>,
}

pub struct Qwen35Runtime {
    cfg: Qwen35Config,
    archive: Arc<GgufArchive>,
    pub(super) tokenizer: Arc<LoadedPromptTokenizer>,
    attn_kv: AttnKvCache,
    rec: Vec<RecurrentState>,
    // Drop graphs borrowing layer contexts before those contexts and their weights.
    gpu_full: Option<crate::native::qwen_full::GpuFull>,
    gpu_recurrent: Vec<Option<crate::native::qwen_recurrent::GpuRecurrent>>,
    gpu_head: Option<crate::native::head::GpuHead>,
    weights: Weights,
    tok_embd: GgufTensorInfo,
    out_norm: GgufTensorInfo,
    out_head: GgufTensorInfo,
    trace_layer_timings: bool,
    debug_max_layers: Option<usize>,
    debug_moe_topk: Option<usize>,
    prefill_chunk_tokens: usize,
    cuda_graph_enabled: bool,
    prefixes: PrefixStore<SavedPrefix>,
    prefix_supported: bool,
    draft: Option<spec_serving::DraftRuntime>,
    context_tier: Option<crate::context_native::Handle>,
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

fn resolve_lm_head(archive: &GgufArchive, tie: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for &n in &["output.weight", "lm_head.weight"] {
        if archive.tensor_by_name(n).is_some() {
            return must_tensor(archive, n);
        }
    }
    Ok(tie.clone())
}

fn env_opt_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

fn rms_combine(
    x: &[f32],
    w_info: &GgufTensorInfo,
    archive: &GgufArchive,
    eps: f32,
) -> Result<Vec<f32>> {
    let w = tensor_f32_flat(archive, w_info)?;
    if w.len() != x.len() {
        return Err(BitNetError::Inference(
            "rmsnorm width mismatch vs hidden".into(),
        ));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let sc = 1.0 / (s + eps).sqrt();
    Ok(x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * sc)
        .collect())
}

impl Qwen35Runtime {
    pub(crate) fn has_full_gpu_pipeline(&self) -> bool {
        self.gpu_full.is_some()
    }
    pub(crate) fn gpu_prefill_capacity(&self) -> usize {
        self.gpu_full.as_ref().map_or(1, |g| g.prefill_capacity())
    }
    pub(crate) fn gpu_execution_summary(&self) -> (usize, usize, bool) {
        (
            self.gpu_recurrent.iter().flatten().count(),
            self.cfg.recurrent_layers.iter().filter(|&&v| v).count(),
            self.gpu_head.is_some() || self.gpu_full.is_some(),
        )
    }
    pub(crate) fn resident_weights_bytes(&self) -> usize {
        self.weights.resident_bytes
            + self
                .gpu_recurrent
                .iter()
                .flatten()
                .map(|g| g.extra_weights_bytes)
                .sum::<usize>()
            + self.draft_resident_weights_bytes()
    }
    pub(crate) fn context_capacity(&self) -> usize {
        self.cfg.max_seq
    }

    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        Self::load_with_optional_draft(archive, tokenizer_path, backend_kind)
    }

    fn load_without_draft(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let cfg = Qwen35Config::from_gguf(archive.as_ref())?;
        let trace_layer_timings = std::env::var("RBITNET_TRACE_LAYER_TIMINGS")
            .ok()
            .map(|v| {
                let t = v.trim();
                t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false);
        let debug_max_layers = env_opt_usize("RBITNET_DEBUG_MAX_LAYERS");
        let debug_moe_topk = env_opt_usize("RBITNET_DEBUG_MOE_TOPK");
        let prefill_chunk_tokens = env_opt_usize("RBITNET_PREFILL_CHUNK_TOKENS").unwrap_or(128);
        let cuda_graph_enabled = matches!(
            std::env::var("RBITNET_CUDA_GRAPH").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let tokenizer = Arc::new(LoadedPromptTokenizer::from_path_for_gguf(
            tokenizer_path,
            &archive,
        )?);

        let tok_embd = must_tensor(
            archive.as_ref(),
            if archive.tensor_by_name("token_embd.weight").is_some() {
                "token_embd.weight"
            } else {
                "token_embd"
            },
        )?;
        let out_norm = must_tensor(archive.as_ref(), "output_norm.weight")?;
        let out_head = resolve_lm_head(archive.as_ref(), &tok_embd)?;
        let weights = Weights::new(Arc::clone(&archive), backend_kind)?;
        // Validate the complete graph before exposing readiness.
        for il in 0..cfg.n_layer {
            for suffix in ["attn_norm.weight", "post_attention_norm.weight"] {
                weights.tensor(&format!("blk.{il}.{suffix}"))?;
            }
            let matrices: &[&str] = if cfg.is_recurrent_layer(il) {
                &[
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_out.weight",
                ]
            } else {
                &[
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ]
            };
            for suffix in matrices {
                weights.tensor(&format!("blk.{il}.{suffix}"))?;
            }
            for suffix in if cfg.n_expert == 0 {
                ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
            } else {
                [
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
            } {
                weights.tensor(&format!("blk.{il}.{suffix}"))?;
            }
        }

        let d_conv = cfg.ssm_d_conv.max(2);
        let d_inner = cfg.ssm_d_inner.max(1);
        let gate_out = first_recurrent_gate_out_dim(archive.as_ref(), &cfg)?;
        let num_v = cfg.ssm_dt_rank.max(1);
        if gate_out % num_v != 0 {
            return Err(BitNetError::Inference(format!(
                "recurrent attn_gate output width {gate_out} not divisible by ssm.time_step_rank ({num_v})"
            )));
        }
        let sv_state = (gate_out / num_v).max(1);
        let mut extra_budget = weights
            .residency_budget_bytes
            .saturating_sub(weights.resident_bytes);
        let gpu_recurrent: Vec<_> = (0..cfg.n_layer)
            .map(|il| {
                if matches!(backend_kind, BackendKind::Cuda | BackendKind::Hybrid)
                    && cfg.n_expert == 0
                    && cfg.is_recurrent_layer(il)
                    && sv_state == cfg.ssm_d_state
                {
                    let gpu = crate::native::qwen_recurrent::GpuRecurrent::new(
                        &weights,
                        il,
                        sv_state,
                        cfg.ssm_n_group,
                        num_v,
                        cfg.norm_eps,
                        extra_budget,
                    );
                    if let Some(gpu) = &gpu {
                        extra_budget -= gpu.extra_weights_bytes;
                    }
                    gpu
                } else {
                    None
                }
            })
            .collect();
        tracing::info!(
            layers = gpu_recurrent.iter().flatten().count(),
            "resident Qwen recurrent blocks"
        );
        let mut rec = Vec::with_capacity(cfg.n_layer);
        for (il, gpu) in gpu_recurrent.iter().enumerate() {
            rec.push(if gpu.is_some() || !cfg.is_recurrent_layer(il) {
                RecurrentState::new(0, 0, 0, 0)
            } else {
                RecurrentState::new(d_conv, d_inner, sv_state, cfg.ssm_dt_rank.max(1))
            });
        }

        let gpu_full =
            if debug_max_layers.is_none() && debug_moe_topk.is_none() && !trace_layer_timings {
                crate::native::qwen_full::GpuFull::new(
                    &weights,
                    &cfg,
                    &gpu_recurrent,
                    &out_head.name,
                    backend_kind,
                )
            } else {
                None
            };
        if std::env::var("RBITNET_REQUIRE_QWEN_FULL").as_deref() == Ok("1") && gpu_full.is_none() {
            return Err(BitNetError::Inference("required Qwen full GPU pipeline unavailable (backend, tensors, options or CUDA DLL)".into()));
        }
        tracing::info!(enabled = gpu_full.is_some(), "full Qwen GPU token pipeline");
        let attn_kv = if gpu_full.is_some() {
            AttnKvCache::default()
        } else {
            AttnKvCache::new(&cfg, cfg.max_seq)
        };
        let gpu_head = if gpu_full.is_none() {
            crate::native::head::GpuHead::new(&weights, &out_head.name, cfg.norm_eps)
        } else {
            None
        };
        let prefix_supported = gpu_recurrent
            .iter()
            .flatten()
            .all(|g| g.supports_snapshot());
        if prefix::enabled() && !prefix_supported {
            tracing::warn!("Qwen prefix reuse unavailable with this CUDA DLL");
        }
        let context_tier = if crate::context_native::enabled() {
            if gpu_full.is_none()
                || std::env::var("RBITNET_QWEN_SPECULATIVE").as_deref() == Ok("1")
                || std::env::var("RBITNET_CUDA_PREFILL_TF32X3").as_deref() == Ok("1")
                || cfg.max_seq > 8192
            {
                return Err(BitNetError::NotImplemented("context tiers require full dense Native Qwen CUDA without TF32/speculative decoding and capacity <= 8192"));
            }
            let configuration = format!(
                "{:?};split={:?};tensor={:?};head={:?}",
                cfg,
                std::env::var("RBITNET_CUDA_SPLIT_KV"),
                std::env::var("RBITNET_CUDA_PREFILL_TF32X3"),
                std::env::var("RBITNET_CUDA_HEAD")
            );
            crate::context_native::Handle::new("qwen", &archive, tokenizer_path, &configuration)
                .map_err(|error| BitNetError::Inference(format!("context tiers: {error}")))?
        } else {
            None
        };
        Ok(Self {
            cfg,
            archive,
            tokenizer,
            attn_kv,
            rec,
            gpu_full,
            gpu_recurrent,
            gpu_head,
            weights,
            tok_embd,
            out_norm,
            out_head,
            trace_layer_timings,
            debug_max_layers,
            debug_moe_topk,
            prefill_chunk_tokens,
            cuda_graph_enabled,
            prefixes: PrefixStore::from_env(),
            prefix_supported,
            draft: None,
            context_tier,
        })
    }

    pub fn generate_with_timings(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        self.generate_inner(prompt, max_tokens, sampling, None)
    }

    pub(crate) fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
        mut events: Option<&mut (dyn FnMut(crate::stream::StreamEvent) -> Result<()> + Send)>,
    ) -> Result<(String, PhaseTimings)> {
        if self.draft.is_some() {
            return self.generate_with_owned_draft(prompt, max_tokens, sampling, events);
        }
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        self.attn_kv.clear();
        for st in &mut self.rec {
            st.conv_hist.fill(0f32);
            st.ssm_state.fill(0f32);
        }
        let arch = Arc::clone(&self.archive);

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
        let mut matched = 0;
        if prefix::enabled() && self.prefix_supported {
            // Leave at least one token to recompute logits; a checkpoint can
            // only restore its complete recurrent state at its original length.
            let reusable = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
            if let Some((saved, length)) =
                self.prefixes
                    .lookup(reusable, false, prefix::minimum_tokens())
            {
                if let Some(attention) = &saved.attention {
                    self.attn_kv.restore(attention)?;
                }
                for (dst, src) in self.rec.iter_mut().zip(&saved.recurrent) {
                    *dst = src.clone();
                }
                for (gpu, state) in self.gpu_recurrent.iter_mut().zip(&saved.device) {
                    if let (Some(gpu), Some(state)) = (gpu, state) {
                        gpu.restore(state, length)?;
                    }
                }
                if let Some(gpu) = &mut self.gpu_full {
                    gpu.restore_attention(&saved.device_attention, length)?;
                }
                matched = length;
                crate::perf::record_prefix_hit(
                    self.gpu_full
                        .as_ref()
                        .map(|g| g.prefix_bytes(length))
                        .unwrap_or_else(|| self.attn_kv.prefix_bytes(length)),
                );
            } else {
                crate::perf::record_prefix_cache_miss();
            }
        }
        if matched == 0 {
            if let (Some(tier), Some(gpu)) = (&mut self.context_tier, &self.gpu_full) {
                let reusable = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
                match unsafe {
                    tier.restore(gpu.portable_context() as *mut std::ffi::c_void, reusable)
                } {
                    Ok(length) => matched = length,
                    Err(error) => {
                        tracing::warn!(%error, "Qwen context restore refused; recomputing prompt")
                    }
                }
            }
        }
        let mut logits = Vec::new();
        let mut next_token = None;
        let gpu_greedy = (self.gpu_head.is_some() || self.gpu_full.is_some())
            && sampling.device_greedy_eligible();
        let prefill_chunk = self.prefill_chunk_tokens.max(1);
        let checkpoint_interval = env_opt_usize("RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS")
            .unwrap_or(256)
            .max(1);
        let checkpoint_enabled =
            (prefix::enabled() && self.prefix_supported) || self.context_tier.is_some();
        let capacity = self.gpu_full.as_ref().map_or(1, |g| g.prefill_capacity());
        let mut pos = matched;
        while pos < prompt_ids.len() {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let mut end = pos
                .saturating_add(prefill_chunk.min(capacity))
                .min(prompt_ids.len());
            if checkpoint_enabled && pos < prompt_ids.len() - 1 {
                // An exact GDN checkpoint cannot be reconstructed by truncating
                // the end state of a block. Stop at each intended checkpoint.
                let checkpoint =
                    (pos / checkpoint_interval + 1).saturating_mul(checkpoint_interval);
                if prefix::enabled() && self.prefix_supported {
                    end = end.min(checkpoint);
                }
                end = end.min(prompt_ids.len() - 1);
            }
            let count = end - pos;
            (logits, next_token) = if count > 1 {
                self.forward_block(
                    &prompt_ids[pos..end],
                    pos,
                    &arch,
                    end == prompt_ids.len(),
                    gpu_greedy,
                )?
            } else {
                self.forward_inner(
                    prompt_ids[pos],
                    pos,
                    &arch,
                    end == prompt_ids.len(),
                    gpu_greedy,
                )?
            };
            pos = end;
            if checkpoint_enabled
                && pos < prompt_ids.len()
                && (pos == prompt_ids.len() - 1
                    || (prefix::enabled()
                        && self.prefix_supported
                        && pos % checkpoint_interval == 0))
            {
                self.save_prefix(&prompt_ids[..pos])?;
            }
            if self.trace_layer_timings {
                tracing::debug!(
                    prefill_chunk_tokens = count,
                    cuda_graph_enabled = self.cuda_graph_enabled,
                    "qwen35 prefill chunk processed"
                );
            }
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        let eos_ids = self.tokenizer.eos_token_ids();

        let mut previous = String::new();
        let t_dec = Instant::now();
        let mut gen = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();

        let mut finish_reason = crate::timings::GenerationFinishReason::Length;
        for step in 0..max_tokens {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next_id = next_token
                .take()
                .unwrap_or_else(|| sample_token(&logits, &sampling, &gen, &mut rng));
            if eos_ids.contains(&next_id) {
                finish_reason = crate::timings::GenerationFinishReason::Stop;
                break;
            }
            gen.push(next_id);
            let text = self.tokenizer.decode_ids(&gen, true)?;
            if let Some(callback) = events.as_deref_mut() {
                emit_text_delta(&text, &mut previous, false, callback)?;
            }
            if step + 1 < max_tokens {
                (logits, next_token) = self.forward_inner(next_id, pos, &arch, true, gpu_greedy)?;
            }
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;

        let text = self.tokenizer.decode_ids(&gen, true)?;
        if let Some(callback) = events {
            emit_text_delta(&text, &mut previous, true, callback)?;
        }
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms,
            prompt_tokens: prompt_ids.len() as u32,
            completion_tokens: gen.len() as u32,
            finish_reason,
        };
        Ok((text, phases))
    }

    fn save_prefix(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.len() >= 8 {
            if let (Some(tier), Some(gpu)) = (&mut self.context_tier, &self.gpu_full) {
                if let Err(error) =
                    unsafe { tier.capture(gpu.portable_context() as *mut std::ffi::c_void, tokens) }
                {
                    tracing::warn!(%error, "Qwen context capture refused; normal decoding remains available");
                }
            }
        }
        if self.prefixes.contains(tokens) {
            return Ok(());
        }
        if !prefix::enabled() || !self.prefix_supported || tokens.len() < prefix::minimum_tokens() {
            return Ok(());
        }
        let bytes = self
            .gpu_full
            .as_ref()
            .map(|g| g.prefix_bytes(tokens.len()))
            .unwrap_or_else(|| self.attn_kv.prefix_bytes(tokens.len()))
            .saturating_add(
                self.rec
                    .iter()
                    .map(|s| (s.conv_hist.len() + s.ssm_state.len()) * 4)
                    .sum::<usize>(),
            )
            .saturating_add(
                self.gpu_recurrent
                    .iter()
                    .flatten()
                    .map(|g| g.state_bytes)
                    .sum::<usize>(),
            );
        if !self.prefixes.reserve(tokens, bytes) {
            return Ok(());
        }
        let mut device = Vec::with_capacity(self.gpu_recurrent.len());
        for gpu in &self.gpu_recurrent {
            let state = if let Some(gpu) = gpu {
                let Some(state) = gpu.snapshot() else {
                    tracing::warn!("Qwen checkpoint allocation failed; skipping prefix insertion");
                    return Ok(());
                };
                Some(state)
            } else {
                None
            };
            device.push(state);
        }
        let device_attention = if let Some(gpu) = &self.gpu_full {
            let Some(saved) = gpu.snapshots() else {
                tracing::warn!(
                    "Qwen attention checkpoint allocation failed; skipping prefix insertion"
                );
                return Ok(());
            };
            saved
        } else {
            Vec::new()
        };
        let saved = SavedPrefix {
            attention: if self.gpu_full.is_none() {
                Some(self.attn_kv.snapshot(tokens.len())?)
            } else {
                None
            },
            device_attention,
            recurrent: self.rec.clone(),
            device,
        };
        self.prefixes.insert(tokens.to_vec(), saved, bytes);
        Ok(())
    }

    #[cfg(test)]
    fn forward_one(
        &mut self,
        token: u32,
        pos: usize,
        archive: &Arc<GgufArchive>,
        logits_required: bool,
    ) -> Result<Vec<f32>> {
        self.forward_inner(token, pos, archive, logits_required, false)
            .map(|result| result.0)
    }

    fn forward_block(
        &mut self,
        tokens: &[u32],
        pos: usize,
        archive: &Arc<GgufArchive>,
        output: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        if pos >= self.cfg.max_seq || tokens.len() > self.cfg.max_seq - pos {
            return Err(BitNetError::Inference("prefill exceeds max_seq".into()));
        }
        let mut embeddings = Vec::with_capacity(tokens.len() * self.cfg.n_embd);
        for &token in tokens {
            if token as usize >= self.cfg.n_vocab {
                return Err(BitNetError::Inference("token id out of range".into()));
            }
            embeddings.extend(token_embedding_row(
                archive,
                &self.tok_embd,
                token as usize,
                self.cfg.n_embd,
                self.cfg.n_vocab,
            )?);
        }
        self.gpu_full
            .as_mut()
            .ok_or_else(|| BitNetError::Inference("Qwen block pipeline unavailable".into()))?
            .prefill(&embeddings, pos, tokens.len(), output, greedy)
    }

    fn forward_inner(
        &mut self,
        token: u32,
        pos: usize,
        archive: &Arc<GgufArchive>,
        logits_required: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        let cfg = &self.cfg;
        let step_t0 = if self.trace_layer_timings {
            Some(Instant::now())
        } else {
            None
        };
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let mut x = token_embedding_row(archive, &self.tok_embd, tok, cfg.n_embd, cfg.n_vocab)?;
        if let Some(gpu) = &mut self.gpu_full {
            return gpu.run(&x, pos, logits_required, greedy);
        }

        let layer_limit = self
            .debug_max_layers
            .map(|v| v.min(cfg.n_layer))
            .unwrap_or(cfg.n_layer);
        for il in 0..layer_limit {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let layer_t0 = if self.trace_layer_timings {
                Some(Instant::now())
            } else {
                None
            };
            if let Some(gpu) = &mut self.gpu_recurrent[il] {
                x = gpu.run(&x, pos)?;
                if self.trace_layer_timings {
                    tracing::info!(
                        pos,
                        layer = il,
                        total_layer_us = layer_t0.unwrap().elapsed().as_micros(),
                        "qwen35 resident layer timing"
                    );
                }
                continue;
            }
            let residual = x.clone();
            let h = rms_combine(
                &x,
                &must_tensor(archive, &format!("blk.{il}.attn_norm.weight"))?,
                archive,
                cfg.norm_eps,
            )?;

            let y = if cfg.is_recurrent_layer(il) {
                let wqkv = must_tensor(archive, &format!("blk.{il}.attn_qkv.weight"))?;
                let wgate = must_tensor(archive, &format!("blk.{il}.attn_gate.weight"))?;
                let conv = must_tensor(archive, &format!("blk.{il}.ssm_conv1d.weight"))?;
                let conv_f = tensor_f32_flat(archive, &conv)?;
                let d0 = usize::try_from(conv.dimensions[0]).unwrap_or(0);
                let d1 = usize::try_from(conv.dimensions[1]).unwrap_or(0);
                let ssm_b = must_tensor(archive, &format!("blk.{il}.ssm_beta.weight"))?;
                let ssm_al = must_tensor(archive, &format!("blk.{il}.ssm_alpha.weight"))?;
                let ssm_dt = must_tensor(archive, &format!("blk.{il}.ssm_dt.bias"))?;
                let ssm_a = archive
                    .tensor_first_of(&[
                        &format!("blk.{il}.ssm_a_noscan.weight"),
                        &format!("blk.{il}.ssm_a.weight"),
                        &format!("blk.{il}.ssm_a"),
                    ])
                    .cloned()
                    .ok_or_else(|| {
                        BitNetError::Inference(format!(
                            "layer {il}: missing ssm_a / ssm_a_noscan tensor"
                        ))
                    })?;
                let ssm_no = must_tensor(archive, &format!("blk.{il}.ssm_norm.weight"))?;
                let ssm_out = must_tensor(archive, &format!("blk.{il}.ssm_out.weight"))?;
                let dt_b = tensor_f32_flat(archive, &ssm_dt)?;
                let a_b = tensor_f32_flat(archive, &ssm_a)?;
                let norm_b = tensor_f32_flat(archive, &ssm_no)?;
                let wqkv_p = archive.tensor_payload(&wqkv)?;
                let wg_p = archive.tensor_payload(&wgate)?;
                let sb_p = archive.tensor_payload(&ssm_b)?;
                let sa_p = archive.tensor_payload(&ssm_al)?;
                let so_p = archive.tensor_payload(&ssm_out)?;
                recurrent_forward(
                    &self.weights,
                    archive,
                    cfg,
                    &mut self.rec[il],
                    il,
                    &h,
                    (
                        wqkv_p,
                        wqkv.ggml_type,
                        usize::try_from(wqkv.dimensions[0]).unwrap_or(0),
                        usize::try_from(wqkv.dimensions[1]).unwrap_or(0),
                    ),
                    (
                        wg_p,
                        wgate.ggml_type,
                        usize::try_from(wgate.dimensions[0]).unwrap_or(0),
                        usize::try_from(wgate.dimensions[1]).unwrap_or(0),
                    ),
                    &conv_f,
                    d0,
                    d1,
                    (
                        sb_p,
                        ssm_b.ggml_type,
                        usize::try_from(ssm_b.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_b.dimensions[1]).unwrap_or(0),
                    ),
                    (
                        sa_p,
                        ssm_al.ggml_type,
                        usize::try_from(ssm_al.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_al.dimensions[1]).unwrap_or(0),
                    ),
                    &dt_b,
                    &a_b,
                    &norm_b,
                    (
                        so_p,
                        ssm_out.ggml_type,
                        usize::try_from(ssm_out.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_out.dimensions[1]).unwrap_or(0),
                    ),
                )?
            } else {
                let wq = must_tensor(archive, &format!("blk.{il}.attn_q.weight"))?;
                let wk = must_tensor(archive, &format!("blk.{il}.attn_k.weight"))?;
                let wv = must_tensor(archive, &format!("blk.{il}.attn_v.weight"))?;
                let wo = must_tensor(archive, &format!("blk.{il}.attn_output.weight"))?;
                let qn = must_tensor(archive, &format!("blk.{il}.attn_q_norm.weight"))?;
                let kn = must_tensor(archive, &format!("blk.{il}.attn_k_norm.weight"))?;
                let qnw = tensor_f32_flat(archive, &qn)?;
                let knw = tensor_f32_flat(archive, &kn)?;
                block_full_attention(
                    &self.weights,
                    cfg,
                    &mut self.attn_kv,
                    il,
                    pos,
                    &h,
                    archive.tensor_payload(&wq)?,
                    wq.ggml_type,
                    usize::try_from(wq.dimensions[0]).unwrap_or(0),
                    usize::try_from(wq.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wk)?,
                    wk.ggml_type,
                    usize::try_from(wk.dimensions[0]).unwrap_or(0),
                    usize::try_from(wk.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wv)?,
                    wv.ggml_type,
                    usize::try_from(wv.dimensions[0]).unwrap_or(0),
                    usize::try_from(wv.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wo)?,
                    wo.ggml_type,
                    usize::try_from(wo.dimensions[0]).unwrap_or(0),
                    usize::try_from(wo.dimensions[1]).unwrap_or(0),
                    &qnw,
                    &knw,
                )?
            };
            let attn_ms = layer_t0.map(|t| t.elapsed().as_millis()).unwrap_or(0);

            for i in 0..cfg.n_embd {
                x[i] = residual[i] + y[i];
            }

            let ffn_residual = x.clone();

            let post_name = archive
                .tensor_by_name(&format!("blk.{il}.post_attention_norm.weight"))
                .or_else(|| archive.tensor_by_name(&format!("blk.{il}.attn_post_norm.weight")))
                .or_else(|| {
                    archive.tensor_by_name(&format!("blk.{il}.attention_norm_after.weight"))
                })
                .ok_or_else(|| {
                    BitNetError::Inference(format!(
                        "layer {il}: missing post-attention RMS norm tensor"
                    ))
                })?;

            let h2 = rms_combine(&x, post_name, archive, cfg.norm_eps)?;

            let moe_delta = if cfg.n_expert == 0 {
                let gate = self
                    .weights
                    .matvec(&format!("blk.{il}.ffn_gate.weight"), &h2)?;
                let up = self
                    .weights
                    .matvec(&format!("blk.{il}.ffn_up.weight"), &h2)?;
                let hidden: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                self.weights
                    .matvec(&format!("blk.{il}.ffn_down.weight"), &hidden)?
            } else {
                let gate_in = must_tensor(archive, &format!("blk.{il}.ffn_gate_inp.weight"))?;
                let up = must_tensor(archive, &format!("blk.{il}.ffn_up_exps.weight"))?;
                let gate = must_tensor(archive, &format!("blk.{il}.ffn_gate_exps.weight"))?;
                let down = must_tensor(archive, &format!("blk.{il}.ffn_down_exps.weight"))?;
                let gate_up_fused = archive.tensor_first_of(&[
                    &format!("blk.{il}.ffn_gate_up_exps.weight"),
                    &format!("blk.{il}.ffn_gate_up_exps"),
                ]);

                let mut moe_delta = moe_forward(
                    &self.weights,
                    archive,
                    cfg,
                    &h2,
                    &gate_in,
                    &up,
                    &gate,
                    &down,
                    self.debug_moe_topk,
                    gate_up_fused,
                )?;

                if let Some(gs) = archive.tensor_first_of(&[
                    &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
                    &format!("blk.{il}.ffn_gate_inp_shexp"),
                ]) {
                    let gate_w = must_tensor(archive, &format!("blk.{il}.ffn_gate_shexp.weight"))?;
                    let up_w = must_tensor(archive, &format!("blk.{il}.ffn_up_shexp.weight"))?;
                    let down_w = must_tensor(archive, &format!("blk.{il}.ffn_down_shexp.weight"))?;
                    let sh = shared_expert_forward(
                        &self.weights,
                        archive,
                        cfg,
                        &h2,
                        gs,
                        &gate_w,
                        &up_w,
                        &down_w,
                    )?;
                    for i in 0..cfg.n_embd {
                        moe_delta[i] += sh[i];
                    }
                }

                moe_delta
            };

            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + moe_delta[i];
            }
            if self.trace_layer_timings {
                let total_ms = layer_t0.map(|t| t.elapsed().as_millis()).unwrap_or(0);
                let moe_ms = total_ms.saturating_sub(attn_ms);
                tracing::info!(
                    pos = pos,
                    layer = il,
                    layer_limit = layer_limit,
                    recurrent = cfg.is_recurrent_layer(il),
                    moe_topk_override = self.debug_moe_topk.unwrap_or(0),
                    attn_or_ssm_ms = attn_ms,
                    moe_ms = moe_ms,
                    total_layer_ms = total_ms,
                    "qwen35 layer timing"
                );
            }
        }

        if !logits_required {
            return Ok((Vec::new(), None));
        }
        let logits_t0 = if self.trace_layer_timings {
            Some(Instant::now())
        } else {
            None
        };
        let output = if let Some(head) = &mut self.gpu_head {
            head.run(&x, greedy)?
        } else {
            let xn = rms_combine(&x, &self.out_norm, archive, cfg.norm_eps)?;
            let head = &self.out_head;
            (
                self.weights.payload(
                    archive.tensor_payload(head)?,
                    head.ggml_type,
                    cfg.n_embd,
                    cfg.n_vocab,
                    &xn,
                )?,
                None,
            )
        };
        if self.trace_layer_timings {
            let logits_ms = logits_t0.map(|t| t.elapsed().as_millis()).unwrap_or(0);
            let step_ms = step_t0.map(|t| t.elapsed().as_millis()).unwrap_or(0);
            tracing::info!(
                pos = pos,
                logits_ms = logits_ms,
                token_total_ms = step_ms,
                "qwen35 token timing"
            );
        }
        Ok(output)
    }
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod sequence_tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        format: String,
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        prompt: String,
        prompt_ids: Vec<u32>,
        greedy_ids: Vec<u32>,
        text: String,
        stop_type: String,
    }

    #[test]
    fn optional_real_block_prefill_cancellation_keeps_checkpoints_and_restarts() {
        if std::env::var("RBITNET_QWEN_BLOCK_TEST").as_deref() != Ok("1") {
            return;
        }
        let gguf = std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap();
        let tokenizer = std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap();
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let previous =
            std::env::var("RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS").unwrap_or("256".into());
        std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "1");
        std::env::set_var("RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS", "128");
        let mut rt = Qwen35Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
        )
        .unwrap();
        assert_eq!(rt.gpu_prefill_capacity(), 128);
        let prompt=format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nQuelle est la capitale de la France ? Réponds en un mot.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","Les bibliothèques conservent des livres et les villes ont des jardins.\n".repeat(72));
        let total = rt.tokenizer.encode_ids(&prompt, true).unwrap().len();
        assert!(total > 1024);
        let before = crate::perf::snapshot().gpu_prefill_blocks;
        crate::cancel::clear_inference_cancel();
        let interrupt = std::thread::spawn(move || {
            for _ in 0..5000 {
                if crate::perf::snapshot().gpu_prefill_blocks > before {
                    crate::cancel::request_inference_cancel();
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            false
        });
        let cancelled =
            rt.generate_with_timings(&prompt, 16, SamplingOptions::from_temperature(0.0));
        let armed = interrupt.join().unwrap();
        crate::cancel::clear_inference_cancel();
        assert!(armed && cancelled.unwrap_err().to_string().contains("cancelled"));
        let blocks = crate::perf::snapshot().gpu_prefill_blocks - before;
        assert!(
            blocks > 0 && blocks < (total.div_ceil(128)) as u64,
            "cancelled after {blocks} blocks for {total} tokens"
        );
        let resumed = rt
            .generate_with_timings(&prompt, 16, SamplingOptions::from_temperature(0.0))
            .unwrap()
            .0;
        let mut cold =
            Qwen35Runtime::load(archive, Path::new(&tokenizer), BackendKind::Cuda).unwrap();
        assert_eq!(
            resumed,
            cold.generate_with_timings(&prompt, 16, SamplingOptions::from_temperature(0.0))
                .unwrap()
                .0
        );
        std::env::set_var("RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS", previous);
        eprintln!("Cancelled during actual block prefill after {blocks} blocks / {total} tokens; checkpoint resume matches cold generation");
    }

    #[test]
    fn optional_real_block_prefill_teacher_forcing_matches_serial_logits_and_state() {
        if std::env::var("RBITNET_QWEN_BLOCK_TEST").as_deref() != Ok("1") {
            return;
        }
        let gguf = std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap();
        let tokenizer = std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap();
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let previous_graph = std::env::var("RBITNET_CUDA_QWEN_FULL_GRAPH").unwrap_or("1".into());
        let previous_tensor = std::env::var("RBITNET_CUDA_PREFILL_TF32X3").unwrap_or("0".into());
        std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "0");
        let mut reference = Qwen35Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
        )
        .unwrap();
        assert_eq!(reference.gpu_prefill_capacity(), 1);
        let mut checked = 0;
        let mut worst_kl = 0.0f64;
        let mut worst_nll = 0.0f64;
        let before = crate::perf::snapshot();
        for graphs in ["0", "1"] {
            std::env::set_var("RBITNET_CUDA_QWEN_FULL_GRAPH", graphs);
            for tensor in ["0", "1"] {
                std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "1");
                std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", tensor);
                let mut block = Qwen35Runtime::load(
                    Arc::clone(&archive),
                    Path::new(&tokenizer),
                    BackendKind::Cuda,
                )
                .unwrap();
                assert_eq!(block.gpu_prefill_capacity(), 128);
                for sentence in ["Paris est la capitale de la France. Ses bibliothèques contiennent des livres. ","def add(a, b):\n    return a + b\nprint(add(17, 25))\n"] {
                    let ids=reference.tokenizer.encode_ids(&sentence.repeat(60),true).unwrap();
                    for count in [7,16,63,64,127,128,129,257] {
                        let mut expected=Vec::new();
                        for (pos,&id) in ids[..count].iter().enumerate() {
                            expected=reference.forward_one(id,pos,&archive,pos+1==count).unwrap();
                        }
                        let mut actual=Vec::new();
                        for (index,chunk) in ids[..count].chunks(128).enumerate() {
                            actual=block.forward_block(chunk,index*128,&archive,index*128+chunk.len()==count,false).unwrap().0;
                        }
                        for pos in count-1..count+5 {
                            assert_eq!(actual.len(),expected.len());
                            let argmax=|x:&[f32]| x.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
                            assert_eq!(argmax(&actual),argmax(&expected),"graphs={graphs} tensor={tensor} prefix={count} pos={pos}");
                            for (&a,&e) in actual.iter().zip(&expected) {
                                assert!(a.is_finite() && e.is_finite() && (a-e).abs()<=0.003*(1.0+e.abs()),"graphs={graphs} tensor={tensor} prefix={count} pos={pos}: {a} vs {e}");
                            }
                            let log_probs=|x:&[f32]| {
                                let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
                                let z=x.iter().map(|&v|(v as f64-max).exp()).sum::<f64>().ln()+max;
                                x.iter().map(|&v|v as f64-z).collect::<Vec<_>>()
                            };
                            let p=log_probs(&expected);let q=log_probs(&actual);
                            let kl=p.iter().zip(&q).map(|(&p,&q)|p.exp()*(p-q)).sum::<f64>();
                            let nll=(p[ids[pos+1] as usize]-q[ids[pos+1] as usize]).abs();
                            worst_kl=worst_kl.max(kl);worst_nll=worst_nll.max(nll);
                            assert!(kl<=1e-5 && nll<=1e-3,"KL={kl} NLL delta={nll}");checked+=1;
                            if pos<count+4 {
                                expected=reference.forward_one(ids[pos+1],pos+1,&archive,true).unwrap();
                                actual=block.forward_one(ids[pos+1],pos+1,&archive,true).unwrap();
                            }
                        }
                    }
                }
            }
        }
        let after = crate::perf::snapshot();
        assert!(
            after.gpu_tensor_gemm_calls > before.gpu_tensor_gemm_calls
                && after.gpu_prefill_tokens > before.gpu_prefill_tokens
        );
        std::env::set_var("RBITNET_CUDA_QWEN_FULL_GRAPH", previous_graph);
        std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", previous_tensor);
        eprintln!("Qwen teacher-forced positions={checked}, worst KL={worst_kl:.3e}, worst abs NLL delta={worst_nll:.3e}; eager/graph SIMT/TF32 argmax and state continuation passed");
    }

    #[test]
    fn optional_qwen_prefix_checkpoints_restore_kv_gdn_and_convolution_after_divergence() {
        if std::env::var("RBITNET_QWEN_PREFIX_TEST").as_deref() != Ok("1") {
            return;
        }
        assert!(prefix::enabled());
        let gguf = std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap();
        let tokenizer = std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap();
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let backend = if std::env::var("RBITNET_QWEN_SEQUENCE_BACKEND").as_deref() == Ok("cuda") {
            BackendKind::Cuda
        } else {
            BackendKind::Cpu
        };
        let common = "<|im_start|>system\nTu es un assistant précis. Réponds sans raisonnement détaillé.\n<|im_end|>\n<|im_start|>user\n";
        let prompts: Vec<_> = [
            "Quelle est la capitale de la France ?",
            "Quelle est la capitale de l'Italie ?",
            "Quelle est la capitale de la France ?",
            "Calcule 17 + 25.",
            "Répète été, café, résumé, 🙂.",
            "Quelle est la capitale de la France ?",
        ]
        .into_iter()
        .map(|q| format!("{common}{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        .collect();
        for sampling in [
            SamplingOptions::from_temperature(0.0),
            SamplingOptions {
                temperature: 0.7,
                seed: Some(735),
                top_p: Some(0.9),
                ..Default::default()
            },
            SamplingOptions {
                temperature: 0.7,
                seed: Some(735),
                frequency_penalty: 0.2,
                presence_penalty: 0.1,
                ..Default::default()
            },
        ] {
            let mut warm =
                Qwen35Runtime::load(Arc::clone(&archive), Path::new(&tokenizer), backend).unwrap();
            let before = crate::perf::snapshot().prefix_cache_hits;
            for prompt in &prompts {
                let actual = warm
                    .generate_with_timings(prompt, 24, sampling.clone())
                    .unwrap();
                let mut cold =
                    Qwen35Runtime::load(Arc::clone(&archive), Path::new(&tokenizer), backend)
                        .unwrap();
                let expected = cold
                    .generate_with_timings(prompt, 24, sampling.clone())
                    .unwrap();
                assert!(!actual.0.is_empty());
                assert_eq!(actual.0, expected.0);
                assert_eq!(actual.1.completion_tokens, expected.1.completion_tokens);
                if backend == BackendKind::Cuda {
                    assert!(warm
                        .gpu_recurrent
                        .iter()
                        .flatten()
                        .all(|g| g.supports_snapshot()));
                }
            }
            assert!(
                crate::perf::snapshot().prefix_cache_hits > before,
                "prefix checkpoints must be reused"
            );
            let expected = warm
                .generate_with_timings(&prompts[0], 24, sampling)
                .unwrap()
                .0;
            let mut emitted = false;
            let mut callback = |event| {
                if matches!(event, crate::stream::StreamEvent::Delta { .. }) {
                    emitted = true;
                    crate::cancel::request_inference_cancel();
                }
                Ok(())
            };
            let cancelled = warm.generate_inner(&prompts[0], 24, sampling, Some(&mut callback));
            crate::cancel::clear_inference_cancel();
            assert!(emitted && cancelled.unwrap_err().to_string().contains("cancelled"));
            assert_eq!(
                warm.generate_with_timings(&prompts[0], 24, sampling)
                    .unwrap()
                    .0,
                expected
            );
        }
    }

    #[test]
    fn optional_qwen_greedy_ids_text_and_sequence_reset_match_reference() {
        let (Ok(gguf), Ok(tokenizer), Ok(fixture)) = (
            std::env::var("RBITNET_QWEN_TEST_GGUF"),
            std::env::var("RBITNET_QWEN_TEST_TOKENIZER"),
            std::env::var("RBITNET_QWEN_SEQUENCE_JSON"),
        ) else {
            return;
        };
        let reference: Reference =
            serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
        assert_eq!(reference.format, "rbitnet-qwen-sequence-v1");
        assert!(!reference.cases.is_empty());
        let backend = match std::env::var("RBITNET_QWEN_SEQUENCE_BACKEND").as_deref() {
            Ok("cuda") => BackendKind::Cuda,
            Ok("cpu") | Err(_) => BackendKind::Cpu,
            Ok(other) => panic!("unsupported Qwen test backend: {other}"),
        };
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let mut rt =
            Qwen35Runtime::load(Arc::clone(&archive), Path::new(&tokenizer), backend).unwrap();
        if std::env::var("RBITNET_REQUIRE_QWEN_FULL").as_deref() == Ok("1") {
            assert!(rt.gpu_full.is_some(), "full GPU pipeline required");
        }
        if std::env::var("RBITNET_QWEN_REQUIRE_RESIDENT").as_deref() == Ok("1") {
            assert_eq!(
                rt.gpu_recurrent.iter().flatten().count(),
                rt.cfg.recurrent_layers.iter().filter(|&&v| v).count()
            );
            assert!(
                rt.gpu_recurrent.iter().any(|g| g.is_some()),
                "resident recurrent blocks required"
            );
        }
        for case in reference.cases {
            assert_eq!(
                rt.tokenizer.encode_ids(&case.prompt, true).unwrap(),
                case.prompt_ids,
                "{}: prompt IDs",
                case.name
            );
            rt.attn_kv.clear();
            for st in &mut rt.rec {
                st.conv_hist.fill(0.0);
                st.ssm_state.fill(0.0);
            }
            let mut logits = Vec::new();
            for (pos, &id) in case.prompt_ids.iter().enumerate() {
                logits = rt
                    .forward_one(id, pos, &archive, pos + 1 == case.prompt_ids.len())
                    .unwrap();
            }
            for (step, &expected) in case.greedy_ids.iter().enumerate() {
                let got = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                assert_eq!(got, expected, "{}: token {step}", case.name);
                if step + 1 < case.greedy_ids.len() {
                    logits = rt
                        .forward_one(got, case.prompt_ids.len() + step, &archive, true)
                        .unwrap();
                }
            }
            let limit = case.greedy_ids.len() as u32 + if case.stop_type == "eos" { 8 } else { 0 };
            let (text, stats) = rt
                .generate_with_timings(&case.prompt, limit, SamplingOptions::from_temperature(0.0))
                .unwrap();
            assert_eq!(text, case.text, "{}: completion text", case.name);
            assert_eq!(stats.prompt_tokens as usize, case.prompt_ids.len());
            assert_eq!(
                stats.completion_tokens as usize + usize::from(case.stop_type == "eos"),
                case.greedy_ids.len()
            );
        }
    }
}

#[path = "spec_decode.rs"]
mod spec_decode;

#[path = "spec_serving.rs"]
mod spec_serving;

#[cfg(test)]
#[path = "spec_tests.rs"]
mod spec_tests;

#[cfg(test)]
#[path = "context_tests.rs"]
mod context_tests;
#[cfg(test)]
#[path = "spec_decode_tests.rs"]
mod spec_decode_tests;
