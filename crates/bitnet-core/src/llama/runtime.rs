//! Tokenizer + generation loop.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::backend::{make_backend, BackendKind, ComputeBackend};
use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::scratch::ScratchArena;
use crate::timings::PhaseTimings;

use crate::kv_pool;
use crate::kv_sidecar::{KvSidecarClient, KvSidecarConfig, KvSidecarPut, NoopKvSidecar};
use crate::paged_kv::PagedKvCache;
use crate::prefix_kv::{
    DenseKvSnapshot, PagedKvSnapshot, PrefixKvBlockCache, PrefixKvMatch, PrefixKvSnap,
};
use crate::prefix_kv_exec::{
    prefix_scope_for_runtime, restore_dense_kv, restore_paged_kv, shared_prefix_kv_execution_cache,
    snapshot_dense_kv, snapshot_key, snapshot_paged_kv, SharedPrefixKvExecutionCache,
};
use crate::stream::{emit_text_delta, StreamEvent};

use super::config::LlamaConfig;
use super::cuda_graph::CudaDecodeGraph;
use super::kv_storage::KvStorage;
use super::model::LlamaModel;

pub(crate) fn llama_encode_add_special_tokens() -> bool {
    !matches!(
        std::env::var("RBITNET_LLAMA_ENCODE_ADD_SPECIAL").as_deref(),
        Ok("0") | Ok("false") | Ok("no")
    )
}

fn llama_decode_skip_special_tokens() -> bool {
    !matches!(
        std::env::var("RBITNET_LLAMA_DECODE_SKIP_SPECIAL").as_deref(),
        Ok("0") | Ok("false") | Ok("no")
    )
}

/// Loads [`LlamaModel`] from GGUF and a Hugging Face tokenizer file (`tokenizer.json`, or `tokenizer.model` when loadable).
pub struct LlamaRuntime {
    model: LlamaModel,
    resident: Option<super::resident::Resident>,
    tokenizer: LoadedPromptTokenizer,
    kv: KvStorage,
    backend: Box<dyn ComputeBackend>,
    prefill_chunk_tokens: usize,
    scratch: ScratchArena,
    prefix_kv_cache: SharedPrefixKvExecutionCache,
    prefix_radix: std::sync::Mutex<PrefixKvBlockCache>,
    prefix_scope: crate::prefix_kv::PrefixKvScope,
    /// Previous prompt token ids + snap for LCP agent-style reuse (SGLang-lite).
    last_prefix_ids: Option<Vec<u32>>,
    last_prefix_snap: Option<PrefixKvSnap>,
    cuda_graph: CudaDecodeGraph,
    sidecar: Box<dyn KvSidecarClient>,
    last_speculative_attempted: bool,
}

impl LlamaRuntime {
    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let model_id = archive.suggested_openai_model_id();
        let model = LlamaModel::from_gguf_arc_for_backend(archive, backend_kind)?;
        let tokenizer = LoadedPromptTokenizer::from_path(tokenizer_path)?;
        let _ = kv_pool::ensure_global_kv_pool(&model.cfg);
        let kv = llama_kv_from_env(&model.cfg)?;
        let resident = if backend_kind == BackendKind::Cuda && kv.as_paged().is_none() {
            super::resident::Resident::new(&model)
        } else {
            None
        };
        let backend = make_backend(backend_kind);
        let prefill_chunk_tokens = std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(128);
        let tokenizer_id = tokenizer_path.display().to_string();
        let kv_format = kv.stats().quant_format.to_string();
        let chat_format = std::env::var("RBITNET_CHAT_FORMAT").unwrap_or_else(|_| "auto".into());
        let prefix_scope =
            prefix_scope_for_runtime(&model_id, &tokenizer_id, &chat_format, &kv_format);
        let sidecar: Box<dyn KvSidecarClient> = if let Ok(Some(http)) =
            crate::kv_sidecar::HttpKvSidecar::from_config(&KvSidecarConfig::from_env())
        {
            Box::new(http)
        } else {
            Box::new(NoopKvSidecar)
        };
        Ok(Self {
            model,
            resident,
            tokenizer,
            kv,
            backend,
            prefill_chunk_tokens,
            scratch: ScratchArena::default(),
            prefix_kv_cache: shared_prefix_kv_execution_cache(),
            prefix_radix: std::sync::Mutex::new(PrefixKvBlockCache::from_env()),
            prefix_scope,
            last_prefix_ids: None,
            last_prefix_snap: None,
            cuda_graph: CudaDecodeGraph::from_env(),
            sidecar,
            last_speculative_attempted: false,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        self.generate_with_timings(
            prompt,
            max_tokens,
            SamplingOptions::from_temperature(temperature),
        )
        .map(|(s, _)| s)
    }

    pub fn generate_with_timings(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        self.generate_inner(prompt, max_tokens, sampling, None)
    }

    /// Stream decoded text deltas as tokens are generated.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
        on_event: &mut dyn FnMut(StreamEvent) -> Result<()>,
    ) -> Result<()> {
        self.generate_inner(prompt, max_tokens, sampling, Some(on_event))
            .map(|_| ())
    }

    fn generate_inner(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
        mut on_event: Option<&mut dyn FnMut(StreamEvent) -> Result<()>>,
    ) -> Result<(String, PhaseTimings)> {
        self.last_speculative_attempted = false;
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        if crate::native::prefix::enabled()
            && self
                .resident
                .as_ref()
                .is_some_and(|r| !r.supports_prefix_cache())
        {
            // An older native library has no snapshot API: use the existing host cache.
            self.resident = None;
        }
        // The resident cache starts a new sequence at position zero on the device.
        // Clearing the unused host cache would write hundreds of MiB per request.
        if self.resident.is_none() {
            self.kv.clear();
        }
        kv_pool::record_pool_metrics();
        let t_enc = Instant::now();
        let prompt_ids = self
            .tokenizer
            .encode_ids(prompt, llama_encode_add_special_tokens())?;
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

        if super::speculative::enabled()
            && self
                .resident
                .as_ref()
                .is_some_and(|r| r.supports_verification())
        {
            return self.generate_resident_speculative(
                &prompt_ids,
                max_tokens,
                sampling,
                encode_ms,
                on_event,
            );
        }
        if self.resident.is_some() && sampling.device_greedy_eligible() {
            return self.generate_resident_greedy(&prompt_ids, max_tokens, encode_ms, on_event);
        }

        let resident_path = self.resident.is_some();
        let mut prefill_from = if let Some(r) = &mut self.resident {
            r.restore_prefix(&prompt_ids)?
        } else {
            0
        };
        if !resident_path {
            let prefix_enabled = matches!(
                std::env::var("RBITNET_PREFIX_KV").as_deref(),
                Ok("1") | Ok("true") | Ok("yes")
            );
            if prefix_enabled {
                // 1) LCP against previous request (agent system/tools reuse).
                if let (Some(prev_ids), Some(prev_snap)) = (
                    self.last_prefix_ids.as_ref(),
                    self.last_prefix_snap.as_ref(),
                ) {
                    let lcp = longest_common_prefix_tokens(prev_ids, &prompt_ids);
                    let min_lcp = std::env::var("RBITNET_PREFIX_KV_MIN_TOKENS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(8usize);
                    if lcp >= min_lcp {
                        if let Some(truncated) =
                            truncate_prefix_snap(prev_snap, lcp, &self.model.cfg)
                        {
                            let restored = match &truncated {
                                PrefixKvSnap::Dense(d) => restore_dense_kv(&mut self.kv, d),
                                PrefixKvSnap::Paged(p) => restore_paged_kv(&mut self.kv, p),
                            };
                            if restored {
                                prefill_from = lcp.min(prompt_ids.len());
                                crate::perf::record_prefix_hit(lcp.saturating_mul(64));
                            }
                        }
                    }
                }
                // 2) Radix tree lookup when LCP path did not restore.
                if prefill_from == 0 {
                    if let Ok(mut radix) = self.prefix_radix.lock() {
                        if let Some(PrefixKvMatch {
                            matched_tokens,
                            bytes_saved,
                            snap,
                            ..
                        }) = radix.longest_token_prefix(&self.prefix_scope, &prompt_ids)
                        {
                            let restored = match snap.as_ref() {
                                Some(PrefixKvSnap::Dense(d)) => restore_dense_kv(&mut self.kv, d),
                                Some(PrefixKvSnap::Paged(p)) => restore_paged_kv(&mut self.kv, p),
                                None => false,
                            };
                            if restored && matched_tokens > 0 {
                                prefill_from = matched_tokens.min(prompt_ids.len());
                                crate::perf::record_prefix_hit(bytes_saved);
                            } else {
                                crate::perf::record_prefix_cache_miss();
                            }
                        } else {
                            crate::perf::record_prefix_cache_miss();
                        }
                    }
                }
            }
            if let Ok(cache) = self.prefix_kv_cache.lock() {
                if cache.enabled() && prefill_from == 0 {
                    let mut hit = false;
                    for len in (1..=prompt_ids.len()).rev() {
                        let key = snapshot_key(&self.prefix_scope, &prompt_ids[..len]);
                        if let Some(snap) = cache.lookup(&key) {
                            if restore_dense_kv(&mut self.kv, snap) {
                                prefill_from = snap.token_count;
                                crate::perf::record_prefix_hit(snap.token_count.saturating_mul(64));
                                hit = true;
                            }
                            break;
                        }
                    }
                    if !hit {
                        crate::perf::record_prefix_cache_miss();
                    }
                }
            }
        }
        let t_pf = Instant::now();
        let mut logits = Vec::new();
        let chunk_sz = self.prefill_chunk_tokens.max(1);
        if prefill_from < prompt_ids.len() {
            let suffix = &prompt_ids[prefill_from..];
            for (chunk_idx, chunk) in suffix.chunks(chunk_sz).enumerate() {
                if inference_cancelled() {
                    return Err(BitNetError::Inference("inference cancelled".into()));
                }
                let chunk_base = prefill_from + chunk_idx * chunk_sz;
                logits = self.prefill_chunk(chunk, chunk_base)?;
            }
        } else if prefill_from == prompt_ids.len() && !prompt_ids.is_empty() {
            let last = *prompt_ids.last().unwrap();
            logits = self.decode_one(last, prompt_ids.len().saturating_sub(1))?;
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        if let Some(r) = &mut self.resident {
            r.save_prefix(&prompt_ids);
        } else {
            let prefix_enabled = crate::native::prefix::enabled();
            if let Ok(mut cache) = self.prefix_kv_cache.lock() {
                if cache.enabled() {
                    if let Some(snap) = snapshot_dense_kv(&self.kv, prompt_ids.len()) {
                        let key = snapshot_key(&self.prefix_scope, &prompt_ids);
                        cache.store(key, snap);
                    }
                }
            }
            if prefix_enabled {
                let full_snap = if let Some(p) = snapshot_paged_kv(&self.kv, prompt_ids.len()) {
                    Some(PrefixKvSnap::Paged(p))
                } else {
                    snapshot_dense_kv(&self.kv, prompt_ids.len()).map(PrefixKvSnap::Dense)
                };
                if let Ok(mut radix) = self.prefix_radix.lock() {
                    let key = snapshot_key(&self.prefix_scope, &prompt_ids);
                    let blocks: Vec<usize> = (0..prompt_ids.len()).collect();
                    radix.insert_tokens_with_snap(key, &prompt_ids, blocks, full_snap.clone());
                    // Also index the LCP with the previous prompt so the next agent turn can hit.
                    if let Some(prev) = self.last_prefix_ids.as_ref() {
                        let lcp = longest_common_prefix_tokens(prev, &prompt_ids);
                        let min_lcp = std::env::var("RBITNET_PREFIX_KV_MIN_TOKENS")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(8usize);
                        if lcp >= min_lcp && lcp < prompt_ids.len() {
                            if let Some(ref full) = full_snap {
                                if let Some(truncated) =
                                    truncate_prefix_snap(full, lcp, &self.model.cfg)
                                {
                                    let key = snapshot_key(&self.prefix_scope, &prompt_ids[..lcp]);
                                    let blocks: Vec<usize> = (0..lcp).collect();
                                    radix.insert_tokens_with_snap(
                                        key,
                                        &prompt_ids[..lcp],
                                        blocks,
                                        Some(truncated),
                                    );
                                }
                            }
                        }
                    }
                }
                self.last_prefix_ids = Some(prompt_ids.clone());
                self.last_prefix_snap = full_snap;
            }
        }
        if self
            .sidecar
            .put_prefix_blocks(&KvSidecarPut {
                model_id: self.prefix_scope.model_id.clone(),
                prefix_hash: crate::prefix_kv::hash_prefix_tokens(&prompt_ids),
                token_count: prompt_ids.len(),
                block_ids: vec![],
            })
            .is_ok()
        {
            tracing::debug!("kv sidecar notified after prefill");
        }

        if let Some(cb) = on_event.as_deref_mut() {
            let partial = PhaseTimings {
                encode_ms,
                prefill_ms,
                decode_ms: 0,
                prompt_tokens: prompt_ids.len() as u32,
                completion_tokens: 0,
            };
            cb(StreamEvent::FirstToken {
                stats: crate::scheduler::InferenceStats::from_phases(partial, false),
            })?;
        }

        let eos_ids = self.tokenizer.eos_token_ids();

        let t_dec = Instant::now();
        let mut gen = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();
        let mut prev_text = String::new();

        for step in 0..max_tokens {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next_id = sample_token(&logits, &sampling, &gen, &mut rng);
            if eos_ids.contains(&next_id) {
                break;
            }
            gen.push(next_id);
            let full = self
                .tokenizer
                .decode_ids(&gen, llama_decode_skip_special_tokens())?;
            if let Some(cb) = on_event.as_deref_mut() {
                emit_text_delta(&full, &mut prev_text, false, cb)?;
            }
            if step + 1 < max_tokens {
                logits = self.decode_one(next_id, pos)?;
            }
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;
        kv_pool::record_pool_metrics();

        let text = self
            .tokenizer
            .decode_ids(&gen, llama_decode_skip_special_tokens())?;
        if let Some(cb) = on_event.as_deref_mut() {
            emit_text_delta(&text, &mut prev_text, true, cb)?;
        }
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms,
            prompt_tokens: prompt_ids.len() as u32,
            completion_tokens: gen.len() as u32,
        };

        if let Some(cb) = on_event {
            cb(StreamEvent::Done(crate::scheduler::InferenceOutput {
                text: text.clone(),
                stats: crate::scheduler::InferenceStats::from_phases(phases, false),
            }))?;
        }

        Ok((text, phases))
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32], base_pos: usize) -> Result<Vec<f32>> {
        if let Some(resident) = &mut self.resident {
            return resident
                .prefill(&self.model, tokens, base_pos, false)
                .map(|p| p.0);
        }
        let mut logits = Vec::new();
        for (idx, &tid) in tokens.iter().enumerate() {
            logits = self.model.forward_step(
                &mut self.kv,
                tid,
                base_pos + idx,
                self.backend.as_ref(),
                &mut self.scratch,
                idx + 1 == tokens.len(),
            )?;
        }
        Ok(logits)
    }

    pub fn decode_one(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        if let Some(resident) = &mut self.resident {
            return resident.forward(&self.model, token, pos, true);
        }
        self.cuda_graph.record_decode_step(true);
        self.model.forward_with_backend_and_scratch(
            &mut self.kv,
            token,
            pos,
            self.backend.as_ref(),
            &mut self.scratch,
        )
    }

    /// Whether this runtime executes the entire token graph on CUDA.
    pub fn uses_resident_cuda(&self) -> bool {
        self.resident.is_some()
    }
    pub fn speculative_attempted(&self) -> bool {
        self.last_speculative_attempted
    }

    fn generate_resident_speculative(
        &mut self,
        ids: &[u32],
        limit: u32,
        options: SamplingOptions,
        encode_ms: u64,
        mut events: Option<&mut dyn FnMut(StreamEvent) -> Result<()>>,
    ) -> Result<(String, PhaseTimings)> {
        let resident = self
            .resident
            .as_mut()
            .expect("verification eligibility checked");
        let greedy = options.device_greedy_eligible();
        let pf = Instant::now();
        let from = resident.restore_prefix(ids)?;
        let (logits, token) = resident.prefill(&self.model, &ids[from..], from, greedy)?;
        resident.save_prefix(ids);
        let prefill_ms = pf.elapsed().as_millis() as u64;
        if let Some(cb) = events.as_deref_mut() {
            cb(StreamEvent::FirstToken {
                stats: crate::scheduler::InferenceStats::from_phases(
                    PhaseTimings {
                        encode_ms,
                        prefill_ms,
                        prompt_tokens: ids.len() as u32,
                        ..Default::default()
                    },
                    false,
                ),
            })?;
        }
        let decode = Instant::now();
        let mut rng = seeded_rng(options.seed);
        let mut next = if greedy {
            token
        } else {
            sample_token(&logits, &options, &[], &mut rng)
        };
        let eos = self.tokenizer.eos_token_ids();
        let mut generated = Vec::new();
        let mut history = ids.to_vec();
        let mut emitted = String::new();
        let mut text = String::new();
        let mut position = ids.len();
        let mut cost_guard = super::speculative::CostGuard::default();
        let width = std::env::var("RBITNET_SPECULATIVE_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, 15);
        while generated.len() < limit as usize {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            if eos.contains(&next) {
                break;
            }
            generated.push(next);
            history.push(next);
            text = self
                .tokenizer
                .decode_ids(&generated, llama_decode_skip_special_tokens())?;
            if let Some(cb) = events.as_deref_mut() {
                emit_text_delta(&text, &mut emitted, false, cb)?;
            }
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let remaining = limit as usize - generated.len();
            if remaining == 0 {
                break;
            }
            let draft_start = Instant::now();
            let proposals = if cost_guard.permits() {
                super::speculative::propose(
                    &history,
                    width
                        .min(remaining)
                        .min(resident.remaining_capacity(position).saturating_sub(1)),
                )
            } else {
                Vec::new()
            };
            let draft_ns = draft_start.elapsed().as_nanos() as u64;
            if proposals.is_empty() {
                let serial_start = Instant::now();
                next = if greedy {
                    resident.greedy(&self.model, next, position, true)?
                } else {
                    sample_token(
                        &resident.forward(&self.model, next, position, true)?,
                        &options,
                        &generated,
                        &mut rng,
                    )
                };
                position += 1;
                cost_guard.serial(serial_start.elapsed().as_nanos() as u64);
                crate::perf::record_speculative_cost(draft_ns, 0, false);
                continue;
            }
            let mut inputs = Vec::with_capacity(proposals.len() + 1);
            inputs.push(next);
            inputs.extend_from_slice(&proposals);
            let verify_start = Instant::now();
            let verified = resident.verify(&self.model, &inputs, position, greedy)?;
            let verify_ns = verify_start.elapsed().as_nanos() as u64;
            self.last_speculative_attempted = true;
            let decision =
                super::speculative::decide(&proposals, remaining, &eos, &generated, |i, prior| {
                    verified.sample(i, &options, prior, &mut rng)
                });
            let kept = 1 + decision.accepted;
            cost_guard.verification(verify_ns, kept);
            let rollback = kept < inputs.len();
            if rollback {
                resident.truncate(position + kept)?;
            }
            position += kept;
            crate::perf::record_speculative(
                proposals.len() as u32,
                proposals.len() as u32,
                decision.accepted as u32,
            );
            crate::perf::record_speculative_cost(draft_ns, verify_ns, rollback);
            // All confirmed IDs are target decisions. Never publish the discarded tail.
            for token in decision.confirmed {
                if inference_cancelled() {
                    return Err(BitNetError::Inference("inference cancelled".into()));
                }
                generated.push(token);
                history.push(token);
                text = self
                    .tokenizer
                    .decode_ids(&generated, llama_decode_skip_special_tokens())?;
                if let Some(cb) = events.as_deref_mut() {
                    emit_text_delta(&text, &mut emitted, false, cb)?;
                }
            }
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let Some(pending) = decision.pending else {
                break;
            };
            next = pending;
        }
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms: decode.elapsed().as_millis() as u64,
            prompt_tokens: ids.len() as u32,
            completion_tokens: generated.len() as u32,
        };
        if let Some(cb) = events {
            emit_text_delta(&text, &mut emitted, true, cb)?;
            cb(StreamEvent::Done(crate::scheduler::InferenceOutput {
                text: text.clone(),
                stats: crate::scheduler::InferenceStats::from_phases(
                    phases,
                    self.last_speculative_attempted,
                ),
            }))?;
        }
        Ok((text, phases))
    }

    fn generate_resident_greedy(
        &mut self,
        ids: &[u32],
        limit: u32,
        encode_ms: u64,
        mut events: Option<&mut dyn FnMut(StreamEvent) -> Result<()>>,
    ) -> Result<(String, PhaseTimings)> {
        let resident = self
            .resident
            .as_mut()
            .expect("resident eligibility checked");
        let pf = Instant::now();
        let from = resident.restore_prefix(ids)?;
        let (_, mut next) = resident.prefill(&self.model, &ids[from..], from, true)?;
        resident.save_prefix(ids);
        let prefill_ms = pf.elapsed().as_millis() as u64;
        if let Some(cb) = events.as_deref_mut() {
            cb(StreamEvent::FirstToken {
                stats: crate::scheduler::InferenceStats::from_phases(
                    PhaseTimings {
                        encode_ms,
                        prefill_ms,
                        decode_ms: 0,
                        prompt_tokens: ids.len() as u32,
                        completion_tokens: 0,
                    },
                    false,
                ),
            })?;
        }
        let stop = self.tokenizer.eos_token_ids();
        let decode = Instant::now();
        let mut generated = Vec::new();
        let mut previous = String::new();
        let mut emitted = String::new();
        for step in 0..limit {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            let text = self
                .tokenizer
                .decode_ids(&generated, llama_decode_skip_special_tokens())?;
            if let Some(cb) = events.as_deref_mut() {
                emit_text_delta(&text, &mut emitted, false, cb)?;
            }
            previous = text;
            if step + 1 < limit {
                next = resident.greedy(&self.model, next, ids.len() + step as usize, true)?;
            }
        }
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms: decode.elapsed().as_millis() as u64,
            prompt_tokens: ids.len() as u32,
            completion_tokens: generated.len() as u32,
        };
        if let Some(cb) = events {
            emit_text_delta(&previous, &mut emitted, true, cb)?;
            cb(StreamEvent::Done(crate::scheduler::InferenceOutput {
                text: previous.clone(),
                stats: crate::scheduler::InferenceStats::from_phases(phases, false),
            }))?;
        }
        Ok((previous, phases))
    }

    /// After a full prompt, return the **greedy** next token id (temperature 0, no penalties).
    /// Used by golden / doctor checks; not a full chat template.
    pub fn greedy_next_token_id_after_prompt(&mut self, prompt: &str) -> Result<u32> {
        self.kv.clear();
        let prompt_ids = self
            .tokenizer
            .encode_ids(prompt, llama_encode_add_special_tokens())?;
        if prompt_ids.is_empty() {
            return Err(crate::error::BitNetError::Inference(
                "greedy_next_token: empty prompt encoding".into(),
            ));
        }
        let mut logits = Vec::new();
        let chunk_sz = self.prefill_chunk_tokens.max(1);
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            let chunk_base = chunk_idx * chunk_sz;
            logits = self.prefill_chunk(chunk, chunk_base)?;
        }
        let mut rng = seeded_rng(Some(0));
        let tok = crate::sampling::sample_token(
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
        );
        Ok(tok)
    }
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

fn llama_kv_from_env(cfg: &LlamaConfig) -> Result<KvStorage> {
    // `RBITNET_KV_POOL=1` activates shared physical pages end-to-end (implies paged slabs).
    if kv_pool::kv_pool_enabled() {
        return kv_pool::new_runtime_kv(cfg);
    }
    let use_paged = matches!(
        std::env::var("RBITNET_LLAMA_PAGED_KV").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    );
    if use_paged {
        let p = PagedKvCache::from_env();
        KvStorage::new_paged(cfg, p.page_size_tokens, p.max_pages)
    } else {
        Ok(KvStorage::new_dense(cfg))
    }
}

fn longest_common_prefix_tokens(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn truncate_prefix_snap(
    snap: &PrefixKvSnap,
    token_count: usize,
    cfg: &LlamaConfig,
) -> Option<PrefixKvSnap> {
    match snap {
        PrefixKvSnap::Dense(d) => {
            if token_count == 0 || token_count > d.token_count {
                return None;
            }
            if token_count == d.token_count {
                return Some(PrefixKvSnap::Dense(d.clone()));
            }
            let stride = cfg.n_kv.saturating_mul(cfg.head_dim);
            let keep = token_count.saturating_mul(stride);
            let mut k = Vec::with_capacity(d.k.len());
            let mut v = Vec::with_capacity(d.v.len());
            for row in &d.k {
                if row.len() < keep {
                    return None;
                }
                k.push(row[..keep].to_vec());
            }
            for row in &d.v {
                if row.len() < keep {
                    return None;
                }
                v.push(row[..keep].to_vec());
            }
            Some(PrefixKvSnap::Dense(DenseKvSnapshot { k, v, token_count }))
        }
        PrefixKvSnap::Paged(p) => {
            if token_count == 0 || token_count > p.token_count {
                return None;
            }
            Some(PrefixKvSnap::Paged(PagedKvSnapshot {
                block_phys: p.block_phys.clone(),
                token_count,
            }))
        }
    }
}

#[cfg(test)]
mod stream_tests {
    #[test]
    fn utf8_fragments_wait_for_complete_characters_and_flush_at_stop() {
        let mut emitted = String::new();
        let mut received = String::new();
        let mut event = |event| {
            if let crate::stream::StreamEvent::Delta { text } = event {
                received.push_str(&text);
            }
            Ok(())
        };
        for decoded in [
            "r\u{fffd}",
            "ré",
            "ré\u{fffd}",
            "résumé ",
            "résumé \u{fffd}",
            "résumé 🙂",
            "résumé 🙂\u{fffd}",
        ] {
            super::emit_text_delta(decoded, &mut emitted, false, &mut event).unwrap();
        }
        super::emit_text_delta("résumé 🙂\u{fffd}", &mut emitted, true, &mut event).unwrap();
        assert_eq!(received, "résumé 🙂\u{fffd}");
    }
}
