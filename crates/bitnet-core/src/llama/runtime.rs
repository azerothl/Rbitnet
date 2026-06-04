//! Tokenizer + generation loop.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::backend::{make_backend, BackendKind, ComputeBackend};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::scratch::ScratchArena;
use crate::timings::PhaseTimings;

use crate::paged_kv::PagedKvCache;
use crate::prefix_kv_exec::{
    prefix_scope_for_runtime, restore_dense_kv, shared_prefix_kv_execution_cache,
    snapshot_dense_kv, snapshot_key, SharedPrefixKvExecutionCache,
};
use crate::stream::StreamEvent;

use super::config::LlamaConfig;
use super::cuda_graph::CudaDecodeGraph;
use super::kv_storage::KvStorage;
use super::model::LlamaModel;

fn llama_encode_add_special_tokens() -> bool {
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
    tokenizer: LoadedPromptTokenizer,
    kv: KvStorage,
    backend: Box<dyn ComputeBackend>,
    prefill_chunk_tokens: usize,
    scratch: ScratchArena,
    prefix_kv_cache: SharedPrefixKvExecutionCache,
    prefix_scope: crate::prefix_kv::PrefixKvScope,
    cuda_graph: CudaDecodeGraph,
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
        let kv = llama_kv_from_env(&model.cfg)?;
        let backend = make_backend(backend_kind);
        let prefill_chunk_tokens = std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(128);
        let tokenizer_id = tokenizer_path.display().to_string();
        let kv_format = kv.stats().quant_format.to_string();
        let chat_format = std::env::var("RBITNET_CHAT_FORMAT").unwrap_or_else(|_| "auto".into());
        let prefix_scope = prefix_scope_for_runtime(
            &model_id,
            &tokenizer_id,
            &chat_format,
            &kv_format,
        );
        Ok(Self {
            model,
            tokenizer,
            kv,
            backend,
            prefill_chunk_tokens,
            scratch: ScratchArena::default(),
            prefix_kv_cache: shared_prefix_kv_execution_cache(),
            prefix_scope,
            cuda_graph: CudaDecodeGraph::from_env(),
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
        self.kv.clear();
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

        let mut prefill_from = 0usize;
        if let Ok(cache) = self.prefix_kv_cache.lock() {
            if cache.enabled() {
                let mut hit = false;
                for len in (1..=prompt_ids.len()).rev() {
                    let key = snapshot_key(&self.prefix_scope, &prompt_ids[..len]);
                    if let Some(snap) = cache.lookup(&key) {
                        if restore_dense_kv(&mut self.kv, snap) {
                            prefill_from = snap.token_count;
                            crate::perf::record_prefix_cache_hit(
                                snap.token_count.saturating_mul(64),
                            );
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

        let t_pf = Instant::now();
        let mut logits = Vec::new();
        let chunk_sz = self.prefill_chunk_tokens.max(1);
        if prefill_from < prompt_ids.len() {
            let suffix = &prompt_ids[prefill_from..];
            for (chunk_idx, chunk) in suffix.chunks(chunk_sz).enumerate() {
                let chunk_base = prefill_from + chunk_idx * chunk_sz;
                logits = self.prefill_chunk(chunk, chunk_base)?;
            }
        } else if prefill_from == prompt_ids.len() && !prompt_ids.is_empty() {
            let last = *prompt_ids.last().unwrap();
            logits = self.decode_one(last, prompt_ids.len().saturating_sub(1))?;
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        if let Ok(mut cache) = self.prefix_kv_cache.lock() {
            if cache.enabled() {
                if let Some(snap) = snapshot_dense_kv(&self.kv, prompt_ids.len()) {
                    let key = snapshot_key(&self.prefix_scope, &prompt_ids);
                    cache.store(key, snap);
                }
            }
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

        let eos_id = self.tokenizer.eos_token_id();

        let t_dec = Instant::now();
        let mut gen = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();
        let mut prev_text = String::new();

        for _ in 0..max_tokens {
            let next_id = sample_token(&logits, &sampling, &gen, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            gen.push(next_id);
            let full = self
                .tokenizer
                .decode_ids(&gen, llama_decode_skip_special_tokens())?;
            if let Some(cb) = on_event.as_deref_mut() {
                let delta = if full.len() >= prev_text.len() {
                    full[prev_text.len()..].to_string()
                } else {
                    full.clone()
                };
                if !delta.is_empty() {
                    cb(StreamEvent::Delta { text: delta })?;
                }
            }
            prev_text = full;
            logits = self.decode_one(next_id, pos)?;
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;

        let text = self
            .tokenizer
            .decode_ids(&gen, llama_decode_skip_special_tokens())?;
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms,
            prompt_tokens: prompt_ids.len() as u32,
            completion_tokens: gen.len() as u32,
        };

        if let Some(cb) = on_event.as_deref_mut() {
            cb(StreamEvent::Done(crate::scheduler::InferenceOutput {
                text: text.clone(),
                stats: crate::scheduler::InferenceStats::from_phases(phases.clone(), false),
            }))?;
        }

        Ok((text, phases))
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32], base_pos: usize) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for (idx, &tid) in tokens.iter().enumerate() {
            logits = self.decode_one(tid, base_pos + idx)?;
        }
        Ok(logits)
    }

    pub fn decode_one(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        self.cuda_graph.record_decode_step();
        self.model.forward_with_backend_and_scratch(
            &mut self.kv,
            token,
            pos,
            self.backend.as_ref(),
            &mut self.scratch,
        )
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
