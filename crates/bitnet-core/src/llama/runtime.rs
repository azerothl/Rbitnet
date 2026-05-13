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

use super::config::LlamaConfig;
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
}

impl LlamaRuntime {
    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let model = LlamaModel::from_gguf_arc_for_backend(archive, backend_kind)?;
        let tokenizer = LoadedPromptTokenizer::from_path(tokenizer_path)?;
        let kv = llama_kv_from_env(&model.cfg)?;
        let backend = make_backend(backend_kind);
        let prefill_chunk_tokens = std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(128);
        Ok(Self {
            model,
            tokenizer,
            kv,
            backend,
            prefill_chunk_tokens,
            scratch: ScratchArena::default(),
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

        let t_pf = Instant::now();
        let mut logits = Vec::new();
        let chunk_sz = self.prefill_chunk_tokens.max(1);
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            let chunk_base = chunk_idx * chunk_sz;
            logits = self.prefill_chunk(chunk, chunk_base)?;
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        let eos_id = self.tokenizer.eos_token_id();

        let t_dec = Instant::now();
        let mut gen = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();

        for _ in 0..max_tokens {
            let next_id = sample_token(&logits, &sampling, &gen, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            gen.push(next_id);
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
        self.model.forward_with_backend_and_scratch(
            &mut self.kv,
            token,
            pos,
            self.backend.as_ref(),
            &mut self.scratch,
        )
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
