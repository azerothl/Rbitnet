//! Tokenizer + generation loop.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;

use crate::backend::{make_backend, BackendKind, ComputeBackend};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::timings::PhaseTimings;

use crate::paged_kv::PagedKvCache;

use super::config::LlamaConfig;
use super::kv_storage::KvStorage;
use super::model::LlamaModel;

/// Loads [`LlamaModel`] from GGUF and a Hugging Face tokenizer file (`tokenizer.json`, or `tokenizer.model` when loadable).
pub struct LlamaRuntime {
    model: LlamaModel,
    tokenizer: LoadedPromptTokenizer,
    kv: KvStorage,
    backend: Box<dyn ComputeBackend>,
    prefill_chunk_tokens: usize,
}

impl LlamaRuntime {
    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let model = LlamaModel::from_gguf_arc(archive)?;
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
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        self.generate_with_timings(prompt, max_tokens, temperature)
            .map(|(s, _)| s)
    }

    pub fn generate_with_timings(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<(String, PhaseTimings)> {
        self.kv.clear();
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
        let chunk_sz = self.prefill_chunk_tokens.max(1);
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            let chunk_base = chunk_idx * chunk_sz;
            for (idx, &tid) in chunk.iter().enumerate() {
                let pos = chunk_base + idx;
                logits = self.model.forward_with_backend(
                    &mut self.kv,
                    tid,
                    pos,
                    self.backend.as_ref(),
                )?;
            }
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        let eos_id = self.tokenizer.eos_token_id();

        let t_dec = Instant::now();
        let mut gen = Vec::new();
        let mut rng = rand::thread_rng();
        let mut pos = prompt_ids.len();

        for _ in 0..max_tokens {
            let next_id = sample_token(&logits, temperature, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            gen.push(next_id);
            logits = self.model.forward_with_backend(
                &mut self.kv,
                next_id,
                pos,
                self.backend.as_ref(),
            )?;
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;

        let text = self.tokenizer.decode_ids(&gen, true)?;
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms,
            prompt_tokens: prompt_ids.len() as u32,
            completion_tokens: gen.len() as u32,
        };
        Ok((text, phases))
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

fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
                let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
                a.total_cmp(&b)
            })
            .unwrap()
            .0 as u32;
    }
    let scaled: Vec<f32> = logits.iter().map(|z| z / temperature).collect();
    let m = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|z| (z - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * s;
    let mut c = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        c += e;
        if c >= r {
            return i as u32;
        }
    }
    (exps.len().saturating_sub(1)) as u32
}
