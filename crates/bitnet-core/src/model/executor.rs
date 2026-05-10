use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::llama::LlamaRuntime;
use crate::model::ToyLlm;
use crate::timings::PhaseTimings;

pub trait ModelExecutor: Send + Sync {
    fn family(&self) -> &'static str;
    fn backend(&self) -> BackendKind;
    fn backend_accelerated(&self) -> bool;
    fn is_ready(&self) -> bool;
    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String>;

    /// Tokenizer-encoded prompt length (used for HTTP limits).
    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32>;

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<(String, PhaseTimings)>;

    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        self.generate_with_timings(prompt, max_tokens, temperature)
            .map(|(s, _)| s)
    }
}

pub struct LlamaExecutor {
    pub backend_kind: BackendKind,
    pub backend_impl: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
    /// Reported [`ModelExecutor::family`] (defaults to `llama` for standard checkpoints).
    pub family_reported: &'static str,
    runtime: Mutex<Option<LlamaRuntime>>,
}

impl LlamaExecutor {
    pub fn new(
        backend_kind: BackendKind,
        backend: Box<dyn ComputeBackend>,
        gguf: Arc<GgufArchive>,
        tokenizer_path: PathBuf,
    ) -> Self {
        Self {
            backend_kind,
            backend_impl: backend,
            gguf,
            tokenizer_path,
            family_reported: "llama",
            runtime: Mutex::new(None),
        }
    }

    /// Same weights/tokenizer as [`Self::new`], but [`ModelExecutor::family`] reports `architecture_slug`
    /// (for Llama-compatible tensors under a roadmap `general.architecture` tag).
    pub fn new_with_architecture_slug(
        backend_kind: BackendKind,
        backend: Box<dyn ComputeBackend>,
        gguf: Arc<GgufArchive>,
        tokenizer_path: PathBuf,
        architecture_slug: &'static str,
    ) -> Self {
        Self {
            backend_kind,
            backend_impl: backend,
            gguf,
            tokenizer_path,
            family_reported: architecture_slug,
            runtime: Mutex::new(None),
        }
    }
}

impl ModelExecutor for LlamaExecutor {
    fn family(&self) -> &'static str {
        self.family_reported
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        let tok = LoadedPromptTokenizer::from_path(&self.tokenizer_path)?;
        Ok(tok.encode_ids(prompt, true)?.len() as u32)
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
    }
    fn backend_accelerated(&self) -> bool {
        self.backend_impl.is_native_accelerated()
    }

    fn is_ready(&self) -> bool {
        self.tokenizer_path.is_file()
    }

    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String> {
        gguf.map(|g| g.suggested_openai_model_id())
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<(String, PhaseTimings)> {
        let mut slot = self.runtime.lock().map_err(|e| {
            BitNetError::Inference(format!("executor lock poisoned: {e}"))
        })?;
        if slot.is_none() {
            *slot = Some(LlamaRuntime::load(
                Arc::clone(&self.gguf),
                &self.tokenizer_path,
                self.backend_kind,
            )?);
        }
        slot.as_mut()
            .unwrap()
            .generate_with_timings(prompt, max_tokens, temperature)
    }
}

pub struct BitNetExecutor {
    pub backend_kind: BackendKind,
    pub backend_impl: Box<dyn ComputeBackend>,
    toy: ToyLlm,
}

impl BitNetExecutor {
    pub fn new(backend_kind: BackendKind, backend: Box<dyn ComputeBackend>, seed: u64) -> Self {
        Self {
            backend_kind,
            backend_impl: backend,
            toy: ToyLlm::new(seed),
        }
    }
}

impl ModelExecutor for BitNetExecutor {
    fn family(&self) -> &'static str {
        "bitnet"
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        Ok((prompt.len() as u32).saturating_div(4).max(1))
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
    }
    fn backend_accelerated(&self) -> bool {
        self.backend_impl.is_native_accelerated()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String> {
        if let Some(g) = gguf {
            return Some(g.suggested_openai_model_id());
        }
        Some("rbitnet-bitnet".into())
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<(String, PhaseTimings)> {
        let t0 = Instant::now();
        let text = self.toy.generate(prompt, max_tokens, temperature);
        let total_ms = t0.elapsed().as_millis() as u64;
        let completion_tokens = text.split_whitespace().count() as u32;
        Ok((
            text,
            PhaseTimings::from_total_wall_ms(total_ms, completion_tokens),
        ))
    }
}
