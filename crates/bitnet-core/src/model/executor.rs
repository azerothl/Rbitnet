use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::llama::LlamaRuntime;
use crate::model::ToyLlm;

pub trait ModelExecutor: Send + Sync {
    fn family(&self) -> &'static str;
    fn backend(&self) -> BackendKind;
    fn is_ready(&self) -> bool;
    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String>;
    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String>;
}

pub struct LlamaExecutor {
    pub backend_kind: BackendKind,
    pub _backend: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
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
            _backend: backend,
            gguf,
            tokenizer_path,
            runtime: Mutex::new(None),
        }
    }
}

impl ModelExecutor for LlamaExecutor {
    fn family(&self) -> &'static str {
        "llama"
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
    }

    fn is_ready(&self) -> bool {
        self.tokenizer_path.is_file()
    }

    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String> {
        gguf.map(|g| g.suggested_openai_model_id())
    }

    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        let mut slot = self.runtime.lock().map_err(|e| {
            BitNetError::Inference(format!("executor lock poisoned: {e}"))
        })?;
        if slot.is_none() {
            *slot = Some(LlamaRuntime::load(&self.gguf, &self.tokenizer_path)?);
        }
        slot.as_mut()
            .unwrap()
            .generate(prompt, max_tokens, temperature)
    }
}

pub struct BitNetExecutor {
    pub backend_kind: BackendKind,
    pub _backend: Box<dyn ComputeBackend>,
    toy: ToyLlm,
}

impl BitNetExecutor {
    pub fn new(backend_kind: BackendKind, backend: Box<dyn ComputeBackend>, seed: u64) -> Self {
        Self {
            backend_kind,
            _backend: backend,
            toy: ToyLlm::new(seed),
        }
    }
}

impl ModelExecutor for BitNetExecutor {
    fn family(&self) -> &'static str {
        "bitnet"
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
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

    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        Ok(self.toy.generate(prompt, max_tokens, temperature))
    }
}
