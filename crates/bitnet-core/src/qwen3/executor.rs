//! [`crate::model::ModelExecutor`] for dense `general.architecture = qwen3`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use crate::timings::PhaseTimings;

use super::runtime::Qwen3Runtime;

pub struct Qwen3Executor {
    pub backend_kind: BackendKind,
    #[allow(dead_code)]
    pub backend_impl: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
    runtime: Mutex<Option<Qwen3Runtime>>,
}

impl Qwen3Executor {
    pub fn new(
        backend_kind: BackendKind,
        gguf: Arc<GgufArchive>,
        backend: Box<dyn ComputeBackend>,
        tokenizer_path: PathBuf,
    ) -> Self {
        Self {
            backend_kind,
            backend_impl: backend,
            gguf,
            tokenizer_path,
            runtime: Mutex::new(None),
        }
    }
}

impl crate::model::ModelExecutor for Qwen3Executor {
    fn family(&self) -> &'static str {
        "qwen3"
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
            .or_else(|| Some("rbitnet-qwen3".into()))
    }

    fn offload_metadata(&self) -> Option<String> {
        (self.backend_kind == BackendKind::Hybrid)
            .then(|| "hybrid selected; dense Qwen3 currently uses CPU fallback".into())
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        let tok = LoadedPromptTokenizer::from_path(&self.tokenizer_path)?;
        Ok(tok.encode_ids(prompt, true)?.len() as u32)
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        if self.backend_kind == BackendKind::Cuda {
            return Err(BitNetError::Inference(
                "dense qwen3 MVP currently supports CPU backend only".into(),
            ));
        }
        if self.backend_kind == BackendKind::Hybrid {
            tracing::info!(
                "dense qwen3 hybrid selected; using CPU runtime until Qwen3 offload is wired"
            );
        }
        let mut slot = self
            .runtime
            .lock()
            .map_err(|e| BitNetError::Inference(format!("executor lock poisoned: {e}")))?;
        if slot.is_none() {
            *slot = Some(Qwen3Runtime::load(
                Arc::clone(&self.gguf),
                &self.tokenizer_path,
            )?);
        }
        slot.as_mut()
            .unwrap()
            .generate_with_timings(prompt, max_tokens, sampling)
    }
}
