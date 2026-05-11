//! [`crate::model::ModelExecutor`] for `general.architecture = qwen35moe`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use crate::timings::PhaseTimings;

use super::cuda_ctx::QwenCudaContext;
use super::runtime::Qwen35Runtime;

pub struct Qwen35MoeExecutor {
    pub backend_kind: BackendKind,
    #[allow(dead_code)]
    pub backend_impl: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
    cuda_ctx: Option<QwenCudaContext>,
    runtime: Mutex<Option<Qwen35Runtime>>,
}

impl Qwen35MoeExecutor {
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
            cuda_ctx: QwenCudaContext::try_load(),
            runtime: Mutex::new(None),
        }
    }
}

impl crate::model::ModelExecutor for Qwen35MoeExecutor {
    fn family(&self) -> &'static str {
        "qwen35moe"
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
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        if self.backend_kind != BackendKind::Cuda {
            return Err(BitNetError::Inference(
                "native qwen35moe requires CUDA for this phase (`RBITNET_BACKEND=cuda`).".into(),
            ));
        }
        let cuda = self.cuda_ctx.clone().ok_or_else(|| {
            BitNetError::Inference(
                "CUDA backend selected but NVIDIA CUDA/cuBLAS failed to load — see docs/USAGE.md for driver requirements."
                    .into(),
            )
        })?;

        let mut slot_rt = self
            .runtime
            .lock()
            .map_err(|e| BitNetError::Inference(format!("executor lock poisoned: {e}")))?;
        if slot_rt.is_none() {
            *slot_rt = Some(Qwen35Runtime::load(
                Arc::clone(&self.gguf),
                &self.tokenizer_path,
                cuda,
                self.backend_kind,
            )?);
        }

        slot_rt
            .as_mut()
            .unwrap()
            .generate_with_timings(prompt, max_tokens, sampling)
    }
}
