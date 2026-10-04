//! [`crate::model::ModelExecutor`] for `general.architecture = qwen35moe`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use crate::timings::PhaseTimings;

use super::runtime::Qwen35Runtime;

pub struct Qwen35MoeExecutor {
    pub backend_kind: BackendKind,
    #[allow(dead_code)]
    pub backend_impl: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
    runtime: Mutex<Option<Qwen35Runtime>>,
}

impl Qwen35MoeExecutor {
    pub fn new(
        backend_kind: BackendKind,
        gguf: Arc<GgufArchive>,
        backend: Box<dyn ComputeBackend>,
        tokenizer_path: PathBuf,
    ) -> Result<Self> {
        let runtime = Qwen35Runtime::load(Arc::clone(&gguf), &tokenizer_path, backend_kind)?;
        Ok(Self {
            backend_kind,
            backend_impl: backend,
            gguf,
            tokenizer_path,
            runtime: Mutex::new(Some(runtime)),
        })
    }
}

impl crate::model::ModelExecutor for Qwen35MoeExecutor {
    fn family(&self) -> &'static str {
        if self.gguf.normalized_architecture().as_deref() == Some("qwen35") {
            "qwen35"
        } else {
            "qwen35moe"
        }
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

    fn offload_metadata(&self) -> Option<String> {
        let runtime = self.runtime.lock().ok()?;
        let rt = runtime.as_ref()?;
        let bytes = rt.resident_weights_bytes();
        let (gpu, total, head) = rt.gpu_execution_summary();
        if rt.has_full_gpu_pipeline() {
            return Some(format!("native dense Qwen CUDA token pipeline; {} MiB quantized weights resident; all attention/recurrent layers and output head on GPU; one embedding upload per token, logits or token ID download", bytes / (1024 * 1024)));
        }
        Some(format!("native qwen35 graph; {} MiB quantized weights resident on CUDA; {gpu}/{total} recurrent blocks resident; resident output head: {head}; remaining operations use the existing CPU/GPU path", bytes / (1024 * 1024)))
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        let mut slot_rt = self
            .runtime
            .lock()
            .map_err(|e| BitNetError::Inference(format!("executor lock poisoned: {e}")))?;
        slot_rt
            .as_mut()
            .unwrap()
            .generate_with_timings(prompt, max_tokens, sampling)
    }
    fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
        callback: &mut (dyn FnMut(crate::stream::StreamEvent) -> Result<()> + Send),
    ) -> Result<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|e| BitNetError::Inference(format!("executor lock poisoned: {e}")))?;
        let (text, phases) = runtime.as_mut().unwrap().generate_inner(
            prompt,
            max_tokens,
            sampling,
            Some(callback),
        )?;
        callback(crate::stream::StreamEvent::Done(
            crate::scheduler::InferenceOutput {
                text,
                stats: crate::scheduler::InferenceStats::from_phases(phases, false),
            },
        ))
    }
}
