//! [`crate::model::ModelExecutor`] for dense `general.architecture = spark2_5`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::backend::{BackendKind, ComputeBackend};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use crate::timings::PhaseTimings;

use super::runtime::Spark25Runtime;

pub struct Spark25Executor {
    pub backend_kind: BackendKind,
    pub backend_impl: Box<dyn ComputeBackend>,
    pub gguf: Arc<GgufArchive>,
    pub tokenizer_path: PathBuf,
    context_capacity: usize,
    prompt_tokenizer: Arc<LoadedPromptTokenizer>,
    gpu_matvec_resident: bool,
    resident_bytes: usize,
    runtime: Mutex<Option<Spark25Runtime>>,
}

impl Spark25Executor {
    pub fn new(
        backend_kind: BackendKind,
        gguf: Arc<GgufArchive>,
        backend: Box<dyn ComputeBackend>,
        tokenizer_path: PathBuf,
    ) -> Result<Self> {
        let runtime = Spark25Runtime::load(
            Arc::clone(&gguf),
            &tokenizer_path,
            backend_kind,
        )?;
        let gpu_matvec_resident = runtime.weights.gpu_matvec_resident();
        let resident_bytes = runtime.weights.resident_bytes;
        Ok(Self {
            backend_kind,
            backend_impl: backend,
            gguf,
            tokenizer_path,
            context_capacity: runtime.context_capacity(),
            prompt_tokenizer: Arc::clone(&runtime.tokenizer),
            gpu_matvec_resident,
            resident_bytes,
            runtime: Mutex::new(Some(runtime)),
        })
    }
}

impl crate::model::ModelExecutor for Spark25Executor {
    fn context_capacity(&self) -> Option<usize> {
        Some(self.context_capacity)
    }
    fn family(&self) -> &'static str {
        "spark2_5"
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
    }

    fn backend_accelerated(&self) -> bool {
        self.gpu_matvec_resident
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn openai_model_id(&self, gguf: Option<&GgufArchive>) -> Option<String> {
        gguf.map(|g| g.suggested_openai_model_id())
            .or_else(|| Some("rbitnet-spark2_5".into()))
    }

    fn offload_metadata(&self) -> Option<String> {
        if !self.gpu_matvec_resident {
            return match self.backend_kind {
                BackendKind::Hybrid | BackendKind::Cuda => Some(
                    "CUDA/hybrid selected; linear layers fall back to CPU quant matvec until \
                     device-resident weights upload (build cuda_quant, Q4_K GGUF)"
                        .into(),
                ),
                _ => None,
            };
        }
        Some(format!(
            "spark2_5 GPU quant matvec for fused QKV, attn gate, FFN, output; {} MiB device-resident; \
             ISWA attention, dual RoPE, GELU FFN activations on CPU",
            self.resident_bytes / (1024 * 1024)
        ))
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        let tok = &self.prompt_tokenizer;
        Ok(tok.encode_ids(prompt, true)?.len() as u32)
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        let mut slot = self
            .runtime
            .lock()
            .map_err(|e| BitNetError::Inference(format!("executor lock poisoned: {e}")))?;
        if slot.is_none() {
            *slot = Some(Spark25Runtime::load(
                Arc::clone(&self.gguf),
                &self.tokenizer_path,
                self.backend_kind,
            )?);
        }
        slot.as_mut()
            .unwrap()
            .generate_with_timings(prompt, max_tokens, sampling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{make_backend, BackendKind, CudaRuntime};

    #[test]
    fn spark25_cuda_kind_not_rejected_at_executor_gate() {
        if CudaRuntime::try_load().is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spark.gguf");
        crate::spark25::config::test_fixtures::write_minimal_spark(&path).unwrap();
        let gguf = Arc::new(crate::gguf::GgufArchive::mmap_path(&path).unwrap());
        let tok_path = dir.path().join("tokenizer.json");
        std::fs::write(&tok_path, r#"{"version":"1.0","model":{"type":"BPE"}}"#).unwrap();
        let backend = make_backend(BackendKind::Cuda);
        let msg = match Spark25Executor::new(BackendKind::Cuda, gguf, backend, tok_path) {
            Ok(_) => panic!("expected load to fail on minimal fixture"),
            Err(e) => format!("{e}"),
        };
        assert!(
            !msg.contains("GPU backend") && !msg.contains("unsupported"),
            "must not refuse CUDA at executor gate: {msg}"
        );
    }
}
