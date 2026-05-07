//! High-level inference façade.
//!
//! - `RBITNET_STUB=1` — HTTP integration text (Akasha `BitNetProvider`).
//! - `RBITNET_TOY=1` — tiny in-process F32 toy LM (no GGUF).
//! - `RBITNET_MODEL` — load GGUF; full Llama-compatible forward + `tokenizer.json` / `tokenizer.model` / `RBITNET_TOKENIZER`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::model::{LlamaExecutor, ModelExecutor, ToyLlm};
use crate::paged_kv::PagedKvCache;
use crate::registry::KernelRegistry;
use crate::scheduler::{ContinuousBatchScheduler, InferenceOutput, InferenceRequest, InferenceStats};

/// Whether stub responses are enabled (no model required).
pub fn stub_mode_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_STUB").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Tiny toy LM (`RBITNET_TOY=1`), no weights file.
pub fn toy_mode_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_TOY").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn toy_seed() -> u64 {
    std::env::var("RBITNET_TOY_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42)
}

/// Path to GGUF from `RBITNET_MODEL` if set.
pub fn model_path_from_env() -> Option<PathBuf> {
    std::env::var_os("RBITNET_MODEL").map(PathBuf::from)
}

/// Reject paths containing `..` so environment-controlled paths cannot escape the intended directory.
pub fn validate_no_parent_components(path: &Path) -> Result<()> {
    for c in path.components() {
        if matches!(c, std::path::Component::ParentDir) {
            return Err(BitNetError::InvalidGguf(
                "path must not contain '..' components".into(),
            ));
        }
    }
    Ok(())
}

fn tokenizer_path_candidate(pb: &Path) -> bool {
    if !pb.is_file() {
        return false;
    }
    let Some(name) = pb.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == "tokenizer.json" || lower == "tokenizer.model"
}

fn resolve_tokenizer_path(model_path: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("RBITNET_TOKENIZER") {
        let pb = PathBuf::from(p);
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
    }
    if let Some(dir) = model_path.parent() {
        let pb = dir.join("tokenizer.json");
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
        let pb = dir.join("tokenizer.model");
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
    }
    Err(BitNetError::TokenizerMissing)
}

/// Shared engine state.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    #[allow(dead_code)]
    model_path: Option<PathBuf>,
    gguf: Option<Arc<GgufArchive>>,
    toy: Option<ToyLlm>,
    stub: bool,
    backend_kind: BackendKind,
    model_family: String,
    scheduler: ContinuousBatchScheduler,
    _kernel_registry: KernelRegistry,
    _paged_kv: PagedKvCache,
    executor: Option<Box<dyn ModelExecutor>>,
}

fn validate_model_path_for_gguf(p: &Path) -> Result<()> {
    validate_no_parent_components(p)?;
    match std::fs::metadata(p) {
        Ok(m) if m.is_dir() => Err(BitNetError::InvalidGguf(format!(
            "RBITNET_MODEL must be a single .gguf file, not a directory: {}. \
             Example: {}\\model.Q4_K_M.gguf",
            p.display(),
            p.display()
        ))),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(BitNetError::InvalidGguf(format!(
            "RBITNET_MODEL path not found: {}. \
             Use the full path to one .gguf file (not a folder). If you downloaded with `rbitnet models install`, open the folder and pick the .gguf name.",
            p.display()
        ))),
        Err(e) => Err(e.into()),
    }
}

impl Engine {
    /// Load from env: optional GGUF path, optional toy LM.
    pub fn from_env() -> Result<Self> {
        let model_path = model_path_from_env();
        if let Some(ref p) = model_path {
            validate_model_path_for_gguf(p)?;
        }
        if let Ok(tok) = std::env::var("RBITNET_TOKENIZER") {
            validate_no_parent_components(Path::new(&tok))?;
        }
        let gguf = if let Some(ref p) = model_path {
            Some(Arc::new(GgufArchive::mmap_path(p)?))
        } else {
            None
        };
        let toy = if toy_mode_enabled() {
            Some(ToyLlm::new(toy_seed()))
        } else {
            None
        };
        let stub = stub_mode_enabled();
        let backend_kind = BackendKind::from_env();
        let model_family = model_family_from_env(gguf.as_deref()).to_string();
        let scheduler = ContinuousBatchScheduler::from_env();
        let kernel_registry = KernelRegistry::bootstrap_default();
        let paged_kv = PagedKvCache::from_env();
        let executor = build_executor(
            backend_kind,
            &model_family,
            gguf.as_ref().map(Arc::clone),
            model_path.as_ref(),
            stub,
            toy.is_some(),
        )?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf,
                toy,
                stub,
                backend_kind,
                model_family,
                scheduler,
                _kernel_registry: kernel_registry,
                _paged_kv: paged_kv,
                executor,
            }),
        })
    }

    /// Load and parse a GGUF path.
    pub fn load_path(path: &Path) -> Result<Self> {
        let gguf = Arc::new(GgufArchive::mmap_path(path)?);
        let backend_kind = BackendKind::from_env();
        let model_family = model_family_from_env(Some(&gguf)).to_string();
        let model_path = Some(path.to_path_buf());
        let executor = build_executor(
            backend_kind,
            &model_family,
            Some(Arc::clone(&gguf)),
            model_path.as_ref(),
            false,
            false,
        )?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf: Some(gguf),
                toy: None,
                stub: false,
                backend_kind,
                model_family,
                scheduler: ContinuousBatchScheduler::from_env(),
                _kernel_registry: KernelRegistry::bootstrap_default(),
                _paged_kv: PagedKvCache::from_env(),
                executor,
            }),
        })
    }

    pub fn has_gguf(&self) -> bool {
        self.inner.gguf.is_some()
    }

    pub fn tensor_count(&self) -> Option<usize> {
        self.inner.gguf.as_ref().map(|g| g.tensor_count())
    }

    pub fn model_summary(&self) -> Option<String> {
        self.inner.gguf.as_ref().map(|g| g.summary_line())
    }

    pub fn tensor_names_preview(&self, max: usize) -> Option<Vec<String>> {
        self.inner.gguf.as_ref().map(|g| {
            g.tensors
                .iter()
                .take(max)
                .map(|t| t.name.clone())
                .collect()
        })
    }

    /// Whether chat can run without a missing-tokenizer configuration error.
    /// Stub and toy modes are always ready; GGUF mode requires a discoverable `tokenizer.json` or `tokenizer.model`.
    pub fn is_ready(&self) -> bool {
        if self.inner.stub || self.inner.toy.is_some() {
            return true;
        }
        self.inner
            .executor
            .as_ref()
            .map(|e| e.is_ready())
            .unwrap_or(false)
    }

    /// Label for `/v1/models`.
    pub fn openai_model_id(&self) -> Option<String> {
        if self.inner.stub {
            return None;
        }
        if self.inner.toy.is_some() {
            return Some("rbitnet-toy".into());
        }
        self.inner
            .executor
            .as_ref()
            .and_then(|e| e.openai_model_id(self.inner.gguf.as_deref()))
            .or_else(|| {
                self.inner
                    .gguf
                    .as_deref()
                    .map(|g| g.suggested_openai_model_id())
            })
    }

    pub fn backend_kind(&self) -> &'static str {
        self.inner.backend_kind.as_str()
    }

    pub fn model_family(&self) -> &str {
        &self.inner.model_family
    }

    pub fn backend_accelerated(&self) -> bool {
        self.inner
            .executor
            .as_ref()
            .map(|e| e.backend_accelerated())
            .unwrap_or(false)
    }

    /// Generate completion text from a user-facing prompt string.
    pub fn complete(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        if self.inner.stub {
            return Ok(stub_response(prompt, max_tokens));
        }
        if let Some(ref t) = self.inner.toy {
            return Ok(t.generate(prompt, max_tokens, temperature));
        }
        let Some(executor) = self.inner.executor.as_deref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        let req = InferenceRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
        };
        self.inner
            .scheduler
            .run(executor, &req)
            .map(|output| output.text)
    }

    pub fn complete_detailed(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<InferenceOutput> {
        if self.inner.stub {
            let text = stub_response(prompt, max_tokens);
            let completion_tokens = text.split_whitespace().count() as u32;
            return Ok(InferenceOutput {
                text,
                stats: InferenceStats {
                    ttft_ms: 1,
                    tpot_us: 1000,
                    completion_tokens,
                    speculative_attempted: false,
                },
            });
        }
        if let Some(ref t) = self.inner.toy {
            let text = t.generate(prompt, max_tokens, temperature);
            let completion_tokens = text.split_whitespace().count() as u32;
            return Ok(InferenceOutput {
                text,
                stats: InferenceStats {
                    ttft_ms: 1,
                    tpot_us: 1000,
                    completion_tokens,
                    speculative_attempted: false,
                },
            });
        }
        let Some(executor) = self.inner.executor.as_deref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        let req = InferenceRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
        };
        self.inner.scheduler.run(executor, &req)
    }
}

fn model_family_from_env(gguf: Option<&GgufArchive>) -> &'static str {
    match std::env::var("RBITNET_MODEL_FAMILY")
        .unwrap_or_else(|_| "auto".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "llama" => "llama",
        "bitnet" => "bitnet",
        _ => {
            if gguf.and_then(|g| g.architecture()).is_some_and(|a| a.eq_ignore_ascii_case("bitnet"))
            {
                "bitnet"
            } else {
                "llama"
            }
        }
    }
}

fn build_executor(
    backend_kind: BackendKind,
    model_family: &str,
    gguf: Option<Arc<GgufArchive>>,
    model_path: Option<&PathBuf>,
    stub: bool,
    toy: bool,
) -> Result<Option<Box<dyn ModelExecutor>>> {
    if stub || toy {
        return Ok(None);
    }
    let backend = make_backend(backend_kind);
    if model_family == "bitnet" {
        return Err(BitNetError::Inference(
            "BitNet GGUF inference is not yet implemented. \
             Use RBITNET_TOY=1 for a toy model, or provide a Llama-architecture GGUF. \
             Set RBITNET_MODEL_FAMILY=llama to override the auto-detected architecture."
                .into(),
        ));
    }
    let gguf = gguf.ok_or(BitNetError::ModelNotLoaded)?;
    let model_path = model_path.ok_or(BitNetError::ModelNotLoaded)?;
    let tok_path = resolve_tokenizer_path(model_path)?;
    Ok(Some(Box::new(LlamaExecutor::new(
        backend_kind,
        backend,
        gguf,
        tok_path,
    ))))
}

fn stub_response(prompt: &str, max_tokens: u32) -> String {
    let preview: String = prompt.chars().take(400).collect();
    format!(
        "[rbitnet stub] Integrated OK. Prompt (truncated): {preview}\n(max_tokens={max_tokens})"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToyLlm;

    fn stub_engine() -> Engine {
        Engine {
            inner: Arc::new(EngineInner {
                model_path: None,
                gguf: None,
                toy: None,
                stub: true,
                backend_kind: BackendKind::Cpu,
                model_family: "stub".into(),
                scheduler: ContinuousBatchScheduler::from_env(),
                _kernel_registry: KernelRegistry::bootstrap_default(),
                _paged_kv: PagedKvCache::from_env(),
                executor: None,
            }),
        }
    }

    fn toy_engine() -> Engine {
        Engine {
            inner: Arc::new(EngineInner {
                model_path: None,
                gguf: None,
                toy: Some(ToyLlm::new(42)),
                stub: false,
                backend_kind: BackendKind::Cpu,
                model_family: "toy".into(),
                scheduler: ContinuousBatchScheduler::from_env(),
                _kernel_registry: KernelRegistry::bootstrap_default(),
                _paged_kv: PagedKvCache::from_env(),
                executor: None,
            }),
        }
    }

    #[test]
    fn stub_always_works() {
        let e = stub_engine();
        assert!(e.complete("hi", 16, 0.7).unwrap().contains("stub"));
        assert_eq!(e.openai_model_id(), None);
    }

    #[test]
    fn toy_mode_complete_and_model_id() {
        let e = toy_engine();
        assert_eq!(e.openai_model_id(), Some("rbitnet-toy".into()));
        let result = e.complete("hello", 8, 0.7).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn stub_engine_is_ready() {
        let e = stub_engine();
        assert!(e.is_ready());
    }
}
