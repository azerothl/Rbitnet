//! High-level inference façade.
//!
//! - `RBITNET_STUB=1` — HTTP integration text (Akasha `BitNetProvider`).
//! - `RBITNET_TOY=1` — tiny in-process F32 toy LM (no GGUF).
//! - `RBITNET_MODEL` — load GGUF; full Llama-compatible forward + `tokenizer.json` / `tokenizer.model` / `RBITNET_TOKENIZER`.
//! - `RBITNET_ARCHITECTURE` — optional override for `general.architecture` dispatch (see `loaders/`).
//! - `RBITNET_MODEL_FAMILY` — `auto`, `bitnet`, or Llama-lineage overrides (`llama`, `mistral`, `deepseek`, …; see [`crate::loaders::family_override_token`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::backend::BackendKind;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::loaders::{
    dispatch_gguf_executor, dispatch_gguf_executor_for_load, family_override_token,
    normalize_architecture_slug, resolve_architecture_key, resolve_architecture_key_for_load,
};
use crate::memory_budget::check_load_memory_budget;
use crate::model::{ModelExecutor, ToyLlm};
use crate::paged_kv::PagedKvCache;
use crate::paths::validate_no_parent_components;
use crate::registry::KernelRegistry;
use crate::scheduler::{
    ContinuousBatchScheduler, InferenceBatch, InferenceOutput, InferenceRequest, InferenceStats,
    ScheduledRequest,
};

/// Metadata surfaced by the HTTP `/v1/models` endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineModelMetadata {
    pub summary: Option<String>,
    pub model_path: Option<String>,
    pub architecture: String,
    pub context_length: Option<u64>,
    pub quantization: Option<String>,
    pub backend: String,
    pub backend_accelerated: bool,
    pub ready: bool,
    pub tensor_count: Option<usize>,
}

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
    prefix_cache_enabled: bool,
    prefix_cache_max_entries: usize,
    prefix_cache: Mutex<Vec<(String, InferenceOutput)>>,
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
            let g = Arc::new(GgufArchive::mmap_path(p)?);
            check_load_memory_budget(g.as_ref())?;
            Some(g)
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
        let model_family = gguf
            .as_deref()
            .map(resolve_architecture_key)
            .unwrap_or_else(model_family_when_no_gguf);
        let scheduler = ContinuousBatchScheduler::from_env();
        let prefix_cache_enabled = matches!(
            std::env::var("RBITNET_PREFIX_CACHE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let prefix_cache_max_entries = std::env::var("RBITNET_PREFIX_CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64);
        let kernel_registry = KernelRegistry::bootstrap_default();
        let paged_kv = PagedKvCache::from_env();
        let executor = build_executor(
            backend_kind,
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
                prefix_cache_enabled,
                prefix_cache_max_entries,
                prefix_cache: Mutex::new(Vec::new()),
                _kernel_registry: kernel_registry,
                _paged_kv: paged_kv,
                executor,
            }),
        })
    }

    /// Load and parse a GGUF path.
    pub fn load_path(path: &Path) -> Result<Self> {
        let gguf = Arc::new(GgufArchive::mmap_path(path)?);
        check_load_memory_budget(gguf.as_ref())?;
        let backend_kind = BackendKind::from_env();
        let model_family = resolve_architecture_key(&gguf);
        let model_path = Some(path.to_path_buf());
        let executor = build_executor(
            backend_kind,
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
                prefix_cache_enabled: false,
                prefix_cache_max_entries: 0,
                prefix_cache: Mutex::new(Vec::new()),
                _kernel_registry: KernelRegistry::bootstrap_default(),
                _paged_kv: PagedKvCache::from_env(),
                executor,
            }),
        })
    }

    /// Load a single GGUF file with optional tokenizer path and architecture slug overrides.
    ///
    /// Unlike [`Engine::load_path`], this does not read `RBITNET_TOKENIZER` or `RBITNET_ARCHITECTURE`
    /// from the environment, so multiple checkpoints can be loaded safely in one process
    /// (e.g. HTTP `model` selection from a registry).
    pub fn load_path_with_overrides(
        path: &Path,
        tokenizer_override: Option<&Path>,
        architecture_override: Option<&str>,
    ) -> Result<Self> {
        validate_model_path_for_gguf(path)?;
        if let Some(t) = tokenizer_override {
            crate::paths::validate_no_parent_components(t)?;
        }
        let gguf = Arc::new(GgufArchive::mmap_path(path)?);
        check_load_memory_budget(gguf.as_ref())?;
        let backend_kind = BackendKind::from_env();
        let model_family = resolve_architecture_key_for_load(&gguf, architecture_override);
        let model_path = Some(path.to_path_buf());
        let executor = build_executor_for_load(
            backend_kind,
            Arc::clone(&gguf),
            path,
            tokenizer_override,
            architecture_override,
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
                prefix_cache_enabled: false,
                prefix_cache_max_entries: 0,
                prefix_cache: Mutex::new(Vec::new()),
                _kernel_registry: KernelRegistry::bootstrap_default(),
                _paged_kv: PagedKvCache::from_env(),
                executor: Some(executor),
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

    pub fn model_metadata(&self) -> EngineModelMetadata {
        let gguf = self.inner.gguf.as_deref();
        EngineModelMetadata {
            summary: gguf.map(|g| g.summary_line()),
            model_path: self
                .inner
                .model_path
                .as_ref()
                .map(|p| redact_path_for_display(&p.display().to_string())),
            architecture: gguf
                .and_then(|g| g.architecture().map(ToString::to_string))
                .unwrap_or_else(|| self.inner.model_family.clone()),
            context_length: gguf.and_then(|g| g.context_length()),
            quantization: gguf.and_then(|g| g.quantization_summary()),
            backend: self.inner.backend_kind.as_str().to_string(),
            backend_accelerated: self.backend_accelerated(),
            ready: self.is_ready(),
            tensor_count: gguf.map(|g| g.tensor_count()),
        }
    }

    pub fn tensor_names_preview(&self, max: usize) -> Option<Vec<String>> {
        self.inner
            .gguf
            .as_ref()
            .map(|g| g.tensors.iter().take(max).map(|t| t.name.clone()).collect())
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

    pub fn continuous_batching_enabled(&self) -> bool {
        self.inner.scheduler.enabled
    }

    /// tokenizer-encoded prompt length (for HTTP guards). Stub/toy use a rough character heuristic.
    pub fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        if self.inner.stub || self.inner.toy.is_some() {
            return Ok(u32::try_from(prompt.len())
                .unwrap_or(u32::MAX)
                .saturating_div(4)
                .max(1));
        }
        let Some(ex) = self.inner.executor.as_deref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        ex.count_prompt_tokens(prompt)
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
                    encode_ms: 0,
                    prefill_ms: 1,
                    decode_ms: 0,
                    itl_us: 1000,
                    tpot_us: 1000,
                    prompt_tokens: 0,
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
                    encode_ms: 0,
                    prefill_ms: 1,
                    decode_ms: 0,
                    itl_us: 1000,
                    tpot_us: 1000,
                    prompt_tokens: 0,
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
        if self.inner.prefix_cache_enabled {
            let key = format!("{}|{}|{:.3}", prompt, max_tokens, temperature);
            if let Ok(cache) = self.inner.prefix_cache.lock() {
                if let Some((_, hit)) = cache.iter().find(|(k, _)| k == &key) {
                    return Ok(hit.clone());
                }
            }
            let output = self.inner.scheduler.run(executor, &req)?;
            if let Ok(mut cache) = self.inner.prefix_cache.lock() {
                cache.push((key, output.clone()));
                if cache.len() > self.inner.prefix_cache_max_entries {
                    let overflow = cache.len() - self.inner.prefix_cache_max_entries;
                    cache.drain(0..overflow);
                }
            }
            return Ok(output);
        }
        self.inner.scheduler.run(executor, &req)
    }

    pub fn complete_batch_detailed(
        &self,
        requests: &[InferenceRequest],
    ) -> Result<Vec<InferenceOutput>> {
        if self.inner.stub || self.inner.toy.is_some() {
            let mut out = Vec::with_capacity(requests.len());
            for req in requests {
                out.push(self.complete_detailed(&req.prompt, req.max_tokens, req.temperature)?);
            }
            return Ok(out);
        }
        let Some(executor) = self.inner.executor.as_deref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        let batch = InferenceBatch {
            requests: requests
                .iter()
                .enumerate()
                .map(|(i, req)| ScheduledRequest {
                    id: i as u64,
                    request: req.clone(),
                })
                .collect(),
        };
        let mut rows = self.inner.scheduler.run_batch(executor, &batch)?;
        rows.sort_by_key(|(id, _)| *id);
        Ok(rows.into_iter().map(|(_, output)| output).collect())
    }
}

/// Stub engine: no GGUF, returns synthetic completions (same as `RBITNET_STUB=1`).
#[must_use]
pub fn stub_engine() -> Engine {
    Engine {
        inner: Arc::new(EngineInner {
            model_path: None,
            gguf: None,
            toy: None,
            stub: true,
            backend_kind: BackendKind::Cpu,
            model_family: "stub".into(),
            scheduler: ContinuousBatchScheduler::from_env(),
            prefix_cache_enabled: false,
            prefix_cache_max_entries: 0,
            prefix_cache: Mutex::new(Vec::new()),
            _kernel_registry: KernelRegistry::bootstrap_default(),
            _paged_kv: PagedKvCache::from_env(),
            executor: None,
        }),
    }
}

/// Label for [`EngineInner::model_family`] when no GGUF is loaded (stub / toy / no `RBITNET_MODEL`).
fn model_family_when_no_gguf() -> String {
    if let Ok(v) = std::env::var("RBITNET_ARCHITECTURE") {
        let t = normalize_architecture_slug(&v);
        if !t.is_empty() {
            return t;
        }
    }
    match std::env::var("RBITNET_MODEL_FAMILY") {
        Ok(f) => {
            let fam = normalize_architecture_slug(&f);
            if fam == "bitnet" {
                return "bitnet".into();
            }
            family_override_token(&fam).unwrap_or("llama").to_string()
        }
        Err(_) => "llama".into(),
    }
}

fn redact_path_for_display(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    if lower.contains("hf_")
        || lower.contains("token=")
        || lower.contains("apikey")
        || lower.contains("api_key")
    {
        return "<redacted>".into();
    }
    path.to_string()
}

fn build_executor(
    backend_kind: BackendKind,
    gguf: Option<Arc<GgufArchive>>,
    model_path: Option<&PathBuf>,
    stub: bool,
    toy: bool,
) -> Result<Option<Box<dyn ModelExecutor>>> {
    if stub || toy {
        return Ok(None);
    }
    let gguf = gguf.ok_or(BitNetError::ModelNotLoaded)?;
    let model_path = model_path.ok_or(BitNetError::ModelNotLoaded)?;
    dispatch_gguf_executor(backend_kind, gguf, model_path.as_path()).map(Some)
}

fn build_executor_for_load(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
    tokenizer_override: Option<&Path>,
    architecture_override: Option<&str>,
) -> Result<Box<dyn ModelExecutor>> {
    dispatch_gguf_executor_for_load(
        backend_kind,
        gguf,
        model_path,
        tokenizer_override,
        architecture_override,
    )
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
        super::stub_engine()
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
                prefix_cache_enabled: false,
                prefix_cache_max_entries: 0,
                prefix_cache: Mutex::new(Vec::new()),
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
