//! High-level inference façade.
//!
//! - `RBITNET_STUB=1` — HTTP integration text (Akasha `BitNetProvider`).
//! - `RBITNET_TOY=1` — tiny in-process F32 toy LM (no GGUF).
//! - `RBITNET_MODEL` — load GGUF; full Llama-compatible forward + `tokenizer.json` / `tokenizer.model` / `RBITNET_TOKENIZER`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::llama::LlamaRuntime;
use crate::model::ToyLlm;

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
    gguf: Option<GgufArchive>,
    toy: Option<ToyLlm>,
    stub: bool,
    llama: Mutex<Option<LlamaRuntime>>,
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
            Some(GgufArchive::mmap_path(p)?)
        } else {
            None
        };
        let toy = if toy_mode_enabled() {
            Some(ToyLlm::new(toy_seed()))
        } else {
            None
        };
        let stub = stub_mode_enabled();
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf,
                toy,
                stub,
                llama: Mutex::new(None),
            }),
        })
    }

    /// Load and parse a GGUF path.
    pub fn load_path(path: &Path) -> Result<Self> {
        let gguf = GgufArchive::mmap_path(path)?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path: Some(path.to_path_buf()),
                gguf: Some(gguf),
                toy: None,
                stub: false,
                llama: Mutex::new(None),
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
        let Some(ref model_path) = self.inner.model_path else {
            return false;
        };
        self.inner.gguf.is_some() && resolve_tokenizer_path(model_path).is_ok()
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
            .gguf
            .as_ref()
            .map(|g| g.suggested_openai_model_id())
    }

    /// Generate completion text from a user-facing prompt string.
    pub fn complete(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        if self.inner.stub {
            return Ok(stub_response(prompt, max_tokens));
        }
        if let Some(ref t) = self.inner.toy {
            return Ok(t.generate(prompt, max_tokens, temperature));
        }
        let Some(gguf) = self.inner.gguf.as_ref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        let model_path = self
            .inner
            .model_path
            .as_ref()
            .ok_or(BitNetError::ModelNotLoaded)?;
        let mut slot = self.inner.llama.lock().map_err(|e| {
            BitNetError::Inference(format!("engine lock poisoned: {e}"))
        })?;
        if slot.is_none() {
            let tok_path = resolve_tokenizer_path(model_path)?;
            *slot = Some(LlamaRuntime::load(gguf, &tok_path)?);
        }
        slot.as_mut()
            .unwrap()
            .generate(prompt, max_tokens, temperature)
    }
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
                llama: Mutex::new(None),
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
                llama: Mutex::new(None),
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
