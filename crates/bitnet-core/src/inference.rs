//! High-level inference façade. Full BitNet forward pass is not implemented yet.
//!
//! When `RBITNET_STUB=1`, returns deterministic stub text for integration testing
//! (Akasha `BitNetProvider`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

/// Whether stub responses are enabled (no model required).
pub fn stub_mode_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_STUB").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Path to GGUF from `RBITNET_MODEL` if set.
pub fn model_path_from_env() -> Option<PathBuf> {
    std::env::var_os("RBITNET_MODEL").map(PathBuf::from)
}

/// Shared engine state: optional fully parsed GGUF archive (metadata + tensor table).
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    #[allow(dead_code)]
    model_path: Option<PathBuf>,
    gguf: Option<GgufArchive>,
}

impl Engine {
    /// Load model from `RBITNET_MODEL` or return empty engine (stub-only).
    pub fn from_env() -> Result<Self> {
        let model_path = model_path_from_env();
        let gguf = if let Some(ref p) = model_path {
            Some(GgufArchive::mmap_path(p)?)
        } else {
            None
        };
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf,
            }),
        })
    }

    /// Load and parse a GGUF path (validates full file structure).
    pub fn load_path(path: &Path) -> Result<Self> {
        let gguf = GgufArchive::mmap_path(path)?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path: Some(path.to_path_buf()),
                gguf: Some(gguf),
            }),
        })
    }

    pub fn has_gguf(&self) -> bool {
        self.inner.gguf.is_some()
    }

    pub fn tensor_count(&self) -> Option<usize> {
        self.inner.gguf.as_ref().map(|g| g.tensor_count())
    }

    /// Human-readable one-line description (architecture, tensor count).
    pub fn model_summary(&self) -> Option<String> {
        self.inner.gguf.as_ref().map(|g| g.summary_line())
    }

    /// First N tensor names for debugging.
    pub fn tensor_names_preview(&self, max: usize) -> Option<Vec<String>> {
        self.inner.gguf.as_ref().map(|g| {
            g.tensors
                .iter()
                .take(max)
                .map(|t| t.name.clone())
                .collect()
        })
    }

    /// Label for `/v1/models` when a GGUF is loaded.
    pub fn openai_model_id(&self) -> Option<String> {
        self.inner
            .gguf
            .as_ref()
            .map(|g| g.suggested_openai_model_id())
    }

    /// Generate completion text from a single user-facing prompt string.
    pub fn complete(&self, prompt: &str, max_tokens: u32, _temperature: f32) -> Result<String> {
        if stub_mode_enabled() {
            return Ok(stub_response(prompt, max_tokens));
        }
        let Some(_g) = self.inner.gguf.as_ref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        Err(BitNetError::NotImplemented(
            "full BitNet forward pass — GGUF metadata and tensors are loaded; wire kernels + graph next",
        ))
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

    #[test]
    fn stub_always_works() {
        std::env::set_var("RBITNET_STUB", "1");
        let e = Engine {
            inner: Arc::new(EngineInner {
                model_path: None,
                gguf: None,
            }),
        };
        assert!(e.complete("hi", 16, 0.7).unwrap().contains("stub"));
    }
}
