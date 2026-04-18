//! High-level inference façade.
//!
//! - `RBITNET_STUB=1` — HTTP integration text (Akasha `BitNetProvider`).
//! - `RBITNET_TOY=1` — tiny in-process F32 toy LM (no GGUF).
//! - `RBITNET_MODEL` — load GGUF (full BitNet forward still WIP).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
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
}

impl Engine {
    /// Load from env: optional GGUF path, optional toy LM.
    pub fn from_env() -> Result<Self> {
        let model_path = model_path_from_env();
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
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf,
                toy,
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

    /// Label for `/v1/models`.
    pub fn openai_model_id(&self) -> Option<String> {
        if stub_mode_enabled() {
            return None;
        }
        if toy_mode_enabled() && self.inner.toy.is_some() {
            return Some("rbitnet-toy".into());
        }
        self.inner
            .gguf
            .as_ref()
            .map(|g| g.suggested_openai_model_id())
    }

    /// Generate completion text from a user-facing prompt string.
    pub fn complete(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        if stub_mode_enabled() {
            return Ok(stub_response(prompt, max_tokens));
        }
        if let Some(ref t) = self.inner.toy {
            return Ok(t.generate(prompt, max_tokens, temperature));
        }
        let Some(_g) = self.inner.gguf.as_ref() else {
            return Err(BitNetError::ModelNotLoaded);
        };
        Err(BitNetError::NotImplemented(
            "full BitNet forward pass — GGUF loaded; implement quantized kernels + transformer graph",
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
        std::env::remove_var("RBITNET_TOY");
        let e = Engine {
            inner: Arc::new(EngineInner {
                model_path: None,
                gguf: None,
                toy: None,
            }),
        };
        assert!(e.complete("hi", 16, 0.7).unwrap().contains("stub"));
    }
}
