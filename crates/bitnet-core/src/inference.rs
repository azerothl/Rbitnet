//! High-level inference façade. Full BitNet forward pass is not implemented yet.
//!
//! When `RBITNET_STUB=1`, returns deterministic stub text for integration testing
//! (Akasha `BitNetProvider`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{BitNetError, Result};
use crate::gguf::{mmap_gguf_header, GgufFileInfo};

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

/// Shared engine state: optional validated GGUF header (real inference later).
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    #[allow(dead_code)]
    model_path: Option<PathBuf>,
    gguf_info: Option<GgufFileInfo>,
}

impl Engine {
    /// Load model from `RBITNET_MODEL` or return empty engine (stub-only).
    pub fn from_env() -> Result<Self> {
        let model_path = model_path_from_env();
        let gguf_info = if let Some(ref p) = model_path {
            let (_mmap, info) = mmap_gguf_header(p)?;
            Some(info)
        } else {
            None
        };
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path,
                gguf_info,
            }),
        })
    }

    /// Validate without loading: check GGUF header if path is set.
    pub fn load_path(path: &Path) -> Result<Self> {
        let (_mmap, info) = mmap_gguf_header(path)?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                model_path: Some(path.to_path_buf()),
                gguf_info: Some(info),
            }),
        })
    }

    pub fn has_gguf(&self) -> bool {
        self.inner.gguf_info.is_some()
    }

    pub fn tensor_count(&self) -> Option<u64> {
        self.inner.gguf_info.as_ref().map(|i| i.tensor_count)
    }

    /// Generate completion text from a single user-facing prompt string.
    pub fn complete(&self, prompt: &str, max_tokens: u32, _temperature: f32) -> Result<String> {
        if stub_mode_enabled() {
            return Ok(stub_response(prompt, max_tokens));
        }
        if self.inner.gguf_info.is_none() {
            return Err(BitNetError::ModelNotLoaded);
        }
        Err(BitNetError::NotImplemented(
            "full BitNet forward pass — use RBITNET_STUB=1 for testing or wire kernels + graph",
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
                gguf_info: None,
            }),
        };
        assert!(e.complete("hi", 16, 0.7).unwrap().contains("stub"));
    }
}
