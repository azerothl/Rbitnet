//! **Rbitnet** — pure Rust BitNet inference core (work in progress).
//!
//! Modules:
//! - [`gguf`] — GGUF archive parsing + Llama metadata helpers
//! - [`kernels`] — reference ternary linear ops
//! - [`model`] — toy LM for end-to-end smoke tests
//! - [`inference`] — [`Engine`] façade
//! - [`llama`] — Llama-compatible GGUF inference (dequant + transformer)
//! - [`ggml`] — GGML type sizes and dequantization helpers

pub mod backend;
pub mod error;
pub mod ggml;
pub mod gguf;
pub mod inference;
pub mod llama;
pub mod kernels;
pub mod loaders;
pub mod model;
pub mod paged_kv;
pub mod paths;
pub mod registry;
pub mod scheduler;

pub use error::{BitNetError, Result};
pub use gguf::{GgufArchive, GgufFileInfo, GgufTensorInfo, GgufValue, LlamaHyperParams};
pub use inference::Engine;
pub use model::ToyLlm;
pub use paths::validate_no_parent_components;
