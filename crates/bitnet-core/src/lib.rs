//! **Rbitnet** — pure Rust BitNet inference core (work in progress).
//!
//! Modules:
//! - [`gguf`] — GGUF archive parsing + Llama metadata helpers
//! - [`kernels`] — reference ternary linear ops
//! - [`model`] — toy LM for end-to-end smoke tests
//! - [`inference`] — [`Engine`] façade

pub mod error;
pub mod gguf;
pub mod inference;
pub mod kernels;
pub mod model;

pub use error::{BitNetError, Result};
pub use gguf::{GgufArchive, GgufFileInfo, GgufTensorInfo, GgufValue, LlamaHyperParams};
pub use inference::Engine;
pub use model::ToyLlm;
