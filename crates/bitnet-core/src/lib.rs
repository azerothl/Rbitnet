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
pub mod cancel;
pub mod deepseek2;
pub mod error;
pub mod ggml;
pub mod gguf;
pub mod glm4_moe;
pub mod gpt_oss;
pub mod inference;
pub mod kernels;
pub mod llama;
pub mod loaders;
pub mod memory_budget;
pub mod model;
pub mod paged_kv;
pub mod paths;
pub mod prefix_kv;
pub mod qwen35;
pub mod registry;
pub mod scheduler;
pub mod timings;

pub use backend::CudaRuntime;
pub use cancel::{clear_inference_cancel, inference_cancelled, request_inference_cancel};
pub use error::{BitNetError, Result};
pub use gguf::{GgufArchive, GgufFileInfo, GgufTensorInfo, GgufValue, LlamaHyperParams};
pub use inference::Engine;
pub use memory_budget::{check_load_memory_budget, gguf_tensor_payload_bytes};
pub use model::ToyLlm;
pub use paths::validate_no_parent_components;
pub use timings::PhaseTimings;
