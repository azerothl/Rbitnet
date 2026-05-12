//! **Rbitnet** — pure Rust BitNet inference core (work in progress).
//!
//! Modules:
//! - [`gguf`] — GGUF archive parsing + Llama metadata helpers
//! - [`kernels`] — reference ternary linear ops
//! - [`model`] — toy LM for end-to-end smoke tests
//! - [`inference`] — [`Engine`] façade
//! - [`llama`] — Llama-compatible GGUF inference (dequant + transformer)
//! - [`ggml`] — GGML type sizes and dequantization helpers

#![allow(
    clippy::get_first,
    clippy::identity_op,
    clippy::large_enum_variant,
    clippy::len_without_is_empty,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

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
pub mod model_manager;
pub mod paged_kv;
pub mod paths;
pub mod perf;
pub mod prefix_kv;
pub mod qwen3;
pub mod qwen35;
pub mod registry;
pub mod sampling;
pub mod scheduler;
pub mod scratch;
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
