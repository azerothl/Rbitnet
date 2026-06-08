//! Native Qwen3.x MoE (hybrid linear attention / full attention + MoE).
//!
//! Text-only CUDA-first path via [`crate::backend::CudaRuntime`] for heavyweight `gemv`; core math
//! is reference CPU F32 (llama.cpp `ggml` semantics for GDN convolution + gated delta recurrence).

pub mod cuda_ctx;
pub mod executor;

mod attention;
mod config;
pub(crate) mod gdn;
mod moe;
mod qmatvec;
mod recurrent;
mod runtime;

pub use executor::Qwen35MoeExecutor;
