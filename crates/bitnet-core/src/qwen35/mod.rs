//! Native Qwen3.5 dense / MoE (GDN recurrent attention / gated full attention).
//!
//! Text-only CPU/CUDA/hybrid paths keep quantized weights mmap/device resident.
//! Dense recurrent blocks can retain convolution/GDN/FFN activations on CUDA;
//! other layouts use CPU F32. Full attention can keep KV resident on CUDA.

pub mod cuda_ctx;
pub mod executor;

mod attention;
mod config;
pub(crate) mod gdn;
mod moe;
pub(crate) mod qmatvec;
mod recurrent;
mod runtime;

pub use executor::Qwen35MoeExecutor;
