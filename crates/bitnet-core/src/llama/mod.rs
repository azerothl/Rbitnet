//! Llama-compatible transformer (GGUF weights dequantized to F32).

mod blas_runtime;
mod config;
pub mod cuda_graph;
pub mod fusion;
mod ggml_bridge;
pub mod kv_storage;
mod model;
#[cfg(feature = "profile-llama")]
pub mod profile;
mod resident;
mod runtime;
pub mod slim_attention;
mod speculative;

pub use config::LlamaConfig;
pub use cuda_graph::{CudaDecodeGraph, DecodeGraphMode};
pub use kv_storage::{
    KvCache, KvPoolStats, KvQuantFormat, KvStorage, PagedKvPool, PagedSeqKv, SharedPhysKvStore,
};
pub use model::{llama_mmap_quant_supported, LlamaModel, LlamaWeightMode, MatrixWeights};
pub(crate) use runtime::llama_encode_add_special_tokens;
pub use runtime::LlamaRuntime;
pub use slim_attention::{
    attention_baseline, attention_tiled, slim_attention_enabled, tile_tokens_from_env,
    DEFAULT_TILE_TOKENS,
};

pub(crate) use resident::{
    scheduler_fused_options, BatchController, BatchOptions, SchedulerFusedLlama,
    SchedulerFusedOptions,
};
