//! Llama-compatible transformer (GGUF weights dequantized to F32).

mod blas_runtime;
mod config;
pub mod cuda_graph;
pub mod fusion;
mod ggml_bridge;
pub mod kv_storage;
mod model;
mod runtime;
pub mod slim_attention;

pub use config::LlamaConfig;
pub use cuda_graph::{CudaDecodeGraph, DecodeGraphMode};
pub use kv_storage::{
    KvCache, KvPoolStats, KvQuantFormat, KvStorage, PagedKvPool, PagedSeqKv, SharedPhysKvStore,
};
pub use model::{llama_mmap_quant_supported, LlamaModel, LlamaWeightMode};
pub use runtime::LlamaRuntime;
pub use slim_attention::{
    attention_baseline, attention_tiled, slim_attention_enabled, tile_tokens_from_env,
    DEFAULT_TILE_TOKENS,
};
