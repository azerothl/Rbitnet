//! DeepSeek-V2/V3/V4-class MoE GGUF (`general.architecture = deepseek2`).
//!
//! Layout reference: llama.cpp `deepseek2` loader / converters.
//!
//! When tensors match the **Llama-compatible** matrix naming (`blk.*`, `token_embd`, …),
//! inference uses the in-tree Llama runtime while reporting family **`deepseek2`**.
//! Otherwise loading fails immediately with a clear error (no placeholder executor).

mod config;
mod dispatch;

pub use config::DeepSeek2GgufMeta;
pub use dispatch::build_deepseek2_executor;
