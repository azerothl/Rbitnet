//! DeepSeek-V2/V3/V4-class MoE GGUF (`general.architecture = deepseek2`).
//!
//! Layout reference: llama.cpp `deepseek2` loader / converters.
//!
//! Native split MLA with compressed KV, routed experts and optional shared experts.
//! Real-model validation targets GLM-4.7-Flash; alternate/fused projections are not certified.
//! Required shapes are checked before readiness on CPU/CUDA/hybrid.

mod config;
mod dispatch;

pub use config::DeepSeek2GgufMeta;
pub use dispatch::build_deepseek2_executor;
