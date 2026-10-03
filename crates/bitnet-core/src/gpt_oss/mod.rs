//! OpenAI **gpt-oss** GGUF (`general.architecture = gpt-oss`, alias `gptoss`).
//!
//! Native biased attention with sinks, alternating windows, YaRN and MXFP4 MoE.
//! CPU/CUDA/hybrid execution validates the real topology before readiness.

mod config;
mod dispatch;

pub use config::GptOssGgufMeta;
pub use dispatch::build_gptoss_executor;
