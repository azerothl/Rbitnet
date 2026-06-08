//! OpenAI **gpt-oss** GGUF (`general.architecture = gptoss`).
//!
//! MXFP4 weights are supported at the GGML layer via [`crate::ggml::tensor_to_f32`].
//! Llama-shaped tensors run on [`LlamaExecutor`] with reported family **`gptoss`**.
//! Other layouts fail at load time with a clear error.

mod config;
mod dispatch;

pub use config::GptOssGgufMeta;
pub use dispatch::build_gptoss_executor;
