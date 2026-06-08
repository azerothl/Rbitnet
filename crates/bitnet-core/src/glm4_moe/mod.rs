//! Z.ai GLM MoE GGUF (`glm4moe` / `glm4_moe`).
//!
//! Llama-compatible tensor naming uses [`LlamaExecutor`] with reported family **`glm4moe`**.
//! Non-Llama MoE exports fail at load time with a clear error.

mod config;
mod dispatch;

pub use config::Glm4MoeGgufMeta;
pub use dispatch::build_glm4_moe_executor;
