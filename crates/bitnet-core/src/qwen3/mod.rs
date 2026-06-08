//! Dense Qwen3 GGUF runtime (text-only, CPU-first MVP).

mod config;
mod executor;
mod runtime;

pub use config::Qwen3Config;
pub use executor::Qwen3Executor;
