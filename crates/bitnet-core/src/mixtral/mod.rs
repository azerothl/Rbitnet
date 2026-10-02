//! Mixtral-style MoE GGUF (`general.architecture = mixtral`) — CPU-first spike (#25).

pub mod ci_fixture;
mod config;
mod executor;
mod moe;
mod runtime;

pub use config::MixtralConfig;
pub use executor::MixtralExecutor;
pub use runtime::MixtralRuntime;
