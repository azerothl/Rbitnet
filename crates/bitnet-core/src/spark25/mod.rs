//! Dense Spark-X2.5 GGUF runtime (`general.architecture = spark2_5`).
//!
//! CPU-first MVP: fused QKV, ISWA (1 full / 3 SWA), head-wise sigmoid attn gate, GELU FFN.
//! Tracking: <https://github.com/azerothl/Rbitnet/issues/142>.

mod config;
mod executor;
mod runtime;

pub use config::Spark25Config;
pub use executor::Spark25Executor;
pub use runtime::Spark25Runtime;
