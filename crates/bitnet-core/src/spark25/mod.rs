//! Dense Spark-X2.5 GGUF runtime (`general.architecture = spark2_5`).
//!
//! CPU-first MVP: fused QKV, ISWA (1 full / 3 SWA), head-wise sigmoid attn gate, GELU FFN.
//! CUDA/hybrid: device-resident Q4_K (etc.) matvec for linear layers; attention/RoPE on CPU.
//! Tracking: <https://github.com/azerothl/Rbitnet/issues/142>, GPU path #148.

mod config;
mod executor;
mod runtime;
mod weights;

pub use config::Spark25Config;
pub use executor::Spark25Executor;
pub use runtime::Spark25Runtime;
