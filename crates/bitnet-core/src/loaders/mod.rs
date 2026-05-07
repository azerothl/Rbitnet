//! GGUF architecture dispatch (Atlas-style factory over [`crate::model::ModelExecutor`]).
//!
//! Adding a new transformer family: introduce a builder module, match on a normalized
//! `general.architecture` key in [`dispatch_gguf_executor`], or extend the Llama path if the
//! checkpoint remains Llama-shaped.

mod arch_key;
mod llama;
mod qwen35;
pub(crate) mod tokenizer;
mod registry;
#[cfg(test)]
pub(crate) mod test_lock;

pub use arch_key::{normalize_architecture_slug, resolve_architecture_key};
pub use registry::dispatch_gguf_executor;
