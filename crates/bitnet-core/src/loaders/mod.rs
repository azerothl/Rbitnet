//! GGUF architecture dispatch (Atlas-style factory over [`crate::model::ModelExecutor`]).
//!
//! Adding a new transformer family: introduce a builder module, match on a normalized
//! `general.architecture` key in [`dispatch_gguf_executor`], or extend the Llama path if the
//! checkpoint remains Llama-shaped.

mod arch_key;
mod llama;
pub(crate) mod prompt_tokenizer;
mod qwen35;
mod registry;
pub(crate) mod roadmap_unsupported;
#[cfg(test)]
pub(crate) mod test_lock;
pub(crate) mod tokenizer;

pub use arch_key::{
    family_override_token, normalize_architecture_slug, resolve_architecture_key,
    resolve_architecture_key_for_load,
};
pub use registry::{dispatch_gguf_executor, dispatch_gguf_executor_for_load};
