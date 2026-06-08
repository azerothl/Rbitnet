//! GGUF loaders that delegate to [`crate::model::LlamaExecutor`].

use std::path::Path;
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::model::{LlamaExecutor, ModelExecutor};

use super::tokenizer::{resolve_tokenizer_path, resolve_tokenizer_path_for_load};

pub fn build_llama_executor(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
    tokenizer_isolated: bool,
    tokenizer_override: Option<&Path>,
) -> Result<Box<dyn ModelExecutor>> {
    let tokenizer_path = if tokenizer_isolated {
        resolve_tokenizer_path_for_load(model_path, tokenizer_override)?
    } else {
        debug_assert!(tokenizer_override.is_none());
        resolve_tokenizer_path(model_path)?
    };
    let backend = make_backend(backend_kind);
    Ok(Box::new(LlamaExecutor::new(
        backend_kind,
        backend,
        gguf,
        tokenizer_path,
    )))
}
