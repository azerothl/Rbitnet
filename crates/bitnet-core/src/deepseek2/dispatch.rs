//! Dispatch `deepseek2` → Llama-compatible runtime only (no stub).

use std::path::Path;
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::llama::LlamaModel;
use crate::loaders::roadmap_unsupported;
use crate::loaders::tokenizer::{resolve_tokenizer_path, resolve_tokenizer_path_for_load};
use crate::model::{LlamaExecutor, ModelExecutor};

pub fn build_deepseek2_executor(
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
    if LlamaModel::from_gguf_arc(Arc::clone(&gguf)).is_ok() {
        return Ok(Box::new(LlamaExecutor::new_with_architecture_slug(
            backend_kind,
            backend,
            gguf,
            tokenizer_path,
            "deepseek2",
        )));
    }
    Err(roadmap_unsupported::roadmap_architecture_not_supported(
        "deepseek2",
    ))
}
