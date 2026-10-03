//! Dispatch `deepseek2` to the native split-MLA/routed-expert graph.

use std::path::Path;
use std::sync::Arc;

use crate::backend::BackendKind;
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::loaders::tokenizer::{resolve_tokenizer_path, resolve_tokenizer_path_for_load};
use crate::model::ModelExecutor;

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
    Ok(Box::new(crate::native::graph::NativeExecutor::load(
        gguf,
        &tokenizer_path,
        backend_kind,
        crate::native::graph::Family::Mla,
    )?))
}
