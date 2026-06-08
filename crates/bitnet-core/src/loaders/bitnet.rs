//! Native BitNet GGUF loader.
//!
//! Microsoft BitNet b1.58 GGUF exports are Llama-shaped transformer graphs tagged with
//! `general.architecture = bitnet` and ternary GGML tensor types (`TQ1_0` / `TQ2_0`) for
//! projection weights. The executor below keeps the BitNet family identity while reusing the
//! in-tree transformer runtime and GGML mmap GEMV kernels.

use std::path::Path;
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::model::{BitNetNativeExecutor, ModelExecutor};

use super::tokenizer::{resolve_tokenizer_path, resolve_tokenizer_path_for_load};

pub fn build_bitnet_executor(
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
    Ok(Box::new(BitNetNativeExecutor::new(
        backend_kind,
        backend,
        gguf,
        tokenizer_path,
    )))
}
