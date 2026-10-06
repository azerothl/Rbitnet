//! GGUF loaders that delegate to [`crate::model::LlamaExecutor`].

use std::path::Path;
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::mmproj::{resolve_mmproj_path, MmprojEncoder, ResolveMmprojOpts};
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
    let mut exec = LlamaExecutor::new(backend_kind, backend, gguf, tokenizer_path)?;
    if let Some(mm_path) = resolve_mmproj_path(ResolveMmprojOpts {
        explicit: None,
        text_gguf: Some(model_path),
    }) {
        match MmprojEncoder::load(&mm_path) {
            Ok(enc) => {
                tracing::info!(
                    path = %mm_path.display(),
                    n_patches = enc.n_patches(),
                    proj_out = enc.proj_out_dim(),
                    "loaded mmproj vision encoder"
                );
                exec = exec.with_mmproj(Arc::new(enc))?;
            }
            Err(e) => {
                tracing::warn!(
                    path = %mm_path.display(),
                    error = %e,
                    "mmproj path resolved but encoder load failed; vision remains disabled"
                );
            }
        }
    }
    Ok(Box::new(exec))
}
