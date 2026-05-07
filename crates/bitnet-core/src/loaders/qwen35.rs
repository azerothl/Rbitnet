//! Builder for [`crate::qwen35::Qwen35MoeExecutor`] (`general.architecture = qwen35moe`).

use std::path::Path;
use std::sync::Arc;

use crate::backend::{make_backend, BackendKind};
use crate::error::Result;
use crate::gguf::GgufArchive;
use crate::model::ModelExecutor;
use crate::qwen35::Qwen35MoeExecutor;

use super::tokenizer::resolve_tokenizer_path;

pub fn build_qwen35_moe_executor(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
) -> Result<Box<dyn ModelExecutor>> {
    let tokenizer_path = resolve_tokenizer_path(model_path)?;
    let backend = make_backend(backend_kind);
    Ok(Box::new(Qwen35MoeExecutor::new(
        backend_kind,
        gguf,
        backend,
        tokenizer_path,
    )))
}
