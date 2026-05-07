//! Tokenizer resolution next to GGUF (`RBITNET_TOKENIZER` or sibling directory).

use std::path::{Path, PathBuf};

use crate::error::{BitNetError, Result};
use crate::paths::validate_no_parent_components;

fn tokenizer_path_candidate(pb: &Path) -> bool {
    if !pb.is_file() {
        return false;
    }
    let Some(name) = pb.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == "tokenizer.json" || lower == "tokenizer.model"
}

/// Resolve tokenizer path beside the GGUF (or `RBITNET_TOKENIZER`), with `..` traversal rejected upstream.
pub fn resolve_tokenizer_path(model_path: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("RBITNET_TOKENIZER") {
        let pb = PathBuf::from(p);
        validate_no_parent_components(&pb)?;
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
    }
    if let Some(dir) = model_path.parent() {
        let pb = dir.join("tokenizer.json");
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
        let pb = dir.join("tokenizer.model");
        if tokenizer_path_candidate(&pb) {
            return Ok(pb);
        }
    }
    Err(BitNetError::TokenizerMissing)
}
