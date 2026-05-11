//! Path validation for environment-controlled model locations.

use std::path::Path;

use crate::error::{BitNetError, Result};

/// Reject paths containing `..` so environment-controlled paths cannot escape the intended directory.
pub fn validate_no_parent_components(path: &Path) -> Result<()> {
    for c in path.components() {
        if matches!(c, std::path::Component::ParentDir) {
            return Err(BitNetError::InvalidGguf(
                "path must not contain '..' components".into(),
            ));
        }
    }
    Ok(())
}
