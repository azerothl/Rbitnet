//! Resolve a sidecar mmproj GGUF path from env or sibling files.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct ResolveMmprojOpts<'a> {
    /// Explicit path (takes precedence). Typically `RBITNET_MMPROJ`.
    pub explicit: Option<&'a Path>,
    /// Text GGUF path used to search for sibling `mmproj-*.gguf` files.
    pub text_gguf: Option<&'a Path>,
}

/// Resolve an mmproj path.
///
/// Order:
/// 1. `explicit` if set and non-empty
/// 2. `RBITNET_MMPROJ` env if set and non-empty
/// 3. First `mmproj*.gguf` sibling next to `text_gguf` (sorted by name)
pub fn resolve_mmproj_path(opts: ResolveMmprojOpts<'_>) -> Option<PathBuf> {
    if let Some(p) = opts.explicit {
        if !p.as_os_str().is_empty() {
            return Some(p.to_path_buf());
        }
    }
    if let Ok(raw) = std::env::var("RBITNET_MMPROJ") {
        let t = raw.trim();
        if !t.is_empty() {
            return Some(PathBuf::from(t));
        }
    }
    let text = opts.text_gguf?;
    let dir = text.parent()?;
    let mut matches: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        let lower = n.to_ascii_lowercase();
                        lower.starts_with("mmproj") && lower.ends_with(".gguf")
                    })
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    matches.sort();
    matches.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::test_lock::env_test_lock;
    use std::fs;

    #[test]
    fn explicit_path_wins() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_MMPROJ");
        let p = Path::new("/tmp/explicit-mmproj.gguf");
        let got = resolve_mmproj_path(ResolveMmprojOpts {
            explicit: Some(p),
            text_gguf: None,
        });
        assert_eq!(got.as_deref(), Some(p));
    }

    #[test]
    fn env_path_used_when_no_explicit() {
        let _g = env_test_lock();
        std::env::set_var("RBITNET_MMPROJ", "/tmp/env-mmproj.gguf");
        let got = resolve_mmproj_path(ResolveMmprojOpts::default());
        std::env::remove_var("RBITNET_MMPROJ");
        assert_eq!(got.as_deref(), Some(Path::new("/tmp/env-mmproj.gguf")));
    }

    #[test]
    fn sibling_mmproj_discovered() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_MMPROJ");
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("model.gguf");
        fs::write(&text, b"x").unwrap();
        fs::write(dir.path().join("mmproj-model-f16.gguf"), b"y").unwrap();
        let got = resolve_mmproj_path(ResolveMmprojOpts {
            explicit: None,
            text_gguf: Some(&text),
        });
        assert_eq!(
            got.as_deref(),
            Some(dir.path().join("mmproj-model-f16.gguf").as_path())
        );
    }
}
