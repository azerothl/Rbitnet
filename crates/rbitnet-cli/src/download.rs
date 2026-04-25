//! Download model files via `hf-hub` (HF cache) into a user directory.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use hf_hub::api::sync::ApiBuilder;
use serde_json::Value;

use crate::hf_search;

/// Resolve filenames to download: explicit list, or all `.gguf` + `tokenizer.json` / `tokenizer.model` from the repo tree API.
pub fn resolve_download_files(
    repo_id: &str,
    explicit: &[String],
    token: Option<&str>,
) -> Result<Vec<String>, String> {
    if !explicit.is_empty() {
        return Ok(explicit.to_vec());
    }
    list_auto_files(repo_id, token)
}

fn list_auto_files(repo_id: &str, token: Option<&str>) -> Result<Vec<String>, String> {
    let url = hf_search::hf_model_detail_url(repo_id);
    let agent = crate::hub_http::agent()?;
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let v: Value = req
        .call()
        .map_err(|e| format!("HF model info {repo_id}: {e}"))?
        .into_json()
        .map_err(|e| format!("HF model JSON: {e}"))?;

    let siblings: Vec<hf_search::Sibling> =
        if let Ok(si) = serde_json::from_value::<hf_search::ModelInfo>(v.clone()) {
            si.siblings
        } else if let Some(arr) = v.get("siblings").and_then(|x| x.as_array()) {
            arr.iter()
                .filter_map(|item| {
                    item.get("rfilename").and_then(|x| x.as_str()).map(|s| {
                        hf_search::Sibling {
                            rfilename: s.to_string(),
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

    let names = select_auto_files(&siblings);
    if names.is_empty() {
        return Err(
            "no .gguf (and no tokenizer.json/tokenizer.model) found in repo; specify --file"
                .into(),
        );
    }
    Ok(names)
}

/// Select the subset of sibling filenames to auto-download: all `.gguf` files plus tokenizer files.
/// Extracted into a separate function so file-selection logic can be unit-tested without network calls.
fn select_auto_files(siblings: &[hf_search::Sibling]) -> Vec<String> {
    let mut names: Vec<String> = siblings
        .iter()
        .filter_map(|s| {
            let lower = s.rfilename.to_ascii_lowercase();
            let take = lower.ends_with(".gguf")
                || lower == "tokenizer.json"
                || lower.ends_with("/tokenizer.json")
                || lower == "tokenizer.model"
                || lower.ends_with("/tokenizer.model");
            if take { Some(s.rfilename.clone()) } else { None }
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Compute the local destination path for a remote `file` relative to `dest_dir`,
/// preserving subdirectory structure. Returns an error if the path is absolute or
/// contains path-traversal components (`..`).
fn dest_path_for(dest_dir: &Path, file: &str) -> Result<PathBuf, String> {
    let rel = Path::new(file);
    for component in rel.components() {
        match component {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            _ => {
                return Err(format!(
                    "unsafe path component in '{file}': only relative paths without '..' are allowed"
                ))
            }
        }
    }
    Ok(dest_dir.join(rel))
}

/// Download each file into `dest_dir` (created if missing). Uses HF cache then copies.
/// Remote subpaths (e.g. `subdir/tokenizer.json`) are preserved under `dest_dir`, with
/// parent directories created as needed. Paths containing `..` or absolute components
/// are rejected to prevent path traversal.
pub fn download_files(
    repo_id: &str,
    files: &[String],
    dest_dir: &Path,
    token: Option<&str>,
) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(dest_dir).map_err(|e| format!("create {}: {e}", dest_dir.display()))?;

    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t.into()));
    }
    let api = builder.build().map_err(|e| format!("hf-hub Api: {e}"))?;
    let repo = api.model(repo_id.to_string());

    let mut out = Vec::new();
    for file in files {
        let dest = dest_path_for(dest_dir, file)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }

        let mut last_err = String::new();
        let mut copied = false;

        // First try hf-hub cache path (fast path).
        for attempt in 1..=3 {
            match repo.get(file) {
                Ok(cached) => {
                    fs::copy(&cached, &dest).map_err(|e| {
                        format!("copy {} -> {}: {e}", cached.display(), dest.display())
                    })?;
                    copied = true;
                    break;
                }
                Err(e) => {
                    last_err = format!("hf-hub attempt {attempt}/3 failed: {e}");
                    thread::sleep(Duration::from_millis(250 * attempt as u64));
                }
            }
        }

        // Fallback: direct HTTPS download via native-tls agent.
        if !copied {
            for attempt in 1..=2 {
                match download_file_via_http(repo_id, file, &dest, token) {
                    Ok(()) => {
                        copied = true;
                        break;
                    }
                    Err(e) => {
                        last_err = format!("http fallback attempt {attempt}/2 failed: {e}");
                        thread::sleep(Duration::from_millis(300 * attempt as u64));
                    }
                }
            }
        }

        if !copied {
            return Err(format!("download {repo_id} {file}: {last_err}"));
        }
        out.push(dest);
    }
    Ok(out)
}

fn download_file_via_http(
    repo_id: &str,
    file: &str,
    dest: &Path,
    token: Option<&str>,
) -> Result<(), String> {
    let url = hf_search::hf_resolve_main_url(repo_id, file);
    let agent = crate::hub_http::agent()?;
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req
        .call()
        .map_err(|e| format!("request error: {url}: {e}"))?;

    let mut reader = resp.into_reader();

    let dest_dir = dest.parent().unwrap_or(Path::new("."));
    let dest_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");

    let mut temp_path = None;
    let mut f = None;
    for attempt in 0..1000 {
        let candidate =
            dest_dir.join(format!(".{dest_name}.part.{}.{}", std::process::id(), attempt));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temp_path = Some(candidate);
                f = Some(file);
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!("create temp file for {}: {e}", dest.display()));
            }
        }
    }

    let temp_path =
        temp_path.ok_or_else(|| format!("create temp file for {}: exhausted retries", dest.display()))?;
    let mut f = f.expect("temporary file handle must exist when temp_path is set");

    if let Err(e) = io::copy(&mut reader, &mut f) {
        let _ = fs::remove_file(&temp_path);
        return Err(format!("write {}: {e}", dest.display()));
    }

    drop(f);

    if let Err(e) = fs::rename(&temp_path, dest) {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "rename {} -> {}: {e}",
            temp_path.display(),
            dest.display()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sibling(name: &str) -> hf_search::Sibling {
        hf_search::Sibling {
            rfilename: name.to_string(),
        }
    }

    // --- select_auto_files ---

    #[test]
    fn auto_files_selects_gguf_and_tokenizer() {
        let siblings = vec![
            sibling("README.md"),
            sibling("model.Q4_K_M.gguf"),
            sibling("tokenizer.json"),
            sibling("config.json"),
        ];
        let selected = select_auto_files(&siblings);
        assert_eq!(selected, vec!["model.Q4_K_M.gguf", "tokenizer.json"]);
    }

    #[test]
    fn auto_files_selects_nested_tokenizer_json() {
        let siblings = vec![
            sibling("model.gguf"),
            sibling("tokenizer/tokenizer.json"),
            sibling("README.md"),
        ];
        let selected = select_auto_files(&siblings);
        assert_eq!(selected, vec!["model.gguf", "tokenizer/tokenizer.json"]);
    }

    #[test]
    fn auto_files_selects_tokenizer_model() {
        let siblings = vec![sibling("model.gguf"), sibling("tokenizer.model")];
        let selected = select_auto_files(&siblings);
        assert_eq!(selected, vec!["model.gguf", "tokenizer.model"]);
    }

    #[test]
    fn auto_files_empty_when_no_match() {
        let siblings = vec![sibling("README.md"), sibling("config.json")];
        let selected = select_auto_files(&siblings);
        assert!(selected.is_empty());
    }

    #[test]
    fn auto_files_deduplicates() {
        let siblings = vec![sibling("model.Q4_K_M.gguf"), sibling("model.Q4_K_M.gguf")];
        let selected = select_auto_files(&siblings);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn auto_files_case_insensitive_gguf() {
        let siblings = vec![sibling("MODEL.GGUF"), sibling("tokenizer.json")];
        let selected = select_auto_files(&siblings);
        assert_eq!(selected, vec!["MODEL.GGUF", "tokenizer.json"]);
    }

    // --- dest_path_for ---

    #[test]
    fn dest_path_flat_file() {
        let base = Path::new("/tmp/models");
        let p = dest_path_for(base, "model.gguf").unwrap();
        assert_eq!(p, PathBuf::from("/tmp/models/model.gguf"));
    }

    #[test]
    fn dest_path_preserves_subdir() {
        let base = Path::new("/tmp/models");
        let p = dest_path_for(base, "subdir/tokenizer.json").unwrap();
        assert_eq!(p, PathBuf::from("/tmp/models/subdir/tokenizer.json"));
    }

    #[test]
    fn dest_path_rejects_parent_traversal() {
        let base = Path::new("/tmp/models");
        let err = dest_path_for(base, "../evil.gguf").unwrap_err();
        assert!(err.contains("unsafe path component"), "got: {err}");
    }

    #[test]
    fn dest_path_rejects_absolute_path() {
        let base = Path::new("/tmp/models");
        let err = dest_path_for(base, "/etc/passwd").unwrap_err();
        assert!(err.contains("unsafe path component"), "got: {err}");
    }

    #[test]
    fn dest_path_rejects_traversal_in_subdir() {
        let base = Path::new("/tmp/models");
        let err = dest_path_for(base, "subdir/../../evil.gguf").unwrap_err();
        assert!(err.contains("unsafe path component"), "got: {err}");
    }

    #[test]
    fn dest_path_different_subdir_same_basename_no_collision() {
        let base = Path::new("/tmp/models");
        let p1 = dest_path_for(base, "a/tokenizer.json").unwrap();
        let p2 = dest_path_for(base, "b/tokenizer.json").unwrap();
        assert_ne!(p1, p2);
    }
}
