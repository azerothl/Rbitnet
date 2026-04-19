//! Download model files via `hf-hub` (HF cache) into a user directory.

use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use serde_json::Value;

use crate::hf_search;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

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
    let enc = urlencoding::encode(repo_id);
    let url = format!("{HF_MODELS_API}/{enc}");
    let mut req = ureq::get(&url);
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

    let mut names: Vec<String> = Vec::new();
    for s in siblings {
        let n = s.rfilename;
        let lower = n.to_ascii_lowercase();
        let take = lower.ends_with(".gguf")
            || lower == "tokenizer.json"
            || lower.ends_with("/tokenizer.json")
            || lower == "tokenizer.model"
            || lower.ends_with("/tokenizer.model");
        if take {
            names.push(n);
        }
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Err(
            "no .gguf (and no tokenizer.json/tokenizer.model) found in repo; specify --file"
                .into(),
        );
    }
    Ok(names)
}

/// Download each file into `dest_dir` (created if missing). Uses HF cache then copies.
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
        let cached = repo
            .get(file)
            .map_err(|e| format!("download {repo_id} {file}: {e}"))?;
        let dest = dest_dir.join(Path::new(file).file_name().ok_or("invalid file path")?);
        fs::copy(&cached, &dest)
            .map_err(|e| format!("copy {} -> {}: {e}", cached.display(), dest.display()))?;
        out.push(dest);
    }
    Ok(out)
}
