//! Curated Hugging Face bundles: paired GGUF + tokenizer repos and `rbitnet.manifest.json`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::catalog;
use crate::download;
use crate::hf_search;

#[derive(Clone, Copy, Debug)]
pub struct CuratedBundle {
    pub id: &'static str,
    pub description: &'static str,
    pub gguf_repo: &'static str,
    pub tokenizer_repo: &'static str,
}

/// Known layouts documented in `docs/HF_BITNET_RBITNET_GAP.md`.
pub const BUNDLES: &[CuratedBundle] = &[CuratedBundle {
    id: "microsoft-bitnet-b1.58-2b-4t",
    description: "Microsoft BitNet b1.58 2B-4T — GGUF weights from the `-gguf` repo; tokenizer from the main model card (Safetensors repo).",
    gguf_repo: "microsoft/bitnet-b1.58-2B-4T-gguf",
    tokenizer_repo: "microsoft/bitnet-b1.58-2B-4T",
}];

pub fn list_bundles_text() -> String {
    let mut s = String::from("Curated install bundles:\n\n");
    for b in BUNDLES {
        s.push_str(&format!(
            "  {}\n    {}\n    gguf repo:      {}\n    tokenizer repo: {}\n\n",
            b.id, b.description, b.gguf_repo, b.tokenizer_repo
        ));
    }
    s.push_str("Install example:\n  rbitnet models install microsoft-bitnet-b1.58-2b-4t --dir ./models\n");
    s
}

fn pick_tokenizer_paths(siblings: &[String]) -> (Option<String>, Option<String>, Option<String>) {
    let mut json = None;
    let mut model = None;
    let mut cfg = None;
    for p in siblings {
        let l = p.to_ascii_lowercase();
        if l.ends_with("tokenizer.json") {
            json.get_or_insert_with(|| p.clone());
        } else if l.ends_with("tokenizer.model") {
            model.get_or_insert_with(|| p.clone());
        } else if l.ends_with("tokenizer_config.json") {
            cfg.get_or_insert_with(|| p.clone());
        }
    }
    (json, model, cfg)
}

fn fetch_text(repo_id: &str, file: &str, token: Option<&str>) -> Result<String, String> {
    let url = hf_search::hf_resolve_main_url(repo_id, file);
    let agent = crate::hub_http::agent()?;
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| {
        format!(
            "GET {url}: {e} (for gated repos set HF_TOKEN and accept the model license on huggingface.co)"
        )
    })?;
    let status = resp.status();
    if !(200..300).contains(&status) {
        return Err(format!(
            "GET {url} -> HTTP {status}. Gated or missing files often need HF_TOKEN and Hub license acceptance."
        ));
    }
    resp.into_string()
        .map_err(|e| format!("read body {url}: {e}"))
}

fn external_tokenizer_hub_id(cfg: &Value) -> Option<String> {
    for key in ["_name_or_path", "name_or_path"] {
        if let Some(s) = cfg.get(key).and_then(|v| v.as_str()) {
            let t = s.trim();
            if t.contains('/') && !t.contains(' ') && t.len() > 3 {
                return Some(t.to_string());
            }
        }
    }
    None
}

/// Resolve tokenizer file(s) to download from `tokenizer_repo` (same-repo siblings or `tokenizer_file` in config).
fn resolve_tokenizer_downloads(
    tokenizer_repo: &str,
    siblings: &[String],
    token: Option<&str>,
) -> Result<Vec<String>, String> {
    let (tj, tm, tcfg_path) = pick_tokenizer_paths(siblings);
    if let Some(p) = tj {
        return Ok(vec![p]);
    }
    if let Some(p) = tm {
        return Ok(vec![p]);
    }

    let config_file = tcfg_path
        .as_deref()
        .unwrap_or("tokenizer_config.json");
    let cfg_raw = match fetch_text(tokenizer_repo, config_file, token) {
        Ok(s) => s,
        Err(e) => {
            return Err(format!(
                "{e}\n\
                 No tokenizer.json / tokenizer.model in Hub siblings for `{tokenizer_repo}` and could not fetch `{config_file}`.\n\
                 See docs/HF_BITNET_RBITNET_GAP.md — you may need another repo or HF_TOKEN."
            ));
        }
    };
    let cfg: Value = serde_json::from_str(&cfg_raw)
        .map_err(|e| format!("parse {tokenizer_repo}/{config_file}: {e}"))?;

    if let Some(tf) = cfg.get("tokenizer_file").and_then(|v| v.as_str()) {
        let tf = tf.trim();
        if tf.is_empty() {
            return Err(format!(
                "`tokenizer_file` in `{tokenizer_repo}`/{config_file} is empty; add tokenizer.json to the repo or set RBITNET_TOKENIZER manually."
            ));
        }
        if tf.contains("://") || tf.starts_with("http") {
            return Err(format!(
                "tokenizer_file in `{tokenizer_repo}` is a URL (`{tf}`), which this CLI does not resolve. Download manually or use a repo that lists tokenizer.json as a sibling."
            ));
        }
        return Ok(vec![tf.to_string()]);
    }

    if let Some(ext) = external_tokenizer_hub_id(&cfg) {
        if ext != tokenizer_repo && !ext.is_empty() {
            return Err(format!(
                "`{tokenizer_repo}` tokenizer_config references Hugging Face id `{ext}`.\n\
                 Rbitnet cannot pull gated third-party vocabularies without your access.\n\
                 Accept the license on huggingface.co for that model, export or copy `tokenizer.json` / `tokenizer.model`, set RBITNET_TOKENIZER, or retry with HF_TOKEN if your account already has access.\n\
                 Docs: https://huggingface.co/{ext}"
            ));
        }
    }

    Err(format!(
        "Could not resolve tokenizer for `{tokenizer_repo}`: no tokenizer.json / tokenizer.model in siblings and no usable `tokenizer_file` in tokenizer_config.json.\n\
         See docs/HF_BITNET_RBITNET_GAP.md."
    ))
}

#[derive(Serialize)]
struct RbitnetManifest {
    version: u32,
    bundle_id: String,
    #[serde(rename = "RBITNET_MODEL")]
    rbitnet_model: String,
    #[serde(rename = "RBITNET_TOKENIZER", skip_serializing_if = "Option::is_none")]
    rbitnet_tokenizer: Option<String>,
    #[serde(rename = "_comment")]
    comment: &'static str,
}

/// Download a curated bundle into `dir` and write `rbitnet.manifest.json` with relative env paths.
pub fn install_bundle(bundle_id: &str, dir: &Path, token: Option<&str>) -> Result<(), String> {
    let bundle = BUNDLES
        .iter()
        .find(|b| b.id == bundle_id)
        .ok_or_else(|| {
            format!(
                "unknown bundle id {bundle_id:?}. Run: rbitnet models install --list"
            )
        })?;

    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;

    let tok_siblings = hf_search::fetch_model_sibling_paths(bundle.tokenizer_repo, token)?;
    let tokenizer_files = resolve_tokenizer_downloads(bundle.tokenizer_repo, &tok_siblings, token)?;

    let gguf_siblings = hf_search::fetch_model_sibling_paths(bundle.gguf_repo, token)?;
    let ggufs: Vec<String> = gguf_siblings
        .iter()
        .filter(|n| n.to_ascii_lowercase().ends_with(".gguf"))
        .cloned()
        .collect();
    let primary =
        catalog::pick_primary_gguf(&ggufs).ok_or_else(|| {
            format!(
                "no .gguf files listed for `{}` (Hub siblings empty or unreachable)",
                bundle.gguf_repo
            )
        })?;

    let mut jobs: Vec<(String, String)> = Vec::new();
    jobs.push((bundle.gguf_repo.to_string(), primary.clone()));
    for t in &tokenizer_files {
        jobs.push((bundle.tokenizer_repo.to_string(), t.clone()));
    }

    let mut written: Vec<PathBuf> = Vec::new();
    for (repo, file) in &jobs {
        let paths = download::download_files(repo, &[file.clone()], dir, token)?;
        written.extend(paths);
    }

    let model_rel = path_relative_to_dir(dir, written.iter().find(|p| {
        p.to_string_lossy().to_ascii_lowercase().ends_with(".gguf")
    }).ok_or("internal: no .gguf written")?)?;
    let tok_rel = tokenizer_files
        .iter()
        .find_map(|tf| {
            written.iter().find(|p| {
                p.ends_with(tf.as_str())
                    || p
                        .file_name()
                        .and_then(|n| n.to_str())
                        == Path::new(tf).file_name().and_then(|x| x.to_str())
            })
        })
        .and_then(|p| path_relative_to_dir(dir, p).ok());

    let manifest = RbitnetManifest {
        version: 1,
        bundle_id: bundle.id.to_string(),
        rbitnet_model: model_rel,
        rbitnet_tokenizer: tok_rel,
        comment: "Relative paths from this manifest's directory. Export as env vars or pass absolute paths to rbitnet serve.",
    };
    let manifest_path = dir.join("rbitnet.manifest.json");
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("serialize manifest: {e}"))?;
    fs::write(&manifest_path, json).map_err(|e| format!("write {}: {e}", manifest_path.display()))?;

    eprintln!(
        "Wrote {} file(s) under {} and {}",
        written.len(),
        dir.display(),
        manifest_path.display()
    );
    Ok(())
}

fn path_relative_to_dir(base: &Path, path: &Path) -> Result<String, String> {
    let rel = path.strip_prefix(base).map_err(|_| {
        format!(
            "path {} is not under {}",
            path.display(),
            base.display()
        )
    })?;
    Ok(rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_tokenizer_prefers_json() {
        let s = vec![
            "a.gguf".into(),
            "tokenizer.model".into(),
            "tokenizer.json".into(),
        ];
        let (j, m, c) = pick_tokenizer_paths(&s);
        assert_eq!(j.as_deref(), Some("tokenizer.json"));
        assert_eq!(m.as_deref(), Some("tokenizer.model"));
        assert!(c.is_none());
    }

    #[test]
    fn resolve_tokenizer_from_siblings_json() {
        let s = vec!["x.gguf".into(), "tokenizer.json".into()];
        let r = resolve_tokenizer_downloads("org/repo", &s, None).unwrap();
        assert_eq!(r, vec!["tokenizer.json"]);
    }

    #[test]
    fn external_tokenizer_hub_id_detects_repo_ref() {
        let v: Value =
            serde_json::from_str(r#"{"_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct"}"#)
                .unwrap();
        assert_eq!(
            external_tokenizer_hub_id(&v).as_deref(),
            Some("meta-llama/Meta-Llama-3-8B-Instruct")
        );
    }

    #[test]
    fn external_tokenizer_hub_id_ignores_class_name() {
        let v: Value = serde_json::from_str(r#"{"name_or_path": "LlamaTokenizerFast"}"#).unwrap();
        assert!(external_tokenizer_hub_id(&v).is_none());
    }
}
