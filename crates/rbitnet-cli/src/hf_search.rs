//! Hugging Face Hub model search with `.gguf`-oriented filtering (not project-tested).

use serde::Deserialize;
use serde_json::Value;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

pub const SEARCH_WARNING: &str = "These results are not verified for Rbitnet compatibility. Only entries in the curated catalog (`rbitnet models list`) are project-tested. Repos with only Safetensors are excluded when no `.gguf` is present.";

#[derive(Debug, Deserialize)]
struct SearchHit {
    id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelInfo {
    #[serde(default)]
    pub(crate) siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Sibling {
    pub(crate) rfilename: String,
}

/// Search HF for models whose repo contains at least one `.gguf` file (inspects up to `max_inspect` candidates from the search page).
pub fn search_gguf_models(
    query: &str,
    search_limit: usize,
    max_inspect: usize,
    token: Option<&str>,
) -> Result<Vec<GgufSearchHit>, String> {
    let q = urlencoding::encode(query);
    let url = format!(
        "{HF_MODELS_API}?search={q}&limit={search_limit}",
        search_limit = search_limit.min(100)
    );
    let mut req = ureq::get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let hits: Vec<SearchHit> = req
        .call()
        .map_err(|e| format!("HF search GET: {e}"))?
        .into_json()
        .map_err(|e| format!("HF search JSON: {e}"))?;

    let mut out = Vec::new();
    for hit in hits.into_iter().take(max_inspect) {
        let gguf_files = list_gguf_files_in_repo(&hit.id, token)?;
        if gguf_files.is_empty() {
            continue;
        }
        out.push(GgufSearchHit {
            id: hit.id,
            gguf_files,
        });
        if out.len() >= 20 {
            break;
        }
    }
    Ok(out)
}

pub struct GgufSearchHit {
    pub id: String,
    pub gguf_files: Vec<String>,
}

fn list_gguf_files_in_repo(model_id: &str, token: Option<&str>) -> Result<Vec<String>, String> {
    let enc = urlencoding::encode(model_id);
    let url = format!("{HF_MODELS_API}/{enc}");
    let mut req = ureq::get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| format!("HF model info {model_id}: {e}"))?;
    let v: Value = resp.into_json().map_err(|e| format!("HF model JSON: {e}"))?;

    // Prefer structured parse; fall back to scanning JSON for `siblings`.
    if let Ok(info) = serde_json::from_value::<ModelInfo>(v.clone()) {
        let mut gg: Vec<String> = info
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|n| n.to_ascii_lowercase().ends_with(".gguf"))
            .collect();
        gg.sort();
        gg.dedup();
        return Ok(gg);
    }

    let mut gg = Vec::new();
    if let Some(arr) = v.get("siblings").and_then(|x| x.as_array()) {
        for item in arr {
            if let Some(name) = item.get("rfilename").and_then(|x| x.as_str()) {
                if name.to_ascii_lowercase().ends_with(".gguf") {
                    gg.push(name.to_string());
                }
            }
        }
    }
    gg.sort();
    gg.dedup();
    Ok(gg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_siblings_from_value() {
        let j = serde_json::json!({
            "siblings": [
                { "rfilename": "README.md" },
                { "rfilename": "model.Q4_K_M.gguf" }
            ]
        });
        let v: Value = j;
        let info: ModelInfo = serde_json::from_value(v).unwrap();
        let names: Vec<_> = info
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|n| n.ends_with(".gguf"))
            .collect();
        assert_eq!(names, vec!["model.Q4_K_M.gguf"]);
    }

    /// Optional live Hub call: `cargo test -p rbitnet-cli hf_search_smoke -- --ignored --nocapture`
    #[ignore]
    #[test]
    fn hf_search_smoke_tinyllama() {
        let hits = search_gguf_models("tinyllama", 10, 5, None).expect("search");
        assert!(!hits.is_empty(), "expected at least one .gguf repo");
    }
}
