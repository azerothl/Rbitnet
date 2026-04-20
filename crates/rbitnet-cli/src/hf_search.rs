//! Hugging Face Hub model search with `.gguf`-oriented filtering (not project-tested).

use serde::Deserialize;
use serde_json::Value;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

/// `GET /api/models/{org}/{repo}` — encode each path **segment** only. Encoding the whole id turns `/`
/// into `%2F`, which Hugging Face rejects with **400 Bad Request** (empty catalog / search otherwise).
pub(crate) fn hf_model_detail_url(repo_id: &str) -> String {
    let path = repo_id
        .split('/')
        .map(|seg| urlencoding::encode(seg).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    format!("{HF_MODELS_API}/{path}")
}

pub const SEARCH_WARNING: &str = "Hub search returns repositories that contain `.gguf` files only. This does NOT guarantee BitNet 1-bit weights nor Rbitnet compatibility. Only the curated catalog (`rbitnet models list`) is project-tested.";

pub const GENERATE_CATALOG_WARNING: &str = "This JSON is built from Hugging Face metadata only (repos with `.gguf`). GGUF does NOT imply BitNet 1-bit. Output is NOT validated by Rbitnet CI; review and trim before commit.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitnetConfidence {
    Likely,
    Possible,
    Generic,
}

impl BitnetConfidence {
    pub fn label(self) -> &'static str {
        match self {
            Self::Likely => "likely-bitnet",
            Self::Possible => "possible-bitnet",
            Self::Generic => "generic-gguf",
        }
    }

    pub fn strict_match(self) -> bool {
        matches!(self, Self::Likely | Self::Possible)
    }
}

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

/// All `rfilename` paths from the Hub model API for one repo.
pub fn fetch_model_sibling_paths(model_id: &str, token: Option<&str>) -> Result<Vec<String>, String> {
    let url = hf_model_detail_url(model_id);
    let agent = crate::hub_http::agent()?;
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| format!("HF model info {model_id}: {e}"))?;
    let v: Value = resp.into_json().map_err(|e| format!("HF model JSON: {e}"))?;

    if let Ok(info) = serde_json::from_value::<ModelInfo>(v.clone()) {
        return Ok(info.siblings.into_iter().map(|s| s.rfilename).collect());
    }

    let mut out = Vec::new();
    if let Some(arr) = v.get("siblings").and_then(|x| x.as_array()) {
        for item in arr {
            if let Some(name) = item.get("rfilename").and_then(|x| x.as_str()) {
                out.push(name.to_string());
            }
        }
    }
    Ok(out)
}

/// One Hub repo with `.gguf` siblings and optional tokenizer paths.
#[derive(Debug, Clone)]
pub struct DiscoveredRepo {
    pub id: String,
    pub gguf_files: Vec<String>,
    pub tokenizer_json: Option<String>,
    pub tokenizer_model: Option<String>,
    /// `tokenizer_config.json` when listed as a sibling (Transformers metadata only for Rbitnet).
    pub tokenizer_config_json: Option<String>,
}

fn tokenizer_from_paths(paths: &[String]) -> (Option<String>, Option<String>) {
    let mut tok_j = None;
    let mut tok_m = None;
    for p in paths {
        let lower = p.to_ascii_lowercase();
        if lower.ends_with("tokenizer.json") || lower == "tokenizer.json" {
            tok_j.get_or_insert_with(|| p.clone());
        } else if lower.ends_with("tokenizer.model") || lower == "tokenizer.model" {
            tok_m.get_or_insert_with(|| p.clone());
        }
    }
    (tok_j, tok_m)
}

fn tokenizer_config_from_paths(paths: &[String]) -> Option<String> {
    for p in paths {
        let lower = p.to_ascii_lowercase();
        if lower.ends_with("tokenizer_config.json") || lower == "tokenizer_config.json" {
            return Some(p.clone());
        }
    }
    None
}

/// Walk search results, fetch siblings for each candidate, return repos that contain at least one `.gguf`.
pub fn discover_gguf_repos(
    query: &str,
    search_limit: usize,
    max_inspect: usize,
    max_return: usize,
    hf_other_filter: Option<&str>,
    token: Option<&str>,
) -> Result<Vec<DiscoveredRepo>, String> {
    let q = urlencoding::encode(query);
    let mut url = format!(
        "{HF_MODELS_API}?search={q}&limit={search_limit}",
        search_limit = search_limit.min(100)
    );
    if let Some(other) = hf_other_filter {
        url.push_str("&other=");
        url.push_str(&urlencoding::encode(other));
    }
    let agent = crate::hub_http::agent()?;
    let mut req = agent.get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let hits: Vec<SearchHit> = req
        .call()
        .map_err(|e| format!("HF search GET: {e}"))?
        .into_json()
        .map_err(|e| format!("HF search JSON: {e}"))?;

    // `max_inspect` = max *detail* fetches (`/api/models/{repo}`), not "first N search rows".
    // Search for e.g. `llama` returns many Safetensors-only repos first; we must skip them
    // until we find enough `.gguf` trees or exhaust the probe budget.
    let mut out = Vec::new();
    let mut inspected = 0usize;
    for hit in hits {
        if out.len() >= max_return {
            break;
        }
        if inspected >= max_inspect {
            break;
        }
        inspected += 1;
        let paths = match fetch_model_sibling_paths(&hit.id, token) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let gguf_files: Vec<String> = paths
            .iter()
            .filter(|n| n.to_ascii_lowercase().ends_with(".gguf"))
            .cloned()
            .collect();
        if gguf_files.is_empty() {
            continue;
        }
        let mut gg = gguf_files;
        gg.sort();
        gg.dedup();
        let (tokenizer_json, tokenizer_model) = tokenizer_from_paths(&paths);
        let tokenizer_config_json = tokenizer_config_from_paths(&paths);
        out.push(DiscoveredRepo {
            id: hit.id,
            gguf_files: gg,
            tokenizer_json,
            tokenizer_model,
            tokenizer_config_json,
        });
    }
    Ok(out)
}

/// Search HF for models whose repo contains at least one `.gguf` file (inspects up to `max_inspect` candidates from the search page).
pub fn search_gguf_models(
    query: &str,
    search_limit: usize,
    max_inspect: usize,
    strict_bitnet: bool,
    token: Option<&str>,
) -> Result<Vec<GgufSearchHit>, String> {
    let other_filter = if strict_bitnet { Some("bitnet") } else { None };
    discover_gguf_repos(
        query,
        search_limit,
        max_inspect,
        20,
        other_filter,
        token,
    )
    .map(|repos| {
        repos
            .into_iter()
            .map(|r| {
                let (confidence, score) = classify_bitnet_confidence(&r.id, &r.gguf_files);
                GgufSearchHit {
                    id: r.id,
                    gguf_files: r.gguf_files,
                    tokenizer_json: r.tokenizer_json,
                    tokenizer_model: r.tokenizer_model,
                    tokenizer_config_json: r.tokenizer_config_json,
                    confidence,
                    confidence_score: score,
                }
            })
            .filter(|h| !strict_bitnet || h.confidence.strict_match())
            .collect()
    })
}

pub struct GgufSearchHit {
    pub id: String,
    pub gguf_files: Vec<String>,
    pub tokenizer_json: Option<String>,
    pub tokenizer_model: Option<String>,
    pub tokenizer_config_json: Option<String>,
    pub confidence: BitnetConfidence,
    pub confidence_score: u8,
}

fn classify_bitnet_confidence(repo_id: &str, gguf_files: &[String]) -> (BitnetConfidence, u8) {
    let repo_l = repo_id.to_ascii_lowercase();
    let files_l: Vec<String> = gguf_files.iter().map(|f| f.to_ascii_lowercase()).collect();

    let mut score: u8 = 0;
    if repo_l.contains("bitnet") {
        score = score.saturating_add(45);
    }
    if repo_l.contains("microsoft/bitnet") {
        score = score.saturating_add(20);
    }
    if repo_l.contains("1bit") || repo_l.contains("1-bit") || repo_l.contains("b1.58") {
        score = score.saturating_add(20);
    }
    if files_l
        .iter()
        .any(|f| f.contains("bitnet") || f.contains("b1.58"))
    {
        score = score.saturating_add(25);
    }
    if files_l
        .iter()
        .any(|f| f.contains("1bit") || f.contains("1-bit"))
    {
        score = score.saturating_add(10);
    }

    let confidence = if score >= 70 {
        BitnetConfidence::Likely
    } else if score >= 40 {
        BitnetConfidence::Possible
    } else {
        BitnetConfidence::Generic
    };
    (confidence, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_detail_url_keeps_slash_between_org_and_name() {
        let u = hf_model_detail_url("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
        assert!(
            u.contains("TheBloke/TinyLlama"),
            "unexpected url (must not use %2F for repo slash): {u}"
        );
        assert!(!u.contains("%2F"), "HF returns 400 if / is encoded as %2F: {u}");
    }

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
        let hits = search_gguf_models("tinyllama", 10, 5, false, None).expect("search");
        assert!(!hits.is_empty(), "expected at least one .gguf repo");
    }
}
