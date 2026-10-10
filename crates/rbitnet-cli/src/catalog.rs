//! Curated model index (JSON over HTTPS).

use serde::{Deserialize, Serialize};

/// Default raw URL for [`data/compatible_models.json`](https://github.com/azerothl/Rbitnet/blob/main/data/compatible_models.json) on the default branch.
pub const DEFAULT_MODELS_INDEX_URL: &str =
    "https://raw.githubusercontent.com/azerothl/Rbitnet/main/data/compatible_models.json";

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Catalog {
    pub version: u32,
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CatalogModel {
    pub id: String,
    pub repo: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub use_case: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_ram_gb: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_ram: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tested: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_rbitnet_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoke_test: Option<String>,
    /// Golden parity badge from the curated index: `verified_golden` (documented llama.cpp match) or `best_effort`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden_tier: Option<String>,
    /// Optional expected SHA-256 (hex) of the primary GGUF file for provenance checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Stable short tags (Ollama-style), e.g. `bitnet:2b`, `tinyllama:q4`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional Hub repo for tokenizer sidecars when they live outside the GGUF repo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_repo: Option<String>,
}

/// Pick one GGUF filename when a repo ships many quantizations (prefers common Q4_K_M-style names).
#[must_use]
pub fn pick_primary_gguf(ggufs: &[String]) -> Option<String> {
    if ggufs.is_empty() {
        return None;
    }
    const PREFER: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "IQ4_XS", "Q4_0", "Q8_0", "F16",
    ];
    for needle in PREFER {
        if let Some(f) = ggufs.iter().find(|s| s.contains(needle)) {
            return Some((*f).clone());
        }
    }
    let mut sorted = ggufs.to_vec();
    sorted.sort();
    Some(sorted[0].clone())
}

/// Stable id slug from a Hub repo id (`org/name` → `org-name`).
#[must_use]
pub fn catalog_id_from_repo(repo_id: &str) -> String {
    repo_id
        .chars()
        .map(|c| if c == '/' { '-' } else { c })
        .collect::<String>()
        .to_ascii_lowercase()
}

pub fn fetch_catalog(url: &str) -> Result<Catalog, String> {
    let body = crate::hub_http::agent()?
        .get(url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?
        .into_string()
        .map_err(|e| format!("read body: {e}"))?;
    serde_json::from_str(&body).map_err(|e| format!("parse catalog JSON: {e}"))
}

/// Catalog compiled into the binary. A checkout or an install directory can override it.
pub fn load_bundled_catalog() -> Catalog {
    serde_json::from_str(include_str!("../../../data/compatible_models.json"))
        .expect("embedded data/compatible_models.json")
}

fn catalog_override_paths() -> Vec<std::path::PathBuf> {
    let mut paths = vec![std::path::PathBuf::from("data/compatible_models.json")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("compatible_models.json"));
            paths.push(dir.join("data").join("compatible_models.json"));
        }
    }
    paths
}

/// Checkout file, then the file beside the executable, then the embedded catalog.
pub fn load_local_catalog() -> Option<Catalog> {
    for path in catalog_override_paths() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Ok(catalog) = serde_json::from_str(&text) {
            return Some(catalog);
        }
    }
    Some(load_bundled_catalog())
}

/// `org/name` is a Hub repo. A catalog tag such as `bitnet:2b` is not.
#[cfg(test)]
#[must_use]
fn looks_like_hub_repo(refer: &str) -> bool {
    let key = refer.trim();
    let mut parts = key.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(org), Some(name), None) => {
            !org.is_empty()
                && !name.is_empty()
                && !org.contains(':')
                && !name.contains(':')
                && !org.contains(' ')
                && !name.contains(' ')
        }
        _ => false,
    }
}

/// Resolve a user ref: catalog `id`, stable `tag`, or return `None` (treat as Hub repo id).
#[must_use]
pub fn resolve_catalog_ref<'a>(catalog: &'a Catalog, refer: &str) -> Option<&'a CatalogModel> {
    let key = refer.trim();
    if key.is_empty() {
        return None;
    }
    let lower = key.to_ascii_lowercase();
    catalog
        .models
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(key))
        .or_else(|| {
            catalog.models.iter().find(|m| {
                m.tags
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(key) || t.to_ascii_lowercase() == lower)
            })
        })
}

/// Human-readable list of stable tags for docs / `--list`.
#[must_use]
pub fn format_stable_tags(catalog: &Catalog) -> String {
    let mut lines = Vec::new();
    for m in &catalog.models {
        if m.tags.is_empty() {
            continue;
        }
        let primary = m
            .file
            .clone()
            .or_else(|| pick_primary_gguf(&m.files))
            .unwrap_or_else(|| "(no file)".into());
        let sha = m.sha256.as_deref().unwrap_or("(no sha256)");
        lines.push(format!(
            "  {} → {} / {}\n    id={}  sha256={sha}",
            m.tags.join(", "),
            m.repo,
            primary,
            m.id
        ));
    }
    if lines.is_empty() {
        "  (no stable tags in catalog)\n".into()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_catalog() {
        let j = r#"{"version":1,"models":[{"id":"a","repo":"x/y","description":"d","files":["m.gguf"]}]}"#;
        let c: Catalog = serde_json::from_str(j).unwrap();
        assert_eq!(c.version, 1);
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].id, "a");
        assert_eq!(c.models[0].repo, "x/y");
        assert_eq!(c.models[0].files, vec!["m.gguf"]);
        assert!(c.models[0].use_case.is_empty());
        assert!(c.models[0].min_ram_gb.is_none());
        assert!(c.models[0].verified.is_none());
        assert!(c.models[0].min_rbitnet_version.is_none());
        assert!(c.models[0].smoke_test.is_none());
    }

    #[test]
    fn pick_primary_prefers_q4_k_m() {
        let g = vec![
            "x.Q2_K.gguf".into(),
            "x.Q4_K_M.gguf".into(),
            "x.Q8_0.gguf".into(),
        ];
        assert_eq!(pick_primary_gguf(&g).as_deref(), Some("x.Q4_K_M.gguf"));
    }

    #[test]
    fn embedded_catalog_resolves_bitnet_tag() {
        let cat = load_bundled_catalog();
        let model = resolve_catalog_ref(&cat, "bitnet:2b").expect("bitnet:2b");
        assert_eq!(model.repo, "microsoft/bitnet-b1.58-2B-4T-gguf");
        assert_eq!(model.file.as_deref(), Some("ggml-model-i2_s.gguf"));
        assert!(!looks_like_hub_repo("bitnet:2b"));
        assert!(looks_like_hub_repo("microsoft/bitnet-b1.58-2B-4T-gguf"));
    }

    #[test]
    fn catalog_id_from_repo_sanitizes() {
        assert_eq!(
            catalog_id_from_repo("TheBloke/Llama-2-GGUF"),
            "thebloke-llama-2-gguf"
        );
    }

    #[test]
    fn resolve_ref_by_id_and_tag() {
        let j = r#"{
          "version":1,
          "models":[{
            "id":"tinyllama-1.1b-chat-q4-k-m",
            "repo":"TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "description":"d",
            "file":"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "files":["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"],
            "tags":["tinyllama:q4","tinyllama"],
            "sha256":"9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"
          }]
        }"#;
        let c: Catalog = serde_json::from_str(j).unwrap();
        assert_eq!(
            resolve_catalog_ref(&c, "tinyllama:q4").map(|m| m.id.as_str()),
            Some("tinyllama-1.1b-chat-q4-k-m")
        );
        assert_eq!(
            resolve_catalog_ref(&c, "tinyllama-1.1b-chat-q4-k-m").map(|m| m.repo.as_str()),
            Some("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        );
        assert!(resolve_catalog_ref(&c, "org/missing").is_none());
    }
}
