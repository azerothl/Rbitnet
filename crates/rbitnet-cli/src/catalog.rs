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
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_rbitnet_version: Option<String>,
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
        assert!(c.models[0].min_rbitnet_version.is_none());
    }

    #[test]
    fn pick_primary_prefers_q4_k_m() {
        let g = vec![
            "x.Q2_K.gguf".into(),
            "x.Q4_K_M.gguf".into(),
            "x.Q8_0.gguf".into(),
        ];
        assert_eq!(
            pick_primary_gguf(&g).as_deref(),
            Some("x.Q4_K_M.gguf")
        );
    }

    #[test]
    fn catalog_id_from_repo_sanitizes() {
        assert_eq!(catalog_id_from_repo("TheBloke/Llama-2-GGUF"), "thebloke-llama-2-gguf");
    }
}
