//! Curated model index (JSON over HTTPS).

use serde::Deserialize;

/// Default raw URL for [`data/compatible_models.json`](https://github.com/loicpeaudecerf/Rbitnet/blob/main/data/compatible_models.json) on the default branch.
pub const DEFAULT_MODELS_INDEX_URL: &str =
    "https://raw.githubusercontent.com/loicpeaudecerf/Rbitnet/main/data/compatible_models.json";

#[derive(Debug, Deserialize, PartialEq)]
pub struct Catalog {
    pub version: u32,
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct CatalogModel {
    pub id: String,
    pub repo: String,
    pub description: String,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub min_rbitnet_version: Option<String>,
}

pub fn fetch_catalog(url: &str) -> Result<Catalog, String> {
    let body = ureq::get(url)
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
}
