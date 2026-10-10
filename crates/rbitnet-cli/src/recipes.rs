//! Versioned serve recipes (Atlas sparkrun-style SSOT).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ServeRecipe {
    pub name: String,
    pub version: u32,
    #[serde(default)]
    pub description: Option<String>,
    /// Optional when the recipe only sets env (paths come from manifest / shell).
    #[serde(default)]
    pub model: Option<RecipeModel>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RecipeModel {
    pub gguf: String,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    /// Optional expected SHA-256 (hex, lowercase or uppercase) of the GGUF file.
    #[serde(default)]
    pub sha256: Option<String>,
}

pub fn load_recipe(path: &Path) -> Result<ServeRecipe, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("recipe JSON: {e}"))
}

pub fn apply_recipe_env(recipe: &ServeRecipe) {
    for (k, v) in &recipe.env {
        std::env::set_var(k, v);
    }
    if let Some(model) = recipe.model.as_ref() {
        std::env::set_var("RBITNET_MODEL", &model.gguf);
        if let Some(tok) = model.tokenizer.as_ref() {
            std::env::set_var("RBITNET_TOKENIZER", tok);
        }
        if let Some(arch) = model.architecture.as_ref() {
            std::env::set_var("RBITNET_ARCHITECTURE", arch);
        }
        if let Some(sha) = model.sha256.as_ref() {
            std::env::set_var("RBITNET_MODEL_SHA256", sha);
        }
    }
}

pub fn recipe_plan_text(recipe: &ServeRecipe, path: &Path) -> String {
    let mut out = format!("Recipe: {} (v{})\n", recipe.name, recipe.version);
    if let Some(d) = &recipe.description {
        out.push_str(&format!("  {d}\n"));
    }
    out.push_str(&format!("  file: {}\n", path.display()));
    if let Some(model) = &recipe.model {
        out.push_str(&format!("  RBITNET_MODEL={}\n", model.gguf));
        if let Some(tok) = &model.tokenizer {
            out.push_str(&format!("  RBITNET_TOKENIZER={tok}\n"));
        }
        if let Some(sha) = &model.sha256 {
            out.push_str(&format!("  RBITNET_MODEL_SHA256={sha}\n"));
        }
    } else {
        out.push_str("  (no model paths — set RBITNET_MODEL / tokenizer from manifest)\n");
    }
    for (k, v) in &recipe.env {
        out.push_str(&format!("  {k}={v}\n"));
    }
    if let Some(n) = &recipe.notes {
        out.push_str(&format!("  notes: {n}\n"));
    }
    out.push_str("\nAction\n  rbitnet serve\n");
    out
}

pub fn print_recipe_plan(recipe: &ServeRecipe, path: &Path) {
    print!("{}", recipe_plan_text(recipe, path));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_only_bitnet_recipe_parses() {
        let j = r#"{
            "version": 1,
            "name": "bitnet-b158-latency",
            "description": "test",
            "env": { "RBITNET_PREFIX_KV": "1" },
            "notes": "set paths from manifest"
        }"#;
        let r: ServeRecipe = serde_json::from_str(j).unwrap();
        assert!(r.model.is_none());
        assert_eq!(r.env.get("RBITNET_PREFIX_KV").map(String::as_str), Some("1"));
    }

    #[test]
    fn reference_recipes_require_sha_and_tokenizer() {
        let bitnet = include_str!("../../../recipes/bitnet-b158.recipe.json");
        let tiny = include_str!("../../../recipes/tinyllama-q4.recipe.json");
        for (name, raw) in [("bitnet", bitnet), ("tinyllama", tiny)] {
            let r: ServeRecipe = serde_json::from_str(raw).unwrap_or_else(|e| panic!("{name}: {e}"));
            let model = r.model.as_ref().unwrap_or_else(|| panic!("{name}: missing model"));
            assert!(
                model.tokenizer.as_ref().is_some_and(|t| !t.is_empty()),
                "{name}: tokenizer sidecar required"
            );
            let sha = model.sha256.as_ref().unwrap_or_else(|| panic!("{name}: sha256 required"));
            assert_eq!(sha.len(), 64, "{name}: sha256 must be 64 hex chars");
        }
    }
}
