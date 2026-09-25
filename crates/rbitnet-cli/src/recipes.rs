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

pub fn print_recipe_plan(recipe: &ServeRecipe, path: &Path) {
    println!("Recipe: {} (v{})", recipe.name, recipe.version);
    if let Some(d) = &recipe.description {
        println!("  {d}");
    }
    println!("  file: {}", path.display());
    if let Some(model) = &recipe.model {
        println!("  RBITNET_MODEL={}", model.gguf);
        if let Some(tok) = &model.tokenizer {
            println!("  RBITNET_TOKENIZER={tok}");
        }
        if let Some(sha) = &model.sha256 {
            println!("  RBITNET_MODEL_SHA256={sha}");
        }
    } else {
        println!("  (no model paths — set RBITNET_MODEL / tokenizer from manifest)");
    }
    for (k, v) in &recipe.env {
        println!("  {k}={v}");
    }
    if let Some(n) = &recipe.notes {
        println!("  notes: {n}");
    }
    println!("\nRun: rbitnet serve");
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
}
