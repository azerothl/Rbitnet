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
    pub model: RecipeModel,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct RecipeModel {
    pub gguf: String,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
}

pub fn load_recipe(path: &Path) -> Result<ServeRecipe, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("recipe JSON: {e}"))
}

pub fn apply_recipe_env(recipe: &ServeRecipe) {
    for (k, v) in &recipe.env {
        std::env::set_var(k, v);
    }
    std::env::set_var("RBITNET_MODEL", &recipe.model.gguf);
    if let Some(tok) = recipe.model.tokenizer.as_ref() {
        std::env::set_var("RBITNET_TOKENIZER", tok);
    }
    if let Some(arch) = recipe.model.architecture.as_ref() {
        std::env::set_var("RBITNET_ARCHITECTURE", arch);
    }
}

pub fn print_recipe_plan(recipe: &ServeRecipe, path: &Path) {
    println!("Recipe: {} (v{})", recipe.name, recipe.version);
    if let Some(d) = &recipe.description {
        println!("  {d}");
    }
    println!("  file: {}", path.display());
    println!("  RBITNET_MODEL={}", recipe.model.gguf);
    if let Some(tok) = &recipe.model.tokenizer {
        println!("  RBITNET_TOKENIZER={tok}");
    }
    for (k, v) in &recipe.env {
        println!("  {k}={v}");
    }
    println!("\nRun: rbitnet serve");
}
