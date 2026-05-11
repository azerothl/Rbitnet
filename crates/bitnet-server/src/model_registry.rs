//! Local model registry (JSON): logical id → GGUF path, optional tokenizer / architecture override.
//!
//! Set `RBITNET_MODEL_REGISTRY` to a JSON file path. Startup uses `RBITNET_ACTIVE_MODEL_ID` or the
//! top-level `"default"` key. Each `/v1/chat/completions` request loads the weights for its `model`
//! field when it differs from the active GGUF in memory.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct RegistryModelEntry {
    pub gguf: PathBuf,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub architecture: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct RegistryFile {
    #[serde(rename = "default")]
    default_model: Option<String>,
    models: HashMap<String, RegistryModelEntry>,
}

/// Parsed `RBITNET_MODEL_REGISTRY` JSON.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    pub default_model: Option<String>,
    pub models: HashMap<String, RegistryModelEntry>,
}

impl ModelRegistry {
    /// Read and parse the registry file from `RBITNET_MODEL_REGISTRY`, or `Ok(None)` if unset.
    pub fn load_from_env() -> Result<Option<(Arc<Self>, String)>, String> {
        let path = match std::env::var("RBITNET_MODEL_REGISTRY") {
            Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
            _ => return Ok(None),
        };
        if !path.is_file() {
            return Err(format!(
                "RBITNET_MODEL_REGISTRY is not a file: {}",
                path.display()
            ));
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let reg: RegistryFile =
            serde_json::from_str(&text).map_err(|e| format!("registry JSON: {e}"))?;

        let active = std::env::var("RBITNET_ACTIVE_MODEL_ID")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or(reg.default_model.clone())
            .ok_or_else(|| {
                "RBITNET_MODEL_REGISTRY is set: define \"default\" in JSON or set RBITNET_ACTIVE_MODEL_ID"
                    .to_string()
            })?;

        if !reg.models.contains_key(&active) {
            return Err(format!("registry: unknown model id '{active}'"));
        }

        let mr = ModelRegistry {
            default_model: reg.default_model,
            models: reg.models,
        };
        Ok(Some((Arc::new(mr), active)))
    }
}
