//! HTTP server limits and policy from environment variables and optional local config.

use std::path::PathBuf;
use std::time::Duration;

/// Tunables for production-style guardrails (see `docs/PLAN_PRODUCTION.md`).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Max JSON body size for `/v1/chat/completions` (bytes).
    pub max_body_bytes: usize,
    /// Max UTF-8 characters accepted for the built prompt string.
    pub max_prompt_chars: usize,
    /// Hard cap on client `max_tokens` (after JSON parse).
    pub max_tokens_cap: u32,
    /// Maximum concurrent blocking inference calls.
    pub max_concurrent: usize,
    /// Wall-clock limit for a single `Engine::complete` call (inference + tokenizer).
    pub inference_timeout: Duration,
    /// If set, require `Authorization: Bearer <key>` or `X-API-Key: <key>` on API routes.
    pub api_key: Option<String>,
    /// If true, `/v1/chat/completions` `model` must match the served model id (or registry id).
    pub require_model_match: bool,
    /// If set, `POST /v1/admin/unload` requires header `X-Rbitnet-Admin-Token: <token>`.
    pub admin_token: Option<String>,
    /// After this many seconds without chat traffic, swap the engine for a stub (frees mmap).
    pub idle_unload_secs: Option<u64>,
    /// If set, reject prompts whose tokenizer-encoded length exceeds this value (HTTP 400).
    pub max_prompt_tokens: Option<u32>,
}

fn parse_u64(key: &str, default: u64) -> Result<u64, String> {
    match std::env::var(key) {
        Ok(s) if s.trim().is_empty() => Ok(default),
        Ok(s) => s
            .parse()
            .map_err(|_| format!("{key}: expected a non-negative integer")),
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_runtime_config_flat_toml_subset() {
        let cfg = parse_runtime_config(
            r#"
model = "models/tiny.gguf"
tokenizer = "models/tokenizer.json"
bind = "127.0.0.1:8081"
chat_format = "llama3"
unknown = "ignored"
"#,
        );
        assert_eq!(cfg.model.as_deref(), Some("models/tiny.gguf"));
        assert_eq!(cfg.tokenizer.as_deref(), Some("models/tokenizer.json"));
        assert_eq!(cfg.bind.as_deref(), Some("127.0.0.1:8081"));
        assert_eq!(cfg.chat_format.as_deref(), Some("llama3"));
    }
}

fn parse_usize(key: &str, default: usize) -> Result<usize, String> {
    let v = parse_u64(key, default as u64)?;
    usize::try_from(v).map_err(|_| format!("{key}: value too large for usize"))
}

fn parse_optional_u32_positive(key: &str) -> Result<Option<u32>, String> {
    match std::env::var(key) {
        Ok(s) if s.trim().is_empty() => Ok(None),
        Ok(s) => {
            let v: u64 = s
                .parse()
                .map_err(|_| format!("{key}: expected a positive integer"))?;
            let v = u32::try_from(v).map_err(|_| format!("{key}: value exceeds u32::MAX"))?;
            if v == 0 {
                return Err(format!("{key}: must be >= 1"));
            }
            Ok(Some(v))
        }
        Err(_) => Ok(None),
    }
}

/// Minimal runtime fields accepted from `rbitnet.toml`.
///
/// This deliberately supports only the flat keys Rbitnet can safely translate to existing
/// environment variables. Env vars keep priority so scripts and production services remain stable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub bind: Option<String>,
    pub chat_format: Option<String>,
    pub model_registry: Option<String>,
    pub active_model_id: Option<String>,
}

impl RuntimeConfig {
    fn merge_missing(&mut self, other: Self) {
        if self.model.is_none() {
            self.model = other.model;
        }
        if self.tokenizer.is_none() {
            self.tokenizer = other.tokenizer;
        }
        if self.bind.is_none() {
            self.bind = other.bind;
        }
        if self.chat_format.is_none() {
            self.chat_format = other.chat_format;
        }
        if self.model_registry.is_none() {
            self.model_registry = other.model_registry;
        }
        if self.active_model_id.is_none() {
            self.active_model_id = other.active_model_id;
        }
    }

    fn apply_to_env_if_missing(&self) {
        set_env_if_missing("RBITNET_MODEL", self.model.as_deref());
        set_env_if_missing("RBITNET_TOKENIZER", self.tokenizer.as_deref());
        set_env_if_missing("RBITNET_BIND", self.bind.as_deref());
        set_env_if_missing("RBITNET_CHAT_FORMAT", self.chat_format.as_deref());
        set_env_if_missing("RBITNET_MODEL_REGISTRY", self.model_registry.as_deref());
        set_env_if_missing("RBITNET_ACTIVE_MODEL_ID", self.active_model_id.as_deref());
    }
}

fn set_env_if_missing(key: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return;
    };
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

fn unquote_config_value(raw: &str) -> String {
    let value = raw.trim();
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        value[1..value.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        value.to_string()
    }
}

pub fn parse_runtime_config(text: &str) -> RuntimeConfig {
    let mut cfg = RuntimeConfig::default();
    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() || line.starts_with('[') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = unquote_config_value(value);
        match key.trim() {
            "model" => cfg.model = Some(value),
            "tokenizer" => cfg.tokenizer = Some(value),
            "bind" => cfg.bind = Some(value),
            "chat_format" => cfg.chat_format = Some(value),
            "model_registry" => cfg.model_registry = Some(value),
            "active_model_id" => cfg.active_model_id = Some(value),
            _ => {}
        }
    }
    cfg
}

fn user_config_path() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RBITNET_CONFIG_DIR") {
        return Some(PathBuf::from(dir).join("rbitnet.toml"));
    }
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join("Rbitnet").join("rbitnet.toml"))
    }
    #[cfg(not(windows))]
    {
        if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
            return Some(PathBuf::from(dir).join("rbitnet").join("rbitnet.toml"));
        }
        std::env::var_os("HOME").map(|h| {
            PathBuf::from(h)
                .join(".config")
                .join("rbitnet")
                .join("rbitnet.toml")
        })
    }
}

pub fn runtime_config_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    paths.push(PathBuf::from("rbitnet.toml"));
    if let Some(p) = std::env::var_os("RBITNET_CONFIG") {
        paths.push(PathBuf::from(p));
    }
    if let Some(p) = user_config_path() {
        paths.push(p);
    }
    paths
}

/// Apply `rbitnet.toml` / user config defaults to env for the existing engine path.
pub fn apply_runtime_config_env() -> Result<(), String> {
    let mut merged = RuntimeConfig::default();
    for path in runtime_config_candidates() {
        if !path.is_file() {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        merged.merge_missing(parse_runtime_config(&text));
    }
    merged.apply_to_env_if_missing();
    Ok(())
}

impl ServerConfig {
    /// Load from environment; missing variables use safe defaults.
    pub fn from_env() -> Result<Self, String> {
        let max_body_bytes = parse_usize("RBITNET_MAX_BODY_BYTES", 1024 * 1024)?;
        let max_prompt_chars = parse_usize("RBITNET_MAX_PROMPT_CHARS", 256_000)?;
        let max_tokens_cap_raw = parse_u64("RBITNET_MAX_TOKENS_CAP", 8192)?;
        let max_tokens_cap = u32::try_from(max_tokens_cap_raw).map_err(|_| {
            format!("RBITNET_MAX_TOKENS_CAP: value {max_tokens_cap_raw} exceeds u32::MAX")
        })?;
        let max_concurrent = parse_usize("RBITNET_MAX_CONCURRENT", 4)?.max(1);
        let inference_timeout_secs = parse_u64("RBITNET_INFERENCE_TIMEOUT_SECS", 600)?;
        let api_key = std::env::var("RBITNET_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let require_model_match = matches!(
            std::env::var("RBITNET_REQUIRE_MODEL_MATCH").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let admin_token = std::env::var("RBITNET_ADMIN_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let idle_unload_secs = std::env::var("RBITNET_IDLE_UNLOAD_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&v| v > 0);

        let max_prompt_tokens = parse_optional_u32_positive("RBITNET_MAX_PROMPT_TOKENS")?;

        let cfg = Self {
            max_body_bytes,
            max_prompt_chars,
            max_tokens_cap,
            max_concurrent,
            inference_timeout: Duration::from_secs(inference_timeout_secs.max(1)),
            api_key,
            require_model_match,
            admin_token,
            idle_unload_secs,
            max_prompt_tokens,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.max_body_bytes == 0 {
            return Err("RBITNET_MAX_BODY_BYTES must be >= 1".into());
        }
        if self.max_prompt_chars == 0 {
            return Err("RBITNET_MAX_PROMPT_CHARS must be >= 1".into());
        }
        if self.max_tokens_cap == 0 {
            return Err("RBITNET_MAX_TOKENS_CAP must be >= 1".into());
        }
        Ok(())
    }

    /// Defaults for unit tests (no auth, small limits where irrelevant).
    #[must_use]
    pub fn test_defaults() -> Self {
        Self {
            max_body_bytes: 1024 * 1024,
            max_prompt_chars: 256_000,
            max_tokens_cap: 8192,
            max_concurrent: 8,
            inference_timeout: Duration::from_secs(120),
            api_key: None,
            require_model_match: false,
            admin_token: None,
            idle_unload_secs: None,
            max_prompt_tokens: None,
        }
    }
}
