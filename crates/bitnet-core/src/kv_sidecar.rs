//! Optional external KV cache sidecar (PegaFlow-inspired warm TTFT across workers).
//!
//! Set `RBITNET_KV_SIDECAR_URL` to an HTTP endpoint that accepts prefix block uploads.
//! The default implementation is a no-op stub until a full connector ships.

use crate::error::{BitNetError, Result};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct KvSidecarConfig {
    pub base_url: Option<String>,
    pub timeout: Duration,
}

impl KvSidecarConfig {
    pub fn from_env() -> Self {
        let base_url = std::env::var("RBITNET_KV_SIDECAR_URL")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let timeout_secs = std::env::var("RBITNET_KV_SIDECAR_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        Self {
            base_url,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    pub fn enabled(&self) -> bool {
        self.base_url.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct KvSidecarPut {
    pub model_id: String,
    pub prefix_hash: u64,
    pub token_count: usize,
    pub block_ids: Vec<usize>,
}

pub trait KvSidecarClient: Send + Sync {
    fn put_prefix_blocks(&self, put: &KvSidecarPut) -> Result<()>;
    fn get_prefix_blocks(&self, model_id: &str, prefix_hash: u64) -> Result<Option<Vec<usize>>>;
}

#[derive(Debug, Default)]
pub struct NoopKvSidecar;

impl KvSidecarClient for NoopKvSidecar {
    fn put_prefix_blocks(&self, _put: &KvSidecarPut) -> Result<()> {
        Ok(())
    }

    fn get_prefix_blocks(&self, _model_id: &str, _prefix_hash: u64) -> Result<Option<Vec<usize>>> {
        Ok(None)
    }
}

/// HTTP stub client: validates URL shape; network I/O is intentionally minimal for now.
#[derive(Debug)]
pub struct HttpKvSidecar {
    pub base_url: String,
    pub timeout: Duration,
}

impl HttpKvSidecar {
    pub fn from_config(cfg: &KvSidecarConfig) -> Result<Option<Self>> {
        let Some(url) = cfg.base_url.as_ref() else {
            return Ok(None);
        };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(BitNetError::Inference(format!(
                "RBITNET_KV_SIDECAR_URL must be http(s): got {url}"
            )));
        }
        Ok(Some(Self {
            base_url: url.clone(),
            timeout: cfg.timeout,
        }))
    }
}

impl KvSidecarClient for HttpKvSidecar {
    fn put_prefix_blocks(&self, put: &KvSidecarPut) -> Result<()> {
        tracing::debug!(
            sidecar = %self.base_url,
            model = %put.model_id,
            prefix_hash = put.prefix_hash,
            blocks = put.block_ids.len(),
            "kv sidecar put (stub — no network round-trip yet)"
        );
        Ok(())
    }

    fn get_prefix_blocks(&self, model_id: &str, prefix_hash: u64) -> Result<Option<Vec<usize>>> {
        tracing::debug!(
            sidecar = %self.base_url,
            model = %model_id,
            prefix_hash,
            "kv sidecar get (stub — cache miss)"
        );
        Ok(None)
    }
}
