//! Optional external KV cache sidecar (PegaFlow-inspired warm TTFT across workers).
//!
//! Set `RBITNET_KV_SIDECAR_URL` to an HTTP endpoint that accepts prefix block uploads.

use crate::error::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSidecarPut {
    pub model_id: String,
    pub prefix_hash: u64,
    pub token_count: usize,
    pub block_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvSidecarGetResponse {
    block_ids: Option<Vec<usize>>,
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

/// Minimal HTTP/1.1 client for PUT/GET JSON prefix blocks.
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
            base_url: url.trim_end_matches('/').to_string(),
            timeout: cfg.timeout,
        }))
    }

    fn request(&self, method: &str, path: &str, body: Option<&str>) -> Result<String> {
        let url = format!("{}{}", self.base_url, path);
        let (host, port, use_tls, req_path) = parse_http_url(&url)?;
        let addr = format!("{host}:{port}");
        let mut stream = TcpStream::connect(&addr).map_err(|e| {
            BitNetError::Inference(format!("kv sidecar connect {addr}: {e}"))
        })?;
        let _ = stream.set_read_timeout(Some(self.timeout));
        let _ = stream.set_write_timeout(Some(self.timeout));
        if use_tls {
            return Err(BitNetError::Inference(
                "kv sidecar: https requires TLS (use http://127.0.0.1 for local sidecar)".into(),
            ));
        }
        let body_bytes = body.unwrap_or("");
        let req = format!(
            "{method} {req_path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body_bytes}",
            body_bytes.len()
        );
        stream
            .write_all(req.as_bytes())
            .map_err(|e| BitNetError::Inference(format!("kv sidecar write: {e}")))?;
        let mut resp = Vec::new();
        stream
            .read_to_end(&mut resp)
            .map_err(|e| BitNetError::Inference(format!("kv sidecar read: {e}")))?;
        let text = String::from_utf8_lossy(&resp);
        let status_ok = text.lines().next().is_some_and(|l| l.contains(" 200 ") || l.contains(" 201 "));
        if !status_ok {
            return Err(BitNetError::Inference(format!(
                "kv sidecar HTTP error: {}",
                text.lines().next().unwrap_or("no status")
            )));
        }
        Ok(text.split("\r\n\r\n").nth(1).unwrap_or("").trim().to_string())
    }
}

fn parse_http_url(url: &str) -> Result<(String, u16, bool, String)> {
    let without_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| BitNetError::Inference("bad sidecar url".into()))?;
    let use_tls = url.starts_with("https://");
    let (host_port, path) = match without_scheme.split_once('/') {
        Some((hp, p)) => (hp, format!("/{p}")),
        None => (without_scheme, "/".into()),
    };
    let (host, port) = match host_port.split_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse()
                .map_err(|_| BitNetError::Inference("bad sidecar port".into()))?,
        ),
        None => (
            host_port.to_string(),
            if use_tls { 443 } else { 80 },
        ),
    };
    Ok((host, port, use_tls, path))
}

impl KvSidecarClient for HttpKvSidecar {
    fn put_prefix_blocks(&self, put: &KvSidecarPut) -> Result<()> {
        let path = format!("/kv/prefix/{}", put.prefix_hash);
        let body = serde_json::to_string(put)
            .map_err(|e| BitNetError::Inference(format!("kv sidecar json: {e}")))?;
        let _ = self.request("PUT", &path, Some(&body))?;
        Ok(())
    }

    fn get_prefix_blocks(&self, model_id: &str, prefix_hash: u64) -> Result<Option<Vec<usize>>> {
        let path = format!("/kv/prefix/{prefix_hash}?model_id={model_id}");
        let body = self.request("GET", &path, None)?;
        if body.is_empty() {
            return Ok(None);
        }
        let parsed: KvSidecarGetResponse = serde_json::from_str(&body).map_err(|e| {
            BitNetError::Inference(format!("kv sidecar parse: {e}"))
        })?;
        Ok(parsed.block_ids)
    }
}
