//! OpenAI-compatible HTTP server for Rbitnet (`BitNetProvider` in Akasha).
//!
//! Environment:
//! - `RBITNET_BIND` — host:port (default `127.0.0.1:8080`)
//! - `RBITNET_STUB` — `1` for stub completions
//! - `RBITNET_TOY` — `1` for tiny in-process F32 toy LM (no GGUF)
//! - `RBITNET_MODEL` — path to `.gguf` (parsed; full BitNet inference WIP)
//! - `RBITNET_API_KEY` — optional; if set, require `Authorization: Bearer` or `X-API-Key`
//! - `RBITNET_MAX_BODY_BYTES`, `RBITNET_MAX_PROMPT_CHARS`, `RBITNET_MAX_TOKENS_CAP`,
//!   `RBITNET_MAX_CONCURRENT`, `RBITNET_INFERENCE_TIMEOUT_SECS` — limits (see docs/USAGE.md)

use std::sync::Arc;

use bitnet_core::inference::{stub_mode_enabled, Engine};
use bitnet_server::{create_app_with_config, ServerConfig};
use tracing::{error, info, warn};

fn warn_if_insecure_bind(bind: &str) {
    if bind.starts_with("0.0.0.0:") || bind == "0.0.0.0" {
        warn!(
            %bind,
            "listening on all IPv4 interfaces; use a reverse proxy, TLS, and RBITNET_API_KEY in production"
        );
    }
    if bind.starts_with("[::]:") || bind == "[::]" {
        warn!(
            %bind,
            "listening on all IPv6 interfaces; use a reverse proxy, TLS, and RBITNET_API_KEY in production"
        );
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let server_config = match ServerConfig::from_env() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            error!(%e, "invalid server configuration");
            std::process::exit(1);
        }
    };

    info!(
        max_body_bytes = server_config.max_body_bytes,
        max_prompt_chars = server_config.max_prompt_chars,
        max_tokens_cap = server_config.max_tokens_cap,
        max_concurrent = server_config.max_concurrent,
        inference_timeout_secs = server_config.inference_timeout.as_secs(),
        api_key_set = server_config.api_key.is_some(),
        "rbitnet server limits"
    );

    let bind = std::env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    warn_if_insecure_bind(&bind);

    let engine = match Engine::from_env() {
        Ok(e) => Arc::new(e),
        Err(e) => {
            error!(?e, "failed to init engine from env");
            std::process::exit(1);
        }
    };

    if let Some(summary) = engine.model_summary() {
        info!(%summary, "GGUF loaded (full BitNet inference WIP)");
    } else if !stub_mode_enabled() {
        info!("no RBITNET_MODEL — set RBITNET_STUB=1 or RBITNET_TOY=1 for testing without weights");
    }

    if !engine.is_ready() {
        warn!(
            "readiness: tokenizer not found next to GGUF; /ready will return 503 until tokenizer.json is available"
        );
    }

    let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&server_config));

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    info!("rbitnet-server listening on http://{bind}");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        error!(%e, "server error");
        std::process::exit(1);
    });
}
