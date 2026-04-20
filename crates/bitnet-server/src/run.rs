//! Shared entrypoint for the OpenAI-compatible HTTP server (used by `rbitnet-server` and `rbitnet serve`).

use std::sync::Arc;

use bitnet_core::inference::{stub_mode_enabled, Engine};
use tracing::{error, info, warn};

use crate::{create_app_with_config, ServerConfig};

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

/// Run the HTTP server until shutdown or fatal error. Same behavior as the `rbitnet-server` binary.
pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server_config = match ServerConfig::from_env() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            error!(%e, "invalid server configuration");
            return Err(format!("invalid server configuration: {e}").into());
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
            return Err(format!("failed to init engine from env: {e:?}").into());
        }
    };

    if let Some(summary) = engine.model_summary() {
        info!(%summary, "GGUF loaded (full BitNet inference WIP)");
    } else if !stub_mode_enabled() {
        info!("no RBITNET_MODEL — set RBITNET_STUB=1 or RBITNET_TOY=1 for testing without weights");
    }

    if engine.has_gguf() && !engine.is_ready() {
        warn!(
            "readiness: tokenizer not found next to GGUF; /ready will return 503 until tokenizer.json is available"
        );
    }

    let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&server_config));

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    info!("rbitnet-server listening on http://{bind}");
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("server error: {e}"))?;
    Ok(())
}
