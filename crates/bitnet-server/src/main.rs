//! OpenAI-compatible HTTP server for Rbitnet (`BitNetProvider` in Akasha).
//!
//! Environment:
//! - `RBITNET_BIND` — host:port (default `127.0.0.1:8080`)
//! - `RBITNET_STUB` — `1` for stub completions
//! - `RBITNET_TOY` — `1` for tiny in-process F32 toy LM (no GGUF)
//! - `RBITNET_MODEL` — path to `.gguf` (parsed; full BitNet inference WIP)

use std::sync::Arc;

use bitnet_core::inference::{stub_mode_enabled, Engine};
use bitnet_server::create_app;
use tracing::{error, info};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let bind = std::env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
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

    let app = create_app(engine);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    info!("rbitnet-server listening on http://{bind}");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        error!(%e, "server error");
        std::process::exit(1);
    });
}
