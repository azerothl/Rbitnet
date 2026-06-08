//! Worker subprocess for `rbitnet-proxy`.
//!
//! The parent sets `RBITNET_MODEL`, `RBITNET_BIND`, and optional tokenizer /
//! architecture overrides.  This process then runs the same OpenAI-compatible
//! Axum server as `rbitnet-server`, isolated to one model id.

use tracing::error;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    if let Err(e) = bitnet_server::run_server().await {
        error!(%e, "rbitnet-runner failed");
        std::process::exit(1);
    }
}
