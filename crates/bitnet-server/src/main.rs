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
        error!(%e, "rbitnet-server failed");
        std::process::exit(1);
    }
}
