//! OpenAI-compatible HTTP server for Rbitnet (`BitNetProvider` in Akasha).
//!
//! Environment:
//! - `RBITNET_BIND` — host:port (default `127.0.0.1:8080`)
//! - `RBITNET_STUB` — `1` for stub completions
//! - `RBITNET_TOY` — `1` for tiny in-process F32 toy LM (no GGUF)
//! - `RBITNET_MODEL` — path to `.gguf` (parsed; full BitNet inference WIP)
//! - `RBITNET_API_KEY` — optional; if set, require `Authorization: Bearer` or `X-API-Key`
//! - `RBITNET_MAX_BODY_BYTES`, `RBITNET_MAX_PROMPT_CHARS`, `RBITNET_MAX_PROMPT_TOKENS`,
//!   `RBITNET_MAX_TOKENS_CAP`,
//!   `RBITNET_MAX_CONCURRENT`, `RBITNET_INFERENCE_TIMEOUT_SECS` — limits (see docs/USAGE.md)
//! - `RBITNET_MODEL_REGISTRY` — JSON file: `{ "default": "id", "models": { "id": { "gguf": "...", "tokenizer": "...", "architecture": "..." } } }`
//! - `RBITNET_ACTIVE_MODEL_ID` — registry key when `default` is not set
//! - `RBITNET_REQUIRE_MODEL_MATCH` — require OpenAI `model` to match served id
//! - `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, `RBITNET_MAX_VRAM_MB`, `RBITNET_BUDGET_MAX_SEQ` — load guardrails (bitnet-core)
//! - `RBITNET_ADMIN_TOKEN` — enables `POST /v1/admin/unload` with `X-Rbitnet-Admin-Token` or Bearer
//! - `RBITNET_IDLE_UNLOAD_SECS` — after idle, swap engine for stub (frees mmap)
//!
//! CLI (optional; only applied when the corresponding env var is unset):
//! - `--api-key`, `--bind`

use clap::Parser;
use tracing::error;

#[derive(Parser)]
#[command(name = "rbitnet-server")]
struct Cli {
    /// API key for protected routes (only if `RBITNET_API_KEY` is not already set).
    #[arg(long, env = "RBITNET_API_KEY")]
    api_key: Option<String>,
    /// Listen address host:port (only if `RBITNET_BIND` is not already set).
    #[arg(long, env = "RBITNET_BIND")]
    bind: Option<String>,
}

fn apply_cli_env(cli: &Cli) {
    if let Some(k) = &cli.api_key {
        if std::env::var_os("RBITNET_API_KEY").is_none() {
            std::env::set_var("RBITNET_API_KEY", k);
        }
    }
    if let Some(b) = &cli.bind {
        if std::env::var_os("RBITNET_BIND").is_none() {
            std::env::set_var("RBITNET_BIND", b);
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    apply_cli_env(&cli);

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
