//! `rbitnet-proxy`: parent OpenAI-compatible proxy supervising one runner per model.

use tracing::error;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    if let Err(e) = rbitnet_proxy::run_proxy().await {
        error!(%e, "rbitnet-proxy failed");
        std::process::exit(1);
    }
}
