//! HTTPS to Hugging Face / GitHub raw using **native-tls** (Schannel on Windows).
//!
//! Default `ureq` enables **rustls**, which often fails on Windows behind proxies, AV, or
//! strict TLS inspection (`WSAECONNRESET` / “connexion fermée par l’hôte distant”).

use std::sync::Arc;

use native_tls::TlsConnector;
use once_cell::sync::OnceCell;
use ureq::Agent;

static AGENT: OnceCell<Agent> = OnceCell::new();

fn build_agent() -> Result<Agent, String> {
    let tls = TlsConnector::new().map_err(|e| format!("native_tls: {e}"))?;
    Ok(ureq::AgentBuilder::new()
        .tls_connector(Arc::new(tls))
        .try_proxy_from_env(true)
        .build())
}

/// Shared agent for Hub / raw.githubusercontent.com requests.
pub fn agent() -> Result<&'static Agent, String> {
    AGENT.get_or_try_init(build_agent)
}
