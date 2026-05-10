//! Stub binary documenting the future runner subprocess contract.
//!
//! The parent proxy is not implemented yet. This binary exists so release packages and docs can
//! pin the child-process interface before supervision and HTTP forwarding land.

use std::env;

fn main() {
    let model = env::var("RBITNET_MODEL").unwrap_or_else(|_| "<unset>".into());
    let bind = env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:0".into());
    eprintln!("rbitnet-runner stub");
    eprintln!("contract:");
    eprintln!("  RBITNET_MODEL={model}");
    eprintln!("  RBITNET_BIND={bind}");
    eprintln!("planned:");
    eprintln!("  child serves /health, /ready, /v1/chat/completions on RBITNET_BIND");
    eprintln!("  parent proxy starts/stops this process per model id");
    std::process::exit(64);
}
