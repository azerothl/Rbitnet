//! One-shot completion (`Engine::complete_detailed`) using `RBITNET_*` env (same as the server).
//!
//! ```text
//! set RBITNET_MODEL=C:\\path\\model.gguf
//! set RBITNET_TOKENIZER=C:\\path\\tokenizer.json
//! set RBITNET_SMOKE_MAX_TOKENS=1
//! cargo run -p bitnet-core --example engine_smoke --release
//! ```
//! Optional: **`RBITNET_SMOKE_MAX_TOKENS`** (default `8`) caps generated tokens for long CPU runs.

use bitnet_core::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::from_env()?;
    let max_tokens = std::env::var("RBITNET_SMOKE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(8)
        .max(1);
    let out = engine.complete_detailed("Say OK in one word.", max_tokens, 0.0)?;
    println!("{}", out.text);
    println!(
        "ttft_ms={} encode_ms={} prefill_ms={} decode_ms={} itl_us={} prompt_tokens={} completion_tokens={}",
        out.stats.ttft_ms,
        out.stats.encode_ms,
        out.stats.prefill_ms,
        out.stats.decode_ms,
        out.stats.itl_us,
        out.stats.prompt_tokens,
        out.stats.completion_tokens
    );
    Ok(())
}
