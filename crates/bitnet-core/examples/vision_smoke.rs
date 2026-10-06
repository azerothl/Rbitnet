//! End-to-end vision smoke with peak-RSS split:
//! 1) encode image with mmproj, then drop the encoder
//! 2) load Llama and inject precomputed patches
//!
//! ```bash
//! RBITNET_MAX_SEQ=1024 RBITNET_BACKEND=cpu \
//! cargo run -p bitnet-core --release --example vision_smoke -- /path/image.png
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use bitnet_core::inference::Engine;
use bitnet_core::mmproj::{MmprojEncoder, IMAGE_PLACEHOLDER};
use bitnet_core::sampling::SamplingOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/rbitnet-vision/red.png"));
    let model = env::var("RBITNET_MODEL")
        .unwrap_or_else(|_| "/tmp/rbitnet-vision/llava-v1.5-7b-Q2_K.gguf".into());
    let mmproj = env::var("RBITNET_MMPROJ")
        .unwrap_or_else(|_| "/tmp/rbitnet-vision/mmproj-model-f16.gguf".into());
    let tokenizer = env::var("RBITNET_TOKENIZER")
        .unwrap_or_else(|_| "/tmp/rbitnet-vision/tokenizer.json".into());

    if env::var_os("RBITNET_MAX_SEQ").is_none() {
        std::env::set_var("RBITNET_MAX_SEQ", "1024");
    }
    std::env::set_var("RBITNET_BACKEND", "cpu");
    // Avoid attaching mmproj to the Llama executor (keeps peak RSS lower).
    std::env::remove_var("RBITNET_MMPROJ");

    eprintln!("encoding image with mmproj={}", mmproj);
    let t0 = Instant::now();
    let enc = MmprojEncoder::load(PathBuf::from(&mmproj).as_path())?;
    let bytes = std::fs::read(&image)?;
    let patches = enc.encode_image_bytes(&bytes)?;
    let n_patches = enc.n_patches();
    let proj_out = enc.proj_out_dim();
    drop(enc);
    eprintln!(
        "encoded in {:?}; n_patches={n_patches} proj_out={proj_out} patches_bytes={}",
        t0.elapsed(),
        patches.len() * 4
    );

    eprintln!("loading Llama engine model={model}");
    let t1 = Instant::now();
    let engine = Engine::load_path_with_overrides(
        PathBuf::from(&model).as_path(),
        Some(PathBuf::from(&tokenizer).as_path()),
        None,
    )?;
    let meta = engine.model_metadata();
    eprintln!(
        "loaded in {:?}; supports_vision={} context_capacity={:?} context_length={:?} (expected supports_vision=false)",
        t1.elapsed(),
        engine.supports_vision(),
        meta.context_capacity,
        meta.context_length
    );

    let prompt = format!(
        "USER: {IMAGE_PLACEHOLDER}\nWhat primary color is dominant in this image? Answer with one word.\nASSISTANT:"
    );
    let sampling = SamplingOptions::from_temperature(0.0);
    let max_tokens = env::var("RBITNET_VISION_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6u32);

    eprintln!("complete_with_vision_patches max_tokens={max_tokens}");
    let t2 = Instant::now();
    let out =
        engine.complete_with_vision_patches(&prompt, &patches, n_patches, max_tokens, sampling)?;
    eprintln!(
        "done in {:?}; prompt_tokens={} completion_tokens={}",
        t2.elapsed(),
        out.stats.prompt_tokens,
        out.stats.completion_tokens
    );
    println!("{}", out.text);
    Ok(())
}
