//! CPU smoke for `<image>` patch injection using TinyLlama + synthetic patches.
//!
//! Proves the Llama prefill fusion path without a 7B resident set. Real mmproj
//! encode is covered by `MmprojEncoder` unit/integration tests.

use std::path::Path;

use bitnet_core::inference::Engine;
use bitnet_core::mmproj::IMAGE_PLACEHOLDER;
use bitnet_core::sampling::SamplingOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("RBITNET_BACKEND", "cpu");
    std::env::set_var("RBITNET_MAX_SEQ", "512");
    std::env::set_var("RBITNET_WEIGHT_MODE", "mmap_quant");
    std::env::remove_var("RBITNET_MMPROJ");

    let model = std::env::var("RBITNET_MODEL").unwrap_or_else(|_| {
        "/tmp/rbitnet-models/tinyllama/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf".into()
    });
    let tokenizer = std::env::var("RBITNET_TOKENIZER").unwrap_or_else(|_| {
        let p = Path::new(&model).parent().unwrap().join("tokenizer.json");
        p.display().to_string()
    });

    let eng = Engine::load_path_with_overrides(Path::new(&model), Some(Path::new(&tokenizer)), None)?;
    let n_embd = eng
        .model_metadata()
        .context_capacity
        .map(|_| 2048usize) // TinyLlama; overridden below from a cheap probe
        .unwrap_or(2048);
    // Prefer reading from a short generate that doesn't need patches dims from metadata —
    // TinyLlama n_embd is 2048.
    let n_embd = std::env::var("RBITNET_N_EMBD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_embd);
    let n_patches = 8usize;
    // Non-zero deterministic patches so the path isn't a silent no-op.
    let mut patches = vec![0.0f32; n_patches * n_embd];
    for (i, v) in patches.iter_mut().enumerate() {
        *v = ((i % 97) as f32) * 0.001 - 0.05;
    }

    let prompt = format!(
        "USER: {IMAGE_PLACEHOLDER}\nReply with the single word OK.\nASSISTANT:"
    );
    let out = eng.complete_with_vision_patches(
        &prompt,
        &patches,
        n_patches,
        8,
        SamplingOptions::from_temperature(0.0),
    )?;
    eprintln!(
        "inject_smoke ok prompt_tokens={} completion_tokens={} text={:?}",
        out.stats.prompt_tokens, out.stats.completion_tokens, out.text
    );
    if out.stats.prompt_tokens < n_patches as u32 {
        return Err("prompt_tokens did not expand by n_patches".into());
    }
    println!("{}", out.text);
    Ok(())
}
