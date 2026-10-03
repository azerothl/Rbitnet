//! Opt-in warmed real GPT capture for proving multi-token CUDA launches.
use bitnet_core::inference::Engine;
use std::path::Path;

#[test]
fn optional_gpt_block_warmed_cuda_profile() {
    if std::env::var("RBITNET_GPT_BLOCK_PROFILE").as_deref() != Ok("1") {
        return;
    }
    for (key, value) in [
        ("RBITNET_BACKEND", "cuda"),
        ("RBITNET_MAX_SEQ", "2048"),
        ("RBITNET_MOE_CACHE_MB", "0"),
        ("RBITNET_MOE_EXECUTION", "cache"),
        ("RBITNET_CUDA_GPT_FULL", "1"),
        ("RBITNET_REQUIRE_GPT_FULL", "1"),
        ("RBITNET_CUDA_GPT_SEGMENTED", "0"),
        ("RBITNET_CUDA_GPT_PREFILL", "1"),
        ("RBITNET_REQUIRE_GPT_PREFILL", "1"),
        ("RBITNET_CUDA_GPT_PREFILL_TOKENS", "32"),
        ("RBITNET_CUDA_GPT_PREFILL_TILE", "0"),
        ("RBITNET_CUDA_GPT_FULL_GRAPH", "1"),
        ("RBITNET_CUDA_SPLIT_KV", "1"),
        ("RBITNET_PREFIX_KV", "0"),
    ] {
        std::env::set_var(key, value);
    }
    let gguf = std::env::var("RBITNET_GPT_BLOCK_GGUF").unwrap();
    let tok = std::env::var("RBITNET_GPT_BLOCK_TOKENIZER").unwrap();
    let engine =
        Engine::load_path_with_overrides(Path::new(&gguf), Some(Path::new(&tok)), Some("gptoss"))
            .unwrap();
    let prompt = format!(
        "<|start|>user<|message|>{}Continue en français.<|end|><|start|>assistant<|channel|>final<|message|>",
        "Le robot lit un livre dans un jardin calme. ".repeat(4)
    );
    let expected = engine.complete(&prompt, 32, 0.0).unwrap();
    let before = bitnet_core::perf::snapshot().gpu_prefill_blocks;
    let library =
        unsafe { libloading::Library::new(std::env::var("RBITNET_CUDA_PROFILE_RUNTIME").unwrap()) }
            .unwrap();
    type Profile = unsafe extern "C" fn() -> i32;
    let start = unsafe { *library.get::<Profile>(b"cudaProfilerStart\0").unwrap() };
    let stop = unsafe { *library.get::<Profile>(b"cudaProfilerStop\0").unwrap() };
    assert_eq!(unsafe { start() }, 0);
    let result = engine.complete(&prompt, 32, 0.0);
    assert_eq!(unsafe { stop() }, 0);
    assert_eq!(result.unwrap(), expected);
    let blocks = bitnet_core::perf::snapshot().gpu_prefill_blocks - before;
    assert!(blocks > 1);
    let proof = serde_json::json!({
        "nonce":std::env::var("RBITNET_CUDA_PROFILE_NONCE").unwrap(),
        "same_output":true,"completed_prefill_blocks":blocks,
        "block_capacity":32,"tile":0,"profiler_stop_succeeded":true,
        "reference_output":expected,
    });
    std::fs::write(
        std::env::var("RBITNET_CUDA_PROFILE_PROOF").unwrap(),
        serde_json::to_vec_pretty(&proof).unwrap(),
    )
    .unwrap();
}
