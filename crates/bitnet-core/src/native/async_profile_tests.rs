//! Small warmed real-model range for CUDA timeline evidence, not a speed test.
use super::*;
#[test]
fn optional_async_real_gpt_warmed_cuda_profiler_range() {
    if std::env::var("RBITNET_MOE_ASYNC_PROFILE").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_MOE_POLICY_GGUF").unwrap();
    let tok = std::env::var("RBITNET_MOE_POLICY_TOKENIZER").unwrap();
    for (k, v) in [
        ("RBITNET_MAX_SEQ", "2048"),
        ("RBITNET_MOE_CACHE_MB", "512"),
        ("RBITNET_MOE_EXECUTION", "cache"),
        ("RBITNET_MOE_ASYNC", "1"),
        ("RBITNET_MOE_PINNED_SLOTS", "2"),
        ("RBITNET_MOE_PREFETCH", "previous-pass"),
        ("RBITNET_PREFIX_KV", "0"),
        ("RBITNET_CUDA_GPT_FULL", "1"),
        ("RBITNET_REQUIRE_GPT_FULL", "1"),
        ("RBITNET_CUDA_GPT_SEGMENTED", "1"),
        ("RBITNET_CUDA_GPT_FULL_GRAPH", "1"),
        ("RBITNET_CUDA_SPLIT_KV", "1"),
    ] {
        std::env::set_var(k, v);
    }
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    let mut runtime =
        Runtime::load(archive, Path::new(&tok), BackendKind::Cuda, Family::GptOss).unwrap();
    let prompt=format!("<|start|>user<|message|>{}Continue en français.<|end|><|start|>assistant<|channel|>final<|message|>","Le robot lit un livre dans un jardin calme. ".repeat(4));
    let sample = SamplingOptions::from_temperature(0.0);
    let expected = runtime.generate(&prompt, 32, sample, None).unwrap().0;
    let lib =
        unsafe { libloading::Library::new(std::env::var("RBITNET_CUDA_PROFILE_RUNTIME").unwrap()) }
            .unwrap();
    type Profile = unsafe extern "C" fn() -> i32;
    let start = unsafe { *lib.get::<Profile>(b"cudaProfilerStart\0").unwrap() };
    let stop = unsafe { *lib.get::<Profile>(b"cudaProfilerStop\0").unwrap() };
    assert_eq!(unsafe { start() }, 0);
    let result = runtime.generate(&prompt, 32, sample, None);
    assert_eq!(unsafe { stop() }, 0);
    assert_eq!(result.unwrap().0, expected);
    let metrics = runtime.weights.moe_metrics.as_ref().unwrap();
    assert_eq!(
        metrics
            .async_failed
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let copies = metrics
        .layers
        .iter()
        .map(|l| {
            l.prefetch_requested
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum::<u64>();
    assert!(copies > 0);
    if let Ok(path) = std::env::var("RBITNET_CUDA_PROFILE_PROOF") {
        let proof = serde_json::json!({"nonce":std::env::var("RBITNET_CUDA_PROFILE_NONCE").unwrap(),
            "test":"optional_async_real_gpt_warmed_cuda_profiler_range","cache_mib":512,
            "same_output":true,"prefetch_copies":copies,"async_failed":0,"profiler_stop_succeeded":true,
            "reference_output":expected});
        std::fs::write(path, serde_json::to_vec_pretty(&proof).unwrap()).unwrap();
    }
    println!("ASYNC_PROFILE warmed real GPT range completed with identical output; inspect timeline for actual overlap and allocations");
}
