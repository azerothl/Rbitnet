//! Default-CI Mixtral MoE golden + Engine smoke (Refs #25).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use bitnet_core::gguf::GgufArchive;
use bitnet_core::inference::Engine;
use bitnet_core::mixtral::ci_fixture::{write_tiny_mixtral_gguf, write_wordlevel_tokenizer};
use bitnet_core::mixtral::MixtralRuntime;
use bitnet_core::sampling::SamplingOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenSpecV1 {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    prompt: String,
    expected_greedy_first_token: u32,
}

#[test]
fn mixtral_moe_synthetic_greedy_matches_checked_in_golden() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/data/golden/mixtral-moe-synthetic.golden.json");
    let raw = fs::read_to_string(&root).expect("checked-in mixtral synthetic golden");
    let spec: GoldenSpecV1 = serde_json::from_str(&raw).expect("parse golden");
    assert_eq!(spec.format.as_deref(), Some("rbitnet-golden-v1"));
    assert_eq!(
        spec.architecture.as_deref().unwrap_or("mixtral"),
        "mixtral",
        "synthetic CI golden must target mixtral"
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("mixtral-tiny.gguf");
    let tok = dir.path().join("tokenizer.json");
    write_tiny_mixtral_gguf(&gguf).expect("write tiny mixtral gguf");
    write_wordlevel_tokenizer(&tok).expect("write tokenizer");

    let archive = Arc::new(GgufArchive::mmap_path(&gguf).expect("mmap"));
    let mut rt = MixtralRuntime::load(archive, &tok).expect("MixtralRuntime::load");
    let got = rt
        .greedy_next_token_id_after_prompt(&spec.prompt)
        .expect("greedy");
    assert_eq!(
        got, spec.expected_greedy_first_token,
        "Mixtral MoE synthetic golden mismatch (see docs/GOLDEN_TESTS.md)"
    );
}

#[test]
fn mixtral_moe_engine_complete_one_token() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("mixtral-tiny.gguf");
    let tok = dir.path().join("tokenizer.json");
    write_tiny_mixtral_gguf(&gguf).expect("write tiny mixtral gguf");
    write_wordlevel_tokenizer(&tok).expect("write tokenizer");

    let engine = Engine::load_path_with_overrides(&gguf, Some(&tok), Some("mixtral"))
        .expect("Engine::load_path_with_overrides mixtral");
    assert_eq!(engine.model_metadata().architecture, "mixtral");
    assert!(
        !engine.model_metadata().backend_accelerated,
        "CPU-only Mixtral must not claim GPU work"
    );
    assert!(engine.is_ready());
    let out = engine
        .complete_detailed_with_options("Hello", 1, SamplingOptions::from_temperature(0.0))
        .expect("complete");
    assert_eq!(out.stats.prompt_tokens, 1);
    assert!(out.stats.completion_tokens <= 1);
}

#[test]
fn mixtral_hybrid_is_cpu_and_unsupported_gpu_requests_fail_during_load() {
    use bitnet_core::backend::BackendKind;
    use bitnet_core::loaders::dispatch_gguf_executor_for_load;
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("mixtral-tiny.gguf");
    let tok = dir.path().join("tokenizer.json");
    write_tiny_mixtral_gguf(&gguf).unwrap();
    write_wordlevel_tokenizer(&tok).unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(&gguf).unwrap());
    let load = |backend| {
        dispatch_gguf_executor_for_load(
            backend,
            Arc::clone(&archive),
            &gguf,
            Some(&tok),
            Some("mixtral"),
        )
    };
    for backend in [
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Vulkan,
        BackendKind::Metal,
    ] {
        let error = match load(backend) {
            Ok(_) => panic!("unsupported GPU request loaded"),
            Err(e) => e,
        };
        assert!(
            error.to_string().contains("unsupported")
                || error.to_string().contains("not implemented"),
            "{error}"
        );
    }
    let hybrid = load(BackendKind::Hybrid).unwrap();
    assert!(hybrid.is_ready());
    assert!(!hybrid.backend_accelerated());
    assert!(hybrid.offload_metadata().unwrap().contains("CPU fallback"));
    assert!(!hybrid
        .generate_with_timings("Hello", 1, SamplingOptions::from_temperature(0.0))
        .unwrap()
        .0
        .is_empty());
    std::fs::remove_file(&tok).unwrap();
    assert!(
        load(BackendKind::Cpu).is_err(),
        "readiness must validate the tokenizer during load"
    );
}
