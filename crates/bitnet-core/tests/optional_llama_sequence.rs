//! Real-model greedy sequence and turn-stop regression against a pinned llama.cpp reference.
//! Set RBITNET_LLAMA_SEQUENCE_JSON, RBITNET_TEST_GGUF and RBITNET_TOKENIZER to run.
//! Optionally set RBITNET_SEQUENCE_BACKEND=cuda to exercise native kernels on hardware.

use std::{fs, path::Path, sync::Arc};

use bitnet_core::{backend::BackendKind, gguf::GgufArchive, llama::LlamaRuntime};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Reference {
    format: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    text: String,
}

#[test]
fn optional_resident_prefix_reuse_matches_cold_generations_after_divergence_and_eviction() {
    if std::env::var("RBITNET_CUDA_PREFIX_TEST").as_deref() != Ok("1") {
        return;
    }
    assert!(matches!(
        std::env::var("RBITNET_PREFIX_KV").as_deref(),
        Ok("1" | "true" | "yes")
    ));
    let gguf = std::env::var("RBITNET_TEST_GGUF").expect("real GGUF required");
    let tokenizer = std::env::var("RBITNET_TOKENIZER").expect("matching tokenizer required");
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    let common = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise assistant. Read each question carefully and provide a short factual answer. All questions refer to ordinary geography and arithmetic. Keep punctuation and accents when needed.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n";
    let prompt = |question| {
        format!("{common}{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    };
    let prompts = [prompt("What is the capital of France?"), prompt("What is the capital of Italy?"),
        prompt("What is the capital of France?"), prompt("Compute 13 plus 29."),
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSay bonjour.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        prompt("What is the capital of France?")];
    let mut sampled = bitnet_core::sampling::SamplingOptions::from_temperature(0.7);
    sampled.seed = Some(735);
    sampled.top_p = Some(0.9);
    let mut penalized = sampled;
    penalized.frequency_penalty = 0.25;
    penalized.presence_penalty = 0.15;
    let before = bitnet_core::perf::snapshot().prefix_cache_hits;
    for options in [
        bitnet_core::sampling::SamplingOptions::from_temperature(0.0),
        sampled,
        penalized,
    ] {
        let mut warm =
            LlamaRuntime::load(archive.clone(), Path::new(&tokenizer), BackendKind::Cuda).unwrap();
        assert!(
            warm.uses_resident_cuda(),
            "prefix flag must retain resident CUDA"
        );
        for prompt in &prompts {
            let mut cold =
                LlamaRuntime::load(archive.clone(), Path::new(&tokenizer), BackendKind::Cuda)
                    .unwrap();
            assert!(cold.uses_resident_cuda());
            let expected = cold.generate_with_timings(prompt, 24, options).unwrap();
            let actual = warm.generate_with_timings(prompt, 24, options).unwrap();
            assert_eq!(actual.0, expected.0, "warm/cold divergence for {options:?}");
            assert_eq!(actual.1.completion_tokens, expected.1.completion_tokens);
            assert!(!actual.0.is_empty());
            assert!(warm.uses_resident_cuda());
        }
        let expected = warm
            .generate_with_timings(&prompts[0], 24, options)
            .unwrap()
            .0;
        let mut emitted = false;
        let cancelled = warm.generate_streaming(&prompts[0], 24, options, &mut |event| {
            if matches!(event, bitnet_core::stream::StreamEvent::Delta { .. }) {
                emitted = true;
                bitnet_core::request_inference_cancel();
            }
            Ok(())
        });
        // Clear the process flag before assertions, including a failed cancel.
        bitnet_core::clear_inference_cancel();
        assert!(emitted && cancelled.unwrap_err().to_string().contains("cancelled"));
        assert_eq!(
            warm.generate_with_timings(&prompts[0], 24, options)
                .unwrap()
                .0,
            expected
        );
    }
    assert!(
        bitnet_core::perf::snapshot().prefix_cache_hits > before,
        "must actually restore device KV"
    );
}

#[test]
fn optional_llama_greedy_sequence_and_turn_stop_match_reference() {
    let (Ok(spec_path), Ok(gguf_path), Ok(tokenizer_path)) = (
        std::env::var("RBITNET_LLAMA_SEQUENCE_JSON"),
        std::env::var("RBITNET_TEST_GGUF"),
        std::env::var("RBITNET_TOKENIZER"),
    ) else {
        return;
    };
    let spec: Reference = serde_json::from_str(&fs::read_to_string(spec_path).unwrap()).unwrap();
    assert_eq!(spec.format, "rbitnet-llama-sequence-v1");
    assert!(!spec.cases.is_empty());
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf_path)).unwrap());
    let backend = match std::env::var("RBITNET_SEQUENCE_BACKEND").as_deref() {
        Ok("cuda") => BackendKind::Cuda,
        Ok("hybrid") => BackendKind::Hybrid,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("unsupported RBITNET_SEQUENCE_BACKEND: {other}"),
    };
    for case in spec.cases {
        // The fixture is exported from llama.cpp /tokenize and /completion, including EOT.
        assert!(
            !case.greedy_ids.is_empty(),
            "{}: missing reference sequence",
            case.name
        );
        let encoded = tokenizer.encode(case.prompt.as_str(), false).unwrap();
        assert_eq!(
            encoded.get_ids(),
            case.prompt_ids,
            "{}: tokenizer mismatch",
            case.name
        );
        let mut runtime =
            LlamaRuntime::load(Arc::clone(&archive), Path::new(&tokenizer_path), backend).unwrap();
        if std::env::var("RBITNET_REQUIRE_RESIDENT").as_deref() == Ok("1") {
            assert!(
                runtime.uses_resident_cuda(),
                "{}: CUDA resident path unavailable",
                case.name
            );
        }
        let mut logits = runtime.prefill_chunk(&case.prompt_ids, 0).unwrap();
        for (step, expected) in case.greedy_ids.iter().enumerate() {
            let got = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            assert_eq!(got, *expected, "{}: greedy token {step}", case.name);
            if step + 1 < case.greedy_ids.len() {
                logits = runtime
                    .decode_one(got, case.prompt_ids.len() + step)
                    .unwrap();
            }
        }
        // Also exercise the runtime's own BOS policy and stop handling, with room to overrun.
        let (text, stats) = runtime
            .generate_with_timings(
                &case.prompt,
                case.greedy_ids.len() as u32 + 8,
                bitnet_core::sampling::SamplingOptions::from_temperature(0.0),
            )
            .unwrap();
        assert_eq!(text, case.text, "{}: completion/stop mismatch", case.name);
        assert_eq!(
            stats.prompt_tokens as usize,
            case.prompt_ids.len(),
            "{}: duplicate BOS",
            case.name
        );
        assert_eq!(
            stats.completion_tokens as usize + 1,
            case.greedy_ids.len(),
            "{}: stop token counted as content",
            case.name
        );
    }
}
