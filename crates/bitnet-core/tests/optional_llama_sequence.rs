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
