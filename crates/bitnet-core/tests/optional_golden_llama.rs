//! Optional greedy-first-token golden for Llama **or** dense Qwen3.
//!
//! **Environment**
//! - `RBITNET_GOLDEN_JSON` — path to a spec file (see `tests/data/golden/README.md`).
//! - `RBITNET_TEST_GGUF` — path to the same GGUF used to produce the reference.
//! - `RBITNET_TOKENIZER` — `tokenizer.json` or `tokenizer.model` used with that GGUF.
//!
//! Optional JSON field `architecture` (`llama` default, or `qwen3`) selects the runtime.
//! Override with `RBITNET_ARCHITECTURE` when set.
//!
//! If any required variable is unset, the test is skipped (CI default). Hub Qwen3
//! goldens stay optional; default CI covers a synthetic Qwen3 golden in
//! `qwen3_dense_golden_ci.rs` (Refs #25).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::llama::LlamaRuntime;
use bitnet_core::qwen3::Qwen3Runtime;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenSpecV1 {
    /// Must be `rbitnet-golden-v1` when set.
    #[serde(default)]
    format: Option<String>,
    /// Runtime family: `llama` (default) or `qwen3`.
    #[serde(default)]
    architecture: Option<String>,
    prompt: String,
    expected_greedy_first_token: u32,
}

fn resolve_architecture(spec: &GoldenSpecV1) -> String {
    if let Ok(env) = std::env::var("RBITNET_ARCHITECTURE") {
        let t = env.trim();
        if !t.is_empty() {
            return t.to_ascii_lowercase();
        }
    }
    spec.architecture
        .as_deref()
        .unwrap_or("llama")
        .trim()
        .to_ascii_lowercase()
}

#[test]
fn optional_golden_greedy_first_token_matches() {
    let Ok(golden_path) = std::env::var("RBITNET_GOLDEN_JSON") else {
        return;
    };
    let Ok(gguf_path) = std::env::var("RBITNET_TEST_GGUF") else {
        return;
    };
    let Ok(tok_path) = std::env::var("RBITNET_TOKENIZER") else {
        return;
    };

    let gpath = Path::new(&golden_path);
    let gguf = Path::new(&gguf_path);
    let tok = Path::new(&tok_path);
    assert!(gpath.is_file(), "RBITNET_GOLDEN_JSON not a file: {}", gpath.display());
    assert!(gguf.is_file(), "RBITNET_TEST_GGUF not a file: {}", gguf.display());
    assert!(tok.is_file(), "RBITNET_TOKENIZER not a file: {}", tok.display());

    let raw = fs::read_to_string(gpath).expect("read golden json");
    let spec: GoldenSpecV1 = serde_json::from_str(&raw).expect("parse golden json");
    if let Some(f) = &spec.format {
        assert_eq!(
            f, "rbitnet-golden-v1",
            "unknown golden format {f:?}; expected rbitnet-golden-v1"
        );
    }

    let archive = Arc::new(GgufArchive::mmap_path(gguf).expect("mmap gguf"));
    let arch = resolve_architecture(&spec);
    let got = match arch.as_str() {
        "qwen3" => {
            let mut rt = Qwen3Runtime::load(Arc::clone(&archive), tok).expect("Qwen3Runtime::load");
            rt.greedy_next_token_id_after_prompt(&spec.prompt)
                .expect("qwen3 greedy next token")
        }
        "mixtral" => {
            let mut rt =
                bitnet_core::mixtral::MixtralRuntime::load(Arc::clone(&archive), tok)
                    .expect("MixtralRuntime::load");
            rt.greedy_next_token_id_after_prompt(&spec.prompt)
                .expect("mixtral greedy next token")
        }
        "llama" | "mistral" | "qwen2" => {
            let mut rt =
                LlamaRuntime::load(archive, tok, BackendKind::from_env()).expect("LlamaRuntime::load");
            rt.greedy_next_token_id_after_prompt(&spec.prompt)
                .expect("llama greedy next token")
        }
        other => panic!(
            "unsupported golden architecture `{other}` (supported: llama, mistral, qwen2, qwen3, mixtral)"
        ),
    };
    assert_eq!(
        got,
        spec.expected_greedy_first_token,
        "greedy first token id mismatch (see docs/GOLDEN_TESTS.md for exporting reference with llama-cli)"
    );
}
