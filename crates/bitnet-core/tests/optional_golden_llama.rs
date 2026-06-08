//! Optional Llama golden: compares greedy first token id to a reference JSON file.
//!
//! **Environment**
//! - `RBITNET_GOLDEN_JSON` — path to a spec file (see `tests/data/golden/README.md`).
//! - `RBITNET_TEST_GGUF` — path to the same GGUF used to produce the reference.
//! - `RBITNET_TOKENIZER` — `tokenizer.json` or `tokenizer.model` used with that GGUF.
//!
//! If any variable is unset, the test is skipped (CI default).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::llama::LlamaRuntime;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenSpecV1 {
    /// Must be `rbitnet-golden-v1` when set.
    #[serde(default)]
    format: Option<String>,
    prompt: String,
    expected_greedy_first_token: u32,
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
    let mut rt =
        LlamaRuntime::load(archive, tok, BackendKind::from_env()).expect("LlamaRuntime::load");

    let got = rt
        .greedy_next_token_id_after_prompt(&spec.prompt)
        .expect("greedy next token");
    assert_eq!(
        got,
        spec.expected_greedy_first_token,
        "greedy first token id mismatch (see docs/GOLDEN_TESTS.md for exporting reference with llama-cli)"
    );
}
