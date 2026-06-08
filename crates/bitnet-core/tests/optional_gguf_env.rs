//! Optional smoke test when `RBITNET_TEST_GGUF` points to a GGUF on disk (e.g. BitNet-converted
//! `1bitLLM/bitnet_b1_58-large`). See `docs/MODEL_TESTING.md`.

use std::path::Path;

use bitnet_core::{Engine, GgufArchive};

#[test]
fn optional_gguf_from_env_smoke() {
    let Ok(path) = std::env::var("RBITNET_TEST_GGUF") else {
        // No local GGUF configured; skip heavy I/O in CI and default dev workflows.
        return;
    };
    let p = Path::new(&path);
    assert!(
        p.exists() && p.is_file(),
        "RBITNET_TEST_GGUF path does not exist or is not a file: {}",
        p.display()
    );
    let g = GgufArchive::mmap_path(p).expect("parse GGUF");
    assert!(g.tensor_count() > 0, "expected at least one tensor");
    assert!(
        !g.tensor_data().is_empty(),
        "tensor data region should be non-empty"
    );
    let _hp = g.llama_hyper_params();
}

#[test]
fn optional_engine_load_from_env_smoke() {
    let Ok(path) = std::env::var("RBITNET_TEST_GGUF") else {
        return;
    };
    let p = Path::new(&path);
    assert!(
        p.exists() && p.is_file(),
        "RBITNET_TEST_GGUF path does not exist or is not a file: {}",
        p.display()
    );
    let dir = match p.parent() {
        Some(d) => d,
        None => return,
    };
    let has_tok = dir.join("tokenizer.json").is_file() || dir.join("tokenizer.model").is_file();
    if !has_tok {
        return;
    }
    let engine = Engine::load_path(p).expect("Engine::load_path with tokenizer beside GGUF");
    assert!(engine.has_gguf());
}
