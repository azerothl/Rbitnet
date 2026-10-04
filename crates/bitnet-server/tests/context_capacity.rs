//! Real tiny GGUF forward checks for configured capacity and HTTP error timing.
#![allow(clippy::await_holding_lock)]
use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::{create_app_with_config, ServerConfig};
use http::Request;
use http_body_util::BodyExt;
use std::sync::{Arc, LazyLock, Mutex};
use tower::ServiceExt;
static ENV: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
struct Guard(Vec<(String, Option<std::ffi::OsString>)>);
impl Guard {
    fn set(items: &[(&str, Option<&str>)]) -> Self {
        Self(
            items
                .iter()
                .map(|&(key, value)| {
                    let old = std::env::var_os(key);
                    if let Some(value) = value {
                        std::env::set_var(key, value);
                    } else {
                        std::env::remove_var(key);
                    }
                    (key.into(), old)
                })
                .collect(),
        )
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            if let Some(v) = v {
                std::env::set_var(k, v);
            } else {
                std::env::remove_var(k);
            }
        }
    }
}
fn fixture(path: &std::path::Path) {
    fn text(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    let tensors: [(&str, &[u64]); 12] = [
        ("token_embd.weight", &[8, 8]),
        ("output_norm.weight", &[8]),
        ("output.weight", &[8, 8]),
        ("blk.0.attn_norm.weight", &[8]),
        ("blk.0.attn_q.weight", &[8, 8]),
        ("blk.0.attn_k.weight", &[8, 4]),
        ("blk.0.attn_v.weight", &[8, 4]),
        ("blk.0.attn_output.weight", &[8, 8]),
        ("blk.0.ffn_norm.weight", &[8]),
        ("blk.0.ffn_gate.weight", &[8, 16]),
        ("blk.0.ffn_up.weight", &[8, 16]),
        ("blk.0.ffn_down.weight", &[16, 8]),
    ];
    let integers = [
        ("llama.embedding_length", 8u32),
        ("llama.vocab_size", 8),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1),
        ("llama.feed_forward_length", 16),
        ("llama.context_length", 128),
        ("general.alignment", 32),
    ];
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((integers.len() as u64 + 2).to_le_bytes());
    for (key, value) in [
        ("general.architecture", "llama"),
        ("general.name", "capacity-fixture"),
    ] {
        text(&mut out, key);
        out.extend(8u32.to_le_bytes());
        text(&mut out, value);
    }
    for (key, value) in integers {
        text(&mut out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    let mut offset = 0u64;
    for (name, dims) in tensors {
        text(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(0u32.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += dims.iter().product::<u64>() * 4;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for (index, (name, dims)) in tensors.into_iter().enumerate() {
        for i in 0..dims.iter().product::<u64>() as usize {
            let value = if name.contains("norm") {
                1.0f32
            } else if name == "token_embd.weight" {
                // Every input has a positive, nonzero hidden vector. Only the
                // visible Hello row of the head receives a positive score.
                (i % 8 + 1) as f32 / 32.
            } else if name == "output.weight" {
                if i / 8 == 1 {
                    (i % 8 + 1) as f32 / 32.
                } else {
                    0.
                }
            } else {
                ((i * 7 + index * 13) % 31) as f32 / 512. - 0.025
            };
            out.extend(value.to_le_bytes());
        }
    }
    std::fs::write(path, out).unwrap();
}
fn request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
#[tokio::test]
async fn real_llama_context_is_frozen_before_lazy_load_and_rejects_before_sse() {
    let _lock = ENV.lock().unwrap();
    let _guard = Guard::set(&[
        ("RBITNET_STUB", None),
        ("RBITNET_TOY", None),
        ("RBITNET_BACKEND", Some("cpu")),
        ("RBITNET_MAX_SEQ", Some("16")),
        ("RBITNET_LLAMA_WEIGHT_MODE", Some("dense")),
        ("RBITNET_PREFIX_KV", Some("0")),
        ("RBITNET_CHAT_FORMAT", Some("raw")),
        ("RBITNET_CHAT_TEMPLATE", None),
        ("RBITNET_LLAMA_ENCODE_ADD_SPECIAL", Some("1")),
        ("RBITNET_MEMORY_BUDGET_MB", None),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("model.gguf");
    let tok = dir.path().join("tokenizer.json");
    fixture(&gguf);
    bitnet_core::mixtral::ci_fixture::write_wordlevel_tokenizer(&tok).unwrap();
    let engine =
        Arc::new(Engine::load_path_with_overrides(&gguf, Some(&tok), Some("llama")).unwrap());
    assert_eq!(engine.context_capacity(), Some(16));
    assert_eq!(engine.model_metadata().context_length, Some(128));
    let frozen_count = engine.count_prompt_tokens("Hello").unwrap();
    let backup = dir.path().join("tokenizer.original.json");
    std::fs::rename(&tok, &backup).unwrap();
    assert_eq!(
        engine.count_prompt_tokens("Hello").unwrap(),
        frozen_count,
        "loaded tokenizer remains available after its file moves"
    );
    // Trigger first weight/KV allocation after the environment changes. The
    // already constructed model must retain the capacity reported by its API.
    std::env::set_var("RBITNET_MAX_SEQ", "4");
    let output = engine.complete("Hello", 15, 0.).unwrap();
    assert!(!output.is_empty());
    assert_eq!(engine.context_capacity(), Some(16));
    let error = engine.complete("Hello", 16, 0.).unwrap_err();
    assert_eq!(error.http_status_for_chat_completion(), 400);
    let prompt = "user: Hello";
    let prompt_tokens = engine.count_prompt_tokens(prompt).unwrap();
    assert!(prompt_tokens < 16);
    let app = create_app_with_config(Arc::clone(&engine), Arc::new(ServerConfig::test_defaults()));
    for stream in [false, true] {
        let response=app.clone().oneshot(request("/v1/chat/completions",serde_json::json!({
            "messages":[{"role":"user","content":"Hello"}],"max_tokens":17-prompt_tokens,"stream":stream,
        }))).await.unwrap();
        assert_eq!(response.status(), 400);
        assert!(response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("application/json"));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("capacity 16"));
    }
    // Exact full capacity is allowed, and an over-limit request cannot poison
    // subsequent real inference in either non-streamed or streamed mode.
    for stream in [false, true] {
        let response=app.clone().oneshot(request("/v1/chat/completions",serde_json::json!({
            "messages":[{"role":"user","content":"Hello"}],"max_tokens":16-prompt_tokens,"stream":stream,
        }))).await.unwrap();
        assert_eq!(response.status(), 200);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        if stream {
            assert!(std::str::from_utf8(&body).unwrap().contains("data: [DONE]"));
        } else {
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(json["choices"][0]["message"]["content"].is_string());
        }
    }
    for (path, body) in [
        (
            "/v1/completions",
            serde_json::json!({"prompt":"Hello","max_tokens":17,"stream":false}),
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt":"Hello","max_tokens":17,"stream":true}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"model":"context-fixture","messages":[{"role":"user","content":"Hello"}],"max_tokens":17}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"model":"context-fixture","messages":[{"role":"user","content":"Hello"}],"max_tokens":17,"stream":true}),
        ),
    ] {
        assert_eq!(
            app.clone()
                .oneshot(request(path, body))
                .await
                .unwrap()
                .status(),
            400
        );
    }
    // A fresh model sees the new configuration; existing model metadata stays
    // consistent. Invalid values fail at load, before a model is advertised ready.
    std::fs::write(&tok, b"invalid tokenizer for a new load").unwrap();
    assert!(Engine::load_path_with_overrides(&gguf, Some(&tok), Some("llama")).is_err());
    assert_eq!(engine.count_prompt_tokens("Hello").unwrap(), frozen_count);
    std::fs::remove_file(&tok).unwrap();
    std::fs::rename(&backup, &tok).unwrap();
    let next = Engine::load_path_with_overrides(&gguf, Some(&tok), Some("llama")).unwrap();
    assert_eq!(next.context_capacity(), Some(4));
    assert_eq!(engine.context_capacity(), Some(16));
    for invalid in ["0", "bad", "-1"] {
        std::env::set_var("RBITNET_MAX_SEQ", invalid);
        assert!(Engine::load_path_with_overrides(&gguf, Some(&tok), Some("llama")).is_err());
    }
}
