//! Mixtral MoE e2e via OpenAI `/v1/chat/completions` (Refs #25).
#![allow(clippy::await_holding_lock)]

use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_core::mixtral::ci_fixture::{write_tiny_mixtral_gguf, write_wordlevel_tokenizer};
use bitnet_server::{create_app_with_expected_model, ServerConfig};
use http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct EnvGuard(Vec<(String, Option<String>)>);

impl EnvGuard {
    fn set(pairs: &[(&str, Option<&str>)]) -> Self {
        let saved = pairs
            .iter()
            .map(|&(k, v)| {
                let prev = std::env::var(k).ok();
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
                (k.to_string(), prev)
            })
            .collect();
        Self(saved)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            match v {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }
}

#[tokio::test]
async fn mixtral_moe_v1_chat_completions_e2e() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("mixtral-tiny.gguf");
    let tok = dir.path().join("tokenizer.json");
    write_tiny_mixtral_gguf(&gguf).expect("gguf");
    write_wordlevel_tokenizer(&tok).expect("tok");

    let (app, _guard) = {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::set(&[
            ("RBITNET_STUB", None),
            ("RBITNET_TOY", None),
            ("RBITNET_MODEL", None),
            ("RBITNET_BACKEND", Some("cpu")),
            ("RBITNET_CHAT_FORMAT", Some("raw")),
        ]);
        let engine = Arc::new(
            Engine::load_path_with_overrides(&gguf, Some(&tok), Some("mixtral"))
                .expect("load mixtral engine"),
        );
        let config = Arc::new(ServerConfig::from_env().expect("config"));
        let (app, _state) =
            create_app_with_expected_model(engine, config, Some("rbitnet-mixtral".into()));
        (app, guard)
    };

    let body = serde_json::json!({
        "model": "rbitnet-mixtral",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1,
        "temperature": 0.0
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), 200, "Mixtral MoE /v1 chat should succeed");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(v["object"], "chat.completion");
    assert!(
        v["choices"][0]["message"]["content"].is_string(),
        "expected message content: {v}"
    );
}
