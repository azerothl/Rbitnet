//! Integration tests: OpenAI-shaped routes expected by Akasha `BitNetProvider`.

use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::create_app;
use http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

/// Serialise all tests that mutate process-wide environment variables.
static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// RAII guard: saves the previous values of a set of env vars, sets new values,
/// and restores them on drop (including on panic).
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
async fn openai_stub_models_and_chat() {
    // Hold the lock while setting env vars and constructing the engine so no
    // other test in this binary can race on the same variables.  The guard
    // ensures vars are restored even if an assertion below panics.
    let _guard = {
        let _lock = ENV_MUTEX.lock().unwrap();
        EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_TOY", None),
            ("RBITNET_STUB", Some("1")),
        ])
        // _lock released here; _guard keeps vars alive until end of test
    };

    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app(Arc::clone(&engine));

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("models response");
    assert!(res.status().is_success());
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "rbitnet-stub");

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "temperature": 0.5,
        "stream": false
    });
    let app = create_app(engine);
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat response");
    assert!(res.status().is_success());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(text.contains("stub"));
}
