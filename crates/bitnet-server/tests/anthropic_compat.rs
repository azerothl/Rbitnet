//! Smoke tests for the Anthropic Messages subset (`POST /v1/messages`).
#![allow(clippy::await_holding_lock)]

use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::{
    anthropic_messages_to_prompt, create_app_with_config, AnthropicMessage, ServerConfig,
};
use http::Request;
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

/// Serialise tests that mutate process-wide environment variables.
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

#[test]
fn anthropic_prompt_conversion_string_and_blocks() {
    let messages = [
        AnthropicMessage {
            role: "user".into(),
            content: json!("ping"),
        },
        AnthropicMessage {
            role: "user".into(),
            content: json!([{ "type": "text", "text": "pong" }]),
        },
    ];
    assert_eq!(
        anthropic_messages_to_prompt(&messages),
        "user: ping\nuser: pong"
    );
}

#[tokio::test]
async fn anthropic_stub_messages_non_stream() {
    let (engine, _guard) = {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_TOY", None),
            ("RBITNET_STUB", Some("1")),
        ]);
        let engine = Arc::new(Engine::from_env().expect("engine"));
        (engine, guard)
    };
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let body = json!({
        "model": "rbitnet-stub",
        "max_tokens": 32,
        "temperature": 0.5,
        "messages": [{ "role": "user", "content": "hello" }]
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("messages response");
    assert!(res.status().is_success());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "rbitnet-stub");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["content"][0]["type"], "text");
    let text = v["content"][0]["text"].as_str().expect("text");
    assert!(text.contains("stub"), "unexpected stub text: {text}");
    assert!(v["usage"]["input_tokens"].as_u64().is_some());
    assert!(v["usage"]["output_tokens"].as_u64().is_some());
}

#[tokio::test]
async fn anthropic_stub_messages_minimal_stream() {
    let (engine, _guard) = {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_TOY", None),
            ("RBITNET_STUB", Some("1")),
        ]);
        let engine = Arc::new(Engine::from_env().expect("engine"));
        (engine, guard)
    };
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let body = json!({
        "model": "rbitnet-stub",
        "max_tokens": 32,
        "stream": true,
        "messages": [{ "role": "user", "content": "hello" }]
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("stream response");
    assert!(res.status().is_success());
    let ct = res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"), "ct={ct}");
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("event: message_start"));
    assert!(text.contains("event: content_block_delta"));
    assert!(text.contains("text_delta"));
    assert!(text.contains("event: message_stop"));
    assert!(text.contains("stub"), "unexpected stream body: {text}");
}
