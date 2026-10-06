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

#[tokio::test]
async fn tool_requests_and_tool_blocks_refuse_before_streaming_or_inference() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[("RBITNET_MODEL", None), ("RBITNET_TOY", None), ("RBITNET_STUB", Some("1")), ("RBITNET_STRUCTURED_OUTPUT", None)]);
    let app = create_app_with_config(Arc::new(Engine::from_env().unwrap()), Arc::new(ServerConfig::test_defaults()));
    for fields in [json!({"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}), json!({"tool_choice":{"type":"any"}}), json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"lookup","input":{}}]}]}), json!({"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"result"}]}]})] {
        for streaming in [false, true] {
            let mut body = json!({"model":"any", "messages":[{"role":"user","content":"hello"}], "max_tokens":8, "stream":streaming});
            body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
            let response = app.clone().oneshot(Request::builder().method("POST").uri("/v1/messages").header("content-type","application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), http::StatusCode::NOT_IMPLEMENTED);
            assert!(response.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
            let body: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
            assert_eq!(body["error"]["code"], "tool_generation_not_supported");
        }
    }
    let response = app.oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap()).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(std::str::from_utf8(&bytes).unwrap().lines().any(|line|line == "rbitnet_inference_calls_total 0"));
}

#[tokio::test]
async fn image_blocks_refuse_before_streaming_or_inference() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
        ("RBITNET_STRUCTURED_OUTPUT", None),
    ]);
    let app = create_app_with_config(
        Arc::new(Engine::from_env().unwrap()),
        Arc::new(ServerConfig::test_defaults()),
    );
    for streaming in [false, true] {
        let body = json!({
            "model": "any",
            "max_tokens": 8,
            "stream": streaming,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "aaa"
                    }
                }]
            }]
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), http::StatusCode::NOT_IMPLEMENTED);
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        let body: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["error"]["code"], "vision_not_supported");
    }
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .any(|line| line == "rbitnet_inference_calls_total 0"));
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
async fn anthropic_stub_messages_live_stream() {
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
    assert!(text.contains("event: content_block_start"));
    assert!(text.contains("event: content_block_delta"));
    assert!(text.contains("text_delta"));
    assert!(text.contains("event: content_block_stop"));
    assert!(text.contains("event: message_delta"));
    assert!(text.contains("event: message_stop"));
    assert!(text.contains("stub"), "unexpected stream body: {text}");
}

#[tokio::test]
async fn anthropic_messages_respects_api_key() {
    let (engine, _guard) = {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_TOY", None),
            ("RBITNET_STUB", Some("1")),
            ("RBITNET_API_KEY", Some("secret-test-key")),
        ]);
        let engine = Arc::new(Engine::from_env().expect("engine"));
        (engine, guard)
    };
    let mut cfg = ServerConfig::test_defaults();
    cfg.api_key = Some("secret-test-key".into());
    let app = create_app_with_config(engine, Arc::new(cfg));
    let body = json!({
        "model": "rbitnet-stub",
        "max_tokens": 8,
        "messages": [{ "role": "user", "content": "hello" }]
    });
    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("denied");
    assert_eq!(denied.status(), http::StatusCode::UNAUTHORIZED);

    let ok = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("x-api-key", "secret-test-key")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("authorized");
    assert!(ok.status().is_success());
}
