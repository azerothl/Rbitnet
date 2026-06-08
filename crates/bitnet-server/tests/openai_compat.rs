//! Integration tests: OpenAI-shaped routes expected by Akasha `BitNetProvider`.
#![allow(clippy::await_holding_lock)]

use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::{
    build_prompt_from_messages_with_tokenizer_template, create_app_with_config,
    create_app_with_expected_model, ChatMessage, ServerConfig,
};
use futures::future::join_all;
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

#[test]
fn tokenizer_config_chatml_template_is_detected() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_CHAT_FORMAT", None),
        ("RBITNET_CHAT_TEMPLATE", None),
    ]);
    let messages = [ChatMessage {
        role: "user".into(),
        content: serde_json::json!("hello"),
    }];
    let prompt = build_prompt_from_messages_with_tokenizer_template(
        &messages,
        Some("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>{% endfor %}"),
    );
    assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
}

#[test]
fn explicit_chat_format_overrides_tokenizer_config_template() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_CHAT_FORMAT", Some("raw")),
        ("RBITNET_CHAT_TEMPLATE", None),
    ]);
    let messages = [ChatMessage {
        role: "user".into(),
        content: serde_json::json!("hello"),
    }];
    let prompt = build_prompt_from_messages_with_tokenizer_template(
        &messages,
        Some("<|im_start|>{{ message['role'] }}<|im_end|>"),
    );
    assert_eq!(prompt, "user: hello");
}

#[tokio::test]
async fn openai_stub_models_and_chat() {
    // Hold the lock for the entire duration we care about env vars: setting them
    // AND constructing the engine that reads them.  The EnvGuard (_guard) keeps
    // the variables set for the rest of the test while _lock is released only
    // after the engine is built.
    let (engine, _guard) = {
        let _lock = ENV_MUTEX.lock().unwrap();
        let guard = EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_TOY", None),
            ("RBITNET_STUB", Some("1")),
        ]);
        let engine = Arc::new(Engine::from_env().expect("engine"));
        (engine, guard)
        // _lock released here; _guard keeps vars alive until end of test
    };
    let app = create_app_with_config(Arc::clone(&engine), Arc::new(ServerConfig::test_defaults()));

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
    assert_eq!(v["data"][0]["ready"], true);
    assert_eq!(v["data"][0]["metadata"]["backend"], "cpu");

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "temperature": 0.5,
        "stream": false
    });
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
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

#[tokio::test]
async fn openai_stub_chat_streams_token_deltas() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let (engine, _guard) = {
        let _guard = EnvGuard::set(&[
            ("RBITNET_MODEL", None),
            ("RBITNET_STUB", Some("1")),
        ]);
        let engine = Arc::new(Engine::from_env().expect("engine"));
        (engine, _guard)
    };
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "stream": true
    });
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
        .expect("stream response");
    assert!(res.status().is_success());
    let ct = res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"));
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains(r#""delta""#));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn admin_reload_requires_enabled_admin_token() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_MODEL_REGISTRY", None),
        ("RBITNET_ACTIVE_MODEL_ID", None),
        ("RBITNET_CONFIG", None),
        ("RBITNET_CONFIG_DIR", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/reload")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .expect("reload response");
    assert_eq!(res.status(), http::StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn admin_reload_rejects_wrong_token() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_MODEL_REGISTRY", None),
        ("RBITNET_ACTIVE_MODEL_ID", None),
        ("RBITNET_CONFIG", None),
        ("RBITNET_CONFIG_DIR", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let mut cfg = ServerConfig::test_defaults();
    cfg.admin_token = Some("secret".into());
    let app = create_app_with_config(engine, Arc::new(cfg));
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/reload")
                .header("content-type", "application/json")
                .header("x-rbitnet-admin-token", "wrong")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .expect("reload response");
    assert_eq!(res.status(), http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn admin_reload_swaps_engine_and_records_metrics() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_MODEL_REGISTRY", None),
        ("RBITNET_ACTIVE_MODEL_ID", None),
        ("RBITNET_CONFIG", None),
        ("RBITNET_CONFIG_DIR", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let mut cfg = ServerConfig::test_defaults();
    cfg.admin_token = Some("secret".into());
    let app = create_app_with_config(engine, Arc::new(cfg));
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/reload")
                .header("content-type", "application/json")
                .header("x-rbitnet-admin-token", "secret")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .expect("reload response");
    assert!(res.status().is_success());
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "reloaded");
    assert_eq!(v["model"], "rbitnet-stub");

    let res = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("metrics response");
    assert!(res.status().is_success());
    let text =
        String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    assert!(text.contains("rbitnet_model_reloads_total 1"));
}

#[tokio::test]
async fn admin_reload_failure_keeps_old_engine() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_MODEL_REGISTRY", None),
        ("RBITNET_ACTIVE_MODEL_ID", None),
        ("RBITNET_CONFIG", None),
        ("RBITNET_CONFIG_DIR", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let mut cfg = ServerConfig::test_defaults();
    cfg.admin_token = Some("secret".into());
    let app = create_app_with_config(engine, Arc::new(cfg));
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/reload")
                .header("content-type", "application/json")
                .header("x-rbitnet-admin-token", "secret")
                .body(Body::from(
                    serde_json::json!({ "model": "missing-file.gguf" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("reload response");
    assert_eq!(res.status(), http::StatusCode::INTERNAL_SERVER_ERROR);

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
}

#[tokio::test]
async fn chat_defaults_model_when_omitted() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let (app, _state) = create_app_with_expected_model(
        engine,
        Arc::new(ServerConfig::test_defaults()),
        Some("rbitnet-stub".into()),
    );
    let chat_body = serde_json::json!({
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 16
    });
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
}

#[tokio::test]
async fn completions_endpoint_maps_to_chat_response() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let body = serde_json::json!({
        "model": "rbitnet-stub",
        "prompt": "hello",
        "max_tokens": 16
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("completion response");
    assert!(res.status().is_success());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert!(v["choices"][0]["text"].as_str().unwrap().contains("stub"));
}

#[tokio::test]
async fn chat_applies_stop_sequence_after_generation() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "stop": ["Prompt"],
        "stream": false
    });
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
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        !text.contains("Prompt"),
        "stop sequence should cut output: {text}"
    );
}

#[tokio::test]
async fn chat_format_chatml_changes_prompt_rendering() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
        ("RBITNET_CHAT_FORMAT", Some("chatml")),
        ("RBITNET_CHAT_TEMPLATE", None),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "stream": false
    });
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
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(text.contains("<|im_start|>user"), "text={text}");
}

#[tokio::test]
async fn health_ready_metrics_do_not_require_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("secret-key".into()),
        ..ServerConfig::test_defaults()
    });
    for path in ["/health", "/ready", "/metrics"] {
        let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&config));
        let res = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            res.status().is_success(),
            "{path} expected 2xx got {}",
            res.status()
        );
    }
}

#[tokio::test]
async fn static_ui_is_served_without_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let config = Arc::new(ServerConfig {
        api_key: Some("secret-key".into()),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);
    let res = app
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .expect("ui response");
    assert!(res.status().is_success());
    let content_type = res
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(content_type.starts_with("text/html"));
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("Rbitnet Local UI"));
}

#[tokio::test]
async fn chat_rejects_wrong_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("correct".into()),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    });
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
    assert_eq!(res.status(), http::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn chat_accepts_bearer_api_key() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("correct".into()),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer correct")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat response");
    assert!(res.status().is_success());
}

/// X-API-Key header should be accepted as an alternative to Authorization: Bearer.
#[tokio::test]
async fn chat_accepts_x_api_key_header() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("correct".into()),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-api-key", "correct")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat response");
    assert!(res.status().is_success());
}

/// Authorization Bearer scheme matching must be case-insensitive (RFC 7235).
#[tokio::test]
async fn chat_accepts_uppercase_bearer_scheme() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("correct".into()),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "stream": false
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "BEARER correct")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat response");
    assert!(res.status().is_success());
}

/// Auth must be enforced on GET / and GET /v1/models when an API key is configured.
#[tokio::test]
async fn get_routes_require_api_key_when_configured() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));

    let config = Arc::new(ServerConfig {
        api_key: Some("secret".into()),
        ..ServerConfig::test_defaults()
    });

    for path in ["/", "/v1/models"] {
        // No credentials → 401
        let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&config));
        let res = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(
            res.status(),
            http::StatusCode::UNAUTHORIZED,
            "{path}: expected 401 without credentials"
        );

        // Valid X-API-Key → 2xx
        let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&config));
        let res = app
            .oneshot(
                Request::builder()
                    .uri(path)
                    .header("x-api-key", "secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            res.status().is_success(),
            "{path}: expected 2xx with valid X-API-Key, got {}",
            res.status()
        );
    }
}

/// Phase 1 plan: light parallel load; `max_concurrent=2` should still complete three stub requests.
#[tokio::test]
async fn parallel_stub_chats_under_concurrency_cap() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let config = Arc::new(ServerConfig {
        max_concurrent: 2,
        ..ServerConfig::test_defaults()
    });

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "parallel" }],
        "stream": false
    });
    let body_str = chat_body.to_string();

    let futs: Vec<_> = (0..3)
        .map(|_| {
            let app = create_app_with_config(Arc::clone(&engine), Arc::clone(&config));
            let body = body_str.clone();
            async move {
                app.oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
            }
        })
        .collect();

    let results = join_all(futs).await;
    for res in results {
        let response = res.expect("response");
        assert!(response.status().is_success(), "got {}", response.status());
    }
}

#[tokio::test]
async fn chat_rejects_prompt_over_token_cap() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let config = Arc::new(ServerConfig {
        max_prompt_tokens: Some(5),
        ..ServerConfig::test_defaults()
    });
    let app = create_app_with_config(engine, config);
    let long = "a".repeat(100);
    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": long }],
        "stream": false
    });
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
    assert_eq!(res.status(), http::StatusCode::BAD_REQUEST);
}
