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
        let _guard = EnvGuard::set(&[("RBITNET_MODEL", None), ("RBITNET_STUB", Some("1"))]);
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
async fn load_failed_ready_exposes_error_and_reload_recovers() {
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
    let (app, state) = create_app_with_expected_model(engine, Arc::new(cfg), None);

    // Simulate startup LoadFailed: stub engine (no GGUF) + last_load_error.
    {
        let mut err = state.last_load_error.write().await;
        *err = Some("failed to init engine from env: bad path /no/such.gguf".into());
    }

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("health");
    assert!(health.status().is_success());

    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("ready");
    assert_eq!(ready.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    let ready_text = String::from_utf8(
        ready
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(ready_text.contains("LoadFailed"), "{ready_text}");

    // Neither unary nor streaming routes may manufacture a stub completion.
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/messages"] {
        for streaming in [false, true] {
            let body = serde_json::json!({"model":"rbitnet-stub", "prompt":"hello",
                "messages":[{"role":"user","content":"hello"}], "max_tokens":8,
                "stream":streaming});
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                http::StatusCode::SERVICE_UNAVAILABLE,
                "{path} stream={streaming}"
            );
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"]["code"], "LoadFailed");
            if path == "/v1/messages" {
                assert_eq!(error["type"], "error");
            }
        }
    }

    // Successful reload (stub from env) clears LoadFailed.
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
        .expect("reload ok");
    assert!(res.status().is_success());
    {
        let err = state.last_load_error.read().await;
        assert!(
            err.is_none(),
            "expected cleared last_load_error, got {err:?}"
        );
    }
    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("ready after reload");
    assert!(ready.status().is_success());
    let response = app.oneshot(Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type","application/json")
        .body(Body::from(r#"{"model":"rbitnet-stub","messages":[{"role":"user","content":"hello"}],"max_tokens":8}"#)).unwrap()).await.unwrap();
    assert!(
        response.status().is_success(),
        "explicit stub reload remains usable"
    );
}

#[tokio::test]
async fn unloaded_model_does_not_become_a_completion_stub() {
    // The retained intent is independent of model-match validation, whose id is
    // cleared by single-model idle eviction. Simulate that post-eviction state.
    // Clear model env so idle auto-restore cannot accidentally load a host GGUF.
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", None),
    ]);
    let engine = Arc::new(bitnet_core::inference::stub_engine());
    let (app, state) =
        create_app_with_expected_model(engine, Arc::new(ServerConfig::test_defaults()), None);
    state
        .requires_loaded_model
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    let response = app.oneshot(Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type","application/json")
        .body(Body::from(r#"{"model":"any-id","messages":[{"role":"user","content":"hello"}],"max_tokens":8}"#)).unwrap()).await.unwrap();
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(error["error"]["code"], "ModelUnloaded");
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
async fn streaming_stop_flushes_pending_text_and_terminates_with_done() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let app = create_app_with_config(
        Arc::new(Engine::from_env().unwrap()),
        Arc::new(ServerConfig::test_defaults()),
    );
    for stop in ["Prompt", "Stub", "an unmatched stop sequence"] {
        let body = serde_json::json!({"model":"any", "messages":[{"role":"user", "content":"hello"}], "max_tokens":32, "stop":[stop]});
        let request = |body: serde_json::Value| {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let unary = app.clone().oneshot(request(body.clone())).await.unwrap();
        assert!(unary.status().is_success());
        let unary: serde_json::Value =
            serde_json::from_slice(&unary.into_body().collect().await.unwrap().to_bytes()).unwrap();
        let mut streaming = body;
        streaming["stream"] = true.into();
        let response = app.clone().oneshot(request(streaming)).await.unwrap();
        assert!(response.status().is_success());
        // Allow every producer event to be queued before polling. Consuming an
        // empty delta must still wake the stream for an already queued Done.
        tokio::task::yield_now().await;
        let bytes = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            response.into_body().collect(),
        )
        .await
        .expect("SSE stalled after a suppressed delta")
        .unwrap()
        .to_bytes();
        let raw = std::str::from_utf8(&bytes).unwrap();
        let mut text = String::new();
        for line in raw
            .lines()
            .filter(|line| line.starts_with("data: ") && *line != "data: [DONE]")
        {
            let value: serde_json::Value = serde_json::from_str(&line[6..]).unwrap();
            if let Some(delta) = value["choices"][0]["delta"]["content"].as_str() {
                text.push_str(delta);
            }
        }
        assert!(raw.contains("data: [DONE]"));
        assert_eq!(
            text,
            unary["choices"][0]["message"]["content"].as_str().unwrap()
        );
    }
}

#[tokio::test]
async fn streaming_stop_preempts_the_producer_and_finishes_immediately() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let app = create_app_with_config(
        Arc::new(Engine::from_env().unwrap()),
        Arc::new(ServerConfig::test_defaults()),
    );
    let body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "stop": ["Stub"],
        "stream": true,
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let raw = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(raw.contains(r#""finish_reason":"stop""#), "{raw}");
    assert!(raw.contains("data: [DONE]"), "{raw}");
    assert!(
        !raw.contains(r#""content":"Stub"#),
        "stop text must not reach the client: {raw}"
    );
}

#[tokio::test]
async fn optional_actual_cuda_live_mux_http_stop_preempts_heterogeneous_wave() {
    if std::env::var("RBITNET_LLAMA_CONTINUOUS_TEST").as_deref() != Ok("1") {
        return;
    }
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_STUB", None),
        ("RBITNET_TOY", None),
        ("RBITNET_MAX_SEQ", Some("1024")),
        ("RBITNET_CUDA_PREFILL", Some("1")),
        ("RBITNET_CUDA_PREFILL_TOKENS", Some("128")),
        ("RBITNET_CUDA_PREFILL_TF32X3", Some("0")),
        ("RBITNET_CUDA_SPLIT_KV", Some("0")),
        ("RBITNET_PREFIX_KV", Some("0")),
        ("RBITNET_CUDA_KV_FORMAT", Some("f32")),
        ("RBITNET_CUDA_RESIDENT_GRAPH", Some("1")),
        ("RBITNET_CUDA_CONTINUOUS", Some("1")),
        ("RBITNET_CUDA_LIVE_SSE_MUX", Some("1")),
        ("RBITNET_CUDA_CONTINUOUS_ADMISSION", Some("adaptive")),
        ("RBITNET_CONTINUOUS_BATCHING", Some("1")),
        ("RBITNET_FUSED_MULTI_SEQ", Some("1")),
        ("RBITNET_CUDA_KV_PAGE_LIMIT", None),
    ]);
    let engine = Arc::new(Engine::from_env().expect("real CUDA Llama engine"));
    let stop_prompt = "Give a short answer about rivers.";
    let reference = engine
        .complete_detailed(stop_prompt, 32, 0.0)
        .expect("reference completion");
    let stop = reference
        .text
        .chars()
        .next()
        .expect("reference completion must be non-empty")
        .to_string();
    let mut config = ServerConfig::test_defaults();
    config.max_concurrent = 4;
    let app = create_app_with_config(engine, Arc::new(config));
    let request = |body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let stop_response = app
        .clone()
        .oneshot(request(serde_json::json!({
            "model": "any",
            "messages": [{ "role": "user", "content": stop_prompt }],
            "max_tokens": 64,
            "temperature": 0.0,
            "stop": [stop],
            "stream": true,
        })))
        .await
        .expect("stop request");
    let survivor_response = app
        .clone()
        .oneshot(request(serde_json::json!({
            "model": "any",
            "messages": [{ "role": "user", "content": "Explain a bridge in one sentence." }],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": true,
        })))
        .await
        .expect("survivor request");
    let stopped = String::from_utf8(
        stop_response
            .into_body()
            .collect()
            .await
            .expect("stop SSE body")
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(stopped.contains(r#""finish_reason":"stop""#), "{stopped}");
    assert!(stopped.contains("data: [DONE]"), "{stopped}");

    let replacement_response = app
        .oneshot(request(serde_json::json!({
            "model": "any",
            "messages": [{ "role": "user", "content": "Name one mountain." }],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": true,
        })))
        .await
        .expect("replacement request");
    for response in [survivor_response, replacement_response] {
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .expect("surviving SSE body")
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(r#""content":"#), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    }
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

/// Frozen Akasha scrape surface — keep in sync with docs/AKASHA_METRICS.md.
#[tokio::test]
async fn akasha_contract_metrics_series_present() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
    ]);
    let engine = Arc::new(Engine::from_env().expect("engine"));
    let config = Arc::new(ServerConfig::test_defaults());

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "ping" }],
        "max_tokens": 4,
        "temperature": 0.0
    });
    let res = create_app_with_config(Arc::clone(&engine), Arc::clone(&config))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat");
    assert!(
        res.status().is_success(),
        "chat should succeed in stub mode"
    );

    let metrics = create_app_with_config(Arc::clone(&engine), Arc::clone(&config))
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("metrics");
    assert!(metrics.status().is_success());
    let body = metrics.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&body).unwrap();

    let required = [
        "rbitnet_chat_requests_total",
        "rbitnet_inference_calls_total",
        "rbitnet_inference_ttft_ms_sum",
        "rbitnet_inference_ttft_ms_avg",
        "rbitnet_inference_decode_tokens_per_sec",
        "rbitnet_completion_tokens_total",
        "rbitnet_core_prefix_cache_hits_total",
        "rbitnet_core_prefix_hit",
        "rbitnet_core_speculative_accepted_tokens_total",
        "rbitnet_core_draft_accept",
        "rbitnet_core_scheduler_decode_waves_total",
    ];
    for series in required {
        assert!(
            text.contains(series),
            "Akasha contract missing series `{series}` in /metrics:\n{text}"
        );
    }
    assert!(text.contains("rbitnet_process_vram_measurement_available 0\n"));
    assert!(
        !text.contains("rbitnet_process_vram_bytes "),
        "unmeasured VRAM must not be reported as zero"
    );
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        let rss = text
            .lines()
            .find_map(|line| line.strip_prefix("rbitnet_process_rss_bytes "))
            .expect("OS working set required")
            .parse::<u64>()
            .unwrap();
        assert!(
            rss > 0,
            "working set must be measured from this test process"
        );
    }

    let models = create_app_with_config(engine, config)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("models");
    assert!(models.status().is_success());
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

#[test]
fn structured_output_schema_golden() {
    use bitnet_server::{validate_structured_output, JsonSchemaSpec, ResponseFormat};

    let rf_obj = ResponseFormat {
        format_type: "json_object".into(),
        json_schema: None,
    };
    assert!(validate_structured_output(r#"{"a":1}"#, &rf_obj).is_ok());
    assert!(validate_structured_output("not-json", &rf_obj).is_err());

    let rf_schema = ResponseFormat {
        format_type: "json_schema".into(),
        json_schema: Some(JsonSchemaSpec {
            name: Some("out".into()),
            schema: Some(serde_json::json!({
                "type": "object",
                "required": ["name", "score"]
            })),
            strict: true,
        }),
    };
    assert!(validate_structured_output(r#"{"name":"x","score":1}"#, &rf_schema).is_ok());
    let err = validate_structured_output(r#"{"name":"x"}"#, &rf_schema).unwrap_err();
    assert!(err.contains("required field 'score'"), "{err}");
}

#[tokio::test]
async fn response_format_refuses_unvalidated_grammar_before_generation() {
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
    let body = serde_json::json!({
        "model": "rbitnet-stub",
        "messages": [{"role":"user","content":"hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "response_format": { "type": "json_object" }
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat");
    assert_eq!(res.status(), http::StatusCode::NOT_IMPLEMENTED);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["code"], "structured_output_not_supported");
}

#[tokio::test]
async fn unsupported_structured_requests_are_json_errors_even_for_sse_and_busy_admission() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[("RBITNET_MODEL", None), ("RBITNET_TOY", None), ("RBITNET_STUB", Some("1")), ("RBITNET_STRUCTURED_OUTPUT", None)]);
    let engine = Arc::new(Engine::from_env().unwrap());
    let app = create_app_with_config(Arc::clone(&engine), Arc::new(ServerConfig { max_concurrent: 0, ..ServerConfig::test_defaults() }));
    let call = |body: serde_json::Value| Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    for kind in ["json_object", "json_schema"] {
        for streaming in [false, true] {
            for maximum in [0, 8] {
                let response = app.clone().oneshot(call(serde_json::json!({"model":"any", "messages":[{"role":"user","content":"hello"}], "max_tokens":maximum, "stream":streaming, "response_format":{"type":kind}}))).await.unwrap();
                assert_eq!(response.status(), http::StatusCode::NOT_IMPLEMENTED);
                assert!(response.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
                let body: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
                assert_eq!(body["error"]["code"], "structured_output_not_supported");
            }
        }
    }
    for fields in [serde_json::json!({"tools":[{"type":"function","function":{"name":"lookup"}}]}), serde_json::json!({"tool_choice":"required"}), serde_json::json!({"tool_choice":{"type":"function","function":{"name":"lookup"}}}), serde_json::json!({"functions":[{"name":"lookup"}]}), serde_json::json!({"function_call":{"name":"lookup"}})] {
        for streaming in [false, true] {
            let mut body = serde_json::json!({"model":"any", "messages":[{"role":"user","content":"hello"}], "max_tokens":8, "stream":streaming});
            body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
            let response = app.clone().oneshot(call(body)).await.unwrap();
            assert_eq!(response.status(), http::StatusCode::NOT_IMPLEMENTED);
            let body: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
            assert_eq!(body["error"]["code"], "tool_generation_not_supported");
        }
    }
    let response = app.oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap()).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(std::str::from_utf8(&bytes).unwrap().lines().any(|line| line == "rbitnet_inference_calls_total 0"));
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    for fields in [serde_json::json!({"tools":[], "response_format":{"type":"text"}}), serde_json::json!({"tools":[{"type":"function","function":{"name":"lookup"}}], "tool_choice":"none"}), serde_json::json!({"tool_choice":"auto"})] {
        let mut body = serde_json::json!({"model":"any", "messages":[{"role":"user","content":"hello"}], "max_tokens":8});
        body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
        assert!(app.clone().oneshot(call(body)).await.unwrap().status().is_success());
    }
}

#[tokio::test]
async fn image_url_requests_refuse_before_inference() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("RBITNET_MODEL", None),
        ("RBITNET_TOY", None),
        ("RBITNET_STUB", Some("1")),
        ("RBITNET_STRUCTURED_OUTPUT", None),
    ]);
    let engine = Arc::new(Engine::from_env().unwrap());
    let app = create_app_with_config(
        Arc::clone(&engine),
        Arc::new(ServerConfig {
            max_concurrent: 0,
            ..ServerConfig::test_defaults()
        }),
    );
    let call = |body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    for streaming in [false, true] {
        let response = app
            .clone()
            .oneshot(call(serde_json::json!({
                "model": "any",
                "max_tokens": 8,
                "stream": streaming,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in the image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaa"}}
                    ]
                }]
            })))
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
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("issues/143"));
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
    let app = create_app_with_config(engine, Arc::new(ServerConfig::test_defaults()));
    let text_only = app
        .oneshot(call(serde_json::json!({
            "model": "any",
            "messages": [{"role":"user","content":[{"type":"text","text":"hello"}]}],
            "max_tokens": 8
        })))
        .await
        .unwrap();
    assert!(text_only.status().is_success());
}

#[tokio::test]
async fn structured_environment_refuses_all_six_routes_before_opening_sse() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let _guard = EnvGuard::set(&[("RBITNET_MODEL", None), ("RBITNET_TOY", None), ("RBITNET_STUB", Some("1")), ("RBITNET_STRUCTURED_OUTPUT", None)]);
    let engine = Arc::new(Engine::from_env().unwrap());
    let app = create_app_with_config(Arc::clone(&engine), Arc::new(ServerConfig::test_defaults()));
    for mode in ["json", "tool", "tool-call", "tool_call", " JSON ", " TOOL "] {
        std::env::set_var("RBITNET_STRUCTURED_OUTPUT", mode);
        assert!(matches!(engine.complete("hello", 8, 0.0), Err(bitnet_core::error::BitNetError::NotImplemented(_))));
        for endpoint in ["/v1/chat/completions", "/v1/completions", "/v1/messages"] {
            for streaming in [false, true] {
                let body = serde_json::json!({"model":"any", "messages":[{"role":"user","content":"hello"}], "prompt":"hello", "max_tokens":8, "stream":streaming});
                let response = app.clone().oneshot(Request::builder().method("POST").uri(endpoint).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), http::StatusCode::NOT_IMPLEMENTED, "{mode} {endpoint} stream={streaming}");
                assert!(response.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
                let body: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
                assert_eq!(body["error"]["code"], "structured_output_not_supported");
            }
        }
    }
}

#[tokio::test]
async fn idle_unload_skips_stub_without_gguf() {
    use bitnet_server::{create_app_with_expected_model, try_idle_unload};
    use std::sync::atomic::Ordering;

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
    let (_app, state) =
        create_app_with_expected_model(engine, Arc::new(ServerConfig::test_defaults()), None);
    // Force "idle" timestamp.
    state.last_inference_activity_ms.store(0, Ordering::Relaxed);
    assert!(
        !try_idle_unload(&state, 1).await,
        "stub has no GGUF — idle unload must no-op"
    );
    assert_eq!(state.metrics.model_unloads_total.load(Ordering::Relaxed), 0);
}
