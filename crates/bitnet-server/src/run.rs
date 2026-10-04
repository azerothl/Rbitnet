//! Shared entrypoint for the OpenAI-compatible HTTP server (used by `rbitnet-server` and `rbitnet serve`).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use bitnet_core::inference::{stub_engine, stub_mode_enabled, Engine};
use tracing::{error, info, warn};

use crate::config::apply_runtime_config_env;
use crate::model_registry::ModelRegistry;
use crate::{
    create_app_with_expected_model, create_app_with_registry, unix_now_ms, AppState, ServerConfig,
};

type AppFactory = Box<dyn FnOnce() -> (axum::Router, AppState)>;

/// A proxy runner is scoped to its registry key even without a child registry.
pub(crate) fn standalone_model_id(
    engine_id: Option<String>,
    active_id: Option<&str>,
) -> Option<String> {
    active_id
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .or(engine_id)
}

fn warn_if_insecure_bind(bind: &str) {
    if bind.starts_with("0.0.0.0:") || bind == "0.0.0.0" {
        warn!(
            %bind,
            "listening on all IPv4 interfaces; use a reverse proxy, TLS, and RBITNET_API_KEY in production"
        );
    }
    if bind.starts_with("[::]:") || bind == "[::]" {
        warn!(
            %bind,
            "listening on all IPv6 interfaces; use a reverse proxy, TLS, and RBITNET_API_KEY in production"
        );
    }
}

fn annotate_startup_load_error(message: &str) -> String {
    let lower = message.to_ascii_lowercase();
    if (lower.contains("moe") || lower.contains("mla") || lower.contains("architecture"))
        && !message.contains("#25")
    {
        return format!(
            "{message} — if this is a MoE/MLA GGUF, see GitHub issue #25 (not yet supported on the default Llama path)."
        );
    }
    if lower.contains("no such file") || lower.contains("not found") || lower.contains("os error 2")
    {
        return format!(
            "{message} — check RBITNET_MODEL / recipe path; HTTP stays up — retry with POST /v1/admin/reload (no process restart)."
        );
    }
    format!("{message} — HTTP stays up in LoadFailed; retry with POST /v1/admin/reload (set RBITNET_ADMIN_TOKEN).")
}

/// Run the HTTP server until shutdown or fatal error. Same behavior as the `rbitnet-server` binary.
pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Err(e) = apply_runtime_config_env() {
        error!(%e, "invalid runtime config");
        return Err(format!("invalid runtime config: {e}").into());
    }

    let server_config = match ServerConfig::from_env() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            error!(%e, "invalid server configuration");
            return Err(format!("invalid server configuration: {e}").into());
        }
    };

    info!(
        max_body_bytes = server_config.max_body_bytes,
        max_prompt_chars = server_config.max_prompt_chars,
        max_prompt_tokens = ?server_config.max_prompt_tokens,
        max_tokens_cap = server_config.max_tokens_cap,
        max_concurrent = server_config.max_concurrent,
        inference_timeout_secs = server_config.inference_timeout.as_secs(),
        api_key_set = server_config.api_key.is_some(),
        "rbitnet server limits"
    );

    let bind = std::env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    warn_if_insecure_bind(&bind);

    let mut startup_load_error: Option<String> = None;

    let (engine, app_factory): (Arc<Engine>, AppFactory) = match ModelRegistry::load_from_env() {
        Ok(Some((reg, active_id))) => {
            let entry = reg
                .models
                .get(&active_id)
                .expect("registry load validates active id");
            let load_result = Engine::load_path_with_overrides(
                &entry.gguf,
                entry.tokenizer.as_deref(),
                entry.architecture.as_deref(),
            );
            let (engine, loaded_id) = match load_result {
                Ok(eng) => (Arc::new(eng), Some(active_id.clone())),
                Err(e) => {
                    let msg = annotate_startup_load_error(&format!(
                        "failed to load registry model '{active_id}': {e:?}"
                    ));
                    error!(%msg, "LoadFailed at startup — binding HTTP with stub for retry");
                    startup_load_error = Some(msg);
                    (Arc::new(stub_engine()), None)
                }
            };
            let reg_arc = Arc::clone(&reg);
            let cfg = Arc::clone(&server_config);
            let eng = Arc::clone(&engine);
            let loaded_for_app = loaded_id.clone();
            (
                engine,
                Box::new(move || create_app_with_registry(eng, cfg, reg_arc, loaded_for_app, None)),
            )
        }
        Ok(None) => {
            let load_result = Engine::from_env();
            let engine = match load_result {
                Ok(eng) => Arc::new(eng),
                Err(e) => {
                    let msg = annotate_startup_load_error(&format!(
                        "failed to init engine from env: {e:?}"
                    ));
                    error!(%msg, "LoadFailed at startup — binding HTTP with stub for retry");
                    startup_load_error = Some(msg);
                    Arc::new(stub_engine())
                }
            };
            let expected_request_model_id = if server_config.require_model_match {
                standalone_model_id(
                    engine.openai_model_id(),
                    std::env::var("RBITNET_ACTIVE_MODEL_ID").ok().as_deref(),
                )
            } else {
                None
            };
            let eng = Arc::clone(&engine);
            let cfg = Arc::clone(&server_config);
            (
                Arc::clone(&engine),
                Box::new(move || {
                    create_app_with_expected_model(eng, cfg, expected_request_model_id)
                }),
            )
        }
        Err(e) => {
            error!(%e, "model registry configuration");
            return Err(e.into());
        }
    };

    if let Some(summary) = engine.model_summary() {
        info!(%summary, "GGUF loaded (full BitNet inference WIP)");
    } else if startup_load_error.is_some() {
        info!("LoadFailed: serving stub until POST /v1/admin/reload succeeds");
    } else if !stub_mode_enabled() {
        info!("no RBITNET_MODEL — set RBITNET_STUB=1 or RBITNET_TOY=1 for testing without weights");
    }

    if engine.has_gguf() && !engine.is_ready() {
        warn!(
            "readiness: tokenizer not found next to GGUF; /ready will return 503 until tokenizer.json is available"
        );
    }

    // The factory transfers its owned clone to AppState. Keeping the startup
    // clone across axum::serve would retain the first model after unload/reload.
    drop(engine);
    let (app, app_state) = app_factory();
    if let Some(msg) = startup_load_error {
        let mut err = app_state.last_load_error.write().await;
        *err = Some(msg);
    }

    if let Some(reg) = app_state.registry.read().await.as_ref() {
        info!(
            models = ?reg.models.keys().collect::<Vec<_>>(),
            "multi-model registry: chat `model` selects GGUF; initial load follows default / RBITNET_ACTIVE_MODEL_ID"
        );
    } else if let Some(ref id) = *app_state.expected_request_model_id.read().await {
        info!(%id, "chat requests must use this model id (RBITNET_REQUIRE_MODEL_MATCH)");
    }

    if let Some(idle_secs) = server_config.idle_unload_secs {
        spawn_idle_unload_watcher(app_state, idle_secs);
        info!(idle_secs, "idle unload enabled (RBITNET_IDLE_UNLOAD_SECS)");
    }

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    info!("rbitnet-server listening on http://{bind}");
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("server error: {e}"))?;
    Ok(())
}

fn spawn_idle_unload_watcher(state: AppState, idle_secs: u64) {
    let idle_ms = idle_secs.saturating_mul(1000).max(1);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(10));
        loop {
            tick.tick().await;
            let _ = try_idle_unload(&state, idle_ms).await;
        }
    });
}

/// If idle longer than `idle_ms` and a GGUF is loaded, swap to stub and bump unload metrics.
/// Returns `true` when an unload occurred.
pub async fn try_idle_unload(state: &AppState, idle_ms: u64) -> bool {
    let last = state.last_inference_activity_ms.load(Ordering::Relaxed);
    let now = unix_now_ms();
    if now.saturating_sub(last) <= idle_ms {
        return false;
    }
    let eng = state.engine.read().await;
    if !eng.has_gguf() {
        return false;
    }
    drop(eng);
    {
        let mut w = state.engine.write().await;
        *w = Arc::new(stub_engine());
        *state.last_load_error.write().await = None;
    }
    {
        let mut lid = state.loaded_registry_model_id.write().await;
        *lid = None;
    }
    if state.registry.read().await.is_none() {
        let mut exp = state.expected_request_model_id.write().await;
        *exp = None;
    }
    state
        .metrics
        .model_unloads_total
        .fetch_add(1, Ordering::Relaxed);
    tracing::info!("idle unload: engine swapped for stub");
    true
}

#[cfg(test)]
mod runner_identity_tests {
    use super::{standalone_model_id, stub_engine, Arc, ServerConfig};

    #[test]
    fn registry_alias_takes_precedence_and_blank_alias_keeps_engine_identity() {
        assert_eq!(
            standalone_model_id(Some("rbitnet-llama".into()), Some(" llama32-1b ")),
            Some("llama32-1b".into())
        );
        for value in [None, Some(""), Some(" \t ")] {
            assert_eq!(
                standalone_model_id(Some("rbitnet-qwen35".into()), value),
                Some("rbitnet-qwen35".into())
            );
        }
        assert_eq!(standalone_model_id(None, None), None);
    }

    #[tokio::test]
    async fn runner_catalog_exposes_its_alias_and_rejects_a_different_model() {
        use axum::body::Body;
        use http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let (app, _) = crate::create_app_with_expected_model(
            Arc::new(stub_engine()),
            Arc::new(ServerConfig::test_defaults()),
            Some("my-registry-alias".into()),
        );
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_success());
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let catalog: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(catalog["data"][0]["id"], "my-registry-alias");

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"wrong-alias","messages":[{"role":"user","content":"hello"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("my-registry-alias"));
    }
}
