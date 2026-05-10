//! Shared entrypoint for the OpenAI-compatible HTTP server (used by `rbitnet-server` and `rbitnet serve`).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use bitnet_core::inference::{stub_engine, stub_mode_enabled, Engine};
use tracing::{error, info, warn};

use crate::model_registry::ModelRegistry;
use crate::{
    create_app_with_expected_model, create_app_with_registry, unix_now_ms, AppState, ServerConfig,
};

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

/// Run the HTTP server until shutdown or fatal error. Same behavior as the `rbitnet-server` binary.
pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    let (engine, app_factory): (
        Arc<Engine>,
        Box<dyn FnOnce() -> (axum::Router, AppState)>,
    ) = match ModelRegistry::load_from_env() {
        Ok(Some((reg, active_id))) => {
            let entry = reg
                .models
                .get(&active_id)
                .expect("registry load validates active id");
            let engine = Arc::new(
                Engine::load_path_with_overrides(
                    &entry.gguf,
                    entry.tokenizer.as_deref(),
                    entry.architecture.as_deref(),
                )
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("failed to load registry model '{active_id}': {e:?}").into()
                })?,
            );
            let reg_arc = Arc::clone(&reg);
            let cfg = Arc::clone(&server_config);
            (
                Arc::clone(&engine),
                Box::new(move || {
                    create_app_with_registry(engine, cfg, reg_arc, Some(active_id.clone()), None)
                }),
            )
        }
        Ok(None) => {
            let engine = Arc::new(
                Engine::from_env().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("failed to init engine from env: {e:?}").into()
                })?,
            );
            let expected_request_model_id = if server_config.require_model_match {
                engine.openai_model_id()
            } else {
                None
            };
            let eng = Arc::clone(&engine);
            let cfg = Arc::clone(&server_config);
            (
                Arc::clone(&engine),
                Box::new(move || create_app_with_expected_model(eng, cfg, expected_request_model_id)),
            )
        }
        Err(e) => {
            error!(%e, "model registry configuration");
            return Err(e.into());
        }
    };

    if let Some(summary) = engine.model_summary() {
        info!(%summary, "GGUF loaded (full BitNet inference WIP)");
    } else if !stub_mode_enabled() {
        info!("no RBITNET_MODEL — set RBITNET_STUB=1 or RBITNET_TOY=1 for testing without weights");
    }

    if engine.has_gguf() && !engine.is_ready() {
        warn!(
            "readiness: tokenizer not found next to GGUF; /ready will return 503 until tokenizer.json is available"
        );
    }

    let (app, app_state) = app_factory();

    if let Some(reg) = app_state.registry.as_ref() {
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
            let last = state.last_inference_activity_ms.load(Ordering::Relaxed);
            let now = unix_now_ms();
            if now.saturating_sub(last) <= idle_ms {
                continue;
            }
            let eng = state.engine.read().await;
            if !eng.has_gguf() {
                continue;
            }
            drop(eng);
            {
                let mut w = state.engine.write().await;
                *w = Arc::new(stub_engine());
            }
            {
                let mut lid = state.loaded_registry_model_id.write().await;
                *lid = None;
            }
            if state.registry.is_none() {
                let mut exp = state.expected_request_model_id.write().await;
                *exp = None;
            }
            state
                .metrics
                .model_unloads_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::info!(idle_secs, "idle unload: engine swapped for stub");
        }
    });
}
