//! OpenAI-compatible HTTP surface for Rbitnet (Akasha `BitNetProvider`).
//!
//! See `docs/PLAN_PRODUCTION.md` for limits, metrics, and health endpoints.

mod anthropic;
mod chat_template;
mod config;
mod metrics;
mod model_registry;
mod run;
mod stream_stop;

pub use anthropic::{anthropic_messages_to_prompt, AnthropicMessage};
pub use run::{run_server, try_idle_unload};

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use bitnet_core::inference::{stub_engine, Engine};
use bitnet_core::sampling::SamplingOptions;
use bitnet_core::scheduler::{InferenceRequest, InferenceStats};
use bitnet_core::stream::StreamEvent;
use bitnet_core::BitNetError;
use bitnet_core::{clear_inference_cancel, request_inference_cancel};
use futures::stream::{self, poll_fn, StreamExt};
use futures::Future;
use serde::Deserialize;
use serde_json::json;
use std::pin::pin;
use std::task::Poll;
use tokio::sync::{RwLock, Semaphore};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use config::apply_runtime_config_env;
pub use config::ServerConfig;
use metrics::ServerMetrics;
pub use model_registry::ModelRegistry;

/// Shared HTTP state (engine may be swapped after admin unload or idle eviction).
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<Arc<Engine>>>,
    pub config: Arc<ServerConfig>,
    pub metrics: Arc<ServerMetrics>,
    pub semaphore: Arc<Semaphore>,
    /// When `Some` and [`AppState::registry`] is `None`, `/v1/chat/completions` must use this exact `model` string.
    pub expected_request_model_id: Arc<RwLock<Option<String>>>,
    /// When set, `model` must be a key in this registry and weights are loaded per request id.
    pub registry: Arc<RwLock<Option<Arc<ModelRegistry>>>>,
    /// Which registry key’s GGUF is currently in [`AppState::engine`] (`None` after stub unload).
    pub loaded_registry_model_id: Arc<RwLock<Option<String>>>,
    pub last_inference_activity_ms: Arc<AtomicU64>,
    /// Retained across idle/admin eviction; a replacement stub cannot serve model requests.
    pub requires_loaded_model: Arc<AtomicBool>,
    /// Last model load failure (startup or admin reload). Cleared on successful load or unload.
    /// When set with a stub/non-ready engine, `/ready` reports `LoadFailed`.
    pub last_load_error: Arc<RwLock<Option<String>>>,
}

/// Build [`AppState`] for tests or custom embedders.
#[must_use]
pub fn build_app_state(
    engine: Arc<Engine>,
    config: Arc<ServerConfig>,
    expected_request_model_id: Option<String>,
) -> AppState {
    build_app_state_with_registry(engine, config, expected_request_model_id, None, None)
}

/// Build state with an optional model registry (multi-model `model` selection).
#[must_use]
pub fn build_app_state_with_registry(
    engine: Arc<Engine>,
    config: Arc<ServerConfig>,
    expected_request_model_id: Option<String>,
    registry: Option<Arc<ModelRegistry>>,
    loaded_registry_model_id: Option<String>,
) -> AppState {
    let max_concurrent = config.max_concurrent;
    let requires_loaded_model = engine.has_gguf();
    AppState {
        engine: Arc::new(RwLock::new(engine)),
        config: Arc::clone(&config),
        metrics: Arc::new(ServerMetrics::default()),
        semaphore: Arc::new(Semaphore::new(max_concurrent)),
        expected_request_model_id: Arc::new(RwLock::new(expected_request_model_id)),
        registry: Arc::new(RwLock::new(registry)),
        loaded_registry_model_id: Arc::new(RwLock::new(loaded_registry_model_id)),
        last_inference_activity_ms: Arc::new(AtomicU64::new(crate::unix_now_ms())),
        requires_loaded_model: Arc::new(AtomicBool::new(requires_loaded_model)),
        last_load_error: Arc::new(RwLock::new(None)),
    }
}

/// Build the Axum app using [`ServerConfig::from_env`].
///
/// Returns an error if any `RBITNET_*` environment variable contains an invalid value.
pub fn create_app(engine: Arc<Engine>) -> Result<Router, String> {
    let config = Arc::new(ServerConfig::from_env()?);
    Ok(create_app_with_config(engine, config))
}

/// Build the Axum app with an explicit config (tests and embedders).
pub fn create_app_with_config(engine: Arc<Engine>, config: Arc<ServerConfig>) -> Router {
    let state = build_app_state(engine, Arc::clone(&config), None);
    router_with_state(state, config.max_body_bytes)
}

/// Same as [`create_app_with_config`] but fixes the expected OpenAI `model` field and exposes state for background tasks.
pub fn create_app_with_expected_model(
    engine: Arc<Engine>,
    config: Arc<ServerConfig>,
    expected_request_model_id: Option<String>,
) -> (Router, AppState) {
    let max_body = config.max_body_bytes;
    let state = build_app_state(engine, Arc::clone(&config), expected_request_model_id);
    let router = router_with_state(state.clone(), max_body);
    (router, state)
}

/// Full server state including optional [`ModelRegistry`] for per-request model loads.
pub fn create_app_with_registry(
    engine: Arc<Engine>,
    config: Arc<ServerConfig>,
    registry: Arc<ModelRegistry>,
    loaded_registry_model_id: Option<String>,
    expected_request_model_id: Option<String>,
) -> (Router, AppState) {
    let max_body = config.max_body_bytes;
    let state = build_app_state_with_registry(
        engine,
        Arc::clone(&config),
        expected_request_model_id,
        Some(registry),
        loaded_registry_model_id,
    );
    let router = router_with_state(state.clone(), max_body);
    (router, state)
}

fn router_with_state(state: AppState, max_body_bytes: usize) -> Router {
    let cors = if std::env::var("RBITNET_CORS_ANY").as_deref() == Ok("1") {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        CorsLayer::new()
            .allow_origin(AllowOrigin::list([
                HeaderValue::from_static("http://localhost:3000"),
                HeaderValue::from_static("http://127.0.0.1:3000"),
                HeaderValue::from_static("http://localhost:8080"),
                HeaderValue::from_static("http://127.0.0.1:8080"),
            ]))
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                HeaderName::from_static("x-api-key"),
                HeaderName::from_static("x-request-id"),
            ])
    };

    let public = Router::new()
        .route("/health", get(liveness))
        .route("/ready", get(readiness))
        .route("/metrics", get(metrics_handler))
        .route("/ui", get(ui_app))
        .route("/app", get(ui_app));

    let api = Router::new()
        .route("/", get(root_health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/completions", post(completions))
        .route("/v1/admin/unload", post(admin_unload))
        .route("/v1/admin/reload", post(admin_reload));

    Router::new()
        .merge(public)
        .merge(api)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
        .layer(cors)
        .layer(
            TraceLayer::new_for_http().make_span_with(|req: &Request<Body>| {
                let id = req
                    .headers()
                    .get("x-request-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("-");
                tracing::info_span!(
                    "http_request",
                    method = %req.method(),
                    path = %req.uri().path(),
                    request_id = %id,
                )
            }),
        )
        // Outermost: assign `x-request-id` before trace/logging (see PLAN_PRODUCTION observability).
        .layer(middleware::from_fn(add_request_id_if_missing))
}

async fn add_request_id_if_missing(mut req: Request<Body>, next: Next) -> Response {
    if req.headers().get("x-request-id").is_none() {
        if let Ok(v) = HeaderValue::from_str(&Uuid::new_v4().to_string()) {
            req.headers_mut().insert("x-request-id", v);
        }
    }
    next.run(req).await
}

pub(crate) fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    let key = match &state.config.api_key {
        None => return Ok(()),
        Some(k) => k,
    };
    let ok = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            // RFC 7235: auth-scheme tokens are case-insensitive.
            let mut parts = s.splitn(2, ' ');
            let scheme = parts.next()?;
            let token = parts.next()?.trim();
            if scheme.eq_ignore_ascii_case("bearer") {
                Some(token)
            } else {
                None
            }
        })
        .map(|t| t == key.as_str())
        .unwrap_or(false)
        || headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(|t| t == key.as_str())
            .unwrap_or(false);
    if ok {
        return Ok(());
    }
    state
        .metrics
        .unauthorized_total
        .fetch_add(1, Ordering::Relaxed);
    Err(Box::new(
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "invalid or missing API key",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response(),
    ))
}

async fn liveness() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn ui_app() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        include_str!("../static/app.html"),
    )
}

async fn readiness(State(state): State<AppState>) -> impl IntoResponse {
    match available_engine(&state, None).await {
        Ok(_) => (StatusCode::OK, "ready\n".to_string()),
        Err((code, message)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("not ready: {code} — {message}\nretry: POST /v1/admin/reload with RBITNET_ADMIN_TOKEN\n"),
        ),
    }
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mut text = state.metrics.prometheus_text();
    text.push_str(&bitnet_core::perf::prometheus_text());
    text.push_str(&bitnet_core::memory_budget::prometheus_text());
    if let Some(rss) = process_rss_bytes() {
        text.push_str("# HELP rbitnet_process_rss_bytes Process resident working set in bytes (Linux proc or Windows PSAPI; omitted if unavailable)\n");
        text.push_str("# TYPE rbitnet_process_rss_bytes gauge\n");
        text.push_str(&format!("rbitnet_process_rss_bytes {rss}\n"));
    }
    // No process-scoped VRAM query is implemented. An unavailable measurement
    // must not masquerade as an observed zero on an active GPU runtime.
    text.push_str("# HELP rbitnet_process_vram_measurement_available Whether process VRAM is measured (currently unavailable)\n");
    text.push_str("# TYPE rbitnet_process_vram_measurement_available gauge\n");
    text.push_str("rbitnet_process_vram_measurement_available 0\n");
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        text,
    )
}

/// A replacement engine after LoadFailed/idle unload is not an intentional
/// model completion stub. Keep explicit smoke/toy modes usable, and allow the
/// last valid GGUF to serve when a reload of another model fails.
pub(crate) async fn available_engine(
    state: &AppState,
    selected: Option<Arc<Engine>>,
) -> Result<Arc<Engine>, (&'static str, String)> {
    // Pin the validated engine for the entire request, including prompt encoding
    // and SSE. An unload/reload must not turn a pending request into a stub response.
    let shared = state.engine.read().await;
    let engine = selected.unwrap_or_else(|| Arc::clone(&shared));
    let no_model = !engine.has_gguf() && engine.openai_model_id().is_none();
    let load_error = state.last_load_error.read().await.clone();
    if no_model {
        if let Some(message) = load_error {
            return Err(("LoadFailed", message));
        }
        if state.requires_loaded_model.load(Ordering::Relaxed) {
            return Err((
                "ModelUnloaded",
                "model unloaded; reload a model before requesting inference".into(),
            ));
        }
    }
    if !engine.is_ready() {
        Err((
            "ModelNotReady",
            "model is not ready; configure and load a supported GGUF/tokenizer".into(),
        ))
    } else {
        Ok(engine)
    }
}

/// Resident working set from the host OS; no synthetic zero for unavailable APIs.
fn process_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_working_set: usize,
            working_set: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_nonpaged: usize,
            quota_nonpaged: usize,
            pagefile: usize,
            peak_pagefile: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                counters: *mut Counters,
                size: u32,
            ) -> i32;
        }
        let mut counters: Counters = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<Counters>() as u32;
        let status =
            unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (status != 0).then_some(counters.working_set as u64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

async fn root_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return *r;
    }
    (StatusCode::OK, "rbitnet OK\n").into_response()
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return *r;
    }
    let registry = state.registry.read().await.clone();
    if let Some(reg) = registry {
        let mut ids: Vec<&String> = reg.models.keys().collect();
        ids.sort();
        let loaded_id = state.loaded_registry_model_id.read().await.clone();
        let active_metadata = state.engine.read().await.model_metadata();
        let data: Vec<serde_json::Value> = ids
            .into_iter()
            .map(|id| {
                let entry = reg
                    .models
                    .get(id)
                    .expect("ids are collected from registry map");
                let loaded = loaded_id.as_deref() == Some(id.as_str());
                let metadata = if loaded {
                    serde_json::to_value(&active_metadata).unwrap_or_else(|_| json!({}))
                } else {
                    json!({
                        "summary": null,
                        "model_path": redact_display_path(&entry.gguf.display().to_string()),
                        "architecture": entry.architecture.as_deref().unwrap_or("auto"),
                        "context_length": null,
                        "quantization": null,
                        "backend": active_metadata.backend,
                        "backend_accelerated": active_metadata.backend_accelerated,
                        "profile": {
                            "backend": entry.backend.as_deref(),
                            "context_length": entry.context_length,
                            "chat_template": entry.chat_template.as_deref(),
                            "max_vram_mb": entry.max_vram_mb,
                            "max_ram_mb": entry.max_ram_mb,
                            "hybrid_layers": entry.hybrid_layers.as_deref(),
                            "hybrid_policy": entry.hybrid_policy.as_deref(),
                            "warmup": entry.warmup
                        },
                        "perf": serde_json::Value::Null,
                        "ready": false,
                        "loaded": false,
                        "tokenizer_path": entry.tokenizer.as_ref().map(|p| redact_display_path(&p.display().to_string()))
                    })
                };
                json!({
                    "id": id,
                    "object": "model",
                    "created": unix_now(),
                    "owned_by": "rbitnet",
                    "ready": loaded && active_metadata.ready,
                    "loaded": loaded,
                    "metadata": metadata
                })
            })
            .collect();
        return Json(json!({
            "object": "list",
            "data": data
        }))
        .into_response();
    }
    let eng = state.engine.read().await;
    let model_id = state
        .expected_request_model_id
        .read()
        .await
        .clone()
        .or_else(|| eng.openai_model_id())
        .unwrap_or_else(|| "rbitnet-stub".into());
    let mut metadata = eng.model_metadata();
    if !eng.has_gguf() && eng.openai_model_id().is_none() {
        metadata.ready &= !state.requires_loaded_model.load(Ordering::Relaxed)
            && state.last_load_error.read().await.is_none();
    }
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": unix_now(),
                "owned_by": "rbitnet",
                "ready": metadata.ready,
                "loaded": metadata.model_path.is_some(),
                "metadata": metadata
            }
        ]
    }))
    .into_response()
}

fn redact_display_path(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    if lower.contains("hf_")
        || lower.contains("token=")
        || lower.contains("apikey")
        || lower.contains("api_key")
    {
        return "<redacted>".into();
    }
    path.to_string()
}

fn admin_token_ok(config: &ServerConfig, headers: &HeaderMap) -> bool {
    let Some(expected) = config.admin_token.as_ref() else {
        return false;
    };
    let from_header = headers
        .get("x-rbitnet-admin-token")
        .and_then(|v| v.to_str().ok());
    let from_bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            let mut parts = s.splitn(2, ' ');
            let scheme = parts.next()?;
            let token = parts.next()?.trim();
            scheme.eq_ignore_ascii_case("bearer").then_some(token)
        });
    from_header == Some(expected.as_str()) || from_bearer == Some(expected.as_str())
}

/// Enrich load failures with actionable hints (path, MoE/#25).
fn annotate_load_error(message: &str) -> String {
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
            "{message} — check RBITNET_MODEL / recipe path; then POST /v1/admin/reload without restarting the process."
        );
    }
    message.to_string()
}

async fn admin_unload(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, Infallible> {
    if state.config.admin_token.is_none() {
        return Ok((
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "admin unload disabled (set RBITNET_ADMIN_TOKEN)",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response());
    }
    if !admin_token_ok(&state.config, &headers) {
        state
            .metrics
            .unauthorized_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "invalid or missing admin token",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response());
    }
    {
        let mut eng = state.engine.write().await;
        *eng = Arc::new(stub_engine());
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
    Ok((StatusCode::OK, "unloaded\n").into_response())
}

#[derive(Debug, Default, Deserialize)]
pub struct AdminReloadRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub active_model_id: Option<String>,
    #[serde(default)]
    pub clear_cache: bool,
}

async fn admin_reload(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<AdminReloadRequest>>,
) -> Result<Response, Infallible> {
    let started = Instant::now();
    if state.config.admin_token.is_none() {
        return Ok((
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "admin reload disabled (set RBITNET_ADMIN_TOKEN)",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response());
    }
    if !admin_token_ok(&state.config, &headers) {
        state
            .metrics
            .unauthorized_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "invalid or missing admin token",
                    "type": "authentication_error"
                }
            })),
        )
            .into_response());
    }

    let req = body.map(|Json(v)| v).unwrap_or_default();
    let result = if req.registry.is_some()
        || std::env::var_os("RBITNET_MODEL_REGISTRY").is_some()
        || state.registry.read().await.is_some()
    {
        reload_registry_engine(&state, &req).await
    } else {
        reload_single_engine(&state, &req).await
    };

    match result {
        Ok(model_id) => {
            let elapsed = started.elapsed().as_millis() as u64;
            state
                .metrics
                .model_reloads_total
                .fetch_add(1, Ordering::Relaxed);
            state
                .metrics
                .model_reload_ms_total
                .fetch_add(elapsed, Ordering::Relaxed);
            {
                let mut err = state.last_load_error.write().await;
                *err = None;
            }
            Ok((
                StatusCode::OK,
                Json(json!({
                    "status": "reloaded",
                    "model": model_id,
                    "elapsed_ms": elapsed,
                    "clear_cache": req.clear_cache
                })),
            )
                .into_response())
        }
        Err(message) => {
            state
                .metrics
                .model_reload_failures_total
                .fetch_add(1, Ordering::Relaxed);
            let message = annotate_load_error(&message);
            {
                let mut err = state.last_load_error.write().await;
                *err = Some(message.clone());
            }
            Ok((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": message,
                        "type": "rbitnet_error",
                        "code": "LoadFailed"
                    }
                })),
            )
                .into_response())
        }
    }
}

async fn reload_registry_engine(
    state: &AppState,
    req: &AdminReloadRequest,
) -> Result<String, String> {
    let loaded_registry = if let Some(path) = req.registry.as_deref() {
        let (reg, _) = ModelRegistry::load_path(PathBuf::from(path))?;
        Some(reg)
    } else if let Ok(path) = std::env::var("RBITNET_MODEL_REGISTRY") {
        let (reg, _) = ModelRegistry::load_path(PathBuf::from(path.trim()))?;
        Some(reg)
    } else {
        state.registry.read().await.clone()
    };
    let Some(reg) = loaded_registry else {
        return Err("no registry available for reload".into());
    };
    let active = req
        .active_model_id
        .as_deref()
        .or(req.model.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .or_else(|| reg.default_model.clone())
        .ok_or_else(|| {
            "reload registry requires active_model_id, model, or registry default".to_string()
        })?;
    let entry = reg
        .models
        .get(&active)
        .ok_or_else(|| format!("registry: unknown model id '{active}'"))?;
    let new_engine = Arc::new(
        Engine::load_path_with_overrides(
            &entry.gguf,
            entry.tokenizer.as_deref(),
            entry.architecture.as_deref(),
        )
        .map_err(|e| format!("failed to load model '{active}': {e:?}"))?,
    );
    {
        let mut eng = state.engine.write().await;
        state
            .requires_loaded_model
            .store(new_engine.has_gguf(), Ordering::Relaxed);
        *eng = new_engine;
    }
    {
        let mut registry = state.registry.write().await;
        *registry = Some(reg);
    }
    {
        let mut lid = state.loaded_registry_model_id.write().await;
        *lid = Some(active.clone());
    }
    Ok(active)
}

async fn reload_single_engine(
    state: &AppState,
    req: &AdminReloadRequest,
) -> Result<String, String> {
    let new_engine = if let Some(model) = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let model_path = PathBuf::from(model);
        Arc::new(
            Engine::load_path_with_overrides(
                &model_path,
                req.tokenizer.as_deref().map(std::path::Path::new),
                req.architecture.as_deref(),
            )
            .map_err(|e| format!("failed to load model '{}': {e:?}", model))?,
        )
    } else {
        apply_runtime_config_env().map_err(|e| format!("invalid runtime config: {e}"))?;
        Arc::new(Engine::from_env().map_err(|e| format!("failed to init engine from env: {e:?}"))?)
    };
    let model_id = if state.config.require_model_match {
        crate::run::standalone_model_id(
            new_engine.openai_model_id(),
            std::env::var("RBITNET_ACTIVE_MODEL_ID").ok().as_deref(),
        )
    } else {
        new_engine.openai_model_id()
    }
    .unwrap_or_else(|| "rbitnet-stub".into());
    {
        let mut eng = state.engine.write().await;
        state
            .requires_loaded_model
            .store(new_engine.has_gguf(), Ordering::Relaxed);
        *eng = new_engine;
    }
    {
        let mut registry = state.registry.write().await;
        *registry = None;
    }
    {
        let mut lid = state.loaded_registry_model_id.write().await;
        *lid = None;
    }
    if state.config.require_model_match {
        let mut exp = state.expected_request_model_id.write().await;
        *exp = Some(model_id.clone());
    }
    Ok(model_id)
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// OpenAI-compatible structured output (`json_object` / `json_schema`).
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    /// When true (default), reject non-conforming completions with 422.
    #[serde(default = "default_strict_true")]
    pub strict: bool,
}

fn default_strict_true() -> bool {
    true
}

/// Validate assistant text against a simple OpenAI-style JSON schema subset
/// (`type: object` + `required` string field names). Full JSON Schema is out of scope.
pub fn validate_structured_output(
    text: &str,
    response_format: &ResponseFormat,
) -> Result<(), String> {
    let kind = response_format.format_type.trim().to_ascii_lowercase();
    match kind.as_str() {
        "text" | "" => Ok(()),
        "json_object" => {
            let trimmed = text.trim();
            serde_json::from_str::<serde_json::Value>(trimmed)
                .map(|_| ())
                .map_err(|e| format!("structured output is not valid JSON: {e}"))
        }
        "json_schema" => {
            let strict = response_format
                .json_schema
                .as_ref()
                .map(|s| s.strict)
                .unwrap_or(true);
            let trimmed = text.trim();
            let value: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("structured output is not valid JSON: {e}"))?;
            let Some(spec) = response_format.json_schema.as_ref() else {
                return Ok(());
            };
            let Some(schema) = spec.schema.as_ref() else {
                return Ok(());
            };
            if !strict {
                return Ok(());
            }
            validate_against_simple_schema(&value, schema)
        }
        other => Err(format!(
            "unsupported response_format.type '{other}' (supported: json_object, json_schema, text)"
        )),
    }
}

fn validate_against_simple_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    let ty = schema
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("object");
    if ty == "object" {
        let obj = value
            .as_object()
            .ok_or_else(|| "structured output must be a JSON object".to_string())?;
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for key in required {
                let Some(name) = key.as_str() else {
                    continue;
                };
                if !obj.contains_key(name) {
                    return Err(format!("structured output missing required field '{name}'"));
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[derive(Clone)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>),
}

impl StopSequence {
    fn as_strings(&self) -> Vec<&str> {
        match self {
            Self::One(s) => vec![s.as_str()],
            Self::Many(items) => items.iter().map(String::as_str).collect(),
        }
    }
}

fn message_content_to_string(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(t);
                }
            }
            out
        }
        _ => content.to_string(),
    }
}

pub fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    build_prompt_from_messages_with_tokenizer_template(messages, None)
}

pub fn build_prompt_from_messages_with_tokenizer_template(
    messages: &[ChatMessage],
    tokenizer_template: Option<&str>,
) -> String {
    try_build_prompt_from_messages_with_tokenizer_template(messages, tokenizer_template)
        .unwrap_or_else(|error| {
            tracing::warn!(%error,"chat template render failed");
            build_raw_prompt(messages)
        })
}

fn try_build_prompt_from_messages_with_tokenizer_template(
    messages: &[ChatMessage],
    tokenizer_template: Option<&str>,
) -> Result<String, String> {
    if let Ok(template) = std::env::var("RBITNET_CHAT_TEMPLATE") {
        let template = template.trim();
        if !template.is_empty() {
            return try_apply_chat_template(template, messages);
        }
    }
    let chat_format = std::env::var("RBITNET_CHAT_FORMAT")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if chat_format.is_none() {
        if let Some(template) = tokenizer_template.map(str::trim).filter(|s| !s.is_empty()) {
            return try_apply_chat_template(template, messages);
        }
    }
    Ok(
        match chat_format
            .unwrap_or_else(|| "raw".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "llama3" => build_llama3_prompt(messages),
            "chatml" => build_chatml_prompt(messages),
            _ => build_raw_prompt(messages),
        },
    )
}

fn build_raw_prompt(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for m in messages {
        let text = message_content_to_string(&m.content);
        if text.is_empty() {
            continue;
        }
        parts.push(format!("{}: {}", m.role, text));
    }
    parts.join("\n\n")
}

fn build_llama3_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in messages {
        let text = message_content_to_string(&m.content);
        if text.is_empty() {
            continue;
        }
        out.push_str("<|start_header_id|>");
        out.push_str(m.role.trim());
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(text.trim());
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn build_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let text = message_content_to_string(&m.content);
        if text.is_empty() {
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(m.role.trim());
        out.push('\n');
        out.push_str(text.trim());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn apply_simple_chat_template(template: &str, messages: &[ChatMessage]) -> String {
    let raw = build_raw_prompt(messages);
    let prompt = messages
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| message_content_to_string(&m.content))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let system = last_role_content(messages, "system");
    let user = last_role_content(messages, "user");
    let assistant = last_role_content(messages, "assistant");
    template
        .replace("{{messages}}", &raw)
        .replace("{messages}", &raw)
        .replace("{{prompt}}", &prompt)
        .replace("{prompt}", &prompt)
        .replace("{{system}}", &system)
        .replace("{system}", &system)
        .replace("{{user}}", &user)
        .replace("{user}", &user)
        .replace("{{assistant}}", &assistant)
        .replace("{assistant}", &assistant)
}

fn try_apply_chat_template(template: &str, messages: &[ChatMessage]) -> Result<String, String> {
    if template.contains("{%") || template.contains("{{") {
        return chat_template::render(template, messages);
    }
    if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
        return Ok(build_llama3_prompt(messages));
    }
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        return Ok(build_chatml_prompt(messages));
    }
    Ok(apply_simple_chat_template(template, messages))
}

fn last_role_content(messages: &[ChatMessage], role: &str) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == role)
        .map(|m| message_content_to_string(&m.content))
        .unwrap_or_default()
}

fn apply_stop_sequences(mut text: String, stop: Option<&StopSequence>) -> String {
    let Some(stop) = stop else {
        return text;
    };
    let mut cut = None;
    for s in stop.as_strings() {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s) {
            cut = Some(cut.map_or(pos, |prev: usize| prev.min(pos)));
        }
    }
    if let Some(pos) = cut {
        text.truncate(pos);
    }
    text
}

/// Load GGUF for `requested` when it differs from the in-memory registry selection (engine-first lock order).
async fn ensure_registry_model_loaded(
    state: &AppState,
    requested: &str,
) -> Result<Arc<Engine>, Response> {
    let Some(reg) = state.registry.read().await.clone() else {
        return Ok(state.engine.read().await.clone());
    };
    let Some(entry) = reg.models.get(requested) else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "internal: registry model missing after validation",
                    "type": "rbitnet_error"
                }
            })),
        )
            .into_response());
    };
    {
        let eng = state.engine.read().await;
        let lid = state.loaded_registry_model_id.read().await;
        if lid.as_deref() == Some(requested) && eng.has_gguf() {
            return Ok(Arc::clone(&eng));
        }
    }
    let new_engine = match Engine::load_path_with_overrides(
        &entry.gguf,
        entry.tokenizer.as_deref(),
        entry.architecture.as_deref(),
    ) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": format!("failed to load model '{requested}': {e:?}"),
                        "type": "rbitnet_error"
                    }
                })),
            )
                .into_response());
        }
    };
    {
        let mut eng = state.engine.write().await;
        let mut lid = state.loaded_registry_model_id.write().await;
        state
            .requires_loaded_model
            .store(new_engine.has_gguf(), Ordering::Relaxed);
        *eng = Arc::clone(&new_engine);
        *lid = Some(requested.to_string());
    }
    Ok(new_engine)
}

async fn default_request_model(state: &AppState, requested: Option<&str>) -> String {
    if let Some(model) = requested.map(str::trim).filter(|s| !s.is_empty()) {
        return model.to_string();
    }
    if let Some(reg) = state.registry.read().await.clone() {
        if let Some(default) = reg.default_model.as_ref() {
            return default.clone();
        }
        if let Some(loaded) = state.loaded_registry_model_id.read().await.clone() {
            return loaded;
        }
        if let Some(first) = reg.models.keys().min() {
            return first.clone();
        }
    }
    if let Some(expected) = state.expected_request_model_id.read().await.clone() {
        return expected;
    }
    state
        .engine
        .read()
        .await
        .openai_model_id()
        .unwrap_or_else(|| "rbitnet-stub".into())
}

fn completion_prompt_to_string(prompt: &serde_json::Value) -> String {
    match prompt {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| v.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => prompt.to_string(),
    }
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, Infallible> {
    let model = default_request_model(&state, req.model.as_deref()).await;
    let chat = ChatCompletionRequest {
        model: Some(model.clone()),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String(completion_prompt_to_string(&req.prompt)),
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: req.stream,
        stop: req.stop,
        top_p: req.top_p,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        seed: req.seed,
        response_format: None,
    };
    let response = chat_completions(State(state), headers, Json(chat)).await?;
    if req.stream == Some(true) {
        return Ok(response);
    }
    let (parts, body) = response.into_parts();
    if !parts.status.is_success() {
        return Ok(Response::from_parts(parts, body));
    }
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return Ok((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": { "message": format!("completion response conversion failed: {e}"), "type": "rbitnet_error" }
                })),
            )
                .into_response());
        }
    };
    let chat_body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
    let text = chat_body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    Ok(Json(json!({
        "id": chat_body["id"].clone(),
        "object": "text_completion",
        "created": chat_body["created"].clone(),
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": serde_json::Value::Null,
            "finish_reason": chat_body["choices"][0]["finish_reason"].clone()
        }],
        "usage": chat_body["usage"].clone()
    }))
    .into_response())
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, Infallible> {
    if let Err(r) = check_auth(&state, &headers) {
        return Ok(*r);
    }

    state
        .metrics
        .chat_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let request_model = default_request_model(&state, req.model.as_deref()).await;

    let selected_engine = if let Some(reg) = state.registry.read().await.clone() {
        if !reg.models.contains_key(&request_model) {
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!(
                            "unknown model '{}'; allowed ids are defined in RBITNET_MODEL_REGISTRY",
                            request_model
                        ),
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response());
        }
        match ensure_registry_model_loaded(&state, &request_model).await {
            Ok(engine) => Some(engine),
            Err(r) => {
                state
                    .metrics
                    .chat_errors_total
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(r);
            }
        }
    } else {
        let expected = state.expected_request_model_id.read().await;
        if let Some(ref id) = *expected {
            if request_model != *id {
                state
                    .metrics
                    .chat_errors_total
                    .fetch_add(1, Ordering::Relaxed);
                return Ok((
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": format!(
                                "model must be '{id}' for this server (RBITNET_REQUIRE_MODEL_MATCH)"
                            ),
                            "type": "invalid_request_error"
                        }
                    })),
                )
                    .into_response());
            }
        }
        None
    };

    state
        .last_inference_activity_ms
        .store(unix_now_ms(), Ordering::Relaxed);

    let engine = match available_engine(&state, selected_engine).await {
        Ok(engine) => engine,
        Err((code, message)) => {
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":{"message":message,"type":"rbitnet_error","code":code}})),
            )
                .into_response());
        }
    };
    let tokenizer_chat_template = engine.tokenizer_chat_template();
    let prompt = match try_build_prompt_from_messages_with_tokenizer_template(
        &req.messages,
        tokenizer_chat_template.as_deref(),
    ) {
        Ok(prompt) => prompt,
        Err(message) => {
            return Ok((
                StatusCode::BAD_REQUEST,
                Json(json!({"error":{"message":message,"type":"invalid_request_error"}})),
            )
                .into_response())
        }
    };
    let prompt_chars = prompt.chars().count();
    if prompt_chars > state.config.max_prompt_chars {
        state
            .metrics
            .chat_errors_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!(
                        "prompt too long ({} chars, max {})",
                        prompt_chars, state.config.max_prompt_chars
                    ),
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response());
    }

    let max_tokens = req.max_tokens.unwrap_or(256);
    if max_tokens > state.config.max_tokens_cap {
        state
            .metrics
            .chat_errors_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!(
                        "max_tokens exceeds cap ({}, max {})",
                        max_tokens, state.config.max_tokens_cap
                    ),
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response());
    }

    if let Err((status, msg)) =
        validate_request_context(&engine, &state.config, &prompt, max_tokens)
    {
        state
            .metrics
            .chat_errors_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            status,
            Json(json!({"error":{"message":msg,"type":"invalid_request_error"}})),
        )
            .into_response());
    }

    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": {
                        "message": "too many concurrent inference requests",
                        "type": "rate_limit_error"
                    }
                })),
            )
                .into_response());
        }
    };

    let temperature = req.temperature.unwrap_or(0.7);
    let structured_json = req.response_format.as_ref().is_some_and(|rf| {
        matches!(
            rf.format_type.trim().to_ascii_lowercase().as_str(),
            "json_object" | "json_schema"
        )
    });
    let sampling = SamplingOptions {
        temperature,
        top_p: req.top_p,
        seed: req.seed,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        structured_json,
    };
    clear_inference_cancel();

    if req.stream == Some(true) {
        return Ok(live_stream_chat_completion(
            state,
            headers,
            permit,
            engine,
            request_model,
            prompt,
            max_tokens,
            sampling,
            req.stop.clone(),
        )
        .await);
    }
    let backend_kind = engine.backend_kind().to_string();
    let model_family = engine.model_family().to_string();
    let backend_accelerated = engine.backend_accelerated();
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let metrics = state.metrics.clone();
    let timeout_dur = state.config.inference_timeout;
    let prompt_owned = prompt;
    let prompt_chars = prompt_chars as u64;
    let continuous_batching = engine.continuous_batching_enabled();
    tracing::info!(
        request_id = %request_id,
        model = %request_model,
        backend = %backend_kind,
        family = %model_family,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = ?req.top_p,
        frequency_penalty = ?req.frequency_penalty,
        presence_penalty = ?req.presence_penalty,
        seed = ?req.seed,
        stop = req.stop.is_some(),
        prompt_chars = prompt_chars,
        continuous_batching = continuous_batching,
        timeout_secs = timeout_dur.as_secs(),
        "inference start"
    );
    let mut join = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        if continuous_batching {
            let req = InferenceRequest {
                prompt: prompt_owned.clone(),
                max_tokens,
                sampling,
            };
            let mut rows = engine.complete_batch_detailed(&[req])?;
            rows.pop()
                .ok_or_else(|| BitNetError::Inference("empty batch result".into()))
        } else {
            engine.complete_detailed_with_options(&prompt_owned, max_tokens, sampling)
        }
    });

    let start = Instant::now();
    let output_result = tokio::select! {
        joined = &mut join => {
            match joined {
                Ok(Ok(output)) => {
                    let ms = start.elapsed().as_millis() as u64;
                    metrics
                        .inference_ms_total
                        .fetch_add(ms, Ordering::Relaxed);
                    metrics
                        .inference_calls_total
                        .fetch_add(1, Ordering::Relaxed);
                    metrics.record_backend_family_call(&backend_kind, &model_family);
                    metrics
                        .inference_ttft_ms_total
                        .fetch_add(output.stats.ttft_ms, Ordering::Relaxed);
                    metrics
                        .inference_encode_ms_total
                        .fetch_add(output.stats.encode_ms, Ordering::Relaxed);
                    metrics
                        .inference_prefill_ms_total
                        .fetch_add(output.stats.prefill_ms, Ordering::Relaxed);
                    metrics
                        .inference_decode_ms_total
                        .fetch_add(output.stats.decode_ms, Ordering::Relaxed);
                    metrics
                        .inference_itl_us_total
                        .fetch_add(output.stats.itl_us, Ordering::Relaxed);
                    metrics
                        .inference_tpot_us_total
                        .fetch_add(output.stats.tpot_us, Ordering::Relaxed);
                    metrics
                        .completion_tokens_total
                        .fetch_add(output.stats.completion_tokens as u64, Ordering::Relaxed);
                    if output.stats.speculative_attempted {
                        metrics
                            .speculative_requests_total
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    if backend_accelerated {
                        metrics
                            .native_accelerated_calls_total
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(output)
                }
                Ok(Err(e)) => {
                    let ms = start.elapsed().as_millis() as u64;
                    metrics
                        .inference_ms_total
                        .fetch_add(ms, Ordering::Relaxed);
                    metrics
                        .inference_calls_total
                        .fetch_add(1, Ordering::Relaxed);
                    metrics.record_backend_family_call(&backend_kind, &model_family);
                    Err(e)
                }
                Err(_join_err) => {
                    metrics
                        .chat_errors_total
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({
                            "error": { "message": "inference task failed", "type": "rbitnet_error" }
                        })),
                    )
                        .into_response());
                }
            }
        }
        _ = tokio::time::sleep(timeout_dur) => {
            request_inference_cancel();
            join.abort();
            tracing::warn!(
                request_id = %request_id,
                model = %request_model,
                backend = %backend_kind,
                family = %model_family,
                timeout_secs = timeout_dur.as_secs(),
                "inference timeout reached; task aborted"
            );
            metrics
                .inference_timeouts_total
                .fetch_add(1, Ordering::Relaxed);
            metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::GATEWAY_TIMEOUT,
                Json(json!({
                    "error": {
                        "message": "inference timed out",
                        "type": "timeout_error"
                    }
                })),
            )
                .into_response());
        }
    };

    let output = match output_result {
        Ok(t) => t,
        Err(e) => {
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            let status = StatusCode::from_u16(e.http_status_for_chat_completion())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut msg = e.to_string();
            if let Some(hint) = e.user_troubleshooting_hint() {
                msg.push_str(" — ");
                msg.push_str(hint);
            }
            let err_type = if status.is_server_error() {
                "rbitnet_error"
            } else {
                "invalid_request_error"
            };
            return Ok((
                status,
                Json(json!({
                    "error": {
                        "message": msg,
                        "type": err_type
                    }
                })),
            )
                .into_response());
        }
    };

    let original_len = output.text.len();
    let text = apply_stop_sequences(output.text, req.stop.as_ref());
    let mut stats = output.stats;
    if text.len() < original_len {
        stats.finish_reason = bitnet_core::timings::GenerationFinishReason::Stop;
    }
    if let Some(rf) = req.response_format.as_ref() {
        if let Err(msg) = validate_structured_output(&text, rf) {
            state
                .metrics
                .structured_output_validation_failures_total
                .fetch_add(1, Ordering::Relaxed);
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return Ok((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({
                    "error": {
                        "message": msg,
                        "type": "invalid_request_error",
                        "code": "structured_output_validation_failed"
                    }
                })),
            )
                .into_response());
        }
    }
    Ok(json_completion(&request_model, &text, &stats).into_response())
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[must_use]
pub fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn json_completion(model: &str, text: &str, stats: &InferenceStats) -> impl IntoResponse {
    let created = unix_now();
    let id = format!("chatcmpl-{}", created);
    let pt = stats.prompt_tokens as u64;
    let mut ct = stats.completion_tokens as u64;
    if pt == 0 && ct == 0 && !text.is_empty() {
        ct = text.split_whitespace().count() as u64;
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": stats.finish_reason.openai()
        }],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct
        }
    }))
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_chat_completion(
    state: AppState,
    headers: HeaderMap,
    _permit: tokio::sync::OwnedSemaphorePermit,
    engine: Arc<Engine>,
    model: String,
    prompt: String,
    max_tokens: u32,
    sampling: SamplingOptions,
    stop: Option<StopSequence>,
) -> Response {
    let created = unix_now();
    let id = format!("chatcmpl-stream-{}", created);
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let metrics = state.metrics.clone();
    let timeout_dur = state.config.inference_timeout;
    let backend_kind = engine.backend_kind().to_string();
    let model_family = engine.model_family().to_string();
    let backend_accelerated = engine.backend_accelerated();

    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<Result<StreamEvent, String>>(64);
    let prompt_for_stop = prompt.clone();
    let mut join = tokio::task::spawn_blocking(move || {
        let handle = tokio::runtime::Handle::current();
        let mut on_event = |ev: StreamEvent| -> bitnet_core::Result<()> {
            handle
                .block_on(event_tx.send(Ok(ev)))
                .map_err(|e| BitNetError::Inference(format!("stream send failed: {e}")))?;
            Ok(())
        };
        match engine.complete_streaming(&prompt, max_tokens, sampling, &mut on_event) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = handle.block_on(event_tx.send(Err(e.to_string())));
                Err(e)
            }
        }
    });

    let model_sse = model.clone();
    let id_sse = id.clone();
    let mut stop_filter = stream_stop::StreamStop::new(stop.as_ref());
    let start = Instant::now();
    let mut role_sent = false;
    let mut finished = false;
    let body_stream = poll_fn(move |cx| {
        let join = &mut join;
        if finished {
            return Poll::Ready(None::<Result<String, std::convert::Infallible>>);
        }
        let recv_fut = event_rx.recv();
        let mut recv_fut = pin!(recv_fut);
        match recv_fut.as_mut().poll(cx) {
            Poll::Ready(Some(Ok(StreamEvent::FirstToken { .. }))) => {
                if !role_sent {
                    role_sent = true;
                    let chunk = json!({
                        "id": id_sse.as_str(),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_sse.as_str(),
                        "choices": [{
                            "index": 0,
                            "delta": { "role": "assistant" },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    return Poll::Ready(Some(Ok(format!("data: {}\n\n", chunk))));
                }
                // We consumed an event; recv did not register a pending waker.
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(Ok(StreamEvent::Delta { text }))) => {
                let piece = stop_filter.push(&text);
                if piece.is_empty() {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                let chunk = json!({
                    "id": id_sse.as_str(),
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_sse.as_str(),
                    "choices": [{
                        "index": 0,
                        "delta": { "content": piece },
                        "finish_reason": serde_json::Value::Null
                    }]
                });
                Poll::Ready(Some(Ok(format!("data: {}\n\n", chunk))))
            }
            Poll::Ready(Some(Ok(StreamEvent::Done(output)))) => {
                let ms = start.elapsed().as_millis() as u64;
                metrics.inference_ms_total.fetch_add(ms, Ordering::Relaxed);
                metrics
                    .inference_calls_total
                    .fetch_add(1, Ordering::Relaxed);
                metrics.record_backend_family_call(&backend_kind, &model_family);
                metrics
                    .inference_ttft_ms_total
                    .fetch_add(output.stats.ttft_ms, Ordering::Relaxed);
                metrics
                    .completion_tokens_total
                    .fetch_add(output.stats.completion_tokens as u64, Ordering::Relaxed);
                if backend_accelerated {
                    metrics
                        .native_accelerated_calls_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                finished = true;
                let finish = json!({
                    "id": id_sse.as_str(),
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_sse.as_str(),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": if stop_filter.stopped(){Some("stop")}else{output.stats.finish_reason.openai()}
                    }]
                });
                let tail = stop_filter.finish();
                let mut packet = String::new();
                if !tail.is_empty() {
                    let delta = json!({"id":id_sse.as_str(), "object":"chat.completion.chunk", "created":created,
                        "model":model_sse.as_str(), "choices":[{"index":0, "delta":{"content":tail}, "finish_reason":serde_json::Value::Null}]});
                    packet.push_str(&format!("data: {}\n\n", delta));
                }
                packet.push_str(&format!("data: {}\n\ndata: [DONE]\n\n", finish));
                Poll::Ready(Some(Ok(packet)))
            }
            Poll::Ready(Some(Err(msg))) => {
                metrics.chat_errors_total.fetch_add(1, Ordering::Relaxed);
                finished = true;
                let err = json!({
                    "error": { "message": msg, "type": "rbitnet_error" }
                });
                Poll::Ready(Some(Ok(format!("data: {}\n\n", err))))
            }
            Poll::Ready(None) => {
                let _ = pin!(join).as_mut().poll(cx);
                Poll::Ready(None::<Result<String, std::convert::Infallible>>)
            }
            Poll::Pending => {
                if start.elapsed() > timeout_dur {
                    request_inference_cancel();
                    let _ = pin!(join).as_mut().poll(cx);
                    metrics
                        .inference_timeouts_total
                        .fetch_add(1, Ordering::Relaxed);
                    metrics.chat_errors_total.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        request_id = %request_id,
                        model = %model_sse,
                        "streaming inference timeout"
                    );
                    finished = true;
                    let err = json!({
                        "error": { "message": "inference timed out", "type": "timeout_error" }
                    });
                    return Poll::Ready(Some(Ok(format!("data: {}\n\n", err))));
                }
                Poll::Pending
            }
        }
    });

    let _ = prompt_for_stop;
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

#[allow(dead_code)]
fn stream_completion(model: &str, full_text: &str) -> Response {
    let created = unix_now();
    let id = format!("chatcmpl-stream-{}", created);
    let model_owned = model.to_string();
    let chunks: Vec<String> = chunk_text_for_stream(full_text);
    let id_for_chunks = id.clone();
    let model_for_chunks = model_owned.clone();
    let s = stream::iter(chunks.into_iter().map(move |piece| {
        let delta = json!({
            "id": id_for_chunks.as_str(),
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_for_chunks.as_str(),
            "choices": [{
                "index": 0,
                "delta": { "content": piece },
                "finish_reason": serde_json::Value::Null
            }]
        });
        let line = format!("data: {}\n\n", delta);
        Ok::<_, std::convert::Infallible>(line)
    }));

    let tail = stream::once(async move {
        let finish = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_owned,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": serde_json::Value::Null
            }]
        });
        Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", finish))
    });

    let done =
        stream::once(async { Ok::<_, std::convert::Infallible>("data: [DONE]\n\n".to_string()) });

    let combined = s.chain(tail).chain(done);
    let body = Body::from_stream(combined);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap()
}

fn chunk_text_for_stream(text: &str) -> Vec<String> {
    const CHUNK: usize = 24;
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    for w in chars.chunks(CHUNK) {
        out.push(w.iter().collect());
    }
    if out.is_empty() && !text.is_empty() {
        out.push(text.to_string());
    }
    out
}

/// Shared by OpenAI compatibility routes and the independent Anthropic handler.
/// Reject before admission/stream headers, using the allocated model capacity.
fn validate_request_context(
    engine: &Engine,
    config: &ServerConfig,
    prompt: &str,
    max_tokens: u32,
) -> Result<(), (StatusCode, String)> {
    if config.max_prompt_tokens.is_none() && engine.context_capacity().is_none() {
        return Ok(());
    }
    let map_error = |e: bitnet_core::BitNetError| {
        let status = StatusCode::from_u16(e.http_status_for_chat_completion())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut text = e.to_string();
        if let Some(hint) = e.user_troubleshooting_hint() {
            text.push_str(" — ");
            text.push_str(hint);
        }
        (status, text)
    };
    let count = engine.count_prompt_tokens(prompt).map_err(map_error)?;
    if let Some(cap) = config.max_prompt_tokens {
        if count > cap {
            return Err((StatusCode::BAD_REQUEST,format!("prompt too many tokens after encoding ({count}, max {cap}); raise RBITNET_MAX_PROMPT_TOKENS or shorten the prompt")));
        }
    }
    engine
        .validate_context_tokens(count, max_tokens)
        .map_err(map_error)
}
