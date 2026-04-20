//! OpenAI-compatible HTTP surface for Rbitnet (Akasha `BitNetProvider`).
//!
//! See `docs/PLAN_PRODUCTION.md` for limits, metrics, and health endpoints.

mod config;
mod metrics;
mod run;

pub use run::run_server;

use std::convert::Infallible;
use std::sync::atomic::Ordering;
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
use bitnet_core::inference::Engine;
use bitnet_core::BitNetError;
use futures::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Semaphore;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

pub use config::ServerConfig;
use metrics::ServerMetrics;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub config: Arc<ServerConfig>,
    pub metrics: Arc<ServerMetrics>,
    pub semaphore: Arc<Semaphore>,
}

/// Build the Axum app using [`ServerConfig::from_env`].
///
/// Returns an error if any `RBITNET_*` environment variable contains an invalid value.
pub fn create_app(engine: Arc<Engine>) -> Result<Router, String> {
    let config = Arc::new(ServerConfig::from_env()?);
    Ok(create_app_with_config(engine, config))
}

/// Build the Axum app with an explicit config (tests and embedders).
#[must_use]
pub fn create_app_with_config(engine: Arc<Engine>, config: Arc<ServerConfig>) -> Router {
    let metrics = Arc::new(ServerMetrics::default());
    let max_body_bytes = config.max_body_bytes;
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let state = AppState {
        engine,
        config: Arc::clone(&config),
        metrics: Arc::clone(&metrics),
        semaphore,
    };

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
        .route("/metrics", get(metrics_handler));

    let api = Router::new()
        .route("/", get(root_health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions));

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

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
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
    Err((
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "message": "invalid or missing API key",
                "type": "authentication_error"
            }
        })),
    )
        .into_response())
}

async fn liveness() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn readiness(State(state): State<AppState>) -> impl IntoResponse {
    if state.engine.is_ready() {
        (StatusCode::OK, "ready\n")
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "not ready: tokenizer missing or model not configured\n",
        )
    }
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        state.metrics.prometheus_text(),
    )
}

async fn root_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return r;
    }
    (StatusCode::OK, "rbitnet OK\n").into_response()
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return r;
    }
    let model_id = state
        .engine
        .openai_model_id()
        .unwrap_or_else(|| "rbitnet-stub".into());
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": unix_now(),
                "owned_by": "rbitnet"
            }
        ]
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
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

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, Infallible> {
    if let Err(r) = check_auth(&state, &headers) {
        return Ok(r);
    }

    state
        .metrics
        .chat_requests_total
        .fetch_add(1, Ordering::Relaxed);

    let prompt = build_prompt_from_messages(&req.messages);
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
    let engine = state.engine.clone();
    let metrics = state.metrics.clone();
    let timeout_dur = state.config.inference_timeout;
    let prompt_owned = prompt;
    let join = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        engine.complete(&prompt_owned, max_tokens, temperature)
    });

    let start = Instant::now();
    let text_result: Result<String, BitNetError> = match tokio::time::timeout(timeout_dur, join).await {
        Ok(Ok(Ok(t))) => {
            let ms = start.elapsed().as_millis() as u64;
            metrics
                .inference_ms_total
                .fetch_add(ms, Ordering::Relaxed);
            metrics
                .inference_calls_total
                .fetch_add(1, Ordering::Relaxed);
            Ok(t)
        }
        Ok(Ok(Err(e))) => {
            let ms = start.elapsed().as_millis() as u64;
            metrics
                .inference_ms_total
                .fetch_add(ms, Ordering::Relaxed);
            metrics
                .inference_calls_total
                .fetch_add(1, Ordering::Relaxed);
            Err(e)
        }
        Ok(Err(_join_err)) => {
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
        Err(_elapsed) => {
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

    let text = match text_result {
        Ok(t) => t,
        Err(e) => {
            state
                .metrics
                .chat_errors_total
                .fetch_add(1, Ordering::Relaxed);
            let (status, msg) = match e {
                BitNetError::ModelNotLoaded => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model not loaded: set RBITNET_MODEL, RBITNET_STUB=1, or RBITNET_TOY=1",
                ),
                BitNetError::NotImplemented(m) => (StatusCode::NOT_IMPLEMENTED, m),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "inference error"),
            };
            return Ok((
                status,
                Json(json!({
                    "error": { "message": msg, "type": "rbitnet_error" }
                })),
            )
                .into_response());
        }
    };

    if req.stream == Some(true) {
        Ok(stream_completion(&req.model, &text).into_response())
    } else {
        Ok(json_completion(&req.model, &text).into_response())
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn json_completion(model: &str, text: &str) -> impl IntoResponse {
    let created = unix_now();
    let id = format!("chatcmpl-{}", created);
    let pt = 0u64;
    let ct = text.split_whitespace().count() as u64;
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct
        }
    }))
}

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
                "finish_reason": "stop"
            }]
        });
        Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", finish))
    });

    let done = stream::once(async {
        Ok::<_, std::convert::Infallible>("data: [DONE]\n\n".to_string())
    });

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
