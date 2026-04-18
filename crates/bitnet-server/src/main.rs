//! OpenAI-compatible HTTP server for Rbitnet (`BitNetProvider` in Akasha).
//!
//! Environment:
//! - `RBITNET_BIND` — host:port (default `127.0.0.1:8080`)
//! - `RBITNET_STUB` — `1` to enable stub completions without a full engine
//! - `RBITNET_MODEL` — path to `.gguf` (header validated; full inference WIP)

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bitnet_core::inference::{stub_mode_enabled, Engine};
use bitnet_core::BitNetError;
use futures::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let bind = std::env::var("RBITNET_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let engine = match Engine::from_env() {
        Ok(e) => Arc::new(e),
        Err(e) => {
            error!(?e, "failed to init engine from env");
            std::process::exit(1);
        }
    };

    if engine.has_gguf() {
        info!(
            tensors = engine.tensor_count(),
            "GGUF header loaded (inference not yet wired)"
        );
    } else if !stub_mode_enabled() {
        info!("no RBITNET_MODEL — set RBITNET_STUB=1 for testing without weights");
    }

    let state = AppState { engine };

    let app = Router::new()
        .route("/", get(root_health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    info!("rbitnet-server listening on http://{bind}");
    axum::serve(listener, app).await.unwrap_or_else(|e| {
        error!(%e, "server error");
        std::process::exit(1);
    });
}

async fn root_health() -> impl IntoResponse {
    (StatusCode::OK, "rbitnet OK")
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let model_id = if state.engine.has_gguf() {
        "rbitnet-gguf"
    } else {
        "rbitnet-stub"
    };
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
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: serde_json::Value,
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

fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, Infallible> {
    let prompt = build_prompt_from_messages(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.7);

    let text_result = state.engine.complete(&prompt, max_tokens, temperature);

    let text = match text_result {
        Ok(t) => t,
        Err(e) => {
            let (status, msg) = match e {
                BitNetError::ModelNotLoaded => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model not loaded: set RBITNET_MODEL or RBITNET_STUB=1",
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

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn json_completion(model: &str, text: &str) -> impl IntoResponse {
    let id = format!("chatcmpl-{}", unix_now());
    let pt = 0u64;
    let ct = text.split_whitespace().count() as u64;
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_now(),
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

/// OpenAI-style SSE stream (`data: {...}\n\n`, ends with `[DONE]`).
fn stream_completion(model: &str, full_text: &str) -> Response {
    let model_owned = model.to_string();
    let chunks: Vec<String> = chunk_text_for_stream(full_text);
    let model_iter = model_owned.clone();
    let s = stream::iter(chunks.into_iter().map(move |piece| {
        let delta = json!({
            "id": format!("chatcmpl-stream-{}", unix_now()),
            "object": "chat.completion.chunk",
            "created": unix_now(),
            "model": model_iter,
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
            "id": format!("chatcmpl-stream-{}", unix_now()),
            "object": "chat.completion.chunk",
            "created": unix_now(),
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

/// Split text into small chunks so streaming clients receive multiple deltas.
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
