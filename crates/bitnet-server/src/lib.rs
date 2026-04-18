//! OpenAI-compatible HTTP surface for Rbitnet (Akasha `BitNetProvider`).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bitnet_core::inference::Engine;
use bitnet_core::BitNetError;
use futures::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
}

/// Build the Axum application (for tests and `rbitnet-server` binary).
pub fn create_app(engine: Arc<Engine>) -> Router {
    let state = AppState { engine };
    let cors = if std::env::var("RBITNET_CORS_ANY").as_deref() == Ok("1") {
        // Dev/testing: allow any origin (gated behind env flag)
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        // Default: restrict to localhost origins
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
            ])
    };
    Router::new()
        .route("/", get(root_health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}

async fn root_health() -> impl IntoResponse {
    (StatusCode::OK, "rbitnet OK")
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
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
    // Compute id/created once so all chunks share the same values.
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
