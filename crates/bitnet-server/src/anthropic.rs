//! Anthropic Messages API subset (`POST /v1/messages`) for agent clients.
//!
//! OpenAI-compatible routes remain the primary Akasha contract. See `docs/ANTHROPIC_API.md`.

use std::convert::Infallible;
use std::pin::pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::Poll;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bitnet_core::error::BitNetError;
use bitnet_core::inference::Engine;
use bitnet_core::request_inference_cancel;
use bitnet_core::sampling::SamplingOptions;
use bitnet_core::stream::StreamEvent;
use futures::stream::poll_fn;
use futures::Future;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: Option<bool>,
}

fn default_temperature() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: &'static str,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlock {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, Infallible> {
    if let Err(r) = crate::check_auth(&state, &headers) {
        return Ok(*r);
    }
    let eng = match crate::available_engine(&state, None).await {
        Ok(engine) => engine,
        Err((code, message)) => return Ok((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                json!({"type":"error","error":{"type":"api_error","message":message,"code":code}}),
            ),
        )
            .into_response()),
    };

    let prompt = anthropic_messages_to_prompt(&req.messages);
    if let Err((status, message)) =
        crate::validate_request_context(&eng, &state.config, &prompt, req.max_tokens)
    {
        state
            .metrics
            .chat_errors_total
            .fetch_add(1, Ordering::Relaxed);
        return Ok((
            status,
            Json(
                json!({"type":"error","error":{"type":"invalid_request_error","message":message}}),
            ),
        )
            .into_response());
    }
    let sampling = SamplingOptions::from_temperature(req.temperature);
    let id = format!("msg_{}", uuid::Uuid::new_v4());

    if req.stream == Some(true) {
        return Ok(live_stream_messages(
            state,
            headers,
            eng,
            id,
            req.model,
            prompt,
            req.max_tokens,
            sampling,
        )
        .await);
    }

    let output = match eng.complete_detailed_with_options(&prompt, req.max_tokens, sampling) {
        Ok(o) => o,
        Err(e) => {
            return Ok((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": e.to_string() }
                })),
            )
                .into_response());
        }
    };

    Ok(Json(MessagesResponse {
        id,
        r#type: "message",
        role: "assistant",
        content: vec![ContentBlock {
            r#type: "text",
            text: output.text,
        }],
        model: req.model,
        stop_reason: "end_turn",
        usage: Usage {
            input_tokens: output.stats.prompt_tokens,
            output_tokens: output.stats.completion_tokens,
        },
    })
    .into_response())
}

/// Live Anthropic-shaped SSE: deltas from [`Engine::complete_streaming`].
#[allow(clippy::too_many_arguments)]
async fn live_stream_messages(
    state: AppState,
    headers: HeaderMap,
    engine: Arc<Engine>,
    id: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    sampling: SamplingOptions,
) -> Response {
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
    let start = Instant::now();
    let mut preamble_sent = false;
    let mut finished = false;
    let body_stream = poll_fn(move |cx| {
        let join = &mut join;
        if finished {
            return Poll::Ready(None::<Result<String, Infallible>>);
        }

        if !preamble_sent {
            preamble_sent = true;
            let message_start = json!({
                "type": "message_start",
                "message": {
                    "id": id_sse.as_str(),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_sse.as_str(),
                    "stop_reason": serde_json::Value::Null,
                    "stop_sequence": serde_json::Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            });
            let block_start = json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            });
            return Poll::Ready(Some(Ok(format!(
                "event: message_start\ndata: {}\n\nevent: content_block_start\ndata: {}\n\n",
                message_start, block_start
            ))));
        }

        let recv_fut = event_rx.recv();
        let mut recv_fut = pin!(recv_fut);
        match recv_fut.as_mut().poll(cx) {
            Poll::Ready(Some(Ok(StreamEvent::FirstToken { stats }))) => {
                // Refresh usage on first token; Anthropic clients tolerate late input_tokens.
                let _ = stats;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(Ok(StreamEvent::Delta { text }))) => {
                if text.is_empty() {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                let delta = json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": text }
                });
                Poll::Ready(Some(Ok(format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    delta
                ))))
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
                let message_delta = json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": serde_json::Value::Null
                    },
                    "usage": {
                        "input_tokens": output.stats.prompt_tokens,
                        "output_tokens": output.stats.completion_tokens
                    }
                });
                Poll::Ready(Some(Ok(format!(
                    "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\nevent: message_delta\ndata: {}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                    message_delta
                ))))
            }
            Poll::Ready(Some(Err(msg))) => {
                metrics.chat_errors_total.fetch_add(1, Ordering::Relaxed);
                finished = true;
                let err = json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": msg }
                });
                Poll::Ready(Some(Ok(format!("event: error\ndata: {}\n\n", err))))
            }
            Poll::Ready(None) => {
                let _ = pin!(join).as_mut().poll(cx);
                Poll::Ready(None::<Result<String, Infallible>>)
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
                        "anthropic streaming inference timeout"
                    );
                    finished = true;
                    let err = json!({
                        "type": "error",
                        "error": { "type": "timeout_error", "message": "inference timed out" }
                    });
                    return Poll::Ready(Some(Ok(format!("event: error\ndata: {}\n\n", err))));
                }
                Poll::Pending
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

/// Convert Anthropic messages into the plain `role: text` prompt used by the engine.
pub fn anthropic_messages_to_prompt(messages: &[AnthropicMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let text = match &m.content {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("{}: {}", m.role, text));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prompt_from_string_content() {
        let messages = vec![
            AnthropicMessage {
                role: "user".into(),
                content: json!("hello"),
            },
            AnthropicMessage {
                role: "assistant".into(),
                content: json!("hi"),
            },
        ];
        assert_eq!(
            anthropic_messages_to_prompt(&messages),
            "user: hello\nassistant: hi"
        );
    }

    #[test]
    fn prompt_from_text_blocks_joins_text() {
        let messages = vec![AnthropicMessage {
            role: "user".into(),
            content: json!([
                { "type": "text", "text": "one" },
                { "type": "tool_use", "id": "t1", "name": "x" },
                { "type": "text", "text": "two" }
            ]),
        }];
        assert_eq!(anthropic_messages_to_prompt(&messages), "user: one\ntwo");
    }
}
