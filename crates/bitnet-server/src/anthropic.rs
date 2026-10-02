//! Anthropic Messages API subset (`POST /v1/messages`) for agent clients.
//!
//! OpenAI-compatible routes remain the primary Akasha contract. See `docs/ANTHROPIC_API.md`.

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use bitnet_core::sampling::SamplingOptions;

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
    Json(req): Json<MessagesRequest>,
) -> Result<Response, (StatusCode, String)> {
    let prompt = anthropic_messages_to_prompt(&req.messages);
    let sampling = SamplingOptions::from_temperature(req.temperature);
    let eng = state.engine.read().await;
    let output = eng
        .complete_detailed_with_options(&prompt, req.max_tokens, sampling)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let id = format!("msg_{}", uuid::Uuid::new_v4());

    if req.stream == Some(true) {
        return Ok(stream_messages_response(
            &id,
            &req.model,
            &output.text,
            output.stats.prompt_tokens,
            output.stats.completion_tokens,
        ));
    }

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

/// Minimal Anthropic-shaped SSE: chunk a finished completion (stub/CI-safe).
/// Real token streaming from the engine is a follow-up slice — see `docs/ANTHROPIC_API.md`.
fn stream_messages_response(
    id: &str,
    model: &str,
    text: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> Response {
    let mut events: Vec<String> = Vec::new();

    let message_start = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": serde_json::Value::Null,
            "stop_sequence": serde_json::Value::Null,
            "usage": { "input_tokens": input_tokens, "output_tokens": 0 }
        }
    });
    events.push(format!(
        "event: message_start\ndata: {}\n\n",
        message_start
    ));

    let block_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" }
    });
    events.push(format!(
        "event: content_block_start\ndata: {}\n\n",
        block_start
    ));

    for piece in chunk_text_for_anthropic_stream(text) {
        let delta = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": piece }
        });
        events.push(format!("event: content_block_delta\ndata: {}\n\n", delta));
    }

    events.push(
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
            .into(),
    );

    let message_delta = json!({
        "type": "message_delta",
        "delta": { "stop_reason": "end_turn", "stop_sequence": serde_json::Value::Null },
        "usage": { "output_tokens": output_tokens }
    });
    events.push(format!(
        "event: message_delta\ndata: {}\n\n",
        message_delta
    ));
    events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".into());

    let body = Body::from_stream(stream::iter(
        events
            .into_iter()
            .map(|line| Ok::<_, std::convert::Infallible>(line)),
    ));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap()
}

fn chunk_text_for_anthropic_stream(text: &str) -> Vec<String> {
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

    #[test]
    fn stream_chunks_split_long_text() {
        let chunks = chunk_text_for_anthropic_stream(&"a".repeat(50));
        assert!(chunks.len() >= 2);
        assert_eq!(chunks.concat(), "a".repeat(50));
    }
}
