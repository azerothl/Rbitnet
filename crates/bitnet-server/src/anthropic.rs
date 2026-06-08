//! Anthropic Messages API subset (`POST /v1/messages`) for agent clients.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
) -> Result<Json<MessagesResponse>, (StatusCode, String)> {
    let prompt = anthropic_messages_to_prompt(&req.messages);
    let sampling = SamplingOptions::from_temperature(req.temperature);
    let eng = state.engine.read().await;
    let output = eng
        .complete_detailed_with_options(&prompt, req.max_tokens, sampling)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let id = format!("msg_{}", uuid::Uuid::new_v4());
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
    }))
}

fn anthropic_messages_to_prompt(messages: &[AnthropicMessage]) -> String {
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
