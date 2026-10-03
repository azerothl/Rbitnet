//! Render the chat template stored in GGUF rather than inserting Jinja source into prompts.
use crate::{message_content_to_string, ChatMessage};
use minijinja::{context, Environment, Error, ErrorKind};
use serde_json::json;

pub(crate) fn render(source: &str, messages: &[ChatMessage]) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_fuel(Some(1_000_000));
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_function("strftime_now", |format: String| {
        chrono::Utc::now().format(&format).to_string()
    });
    let messages: Vec<_> = messages
        .iter()
        .map(|m| json!({"role":m.role,"content":message_content_to_string(&m.content)}))
        .collect();
    let bos = if source.contains("[INST]") {
        "<s>"
    } else {
        "<|begin_of_text|>"
    };
    env.add_template("chat", source)
        .map_err(|e| format!("invalid GGUF chat template: {e}"))?;
    env.get_template("chat").map_err(|e|e.to_string())?.render(context!(
        messages=>messages,add_generation_prompt=>true,
        enable_thinking=>false,reasoning_effort=>"low",bos_token=>bos,eos_token=>"<|end_of_text|>",
    )).map_err(|e|format!("GGUF chat template render failed: {e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_tools_do_not_enable_tool_instructions() {
        assert_eq!(
            super::render(
                "{% if tools is defined %}tool mode{% else %}chat mode{% endif %}",
                &[]
            )
            .unwrap(),
            "chat mode"
        );
    }
    #[test]
    fn renders_roles_without_treating_user_content_as_template_code() {
        let messages = [crate::ChatMessage {
            role: "user".into(),
            content: serde_json::json!("{{ 7 * 7 }}"),
        }];
        let template="{% for message in messages %}<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}";
        assert_eq!(
            super::render(template, &messages).unwrap(),
            "<|start|>user<|message|>{{ 7 * 7 }}<|end|><|start|>assistant"
        );
    }
}
