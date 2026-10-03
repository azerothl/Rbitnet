//! Opt-in parity check of GGUF Jinja templates against llama.cpp-exported prompt fixtures.
use bitnet_core::gguf::{GgufArchive, GgufValue};
use bitnet_server::{build_prompt_from_messages_with_tokenizer_template, ChatMessage};
use std::{fs, path::Path};
fn main() {
    std::env::remove_var("RBITNET_CHAT_TEMPLATE");
    std::env::remove_var("RBITNET_CHAT_FORMAT");
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "check_chat_templates MODEL.gguf prompts.json"
    );
    let archive = GgufArchive::mmap_path(Path::new(&args[1])).unwrap();
    let Some(GgufValue::String(template)) = archive.metadata.get("tokenizer.chat_template") else {
        panic!("GGUF chat template missing");
    };
    let fixtures: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(&args[2]).unwrap()).unwrap();
    for fixture in &fixtures {
        let messages: Vec<ChatMessage> =
            serde_json::from_value(fixture["messages"].clone()).unwrap();
        let actual = build_prompt_from_messages_with_tokenizer_template(&messages, Some(template));
        assert_eq!(
            actual,
            fixture["prompt"].as_str().unwrap(),
            "{}: rendered prompt differs",
            fixture["id"]
        );
    }
    println!(
        "{}",
        serde_json::json!({"architecture":archive.normalized_architecture(),"fixtures":fixtures.len(),"all_prompts_exact":true})
    );
}
