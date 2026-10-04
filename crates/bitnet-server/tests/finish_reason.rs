//! Real tiny GGUF forward checks for configured capacity and HTTP error timing.
#![allow(clippy::await_holding_lock)]
use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::{create_app_with_config, ServerConfig};
use http::Request;
use http_body_util::BodyExt;
use std::sync::{Arc, LazyLock, Mutex};
use tower::ServiceExt;
static ENV: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
struct Guard(Vec<(String, Option<std::ffi::OsString>)>);
impl Guard {
    fn set(items: &[(&str, Option<&str>)]) -> Self {
        Self(
            items
                .iter()
                .map(|&(key, value)| {
                    let old = std::env::var_os(key);
                    if let Some(value) = value {
                        std::env::set_var(key, value);
                    } else {
                        std::env::remove_var(key);
                    }
                    (key.into(), old)
                })
                .collect(),
        )
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            if let Some(v) = v {
                std::env::set_var(k, v);
            } else {
                std::env::remove_var(k);
            }
        }
    }
}
fn fixture(path: &std::path::Path, eos: bool) {
    fn text(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    let tensors: [(&str, &[u64]); 12] = [
        ("token_embd.weight", &[8, 8]),
        ("output_norm.weight", &[8]),
        ("output.weight", &[8, 8]),
        ("blk.0.attn_norm.weight", &[8]),
        ("blk.0.attn_q.weight", &[8, 8]),
        ("blk.0.attn_k.weight", &[8, 4]),
        ("blk.0.attn_v.weight", &[8, 4]),
        ("blk.0.attn_output.weight", &[8, 8]),
        ("blk.0.ffn_norm.weight", &[8]),
        ("blk.0.ffn_gate.weight", &[8, 16]),
        ("blk.0.ffn_up.weight", &[8, 16]),
        ("blk.0.ffn_down.weight", &[16, 8]),
    ];
    let integers = [
        ("llama.embedding_length", 8u32),
        ("llama.vocab_size", 8),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1),
        ("llama.feed_forward_length", 16),
        ("llama.context_length", 128),
        ("general.alignment", 32),
    ];
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((integers.len() as u64 + 2).to_le_bytes());
    for (key, value) in [
        ("general.architecture", "llama"),
        ("general.name", "capacity-fixture"),
    ] {
        text(&mut out, key);
        out.extend(8u32.to_le_bytes());
        text(&mut out, value);
    }
    for (key, value) in integers {
        text(&mut out, key);
        out.extend(4u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    let mut offset = 0u64;
    for (name, dims) in tensors {
        text(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(0u32.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += dims.iter().product::<u64>() * 4;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for (index, (name, dims)) in tensors.into_iter().enumerate() {
        for i in 0..dims.iter().product::<u64>() as usize {
            let value = if name.contains("norm") {
                1.0f32
            } else if name == "token_embd.weight" {
                // Every input has a positive, nonzero hidden vector. Only the
                // visible Hello row of the head receives a positive score.
                (i % 8 + 1) as f32 / 32.
            } else if name == "output.weight" {
                if i / 8 == (if eos { 2 } else { 1 }) {
                    (i % 8 + 1) as f32 / 32.
                } else {
                    0.
                }
            } else {
                ((i * 7 + index * 13) % 31) as f32 / 512. - 0.025
            };
            out.extend(value.to_le_bytes());
        }
    }
    std::fs::write(path, out).unwrap();
}
fn request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn actual_tiny_llama_distinguishes_eos_budget_and_explicit_utf8_stops_in_json_and_sse() {
    use bitnet_core::timings::GenerationFinishReason;
    let _lock = ENV.lock().unwrap();
    let _guard = Guard::set(&[
        ("RBITNET_STUB", None),
        ("RBITNET_TOY", None),
        ("RBITNET_BACKEND", Some("cpu")),
        ("RBITNET_MAX_SEQ", Some("64")),
        ("RBITNET_LLAMA_WEIGHT_MODE", Some("dense")),
        ("RBITNET_PREFIX_KV", Some("0")),
        ("RBITNET_CHAT_FORMAT", Some("raw")),
        ("RBITNET_CHAT_TEMPLATE", None),
        ("RBITNET_KV_POOL", Some("0")),
        ("RBITNET_CONTINUOUS_BATCHING", Some("0")),
        ("RBITNET_SPECULATIVE", Some("0")),
        ("RBITNET_SPECULATIVE_PLD", Some("0")),
    ]);
    let mut cases = 0usize;
    for eos in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("model.gguf");
        let tok = dir.path().join("tokenizer.json");
        fixture(&gguf, eos);
        bitnet_core::mixtral::ci_fixture::write_wordlevel_tokenizer(&tok).unwrap();
        let tokenizer = std::fs::read_to_string(&tok)
            .unwrap()
            .replace("\"world\": 2", "\"</s>\": 2");
        std::fs::write(&tok, tokenizer).unwrap();
        let engine =
            Arc::new(Engine::load_path_with_overrides(&gguf, Some(&tok), Some("llama")).unwrap());
        for max_tokens in [0, 1, 2] {
            bitnet_core::clear_inference_cancel();
            let expected = if eos && max_tokens > 0 {
                GenerationFinishReason::Stop
            } else {
                GenerationFinishReason::Length
            };
            let output = engine.complete_detailed("Hello", max_tokens, 0.).unwrap();
            assert_eq!(output.stats.finish_reason, expected);
            assert_eq!(
                output.stats.completion_tokens,
                if eos { 0 } else { max_tokens }
            );
            let mut done = None;
            engine
                .complete_streaming(
                    "Hello",
                    max_tokens,
                    bitnet_core::sampling::SamplingOptions::from_temperature(0.),
                    &mut |ev| {
                        if let bitnet_core::StreamEvent::Done(output) = ev {
                            done = Some(output);
                        }
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(done.unwrap().stats.finish_reason, expected);
            let app = create_app_with_config(
                Arc::clone(&engine),
                Arc::new(ServerConfig::test_defaults()),
            );
            for path in ["/v1/chat/completions", "/v1/completions", "/v1/messages"] {
                for stream in [false, true] {
                    let anthropic = path == "/v1/messages";
                    let body = if path.ends_with("/chat/completions") {
                        serde_json::json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":max_tokens,"temperature":0.,"stream":stream})
                    } else if anthropic {
                        serde_json::json!({"model":"fixture","messages":[{"role":"user","content":"Hello"}],"max_tokens":max_tokens,"temperature":0.,"stream":stream})
                    } else {
                        serde_json::json!({"prompt":"Hello","max_tokens":max_tokens,"temperature":0.,"stream":stream})
                    };
                    let expected_text = if anthropic {
                        Some(if expected == GenerationFinishReason::Stop {
                            "end_turn"
                        } else {
                            "max_tokens"
                        })
                    } else {
                        expected.openai()
                    };
                    let response = app.clone().oneshot(request(path, body)).await.unwrap();
                    assert_eq!(response.status(), 200);
                    let bytes = response.into_body().collect().await.unwrap().to_bytes();
                    if stream {
                        let text = std::str::from_utf8(&bytes).unwrap();
                        let mut finish = None;
                        let mut dones = 0;
                        for line in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
                            if line == "[DONE]" {
                                dones += 1;
                                continue;
                            }
                            let json: serde_json::Value = serde_json::from_str(line).unwrap();
                            assert!(json.get("error").is_none(), "{json}");
                            if anthropic && json["type"] == "message_stop" {
                                dones += 1;
                            }
                            if let Some(value) = (if anthropic {
                                &json["delta"]["stop_reason"]
                            } else {
                                &json["choices"][0]["finish_reason"]
                            })
                            .as_str()
                            {
                                assert!(finish.is_none());
                                finish = Some(value.to_owned());
                            }
                        }
                        assert_eq!(finish.as_deref(), expected_text);
                        assert_eq!(dones, 1);
                    } else {
                        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(
                            (if anthropic {
                                &json["stop_reason"]
                            } else {
                                &json["choices"][0]["finish_reason"]
                            })
                            .as_str(),
                            expected_text
                        );
                    }
                    cases += 1;
                }
            }
        }
        if !eos {
            let app = create_app_with_config(
                Arc::clone(&engine),
                Arc::new(ServerConfig::test_defaults()),
            );
            for (stop, expected) in [("Hello", "stop"), ("Hellox", "length"), ("", "length")] {
                for stream in [false, true] {
                    let body = serde_json::json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":2,"temperature":0.,"stream":stream,"stop":stop});
                    let response = app
                        .clone()
                        .oneshot(request("/v1/chat/completions", body))
                        .await
                        .unwrap();
                    assert_eq!(response.status(), 200);
                    let bytes = response.into_body().collect().await.unwrap().to_bytes();
                    if stream {
                        let rows: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
                            .unwrap()
                            .lines()
                            .filter_map(|l| l.strip_prefix("data: "))
                            .filter(|l| *l != "[DONE]")
                            .map(|l| serde_json::from_str(l).unwrap())
                            .collect();
                        let reasons: Vec<_> = rows
                            .iter()
                            .filter_map(|r| r["choices"][0]["finish_reason"].as_str())
                            .collect();
                        assert_eq!(reasons, [expected]);
                    } else {
                        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(json["choices"][0]["finish_reason"], expected);
                    }
                    cases += 1;
                }
            }
        }
    }
    println!("FINISH_REASON_HTTP_DONE cases={cases}");
    assert_eq!(cases, 42);
}
