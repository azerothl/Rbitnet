//! Integration tests: OpenAI-shaped routes expected by Akasha `BitNetProvider`.

use std::sync::Arc;

use axum::body::Body;
use bitnet_core::inference::Engine;
use bitnet_server::create_app;
use http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
async fn openai_stub_models_and_chat() {
    std::env::remove_var("RBITNET_MODEL");
    std::env::remove_var("RBITNET_TOY");
    std::env::set_var("RBITNET_STUB", "1");

    let engine = Arc::new(Engine::from_env().expect("engine"));
    let app = create_app(Arc::clone(&engine));

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("models response");
    assert!(res.status().is_success());
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "rbitnet-stub");

    let chat_body = serde_json::json!({
        "model": "any",
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 32,
        "temperature": 0.5,
        "stream": false
    });
    let app = create_app(engine);
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_body.to_string()))
                .unwrap(),
        )
        .await
        .expect("chat response");
    assert!(res.status().is_success());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(text.contains("stub"));

    std::env::remove_var("RBITNET_STUB");
}
