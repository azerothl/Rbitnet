use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::model::ModelExecutor;
use bitnet_core::scheduler::{ContinuousBatchScheduler, InferenceRequest};
use bitnet_core::Result;

struct EchoExecutor;

impl ModelExecutor for EchoExecutor {
    fn family(&self) -> &'static str {
        "bitnet"
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn openai_model_id(&self, _gguf: Option<&GgufArchive>) -> Option<String> {
        Some("rbitnet-bitnet".into())
    }

    fn generate(&self, prompt: &str, max_tokens: u32, _temperature: f32) -> Result<String> {
        Ok(format!("{prompt}[{max_tokens}]"))
    }
}

#[test]
fn scheduler_regular_mode_passthrough() {
    let scheduler = ContinuousBatchScheduler {
        enabled: false,
        speculative_enabled: false,
        draft_ratio_num: 1,
        draft_ratio_den: 4,
    };
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: 8,
        temperature: 0.0,
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert_eq!(out, "hello[8]");
}

#[test]
fn scheduler_speculative_combines_draft_and_verify() {
    let scheduler = ContinuousBatchScheduler {
        enabled: true,
        speculative_enabled: true,
        draft_ratio_num: 1,
        draft_ratio_den: 2,
    };
    let req = InferenceRequest {
        prompt: "hi".into(),
        max_tokens: 10,
        temperature: 0.7,
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert!(out.contains("hi[5]"));
    assert!(out.contains("hi"));
}
