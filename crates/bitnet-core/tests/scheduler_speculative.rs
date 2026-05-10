use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::model::ModelExecutor;
use bitnet_core::scheduler::{
    ContinuousBatchScheduler, InferenceBatch, InferenceRequest, ScheduledRequest,
};
use bitnet_core::PhaseTimings;
use bitnet_core::Result;

struct EchoExecutor;

impl ModelExecutor for EchoExecutor {
    fn family(&self) -> &'static str {
        "bitnet"
    }

    fn count_prompt_tokens(&self, _prompt: &str) -> Result<u32> {
        Ok(1)
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn backend_accelerated(&self) -> bool {
        false
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn openai_model_id(&self, _gguf: Option<&GgufArchive>) -> Option<String> {
        Some("rbitnet-bitnet".into())
    }

    fn generate_with_timings(
        &self,
        prompt: &str,
        max_tokens: u32,
        _temperature: f32,
    ) -> Result<(String, PhaseTimings)> {
        Ok((
            format!("{prompt}[{max_tokens}]"),
            PhaseTimings {
                encode_ms: 0,
                prefill_ms: 1,
                decode_ms: 0,
                prompt_tokens: 1,
                completion_tokens: max_tokens,
            },
        ))
    }
}

#[test]
fn scheduler_regular_mode_passthrough() {
    let scheduler = ContinuousBatchScheduler {
        enabled: false,
        speculative_enabled: false,
        draft_ratio_num: 1,
        draft_ratio_den: 4,
        prefill_chunk_tokens: 128,
    };
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: 8,
        temperature: 0.0,
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert_eq!(out.text, "hello[8]");
    assert!(!out.stats.speculative_attempted);
}

#[test]
fn scheduler_speculative_combines_draft_and_verify() {
    let scheduler = ContinuousBatchScheduler {
        enabled: true,
        speculative_enabled: true,
        draft_ratio_num: 1,
        draft_ratio_den: 2,
        prefill_chunk_tokens: 128,
    };
    let req = InferenceRequest {
        prompt: "hi".into(),
        max_tokens: 10,
        temperature: 0.7,
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert!(out.text.contains("hi[5]"));
    assert!(out.text.contains("hi"));
    assert!(out.stats.speculative_attempted);
}

#[test]
fn scheduler_batch_two_preserves_order() {
    let scheduler = ContinuousBatchScheduler {
        enabled: true,
        speculative_enabled: false,
        draft_ratio_num: 1,
        draft_ratio_den: 4,
        prefill_chunk_tokens: 128,
    };
    let batch = InferenceBatch {
        requests: vec![
            ScheduledRequest {
                id: 1,
                request: InferenceRequest {
                    prompt: "a".into(),
                    max_tokens: 2,
                    temperature: 0.0,
                },
            },
            ScheduledRequest {
                id: 2,
                request: InferenceRequest {
                    prompt: "b".into(),
                    max_tokens: 3,
                    temperature: 0.0,
                },
            },
        ],
    };
    let rows = scheduler
        .run_batch(&EchoExecutor, &batch)
        .expect("batch");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[1].0, 2);
}
