use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::model::ModelExecutor;
use bitnet_core::sampling::SamplingOptions;
use bitnet_core::scheduler::{
    ContinuousBatchScheduler, DraftPath, InferenceBatch, InferenceRequest, ScheduledRequest,
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
        _sampling: SamplingOptions,
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

fn test_scheduler(
    enabled: bool,
    speculative: bool,
    draft: DraftPath,
    prefill_chunk: usize,
    budget: usize,
) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler {
        enabled,
        speculative_enabled: speculative,
        draft_ratio_num: 1,
        draft_ratio_den: if speculative { 2 } else { 4 },
        prefill_chunk_tokens: prefill_chunk,
        iteration_token_budget: budget,
        draft_path: draft,
        mtp_k: 1,
    }
}

#[test]
fn scheduler_regular_mode_passthrough() {
    let scheduler = test_scheduler(false, false, DraftPath::TargetModel, 128, 256);
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: 8,
        sampling: SamplingOptions::from_temperature(0.0),
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert_eq!(out.text, "hello[8]");
    assert!(!out.stats.speculative_attempted);
}

#[test]
fn scheduler_speculative_combines_draft_and_verify() {
    let mut scheduler = test_scheduler(true, true, DraftPath::TargetModel, 128, 256);
    scheduler.draft_ratio_num = 1;
    scheduler.draft_ratio_den = 2;
    let req = InferenceRequest {
        prompt: "hi".into(),
        max_tokens: 10,
        sampling: SamplingOptions::from_temperature(0.7),
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert!(out.text.contains("hi[5]"));
    assert!(out.text.contains("hi"));
    assert!(out.stats.speculative_attempted);
}

#[test]
fn scheduler_batch_two_preserves_order() {
    let scheduler = test_scheduler(true, false, DraftPath::TargetModel, 128, 256);
    let batch = InferenceBatch {
        requests: vec![
            ScheduledRequest {
                id: 1,
                request: InferenceRequest {
                    prompt: "a".into(),
                    max_tokens: 2,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
            ScheduledRequest {
                id: 2,
                request: InferenceRequest {
                    prompt: "b".into(),
                    max_tokens: 3,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
        ],
    };
    let rows = scheduler.run_batch(&EchoExecutor, &batch).expect("batch");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[1].0, 2);
}

#[test]
fn continuous_batching_waves_increment_decode_wave_counter() {
    let before = bitnet_core::perf::snapshot().scheduler_decode_waves;
    let scheduler = test_scheduler(true, false, DraftPath::TargetModel, 128, 256);
    let batch = InferenceBatch {
        requests: vec![
            ScheduledRequest {
                id: 10,
                request: InferenceRequest {
                    prompt: "x".into(),
                    max_tokens: 2,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
            ScheduledRequest {
                id: 11,
                request: InferenceRequest {
                    prompt: "y".into(),
                    max_tokens: 2,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
        ],
    };
    let rows = scheduler.run_batch(&EchoExecutor, &batch).expect("waves");
    assert_eq!(rows.len(), 2);
    let after = bitnet_core::perf::snapshot().scheduler_decode_waves;
    assert!(
        after > before,
        "run_batch_waves should record scheduler_decode_waves (before={before} after={after})"
    );
}

/// Counting executor: prompt tokens = whitespace words (for Sarathi budget tests).
struct CountingEcho;

impl ModelExecutor for CountingEcho {
    fn family(&self) -> &'static str {
        "bitnet"
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        Ok(prompt.split_whitespace().count().max(1) as u32)
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
        _sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        Ok((
            format!("#{max_tokens}"),
            PhaseTimings {
                encode_ms: 0,
                prefill_ms: 1,
                decode_ms: 1,
                prompt_tokens: prompt.split_whitespace().count().max(1) as u32,
                completion_tokens: max_tokens,
            },
        ))
    }
}

#[test]
fn sarathi_stall_free_records_prefill_chunks_and_budget_iters() {
    let before_chunks = bitnet_core::perf::snapshot().scheduler_prefill_chunks;
    let before_iters = bitnet_core::perf::snapshot().scheduler_stall_free_iters;
    // Small chunk + budget forces multiple prefill admissions before decode.
    let scheduler = test_scheduler(true, false, DraftPath::TargetModel, 2, 4);
    let batch = InferenceBatch {
        requests: vec![
            ScheduledRequest {
                id: 1,
                request: InferenceRequest {
                    // 8 whitespace tokens → 4 prefill chunks of size 2
                    prompt: "a b c d e f g h".into(),
                    max_tokens: 2,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
            ScheduledRequest {
                id: 2,
                request: InferenceRequest {
                    prompt: "x y".into(),
                    max_tokens: 1,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
        ],
    };
    let rows = scheduler
        .run_batch(&CountingEcho, &batch)
        .expect("sarathi batch");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 1);
    assert_eq!(rows[1].0, 2);
    let snap = bitnet_core::perf::snapshot();
    assert!(
        snap.scheduler_prefill_chunks > before_chunks,
        "expected prefill chunks (before={before_chunks} after={})",
        snap.scheduler_prefill_chunks
    );
    assert!(
        snap.scheduler_stall_free_iters > before_iters,
        "expected stall-free iters (before={before_iters} after={})",
        snap.scheduler_stall_free_iters
    );
    assert_eq!(snap.scheduler_iteration_budget, 4);
    assert!(snap.scheduler_decode_waves > 0);
}

#[test]
fn sarathi_prefill_decode_queue_starts_prefill_only() {
    use bitnet_core::scheduler::PrefillDecodeQueue;
    let batch = InferenceBatch {
        requests: vec![
            ScheduledRequest {
                id: 7,
                request: InferenceRequest {
                    prompt: "hi".into(),
                    max_tokens: 1,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
            ScheduledRequest {
                id: 8,
                request: InferenceRequest {
                    prompt: "yo".into(),
                    max_tokens: 1,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
        ],
    };
    let mut q = PrefillDecodeQueue::from_batch(&batch);
    assert_eq!(q.prefill_seq_ids, vec![7, 8]);
    assert!(q.decode_seq_ids.is_empty());
    q.promote_to_decode(7);
    assert_eq!(q.prefill_seq_ids, vec![8]);
    assert_eq!(q.decode_seq_ids, vec![7]);
    q.mark_done(7);
    assert_eq!(q.decode_seq_ids, Vec::<u64>::new());
}
