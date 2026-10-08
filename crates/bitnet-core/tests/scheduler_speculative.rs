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
                finish_reason: Default::default(),
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
        fused_multi_seq: false,
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
fn scheduler_speculative_unsupported_executor_preserves_one_target_generation() {
    let mut scheduler = test_scheduler(true, true, DraftPath::TargetModel, 128, 256);
    scheduler.draft_ratio_num = 1;
    scheduler.draft_ratio_den = 2;
    let req = InferenceRequest {
        prompt: "hi".into(),
        max_tokens: 10,
        sampling: SamplingOptions::from_temperature(0.7),
    };
    let out = scheduler.run(&EchoExecutor, &req).expect("run");
    assert_eq!(out.text, "hi[10]");
    assert_eq!(out.stats.completion_tokens, 10);
    assert_eq!(out.stats.prefill_ms, 1);
    assert!(!out.stats.speculative_attempted);
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
                finish_reason: Default::default(),
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
    assert_eq!(
        scheduler.iteration_token_budget, 4,
        "test scheduler must use a 4-token iteration budget"
    );
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
    // Global gauge is last-write-wins across parallel tests in this binary; do not
    // require exact equality with this test's budget (races with budget=256 cases).
    assert!(
        snap.scheduler_iteration_budget > 0,
        "stall-free path should record a positive iteration budget gauge"
    );
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

/// Executor that records whether [`ModelExecutor::generate_decode_batch`] was used.
struct BatchAwareEcho {
    batch_calls: std::sync::atomic::AtomicUsize,
    prefill_rows: std::sync::atomic::AtomicUsize,
    seq_calls: std::sync::atomic::AtomicUsize,
}

impl ModelExecutor for BatchAwareEcho {
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
        self.seq_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((
            format!("{prompt}[{max_tokens}]"),
            PhaseTimings {
                encode_ms: 0,
                prefill_ms: 1,
                decode_ms: 0,
                prompt_tokens: 1,
                completion_tokens: max_tokens,
                finish_reason: Default::default(),
            },
        ))
    }

    fn generate_decode_batch(
        &self,
        items: &[(u64, String, u32, SamplingOptions)],
    ) -> Result<Vec<(String, PhaseTimings)>> {
        self.batch_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut out = Vec::with_capacity(items.len());
        for (_, prompt, max_tokens, sampling) in items {
            // Still use per-item generate (toy) but count the batch entrypoint.
            let (text, phases) = self.generate_with_timings(prompt, *max_tokens, *sampling)?;
            // Undo seq_calls inflation from the above — batch path owns the wave.
            self.seq_calls
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            out.push((text, phases));
        }
        Ok(out)
    }

    fn generate_prefill_batch(
        &self,
        items: &[(u64, String, usize, SamplingOptions)],
    ) -> Result<()> {
        self.prefill_rows
            .fetch_add(items.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn fused_multi_seq_decode_uses_generate_decode_batch() {
    let mut scheduler = test_scheduler(true, false, DraftPath::TargetModel, 128, 256);
    scheduler.fused_multi_seq = true;
    let exec = BatchAwareEcho {
        batch_calls: std::sync::atomic::AtomicUsize::new(0),
        prefill_rows: std::sync::atomic::AtomicUsize::new(0),
        seq_calls: std::sync::atomic::AtomicUsize::new(0),
    };
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
                    max_tokens: 2,
                    sampling: SamplingOptions::from_temperature(0.0),
                },
            },
        ],
    };
    let rows = scheduler.run_batch(&exec, &batch).expect("fused batch");
    assert_eq!(rows.len(), 2);
    let batch_calls = exec.batch_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        batch_calls >= 1,
        "fused_multi_seq should call generate_decode_batch (got {batch_calls})"
    );
    assert_eq!(
        exec.prefill_rows.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "fused scheduler should admit both prompt rows through generate_prefill_batch"
    );
}

struct EosStepExecutor {
    calls: std::sync::atomic::AtomicUsize,
}
impl ModelExecutor for EosStepExecutor {
    fn family(&self) -> &'static str {
        "llama"
    }
    fn backend(&self) -> bitnet_core::backend::BackendKind {
        bitnet_core::backend::BackendKind::Cpu
    }
    fn backend_accelerated(&self) -> bool {
        false
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn openai_model_id(&self, _: Option<&GgufArchive>) -> Option<String> {
        None
    }
    fn count_prompt_tokens(&self, _: &str) -> Result<u32> {
        Ok(1)
    }
    fn generate_with_timings(
        &self,
        prompt: &str,
        _: u32,
        _: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        assert!(
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                < 8,
            "EOS queue must not retry an empty completion"
        );
        let eos = prompt.starts_with("eos");
        Ok((
            if eos { String::new() } else { "x".into() },
            PhaseTimings {
                prompt_tokens: 1,
                completion_tokens: if eos { 0 } else { 1 },
                finish_reason: if eos {
                    bitnet_core::timings::GenerationFinishReason::Stop
                } else {
                    bitnet_core::timings::GenerationFinishReason::Length
                },
                ..Default::default()
            },
        ))
    }
}

struct BurstFinishExecutor {
    first_eos: bool,
    calls: std::sync::atomic::AtomicUsize,
}
impl ModelExecutor for BurstFinishExecutor {
    fn family(&self) -> &'static str {
        "llama"
    }
    fn backend(&self) -> bitnet_core::backend::BackendKind {
        bitnet_core::backend::BackendKind::Cpu
    }
    fn backend_accelerated(&self) -> bool {
        false
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn openai_model_id(&self, _: Option<&GgufArchive>) -> Option<String> {
        None
    }
    fn count_prompt_tokens(&self, _: &str) -> Result<u32> {
        Ok(1)
    }
    fn generate_with_timings(
        &self,
        _: &str,
        max_tokens: u32,
        _: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert!(call < 2, "a completed burst must never restart generation");
        let stop = self.first_eos || call == 1;
        let count = if self.first_eos {
            0
        } else if call == 0 {
            max_tokens
        } else {
            1
        };
        Ok((
            "x".repeat(count as usize),
            PhaseTimings {
                prompt_tokens: 1,
                completion_tokens: count,
                finish_reason: if stop {
                    bitnet_core::timings::GenerationFinishReason::Stop
                } else {
                    bitnet_core::timings::GenerationFinishReason::Length
                },
                ..Default::default()
            },
        ))
    }
}
#[test]
fn legacy_burst_respects_first_eos_and_uses_the_last_actual_termination_reason() {
    use bitnet_core::timings::GenerationFinishReason;
    for first_eos in [true, false] {
        let executor = BurstFinishExecutor {
            first_eos,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut scheduler = test_scheduler(false, false, DraftPath::TargetModel, 1, 8);
        scheduler.mtp_k = 2;
        let result = scheduler
            .run(
                &executor,
                &InferenceRequest {
                    prompt: "burst".into(),
                    max_tokens: 5,
                    sampling: SamplingOptions::from_temperature(0.),
                },
            )
            .unwrap();
        assert_eq!(result.stats.finish_reason, GenerationFinishReason::Stop);
        assert_eq!(
            result.stats.completion_tokens,
            if first_eos { 0 } else { 3 }
        );
        assert_eq!(
            executor.calls.load(std::sync::atomic::Ordering::Relaxed),
            if first_eos { 1 } else { 2 }
        );
    }
}
#[test]
fn eos_zero_output_finishes_both_queue_modes_while_other_members_reach_their_budget() {
    use bitnet_core::scheduler::{InferenceBatch, ScheduledRequest};
    for fused in [false, true] {
        let executor = EosStepExecutor {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut scheduler = test_scheduler(true, false, DraftPath::TargetModel, 1, 8);
        scheduler.fused_multi_seq = fused;
        let batch = InferenceBatch {
            requests: [("eos", 5), ("normal", 3), ("zero", 0)]
                .into_iter()
                .enumerate()
                .map(|(id, (prompt, max_tokens))| ScheduledRequest {
                    id: id as u64,
                    request: InferenceRequest {
                        prompt: prompt.into(),
                        max_tokens,
                        sampling: SamplingOptions::from_temperature(0.),
                    },
                })
                .collect(),
        };
        let result = scheduler.run_batch_waves(&executor, &batch).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].1.stats.finish_reason,
            bitnet_core::timings::GenerationFinishReason::Stop
        );
        assert!(result[0].1.text.is_empty());
        assert_eq!(
            result[1].1.stats.finish_reason,
            bitnet_core::timings::GenerationFinishReason::Length
        );
        assert_eq!(result[1].1.text, "xxx");
        assert_eq!(
            result[2].1.stats.finish_reason,
            bitnet_core::timings::GenerationFinishReason::Length
        );
        assert_eq!(result[2].1.stats.completion_tokens, 0);
        assert_eq!(executor.calls.load(std::sync::atomic::Ordering::Relaxed), 4);
    }
}
