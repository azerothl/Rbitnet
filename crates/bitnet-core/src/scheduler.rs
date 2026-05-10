//! Lightweight scheduler primitives for continuous batching and speculative decode.

use crate::error::Result;
use crate::model::ModelExecutor;
use crate::timings::PhaseTimings;

/// Placeholder queue for phase B.2 (prefill vs decode interleaving across sequences).
#[derive(Debug, Default, Clone)]
pub struct PrefillDecodeQueue {
    pub prefill_seq_ids: Vec<u64>,
    pub decode_seq_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct InferenceStats {
    /// Time from request start until first output token is ready (encode + prefill), milliseconds.
    pub ttft_ms: u64,
    pub encode_ms: u64,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    /// Average inter-token latency during decode (microseconds per generated token).
    pub itl_us: u64,
    /// Decode throughput helper: same as `itl_us` for this engine (TPOT-style average).
    pub tpot_us: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub speculative_attempted: bool,
}

impl InferenceStats {
    pub fn from_phases(p: PhaseTimings, speculative_attempted: bool) -> Self {
        let itl = p.itl_us();
        let ttft_ms = p.ttft_ms();
        Self {
            ttft_ms,
            encode_ms: p.encode_ms,
            prefill_ms: p.prefill_ms,
            decode_ms: p.decode_ms,
            itl_us: itl,
            tpot_us: itl,
            prompt_tokens: p.prompt_tokens,
            completion_tokens: p.completion_tokens,
            speculative_attempted,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub text: String,
    pub stats: InferenceStats,
}

#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub id: u64,
    pub request: InferenceRequest,
}

#[derive(Debug, Clone)]
pub struct InferenceBatch {
    pub requests: Vec<ScheduledRequest>,
}

fn merge_speculative_phases(a: PhaseTimings, b: PhaseTimings) -> PhaseTimings {
    PhaseTimings {
        encode_ms: a.encode_ms.saturating_add(b.encode_ms),
        prefill_ms: a.prefill_ms.saturating_add(b.prefill_ms),
        decode_ms: a.decode_ms.saturating_add(b.decode_ms),
        prompt_tokens: a.prompt_tokens,
        completion_tokens: a
            .completion_tokens
            .saturating_add(b.completion_tokens),
    }
}

/// MVP scheduler: keeps API stable while preparing for real batching.
#[derive(Debug, Clone)]
pub struct ContinuousBatchScheduler {
    pub enabled: bool,
    pub speculative_enabled: bool,
    pub draft_ratio_num: u32,
    pub draft_ratio_den: u32,
    pub prefill_chunk_tokens: usize,
}

impl ContinuousBatchScheduler {
    pub fn from_env() -> Self {
        let enabled = matches!(
            std::env::var("RBITNET_CONTINUOUS_BATCHING").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let speculative_enabled = matches!(
            std::env::var("RBITNET_SPECULATIVE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let draft_ratio_num = std::env::var("RBITNET_SPEC_DRAFT_RATIO_NUM")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(1);
        let draft_ratio_den = std::env::var("RBITNET_SPEC_DRAFT_RATIO_DEN")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4);
        let prefill_chunk_tokens = std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(128);
        Self {
            enabled,
            speculative_enabled,
            draft_ratio_num,
            draft_ratio_den,
            prefill_chunk_tokens,
        }
    }

    pub fn run(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<InferenceOutput> {
        if self.speculative_enabled {
            self.run_speculative(executor, req)
        } else {
            let (text, phases) =
                executor.generate_with_timings(&req.prompt, req.max_tokens, req.temperature)?;
            Ok(InferenceOutput {
                text,
                stats: InferenceStats::from_phases(phases, false),
            })
        }
    }

    /// Batch entry point used by server/runtime orchestration.
    ///
    /// Current MVP executes requests sequentially while preserving a stable API for
    /// future continuous batching and per-wave scheduling.
    pub fn run_batch(&self, executor: &dyn ModelExecutor, batch: &InferenceBatch) -> Result<Vec<(u64, InferenceOutput)>> {
        if self.enabled && batch.requests.len() > 1 {
            tracing::debug!(
                batch_len = batch.requests.len(),
                "continuous batching: sequential wave (shared batched forward not yet implemented)"
            );
        }
        let mut out = Vec::with_capacity(batch.requests.len());
        for req in &batch.requests {
            let mut req_local = req.request.clone();
            if self.prefill_chunk_tokens > 0 && req_local.prompt.len() > self.prefill_chunk_tokens {
                // Hook for future chunked prefill planning; no prompt rewrite today.
                req_local.prompt.reserve(0);
            }
            let result = self.run(executor, &req_local)?;
            out.push((req.id, result));
        }
        Ok(out)
    }

    fn run_speculative(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<InferenceOutput> {
        let mut draft_tokens =
            req.max_tokens.saturating_mul(self.draft_ratio_num) / self.draft_ratio_den;
        if draft_tokens == 0 {
            draft_tokens = 1;
        }
        let verify_tokens = req.max_tokens.saturating_sub(draft_tokens);
        let (draft, draft_phases) =
            executor.generate_with_timings(&req.prompt, draft_tokens, req.temperature)?;
        if verify_tokens == 0 {
            let stats = InferenceStats::from_phases(draft_phases, true);
            return Ok(InferenceOutput {
                text: draft,
                stats,
            });
        }
        let verify_prompt = format!("{}\n{}", req.prompt, draft);
        let (verify, verify_phases) =
            executor.generate_with_timings(&verify_prompt, verify_tokens, req.temperature)?;
        let text = format!("{draft}{verify}");
        let merged = merge_speculative_phases(draft_phases, verify_phases);
        Ok(InferenceOutput {
            text,
            stats: InferenceStats::from_phases(merged, true),
        })
    }
}
