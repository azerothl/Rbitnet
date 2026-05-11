//! Lightweight scheduler primitives for continuous batching and speculative decode.

use crate::error::Result;
use crate::model::ModelExecutor;
use crate::sampling::SamplingOptions;
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
    pub sampling: SamplingOptions,
}

#[derive(Debug, Clone)]
pub struct InferenceStats {
    /// Time from request start until first output token is ready (encode + prefill), milliseconds.
    ///
    /// For speculative decoding this reflects the draft phase only, which is the true
    /// wall-clock time until the first generated token is available.
    pub ttft_ms: u64,
    pub encode_ms: u64,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    /// Total wall time across all phases (encode + prefill + decode), milliseconds.
    ///
    /// For speculative decoding this is the sum of draft and verify phase times and
    /// represents the true end-to-end latency of the request.
    pub total_wall_ms: u64,
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
        let total_wall_ms = p.encode_ms.saturating_add(p.prefill_ms).saturating_add(p.decode_ms);
        Self {
            ttft_ms,
            encode_ms: p.encode_ms,
            prefill_ms: p.prefill_ms,
            decode_ms: p.decode_ms,
            total_wall_ms,
            itl_us: itl,
            tpot_us: itl,
            prompt_tokens: p.prompt_tokens,
            completion_tokens: p.completion_tokens,
            speculative_attempted,
        }
    }

    /// Build stats for a completed speculative-decode request.
    ///
    /// `draft` covers the draft generation pass and `verify` covers the
    /// verification pass.  TTFT is taken from the draft phase only, because
    /// that is the moment the first output token becomes available.
    /// `total_wall_ms` accumulates both passes and reflects true end-to-end
    /// latency.
    pub fn from_speculative_phases(draft: PhaseTimings, verify: PhaseTimings) -> Self {
        let ttft_ms = draft.ttft_ms();
        let encode_ms = draft.encode_ms.saturating_add(verify.encode_ms);
        let prefill_ms = draft.prefill_ms.saturating_add(verify.prefill_ms);
        let decode_ms = draft.decode_ms.saturating_add(verify.decode_ms);
        let total_wall_ms = encode_ms.saturating_add(prefill_ms).saturating_add(decode_ms);
        let completion_tokens = draft.completion_tokens.saturating_add(verify.completion_tokens);
        let decode_us = decode_ms.saturating_mul(1000);
        let itl = if completion_tokens == 0 { 0 } else { decode_us / completion_tokens as u64 };
        Self {
            ttft_ms,
            encode_ms,
            prefill_ms,
            decode_ms,
            total_wall_ms,
            itl_us: itl,
            tpot_us: itl,
            prompt_tokens: draft.prompt_tokens,
            completion_tokens,
            speculative_attempted: true,
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

    pub fn run(
        &self,
        executor: &dyn ModelExecutor,
        req: &InferenceRequest,
    ) -> Result<InferenceOutput> {
        if self.speculative_enabled {
            self.run_speculative(executor, req)
        } else {
            let (text, phases) =
                executor.generate_with_timings(&req.prompt, req.max_tokens, req.sampling)?;
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
    pub fn run_batch(
        &self,
        executor: &dyn ModelExecutor,
        batch: &InferenceBatch,
    ) -> Result<Vec<(u64, InferenceOutput)>> {
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

    fn run_speculative(
        &self,
        executor: &dyn ModelExecutor,
        req: &InferenceRequest,
    ) -> Result<InferenceOutput> {
        let mut draft_tokens =
            req.max_tokens.saturating_mul(self.draft_ratio_num) / self.draft_ratio_den;
        if draft_tokens == 0 {
            draft_tokens = 1;
        }
        let verify_tokens = req.max_tokens.saturating_sub(draft_tokens);
        let (draft, draft_phases) =
            executor.generate_with_timings(&req.prompt, draft_tokens, req.sampling)?;
        if verify_tokens == 0 {
            let stats = InferenceStats::from_phases(draft_phases, true);
            return Ok(InferenceOutput { text: draft, stats });
        }
        let verify_prompt = format!("{}\n{}", req.prompt, draft);
        let (verify, verify_phases) =
            executor.generate_with_timings(&verify_prompt, verify_tokens, req.sampling)?;
        let text = format!("{draft}{verify}");
        Ok(InferenceOutput {
            text,
            stats: InferenceStats::from_speculative_phases(draft_phases, verify_phases),
        })
    }
}
