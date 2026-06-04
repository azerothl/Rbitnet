//! Lightweight scheduler primitives for continuous batching and speculative decode.

use std::path::PathBuf;

use crate::error::Result;
use crate::model::ModelExecutor;
use crate::sampling::SamplingOptions;
use crate::stream::StreamEvent;
use crate::timings::PhaseTimings;

/// Placeholder queue for phase B.2 (prefill vs decode interleaving across sequences).
#[derive(Debug, Default, Clone)]
pub struct PrefillDecodeQueue {
    pub prefill_seq_ids: Vec<u64>,
    pub decode_seq_ids: Vec<u64>,
}

impl PrefillDecodeQueue {
    pub fn from_batch(batch: &InferenceBatch) -> Self {
        Self {
            prefill_seq_ids: batch.requests.iter().map(|r| r.id).collect(),
            decode_seq_ids: batch.requests.iter().map(|r| r.id).collect(),
        }
    }
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
        let total_wall_ms = p
            .encode_ms
            .saturating_add(p.prefill_ms)
            .saturating_add(p.decode_ms);
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
        let total_wall_ms = encode_ms
            .saturating_add(prefill_ms)
            .saturating_add(decode_ms);
        let completion_tokens = draft
            .completion_tokens
            .saturating_add(verify.completion_tokens);
        let decode_us = decode_ms.saturating_mul(1000);
        let itl = if completion_tokens == 0 {
            0
        } else {
            decode_us / completion_tokens as u64
        };
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
    pub draft_path: DraftPath,
    /// Multi-token prediction width (Atlas-style MTP). `1` disables MTP bursts.
    pub mtp_k: u32,
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
        let draft_path = DraftPath::from_env();
        let mtp_k = std::env::var("RBITNET_MTP_K")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|v| *v > 1)
            .unwrap_or(1);
        Self {
            enabled,
            speculative_enabled,
            draft_ratio_num,
            draft_ratio_den,
            prefill_chunk_tokens,
            draft_path,
            mtp_k,
        }
    }

    pub fn run(
        &self,
        executor: &dyn ModelExecutor,
        req: &InferenceRequest,
    ) -> Result<InferenceOutput> {
        if self.speculative_enabled {
            self.run_speculative(executor, req)
        } else if self.mtp_k > 1 {
            self.run_mtp_burst(executor, req)
        } else {
            let (text, phases) =
                executor.generate_with_timings(&req.prompt, req.max_tokens, req.sampling)?;
            Ok(InferenceOutput {
                text,
                stats: InferenceStats::from_phases(phases, false),
            })
        }
    }

    fn run_mtp_burst(
        &self,
        executor: &dyn ModelExecutor,
        req: &InferenceRequest,
    ) -> Result<InferenceOutput> {
        let first_burst = req.max_tokens.min(self.mtp_k);
        crate::perf::record_scheduler_decode_wave(1);
        let (mut text, mut phases_acc) =
            executor.generate_with_timings(&req.prompt, first_burst, req.sampling)?;
        let remaining = req.max_tokens.saturating_sub(phases_acc.completion_tokens);
        if remaining > 0 {
            let tail_prompt = format!("{}{}", req.prompt, text);
            let (tail, tail_phases) =
                executor.generate_with_timings(&tail_prompt, remaining, req.sampling)?;
            text.push_str(&tail);
            phases_acc.encode_ms = phases_acc
                .encode_ms
                .saturating_add(tail_phases.encode_ms);
            phases_acc.prefill_ms = phases_acc
                .prefill_ms
                .saturating_add(tail_phases.prefill_ms);
            phases_acc.decode_ms = phases_acc
                .decode_ms
                .saturating_add(tail_phases.decode_ms);
            phases_acc.completion_tokens = phases_acc
                .completion_tokens
                .saturating_add(tail_phases.completion_tokens);
        }
        Ok(InferenceOutput {
            text,
            stats: InferenceStats::from_phases(phases_acc, false),
        })
    }

    pub fn run_streaming(
        &self,
        executor: &dyn ModelExecutor,
        req: &InferenceRequest,
        on_event: &mut (dyn FnMut(StreamEvent) -> Result<()> + Send),
    ) -> Result<()> {
        if self.speculative_enabled {
            let output = self.run_speculative(executor, req)?;
            if !output.text.is_empty() {
                on_event(StreamEvent::Delta {
                    text: output.text.clone(),
                })?;
            }
            on_event(StreamEvent::Done(output))?;
            return Ok(());
        }
        executor.generate_streaming(
            &req.prompt,
            req.max_tokens,
            req.sampling,
            on_event,
        )
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
        crate::perf::record_scheduler_batch(batch.requests.len());
        let queue = PrefillDecodeQueue::from_batch(batch);
        if self.enabled && batch.requests.len() > 1 {
            tracing::debug!(
                batch_len = batch.requests.len(),
                prefill = ?queue.prefill_seq_ids,
                decode = ?queue.decode_seq_ids,
                "continuous batching: decode wave scheduling (fused forward pending)"
            );
            crate::perf::record_scheduler_decode_wave(batch.requests.len());
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
        let (draft, draft_phases) = self
            .draft_path
            .generate(&req.prompt, draft_tokens, req.sampling)
            .unwrap_or_else(|| {
                executor.generate_with_timings(&req.prompt, draft_tokens, req.sampling)
            })?;
        if verify_tokens == 0 {
            crate::perf::record_speculative(draft_tokens, 0, draft_phases.completion_tokens);
            let stats = InferenceStats::from_phases(draft_phases, true);
            return Ok(InferenceOutput { text: draft, stats });
        }
        let verify_prompt = format!("{}\n{}", req.prompt, draft);
        let (verify, verify_phases) =
            executor.generate_with_timings(&verify_prompt, verify_tokens, req.sampling)?;
        let text = format!("{draft}{verify}");
        crate::perf::record_speculative(
            draft_tokens,
            verify_phases.completion_tokens,
            draft_phases.completion_tokens,
        );
        Ok(InferenceOutput {
            text,
            stats: InferenceStats::from_speculative_phases(draft_phases, verify_phases),
        })
    }
}

#[derive(Debug, Clone)]
pub enum DraftPath {
    TargetModel,
    Ngram,
    Toy,
    ExternalGguf(PathBuf),
}

impl DraftPath {
    fn from_env() -> Self {
        if let Ok(path) = std::env::var("RBITNET_DRAFT_MODEL") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Self::ExternalGguf(path);
            }
        }
        match std::env::var("RBITNET_DRAFT_PATH")
            .unwrap_or_else(|_| "target".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ngram" => Self::Ngram,
            "toy" => Self::Toy,
            _ => Self::TargetModel,
        }
    }

    fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        _sampling: SamplingOptions,
    ) -> Option<Result<(String, PhaseTimings)>> {
        match self {
            Self::TargetModel => None,
            Self::ExternalGguf(path) => {
                tracing::warn!(
                    draft_model = %path.display(),
                    "RBITNET_DRAFT_MODEL configured; GGUF draft executor is planned, falling back to n-gram draft"
                );
                Some(Ok(ngram_draft(prompt, max_tokens)))
            }
            Self::Ngram => Some(Ok(ngram_draft(prompt, max_tokens))),
            Self::Toy => Some(Ok(toy_draft(max_tokens))),
        }
    }
}

fn ngram_draft(prompt: &str, max_tokens: u32) -> (String, PhaseTimings) {
    let seed = prompt
        .split_whitespace()
        .rev()
        .find(|s| !s.trim().is_empty())
        .unwrap_or("ok");
    let mut out = String::new();
    for i in 0..max_tokens {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(seed);
    }
    (
        out,
        PhaseTimings {
            completion_tokens: max_tokens,
            ..Default::default()
        },
    )
}

fn toy_draft(max_tokens: u32) -> (String, PhaseTimings) {
    (
        " ok".repeat(max_tokens as usize),
        PhaseTimings {
            completion_tokens: max_tokens,
            ..Default::default()
        },
    )
}
