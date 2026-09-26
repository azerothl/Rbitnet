//! Lightweight scheduler primitives for continuous batching and speculative decode.

use std::collections::HashMap;
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
        if self.enabled && batch.requests.len() > 1 {
            return self.run_batch_waves(executor, batch);
        }
        self.run_batch_sequential(executor, batch)
    }

    fn run_batch_sequential(
        &self,
        executor: &dyn ModelExecutor,
        batch: &InferenceBatch,
    ) -> Result<Vec<(u64, InferenceOutput)>> {
        crate::perf::record_scheduler_batch(batch.requests.len());
        let mut out = Vec::with_capacity(batch.requests.len());
        for req in &batch.requests {
            let mut req_local = req.request.clone();
            if self.prefill_chunk_tokens > 0 && req_local.prompt.len() > self.prefill_chunk_tokens {
                req_local.prompt.reserve(0);
            }
            let result = self.run(executor, &req_local)?;
            out.push((req.id, result));
        }
        Ok(out)
    }

    /// Interleaved decode wave: run each request for one token at a time when batching is enabled.
    pub fn run_batch_waves(
        &self,
        executor: &dyn ModelExecutor,
        batch: &InferenceBatch,
    ) -> Result<Vec<(u64, InferenceOutput)>> {
        crate::perf::record_scheduler_batch(batch.requests.len());
        let queue = PrefillDecodeQueue::from_batch(batch);
        tracing::debug!(
            batch_len = batch.requests.len(),
            prefill = ?queue.prefill_seq_ids,
            decode = ?queue.decode_seq_ids,
            "continuous batching: interleaved decode waves"
        );
        crate::perf::record_scheduler_decode_wave(batch.requests.len());

        if crate::inference_session::sessions_enabled() {
            let store = crate::inference_session::global_sessions();
            if let Ok(mut g) = store.lock() {
                for _req in &batch.requests {
                    let _ = g.open(0);
                }
            }
        }

        let mut pending: Vec<_> = batch
            .requests
            .iter()
            .map(|r| {
                (
                    r.id,
                    InferenceRequest {
                        prompt: r.request.prompt.clone(),
                        max_tokens: 1,
                        sampling: r.request.sampling,
                    },
                )
            })
            .collect();

        let mut acc: HashMap<u64, (String, PhaseTimings)> = HashMap::new();
        while !pending.is_empty() {
            let mut next = Vec::new();
            for (id, mut req) in pending {
                let (chunk, phases) = executor.generate_with_timings(
                    &req.prompt,
                    req.max_tokens,
                    req.sampling,
                )?;
                let entry = acc.entry(id).or_insert_with(|| (String::new(), PhaseTimings::default()));
                entry.0.push_str(&chunk);
                entry.1.encode_ms = entry.1.encode_ms.saturating_add(phases.encode_ms);
                entry.1.prefill_ms = entry.1.prefill_ms.saturating_add(phases.prefill_ms);
                entry.1.decode_ms = entry.1.decode_ms.saturating_add(phases.decode_ms);
                entry.1.prompt_tokens = entry.1.prompt_tokens.max(phases.prompt_tokens);
                entry.1.completion_tokens = entry
                    .1
                    .completion_tokens
                    .saturating_add(phases.completion_tokens);
                let orig = batch.requests.iter().find(|r| r.id == id).unwrap();
                if entry.1.completion_tokens < orig.request.max_tokens {
                    req.prompt = format!("{}{}", orig.request.prompt, entry.0);
                    req.max_tokens = 1;
                    next.push((id, req));
                }
            }
            pending = next;
        }

        let mut out = Vec::with_capacity(batch.requests.len());
        for req in &batch.requests {
            if let Some((text, phases)) = acc.remove(&req.id) {
                out.push((
                    req.id,
                    InferenceOutput {
                        text,
                        stats: InferenceStats::from_phases(phases, false),
                    },
                ));
            }
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
        draft_tokens = draft_tokens.min(req.max_tokens).max(1);

        let (draft, draft_phases) = self
            .draft_path
            .generate(&req.prompt, draft_tokens, req.sampling)
            .unwrap_or_else(|| {
                executor.generate_with_timings(&req.prompt, draft_tokens, req.sampling)
            })?;

        // Verify: generate the same budget from the target and accept the common prefix
        // (lossless frame [2211.17192]; PLD/n-gram drafts need no extra weights).
        let greedy = SamplingOptions {
            temperature: 0.0,
            ..req.sampling
        };
        let (verify_text, verify_phases) =
            executor.generate_with_timings(&req.prompt, draft_tokens, greedy)?;
        let (accepted, accepted_n) = accept_draft_prefix(&draft, &verify_text);
        let accepted_tokens = accepted_n.min(draft_phases.completion_tokens);

        let remaining = req.max_tokens.saturating_sub(accepted_tokens);
        // Continue from accepted prefix when draft was partial.
        let (tail, tail_phases) = if remaining > 0 {
            let continue_prompt = format!("{}{}", req.prompt, accepted);
            let (t, p) =
                executor.generate_with_timings(&continue_prompt, remaining, req.sampling)?;
            (t, p)
        } else {
            (String::new(), PhaseTimings::default())
        };

        let text = format!("{accepted}{tail}");
        let mut verify_acc = verify_phases.clone();
        verify_acc.completion_tokens = verify_acc
            .completion_tokens
            .saturating_add(tail_phases.completion_tokens);
        verify_acc.encode_ms = verify_acc.encode_ms.saturating_add(tail_phases.encode_ms);
        verify_acc.prefill_ms = verify_acc
            .prefill_ms
            .saturating_add(tail_phases.prefill_ms);
        verify_acc.decode_ms = verify_acc.decode_ms.saturating_add(tail_phases.decode_ms);

        crate::perf::record_speculative(
            draft_phases.completion_tokens,
            verify_phases.completion_tokens,
            accepted_tokens,
        );
        Ok(InferenceOutput {
            text,
            stats: InferenceStats::from_speculative_phases(draft_phases, verify_acc),
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
            .unwrap_or_else(|_| {
                // When speculative is on, prefer weight-free PLD/n-gram by default.
                if matches!(
                    std::env::var("RBITNET_SPECULATIVE").as_deref(),
                    Ok("1") | Ok("true") | Ok("yes")
                ) {
                    "ngram".into()
                } else {
                    "target".into()
                }
            })
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ngram" | "pld" | "prompt-lookup" => Self::Ngram,
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
                match crate::gguf::GgufArchive::mmap_path(path) {
                    Ok(gguf) => {
                        let g = std::sync::Arc::new(gguf);
                        if let Ok(ex) = crate::loaders::dispatch_gguf_executor(
                            crate::backend::BackendKind::Cpu,
                            g,
                            path,
                        ) {
                            return Some(
                                ex.generate_with_timings(prompt, max_tokens, _sampling),
                            );
                        }
                    }
                    Err(e) => tracing::warn!(draft_model = %path.display(), error = %e, "draft GGUF mmap failed"),
                }
                tracing::warn!(
                    draft_model = %path.display(),
                    "RBITNET_DRAFT_MODEL falling back to n-gram draft"
                );
                Some(Ok(ngram_draft(prompt, max_tokens)))
            }
            Self::Ngram => Some(Ok(ngram_draft(prompt, max_tokens))),
            Self::Toy => Some(Ok(toy_draft(max_tokens))),
        }
    }
}

fn ngram_draft(prompt: &str, max_tokens: u32) -> (String, PhaseTimings) {
    let draft = prompt_lookup_draft(prompt, max_tokens as usize);
    let completion_tokens = draft.split_whitespace().count().max(1) as u32;
    (
        draft,
        PhaseTimings {
            // PLD is CPU string work — attribute to encode for TTFT accounting.
            encode_ms: 0,
            prefill_ms: 0,
            decode_ms: 0,
            completion_tokens: completion_tokens.min(max_tokens.max(1)),
            ..Default::default()
        },
    )
}

/// Prompt Lookup Decoding (Saxena): copy the continuation after the longest n-gram
/// match of the prompt suffix against earlier prompt windows. No draft model weights.
pub fn prompt_lookup_draft(prompt: &str, max_tokens: usize) -> String {
    let words: Vec<&str> = prompt.split_whitespace().collect();
    if words.is_empty() || max_tokens == 0 {
        return String::new();
    }
    let max_n = words.len().min(5).max(1);
    for n in (1..=max_n).rev() {
        if words.len() < n {
            continue;
        }
        let needle = &words[words.len() - n..];
        // Search earlier windows (exclude the suffix itself).
        let search_end = words.len().saturating_sub(n);
        if search_end == 0 {
            continue;
        }
        let mut best_i: Option<usize> = None;
        for i in (0..search_end).rev() {
            if i + n > words.len() {
                continue;
            }
            if &words[i..i + n] == needle {
                best_i = Some(i);
                break;
            }
        }
        if let Some(i) = best_i {
            let start = i + n;
            if start < words.len() {
                let take = max_tokens.min(words.len() - start);
                if take > 0 {
                    return words[start..start + take].join(" ");
                }
            }
        }
    }
    // Fallback: repeat last word (still weight-free).
    let seed = words.last().copied().unwrap_or("ok");
    std::iter::repeat(seed)
        .take(max_tokens.max(1))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Accept the longest whitespace-token prefix shared by draft and target verify text.
fn accept_draft_prefix(draft: &str, verify: &str) -> (String, u32) {
    let d: Vec<&str> = draft.split_whitespace().collect();
    let v: Vec<&str> = verify.split_whitespace().collect();
    let mut n = 0usize;
    while n < d.len() && n < v.len() && d[n] == v[n] {
        n += 1;
    }
    if n == 0 {
        // No agreement — fall back to verify text (target is source of truth).
        return (verify.to_string(), v.len() as u32);
    }
    (d[..n].join(" "), n as u32)
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

#[cfg(test)]
mod tests {
    use super::{accept_draft_prefix, prompt_lookup_draft};

    #[test]
    fn pld_copies_continuation_after_suffix_ngram() {
        let prompt = "the cat sat on the mat the cat sat";
        let draft = prompt_lookup_draft(prompt, 3);
        // Suffix "the cat sat" matches earlier; continuation "on the mat".
        assert_eq!(draft, "on the mat");
    }

    #[test]
    fn draft_accept_counts_common_prefix() {
        let (text, n) = accept_draft_prefix("hello world foo", "hello world bar");
        assert_eq!(text, "hello world");
        assert_eq!(n, 2);
    }

    #[test]
    fn draft_accept_falls_back_to_verify_on_mismatch() {
        let (text, n) = accept_draft_prefix("aaa bbb", "xxx yyy");
        assert_eq!(text, "xxx yyy");
        assert_eq!(n, 2);
    }
}
