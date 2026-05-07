//! Lightweight scheduler primitives for continuous batching and speculative decode.

use crate::error::Result;
use crate::model::ModelExecutor;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub ttft_ms: u64,
    pub tpot_us: u64,
    pub completion_tokens: u32,
    pub speculative_attempted: bool,
}

#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub text: String,
    pub stats: InferenceStats,
}

/// MVP scheduler: keeps API stable while preparing for real batching.
#[derive(Debug, Clone)]
pub struct ContinuousBatchScheduler {
    pub enabled: bool,
    pub speculative_enabled: bool,
    pub draft_ratio_num: u32,
    pub draft_ratio_den: u32,
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
        Self {
            enabled,
            speculative_enabled,
            draft_ratio_num,
            draft_ratio_den,
        }
    }

    pub fn run(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<InferenceOutput> {
        if self.speculative_enabled {
            self.run_speculative(executor, req)
        } else {
            let start = Instant::now();
            let text = executor.generate(&req.prompt, req.max_tokens, req.temperature)?;
            let elapsed_us = start.elapsed().as_micros() as u64;
            let completion_tokens = text.split_whitespace().count() as u32;
            let tpot_us = if completion_tokens > 0 {
                elapsed_us / completion_tokens as u64
            } else {
                elapsed_us
            };
            Ok(InferenceOutput {
                text,
                stats: InferenceStats {
                    ttft_ms: (elapsed_us / 1000).max(1),
                    tpot_us,
                    completion_tokens,
                    speculative_attempted: false,
                },
            })
        }
    }

    fn run_speculative(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<InferenceOutput> {
        let mut draft_tokens =
            req.max_tokens.saturating_mul(self.draft_ratio_num) / self.draft_ratio_den;
        if draft_tokens == 0 {
            draft_tokens = 1;
        }
        let verify_tokens = req.max_tokens.saturating_sub(draft_tokens);
        let start = Instant::now();
        let draft = executor.generate(&req.prompt, draft_tokens, req.temperature)?;
        let ttft_ms = (start.elapsed().as_millis() as u64).max(1);
        if verify_tokens == 0 {
            let completion_tokens = draft.split_whitespace().count() as u32;
            let elapsed_us = start.elapsed().as_micros() as u64;
            let tpot_us = if completion_tokens > 0 {
                elapsed_us / completion_tokens as u64
            } else {
                elapsed_us
            };
            return Ok(InferenceOutput {
                text: draft,
                stats: InferenceStats {
                    ttft_ms,
                    tpot_us,
                    completion_tokens,
                    speculative_attempted: true,
                },
            });
        }
        let verify_prompt = format!("{}\n{}", req.prompt, draft);
        let verify = executor.generate(&verify_prompt, verify_tokens, req.temperature)?;
        let text = format!("{draft}{verify}");
        let completion_tokens = text.split_whitespace().count() as u32;
        let elapsed_us = start.elapsed().as_micros() as u64;
        let tpot_us = if completion_tokens > 0 {
            elapsed_us / completion_tokens as u64
        } else {
            elapsed_us
        };
        Ok(InferenceOutput {
            text,
            stats: InferenceStats {
                ttft_ms,
                tpot_us,
                completion_tokens,
                speculative_attempted: true,
            },
        })
    }
}
