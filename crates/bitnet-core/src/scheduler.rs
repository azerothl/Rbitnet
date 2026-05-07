//! Lightweight scheduler primitives for continuous batching and speculative decode.

use crate::error::Result;
use crate::model::ModelExecutor;

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
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

    pub fn run(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<String> {
        if self.speculative_enabled {
            self.run_speculative(executor, req)
        } else {
            executor.generate(&req.prompt, req.max_tokens, req.temperature)
        }
    }

    fn run_speculative(&self, executor: &dyn ModelExecutor, req: &InferenceRequest) -> Result<String> {
        let mut draft_tokens =
            req.max_tokens.saturating_mul(self.draft_ratio_num) / self.draft_ratio_den;
        if draft_tokens == 0 {
            draft_tokens = 1;
        }
        let verify_tokens = req.max_tokens.saturating_sub(draft_tokens);
        let draft = executor.generate(&req.prompt, draft_tokens, req.temperature)?;
        if verify_tokens == 0 {
            return Ok(draft);
        }
        let verify_prompt = format!("{}\n{}", req.prompt, draft);
        let verify = executor.generate(&verify_prompt, verify_tokens, req.temperature)?;
        Ok(format!("{draft}{verify}"))
    }
}
