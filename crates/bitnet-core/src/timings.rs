//! Wall-clock phase timings: tokenizer encode vs model prefill vs decode.

/// Why the model stopped. Unknown is reserved for executors that only expose
/// approximate text statistics, never inferred from their output length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GenerationFinishReason {
    Stop,
    Length,
    #[default]
    Unknown,
}
impl GenerationFinishReason {
    pub fn openai(self) -> Option<&'static str> {
        match self {
            Self::Stop => Some("stop"),
            Self::Length => Some("length"),
            Self::Unknown => None,
        }
    }
}

/// Millisecond-resolution timings plus token counts from the tokenizer/runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    pub encode_ms: u64,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: GenerationFinishReason,
}

impl PhaseTimings {
    /// Time to first token: tokenizer encode + prefill (milliseconds).
    pub fn ttft_ms(self) -> u64 {
        self.encode_ms.saturating_add(self.prefill_ms)
    }

    /// Average inter-token latency during decode (microseconds per generated token).
    pub fn itl_us(self) -> u64 {
        if self.completion_tokens == 0 {
            return 0;
        }
        let decode_us = self.decode_ms.saturating_mul(1000);
        decode_us / self.completion_tokens as u64
    }

    /// Fallback when the executor only exposes total wall time (no phase split).
    pub fn from_total_wall_ms(total_ms: u64, completion_tokens_est: u32) -> Self {
        Self {
            encode_ms: 0,
            prefill_ms: 0,
            decode_ms: total_ms,
            prompt_tokens: 0,
            completion_tokens: completion_tokens_est,
            finish_reason: GenerationFinishReason::Unknown,
        }
    }
}

#[cfg(test)]
mod finish_tests {
    use super::*;
    #[test]
    fn legacy_statistics_are_unknown_and_phase_reason_survives_both_combiners() {
        assert_eq!(
            PhaseTimings::from_total_wall_ms(10, 128).finish_reason,
            GenerationFinishReason::Unknown
        );
        assert_eq!(GenerationFinishReason::Unknown.openai(), None);
        for reason in [GenerationFinishReason::Stop, GenerationFinishReason::Length] {
            let phases = PhaseTimings {
                finish_reason: reason,
                completion_tokens: 0,
                ..Default::default()
            };
            assert_eq!(
                crate::scheduler::InferenceStats::from_phases(phases, false).finish_reason,
                reason
            );
            #[allow(deprecated)]
            let stats = crate::scheduler::InferenceStats::from_speculative_phases(
                PhaseTimings::default(),
                phases,
            );
            assert_eq!(stats.finish_reason, reason);
        }
    }
}
