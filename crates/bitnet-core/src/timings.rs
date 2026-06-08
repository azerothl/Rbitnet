//! Wall-clock phase timings: tokenizer encode vs model prefill vs decode.

/// Millisecond-resolution timings plus token counts from the tokenizer/runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    pub encode_ms: u64,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
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
        }
    }
}
