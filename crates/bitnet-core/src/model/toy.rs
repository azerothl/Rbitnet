//! Tiny byte-level LM for end-to-end smoke tests (F32, no GGUF).
//!
//! Not a BitNet-quantized model — proves tokenizer loop + matmul + sampling in-process.

/// Small deterministic LM: embedding + single linear head, vocab = 256 bytes.
pub struct ToyLlm {
    embed: Vec<f32>,
    head: Vec<f32>,
    dim: usize,
}

impl ToyLlm {
    const VOCAB: usize = 256;

    /// Deterministic pseudo-random init from seed (no `rand` crate needed).
    pub fn new(seed: u64) -> Self {
        let dim = 32usize;
        let mut embed = vec![0.0f32; Self::VOCAB * dim];
        let mut head = vec![0.0f32; dim * Self::VOCAB];
        let mut s = seed;
        for e in embed.iter_mut().chain(head.iter_mut()) {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *e = ((s >> 16) & 0xffff) as f32 / 65535.0 - 0.5;
        }
        Self {
            embed,
            head,
            dim,
        }
    }

    fn embed_row(&self, token: u8) -> &[f32] {
        let i = token as usize * self.dim;
        &self.embed[i..i + self.dim]
    }

    fn logits(&self, token: u8) -> Vec<f32> {
        let x = self.embed_row(token);
        let mut out = vec![0.0f32; Self::VOCAB];
        let d = self.dim;
        for v in 0..Self::VOCAB {
            let mut acc = 0.0f32;
            let row = v * d;
            for j in 0..d {
                acc += self.head[row + j] * x[j];
            }
            out[v] = acc;
        }
        out
    }

    fn greedy(logits: &[f32]) -> u8 {
        let mut best = 0usize;
        let mut best_v = logits[0];
        for (i, &l) in logits.iter().enumerate().skip(1) {
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        best as u8
    }

    fn sample_token(logits: &[f32], temperature: f32, step: u64) -> u8 {
        if temperature <= 1e-6 {
            return Self::greedy(logits);
        }
        // Numerically stable softmax: subtract max logit before exp to avoid overflow.
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let s = step.wrapping_mul(0x9E3779B97F4A7C15);
        let mut sum = 0.0f32;
        let mut scaled = vec![0.0f32; Self::VOCAB];
        for (i, l) in logits.iter().enumerate() {
            let v = ((*l - max_logit) / temperature).exp();
            scaled[i] = v;
            sum += v;
        }
        // After subtracting max_logit, the max-shifted term contributes exp(0)=1.0 to
        // the sum, so sum >= 1.0 always holds for finite logits.  We only guard against
        // the (degenerate) case where a non-finite logit slips through.
        if !sum.is_finite() {
            return Self::greedy(logits);
        }
        let r = (s % 10000) as f32 / 10000.0 * sum;
        let mut acc = 0.0f32;
        for (i, v) in scaled.iter().enumerate() {
            acc += *v;
            if acc >= r {
                return i as u8;
            }
        }
        0
    }

    /// Generate bytes as lossy UTF-8 (toy output).
    pub fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> String {
        let max_tokens = max_tokens.max(1).min(256);
        let mut out = String::new();
        let mut tok = prompt.as_bytes().last().copied().unwrap_or(b' ');
        for t in 0..max_tokens {
            let logits = self.logits(tok);
            tok = Self::sample_token(&logits, temperature, t as u64 + 1);
            if tok == 0 {
                break;
            }
            if (32..127).contains(&tok) {
                out.push(tok as char);
            }
        }
        if out.is_empty() {
            out.push_str("(toy)");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_is_deterministic() {
        let m = ToyLlm::new(42);
        let a = m.generate("hi", 8, 0.0);
        let b = m.generate("hi", 8, 0.0);
        assert_eq!(a, b);
    }
}
