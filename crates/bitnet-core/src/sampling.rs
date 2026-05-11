//! Shared token sampling helpers.

use std::collections::HashMap;

use rand::Rng;

/// OpenAI-compatible sampling knobs promoted through the core generation path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl SamplingOptions {
    #[must_use]
    pub fn from_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::default()
        }
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: None,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Sample one token from logits, applying penalties before temperature/top-p.
#[must_use]
pub fn sample_token(
    logits: &[f32],
    options: &SamplingOptions,
    prior_tokens: &[u32],
    rng: &mut impl Rng,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let mut adjusted = logits.to_vec();
    apply_repetition_penalties(
        &mut adjusted,
        prior_tokens,
        options.frequency_penalty,
        options.presence_penalty,
    );

    if options.temperature <= 0.0 {
        return argmax(&adjusted);
    }

    let temperature = options.temperature.max(f32::MIN_POSITIVE);
    let mut scaled: Vec<f32> = adjusted.iter().map(|z| z / temperature).collect();
    let m = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return argmax(&adjusted);
    }

    for z in &mut scaled {
        if z.is_finite() {
            *z = (*z - m).exp();
        } else {
            *z = 0.0;
        }
    }

    if let Some(top_p) = options.top_p {
        sample_top_p(&scaled, top_p, rng).unwrap_or_else(|| sample_multinomial(&scaled, rng))
    } else {
        sample_multinomial(&scaled, rng)
    }
}

fn apply_repetition_penalties(
    logits: &mut [f32],
    prior_tokens: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    if frequency_penalty == 0.0 && presence_penalty == 0.0 {
        return;
    }

    let mut counts = HashMap::<u32, u32>::new();
    for &token in prior_tokens {
        *counts.entry(token).or_default() += 1;
    }

    for (token, count) in counts {
        let idx = token as usize;
        if let Some(logit) = logits.get_mut(idx) {
            *logit -= presence_penalty;
            *logit -= frequency_penalty * count as f32;
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
            a.total_cmp(&b)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn sample_multinomial(weights: &[f32], rng: &mut impl Rng) -> u32 {
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(weights);
    }
    let r = rng.gen::<f32>() * sum;
    let mut c = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        c += w.max(0.0);
        if c >= r {
            return i as u32;
        }
    }
    (weights.len().saturating_sub(1)) as u32
}

fn sample_top_p(weights: &[f32], top_p: f32, rng: &mut impl Rng) -> Option<u32> {
    let top_p = top_p.clamp(0.0, 1.0);
    if top_p >= 1.0 {
        return None;
    }

    let total: f32 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return None;
    }

    let mut ranked: Vec<(usize, f32)> = weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, w)| *w > 0.0 && w.is_finite())
        .collect();
    ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));

    let threshold = total * top_p;
    let mut kept = Vec::new();
    let mut cumulative = 0.0f32;
    for (idx, weight) in ranked {
        kept.push((idx, weight));
        cumulative += weight;
        if cumulative >= threshold {
            break;
        }
    }
    if kept.is_empty() {
        return None;
    }

    let kept_sum: f32 = kept.iter().map(|(_, w)| *w).sum();
    if kept_sum <= 0.0 || !kept_sum.is_finite() {
        return None;
    }

    let r = rng.gen::<f32>() * kept_sum;
    let mut c = 0.0f32;
    for (idx, weight) in kept {
        c += weight;
        if c >= r {
            return Some(idx as u32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn top_p_zero_keeps_argmax_bucket() {
        let options = SamplingOptions {
            temperature: 1.0,
            top_p: Some(0.0),
            ..SamplingOptions::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(sample_token(&[0.0, 4.0, 1.0], &options, &[], &mut rng), 1);
    }

    #[test]
    fn presence_penalty_can_change_greedy_choice() {
        let options = SamplingOptions {
            temperature: 0.0,
            presence_penalty: 2.0,
            ..SamplingOptions::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(sample_token(&[0.0, 1.0, 0.5], &options, &[1], &mut rng), 2);
    }
}
