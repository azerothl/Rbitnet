//! Deterministic token proposals and exact target-coupled verification.
//! For q=delta(d), sample T~p and accept iff T=d. Acceptance is p(d);
//! rejection yields p(t)/(1-p(d)), t!=d, the usual residual correction.
//! One target draw per emitted token also preserves seeded RNG consumption.

pub(super) fn enabled() -> bool {
    matches!(
        std::env::var("RBITNET_SPECULATIVE_PLD").as_deref(),
        Ok("1" | "true" | "yes")
    ) || matches!(
        std::env::var("RBITNET_SPECULATIVE").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

/// Request-local cost guard. Cold graph creation is excluded from the serial
/// estimate. A measured expensive block disables further guesses for this
/// request; every future token still comes from the unchanged target runtime.
#[derive(Default)]
pub(super) struct CostGuard {
    serial_samples: usize,
    serial_ns: f64,
    suppressed: bool,
}
impl CostGuard {
    pub fn permits(&self) -> bool {
        !self.suppressed
    }
    pub fn serial(&mut self, ns: u64) {
        self.serial_samples += 1;
        if self.serial_samples == 1 {
            return;
        }
        self.serial_ns = if self.serial_samples == 2 {
            ns as f64
        } else {
            self.serial_ns * 0.75 + ns as f64 * 0.25
        };
    }
    pub fn verification(&mut self, ns: u64, advance: usize) {
        let adaptive = std::env::var("RBITNET_SPECULATIVE_ADAPTIVE").as_deref() != Ok("0");
        if adaptive
            && self.serial_samples >= 3
            && advance > 0
            && ns as f64 / advance as f64 > self.serial_ns * 1.1
        {
            self.suppressed = true;
        }
    }
}

pub(super) fn propose(history: &[u32], limit: usize) -> Vec<u32> {
    if limit == 0 {
        return Vec::new();
    }
    for n in (2..=8.min(history.len().saturating_sub(1))).rev() {
        let suffix = &history[history.len() - n..];
        for start in (0..history.len() - n).rev() {
            if &history[start..start + n] == suffix {
                return history[start + n..(start + n + limit).min(history.len())].to_vec();
            }
        }
    }
    Vec::new()
}

pub(super) struct Decision {
    pub confirmed: Vec<u32>,
    pub pending: Option<u32>,
    pub accepted: usize,
}

/// The caller has already emitted the first input token of the verification
/// block. Only rows whose preceding proposals match are valid target states.
pub(super) fn decide(
    proposals: &[u32],
    remaining: usize,
    eos: &[u32],
    prior: &[u32],
    mut sample: impl FnMut(usize, &[u32]) -> u32,
) -> Decision {
    let mut history = prior.to_vec();
    let mut decision = Decision {
        confirmed: Vec::new(),
        pending: None,
        accepted: 0,
    };
    for index in 0..=proposals.len() {
        if decision.confirmed.len() == remaining {
            break;
        }
        let target = sample(index, &history);
        if eos.contains(&target) {
            break;
        }
        if proposals.get(index) != Some(&target) {
            decision.pending = Some(target);
            break;
        }
        history.push(target);
        decision.confirmed.push(target);
        decision.accepted += 1;
    }
    decision
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cost_guard_ignores_cold_setup_and_keeps_only_profitable_blocks() {
        if std::env::var("RBITNET_SPECULATIVE_ADAPTIVE").as_deref() == Ok("0") {
            return;
        }
        let mut guard = CostGuard::default();
        guard.serial(100_000);
        guard.serial(100);
        guard.serial(100);
        guard.verification(600, 8);
        assert!(guard.permits());
        guard.verification(600, 2);
        assert!(!guard.permits());
    }
    #[test]
    fn token_lookup_prefers_longest_recent_match_without_inventing_tokens() {
        assert_eq!(propose(&[1, 2, 3, 9, 1, 2], 3), [3, 9, 1]);
        assert_eq!(propose(&[1, 2, 3, 4, 1, 2, 8, 1, 2, 3, 4], 4), [1, 2, 8, 1]);
        assert!(propose(&[1, 2, 3], 8).is_empty());
        assert!(propose(&[1, 2, 1, 2], 0).is_empty());
    }
    #[test]
    fn rollback_boundary_tracks_all_first_and_partial_rejections_eos_and_budget() {
        let run =
            |targets: &[u32], limit| decide(&[2, 3, 4], limit, &[99], &[1], |i, _| targets[i]);
        let all = run(&[2, 3, 4, 5], 8);
        assert_eq!(all.confirmed, [2, 3, 4]);
        assert_eq!(all.pending, Some(5));
        assert_eq!(all.accepted + 1, 4);
        let first = run(&[8], 8);
        assert!(first.confirmed.is_empty());
        assert_eq!(first.pending, Some(8));
        assert_eq!(first.accepted + 1, 1);
        let partial = run(&[2, 8], 8);
        assert_eq!(partial.confirmed, [2]);
        assert_eq!(partial.pending, Some(8));
        assert_eq!(partial.accepted + 1, 2);
        let eos = run(&[2, 99], 8);
        assert_eq!(eos.confirmed, [2]);
        assert_eq!(eos.pending, None);
        assert_eq!(eos.accepted + 1, 2);
        let budget = run(&[2, 3], 2);
        assert_eq!(budget.confirmed, [2, 3]);
        assert_eq!(budget.pending, None);
    }
    #[test]
    fn delta_draft_acceptance_and_correction_preserve_known_target_distribution() {
        use rand::SeedableRng;
        let logits = [0.1f32.ln(), 0.3f32.ln(), 0.6f32.ln()];
        let options = crate::sampling::SamplingOptions::from_temperature(1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(735);
        let mut counts = [0usize; 3];
        let mut accepted = 0;
        for _ in 0..100_000 {
            let decision = decide(&[1], 1, &[], &[], |_, prior| {
                crate::sampling::sample_token(&logits, &options, prior, &mut rng)
            });
            let token = decision
                .confirmed
                .first()
                .copied()
                .or(decision.pending)
                .unwrap();
            counts[token as usize] += 1;
            accepted += decision.accepted;
        }
        for (count, expected) in counts.iter().zip([0.1, 0.3, 0.6]) {
            assert!((*count as f64 / 100_000.0 - expected).abs() < 0.006);
        }
        assert_eq!(accepted, counts[1]);
        assert!((counts[2] as f64 / (counts[0] + counts[2]) as f64 - 6.0 / 7.0).abs() < 0.006);
    }
}
