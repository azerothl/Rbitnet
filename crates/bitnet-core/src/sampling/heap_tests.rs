//! Pinned original sampler reference; exact token and post-draw RNG state.
use super::*;
use rand::{rngs::StdRng, SeedableRng};
fn original_top_p(weights: &[f32], top_p: f32, rng: &mut impl Rng) -> Option<u32> {
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

#[test]
fn heap_top_p_preserves_stable_ties_f32_thresholds_fallback_and_rng() {
    let mut cases = 0;
    for size in [1, 2, 7, 63, 64, 65, 257, 4096, 128256] {
        let mut distributions = vec![vec![1.; size], vec![0.; size]];
        distributions.push(
            (0..size)
                .map(|i| if i < 5 { 1.0 / (i + 1) as f32 } else { 1e-7 })
                .collect(),
        );
        distributions.push(
            (0..size)
                .map(|i| ((i * 17 % 101) + 1) as f32 / 101.)
                .collect(),
        );
        distributions.push(
            (0..size)
                .map(|i| match i % 11 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => -0.0,
                    3 => f32::NEG_INFINITY,
                    _ => 0.1,
                })
                .collect(),
        );
        for weights in distributions {
            for top_p in [0., 0.01, 0.5, 0.9, 0.999, 1., f32::NAN] {
                for seed in [0, 42, 53, 999] {
                    let mut expected = StdRng::seed_from_u64(seed);
                    let mut actual = StdRng::seed_from_u64(seed);
                    assert_eq!(
                        top_p_heap::sample_top_p_heap(&weights, top_p, &mut actual),
                        original_top_p(&weights, top_p, &mut expected),
                        "size={size} top_p={top_p} seed={seed}"
                    );
                    assert_eq!(
                        actual.gen::<u64>(),
                        expected.gen::<u64>(),
                        "RNG consumption changed"
                    );
                    cases += 1;
                }
            }
        }
    }
    // Boundaries just below/above an exactly representable cumulative prefix.
    let weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
    for bits in [
        0.75f32.to_bits() - 1,
        0.75f32.to_bits(),
        0.75f32.to_bits() + 1,
    ] {
        for seed in 0..256 {
            let mut expected = StdRng::seed_from_u64(seed);
            let mut actual = StdRng::seed_from_u64(seed);
            let p = f32::from_bits(bits);
            assert_eq!(
                top_p_heap::sample_top_p_heap(&weights, p, &mut actual),
                original_top_p(&weights, p, &mut expected)
            );
            assert_eq!(actual.gen::<u64>(), expected.gen::<u64>());
            cases += 1;
        }
    }
    println!("TOP_P_HEAP_ORIGINAL_EXACT_DONE cases={cases} stable_ties=true fallback=true rng_exact=true");
}

include!("heap_actual_tests.rs");
