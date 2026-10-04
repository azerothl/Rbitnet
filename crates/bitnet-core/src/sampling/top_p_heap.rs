//! Experimental exact top-p selection: short sorted prefix, full-sort fallback.
//! Opt-in only. Exact tokens/RNG and end-to-end benefit require fresh validation.
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct Ranked {
    id: usize,
    weight: f32,
}
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.weight.to_bits() == other.weight.to_bits()
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Preserve the original sorted weights, tie order, F32 accumulation and RNG draw.
/// Only the way the sorted prefix is found changes. Flat distributions fall back.
pub(super) fn sample_top_p_heap(weights: &[f32], top_p: f32, rng: &mut impl Rng) -> Option<u32> {
    let top_p = top_p.clamp(0.0, 1.0);
    if top_p >= 1.0 {
        return None;
    }
    // The sum remains in vocabulary order, exactly as in the existing sampler.
    let total: f32 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let ranked: Vec<Ranked> = weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight > 0.0 && weight.is_finite())
        .map(|(id, weight)| Ranked { id, weight })
        .collect();
    let threshold = total * top_p;
    // Already sorted distributions need no heap or sort. The total and
    // retained-prefix additions still use the original vocabulary/sorted order.
    let already_sorted = ranked.windows(2).all(|pair| pair[0] >= pair[1]);
    let maximum = ranked
        .iter()
        .map(|value| value.weight)
        .fold(0.0f32, f32::max);
    let use_heap = !already_sorted
        && ranked.len() > 64
        && threshold.is_finite()
        && maximum * 64.0 >= threshold;
    let mut cumulative = 0.0f32;
    let mut kept = Vec::new();
    if !use_heap {
        let mut sorted = ranked;
        if !already_sorted {
            sorted.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        }
        for value in sorted {
            kept.push(value);
            cumulative += value.weight;
            if cumulative >= threshold {
                break;
            }
        }
    } else {
        let mut heap = BinaryHeap::from(ranked);
        let mut reached = false;
        for _ in 0..64 {
            let Some(value) = heap.pop() else {
                break;
            };
            kept.push(value);
            cumulative += value.weight;
            if cumulative >= threshold {
                reached = true;
                break;
            }
        }
        if !reached && !heap.is_empty() {
            // Rebuild the complete sorted prefix, retaining ID order for ties.
            let mut all = heap.into_vec();
            all.extend(kept);
            all.sort_unstable_by(|a, b| {
                b.weight.total_cmp(&a.weight).then_with(|| a.id.cmp(&b.id))
            });
            kept = Vec::new();
            cumulative = 0.0;
            for value in all {
                kept.push(value);
                cumulative += value.weight;
                if cumulative >= threshold {
                    break;
                }
            }
        }
    }
    if kept.is_empty() {
        return None;
    }
    let kept_sum: f32 = kept.iter().map(|value| value.weight).sum();
    if kept_sum <= 0.0 || !kept_sum.is_finite() {
        return None;
    }
    let random = rng.gen::<f32>() * kept_sum;
    let mut accumulated = 0.0f32;
    for value in kept {
        accumulated += value.weight;
        if accumulated >= random {
            return Some(value.id as u32);
        }
    }
    None
}
