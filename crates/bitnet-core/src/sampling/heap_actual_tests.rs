// Real, previously captured full-vocabulary inputs, followed by a quiet CPU ablation.
// Execute only after the shared-machine hardware owners finish.
use std::hint::black_box;
use std::time::Instant;

fn actual_weights(path: &std::path::Path, temperature: f32) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert!(!bytes.is_empty() && bytes.len() % 4 == 0);
    let mut scaled: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()) / temperature)
        .collect();
    let maximum = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(maximum.is_finite());
    for value in &mut scaled {
        *value = if value.is_finite() {
            (*value - maximum).exp()
        } else {
            0.0
        };
    }
    scaled
}

#[test]
fn optional_actual_top_p_original_equality_and_quiet_cpu_cost() {
    if std::env::var("RBITNET_TOP_P_ACTUAL_TEST").as_deref() != Ok("1") {
        return;
    }
    let input = std::fs::read(std::env::var("RBITNET_TOP_P_INPUTS").unwrap()).unwrap();
    let index: serde_json::Value = serde_json::from_slice(&input).unwrap();
    let vectors = index["vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 24);
    let mut exact_cases = 0;
    for row in vectors {
        let path = std::path::Path::new(row["path"].as_str().unwrap());
        for temperature in [0.7, 1.0, 1.4] {
            let weights = actual_weights(path, temperature);
            for p in [0.5, 0.9, 0.99] {
                for seed in [0, 42, 53, 999] {
                    let mut expected = StdRng::seed_from_u64(seed);
                    let mut actual = StdRng::seed_from_u64(seed);
                    assert_eq!(
                        top_p_heap::sample_top_p_heap(&weights, p, &mut actual),
                        original_top_p(&weights, p, &mut expected),
                        "{} temp={temperature} p={p} seed={seed}",
                        path.display()
                    );
                    assert_eq!(actual.gen::<u64>(), expected.gen::<u64>());
                    exact_cases += 1;
                }
            }
        }
    }
    assert_eq!(exact_cases, 864);
    let mut distributions = Vec::<(String, Vec<f32>)>::new();
    for row in vectors.iter().step_by(4) {
        let path = std::path::Path::new(row["path"].as_str().unwrap());
        distributions.push((
            path.file_name().unwrap().to_string_lossy().into_owned(),
            actual_weights(path, 0.7),
        ));
    }
    for size in [128256, 201088] {
        distributions.push((format!("uniform-{size}"), vec![1.0; size]));
        distributions.push((
            format!("concentrated-{size}"),
            (0..size)
                .map(|i| if i < 5 { 1.0 / (i + 1) as f32 } else { 1e-7 })
                .collect(),
        ));
    }
    for (name, weights) in distributions {
        for p in [0.9, 0.99] {
            let mut original_ns = Vec::new();
            let mut heap_ns = Vec::new();
            // Alternating execution order and separately seeded identical RNG streams.
            // One warm cycle, five measured cycles, 16 complete selections per cell.
            for cycle in 0..6 {
                let mut costs = [0; 2];
                let mut checksums = [0u64; 2];
                for variant in if cycle % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let mut rng = StdRng::seed_from_u64(42 + cycle);
                    let start = Instant::now();
                    for _ in 0..16 {
                        let choice = if variant == 0 {
                            original_top_p(black_box(&weights), p, &mut rng)
                        } else {
                            top_p_heap::sample_top_p_heap(black_box(&weights), p, &mut rng)
                        };
                        checksums[variant] += black_box(choice.unwrap()) as u64;
                    }
                    costs[variant] = start.elapsed().as_nanos() as u64 / 16;
                }
                assert_eq!(checksums[0], checksums[1]);
                if cycle != 0 {
                    original_ns.push(costs[0]);
                    heap_ns.push(costs[1]);
                }
            }
            original_ns.sort_unstable();
            heap_ns.sort_unstable();
            println!(
                "TOP_P_COST {}",
                serde_json::json!({"input":name,"vocab":weights.len(),"top_p":p,
                "original_ns":original_ns,"heap_ns":heap_ns,"original_median_ns":original_ns[2],"heap_median_ns":heap_ns[2],
                "measured_repeats":5,"draws_per_cell":16,"includes_weight_construction":false})
            );
        }
    }
    println!(
        "TOP_P_ACTUAL_DONE vectors=24 exact_cases={exact_cases} rng_exact=true cpu_cost_only=true"
    );
}
