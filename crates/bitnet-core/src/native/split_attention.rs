//! Independent F64 checks for the exact CUDA split-KV reduction.
#[test]
fn opt_in_split_kv_matches_f64_across_tiles_graph_positions_and_causality() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    type Check = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        u32,
        u32,
        u32,
        u32,
        u32,
        f32,
        u32,
        *const u32,
        u32,
        u32,
        *mut f32,
    ) -> i32;
    let lib = crate::ggml::load_cuda_quant_library().expect("CUDA library required");
    let check = unsafe {
        *lib.get::<Check>(b"rbitnet_cuda_split_attention_check\0")
            .expect("split-KV ABI required")
    };
    // GQA, non-warp-sized heads, full capacity, causal blocks and windows
    // beginning in the middle of a tile. Decreasing positions reuse a graph.
    for (dim, kv_heads, heads, capacity, count, window, magnitude) in [
        (64, 2, 4, 8192, 1, 0, 1.0),
        (80, 1, 3, 1025, 7, 333, 1.0),
        (256, 2, 4, 2048, 16, 257, 4.0),
        (32, 1, 2, 545, 128, 0, 1.0),
    ] {
        let scale = 1.0 / (dim as f32).sqrt();
        let q: Vec<f32> = (0..count * heads * dim)
            .map(|i| (i as f32 * 0.71).sin() * magnitude)
            .collect();
        let k: Vec<f32> = (0..capacity * kv_heads * dim)
            .map(|i| (i as f32 * 0.17).cos() * magnitude)
            .collect();
        let v: Vec<f32> = (0..capacity * kv_heads * dim)
            .map(|i| (i as f32 * 0.39).sin())
            .collect();
        let positions: Vec<u32> = [0, 254, 255, 256, 257, capacity - count, 31]
            .into_iter()
            .map(|i| i as u32)
            .collect();
        let length = count * heads * dim;
        for graphs in [0, 1] {
            let mut output = vec![f32::NAN; positions.len() * length];
            let status = unsafe {
                check(
                    k.as_ptr(),
                    v.as_ptr(),
                    q.as_ptr(),
                    capacity as u32,
                    kv_heads as u32,
                    heads as u32,
                    dim as u32,
                    window as u32,
                    scale,
                    count as u32,
                    positions.as_ptr(),
                    positions.len() as u32,
                    graphs,
                    output.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0, "native split-KV failed");
            for (step, &position) in positions.iter().enumerate() {
                for token in 0..count {
                    let seq = position as usize + token + 1;
                    let first = if window == 0 {
                        0
                    } else {
                        seq.saturating_sub(window)
                    };
                    for head in 0..heads {
                        let kh = head / (heads / kv_heads);
                        let scores: Vec<f64> = (first..seq)
                            .map(|p| {
                                (0..dim)
                                    .map(|i| {
                                        q[(token * heads + head) * dim + i] as f64
                                            * k[(p * kv_heads + kh) * dim + i] as f64
                                    })
                                    .sum::<f64>()
                                    * scale as f64
                            })
                            .collect();
                        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let weights: Vec<f64> =
                            scores.iter().map(|x| (x - maximum).exp()).collect();
                        let sum = weights.iter().sum::<f64>();
                        for i in 0..dim {
                            let expected = (first..seq)
                                .zip(&weights)
                                .map(|(p, w)| w / sum * v[(p * kv_heads + kh) * dim + i] as f64)
                                .sum::<f64>();
                            let actual =
                                output[step * length + (token * heads + head) * dim + i] as f64;
                            assert!((actual - expected).abs() < 3e-5 * (1.0 + expected.abs()),
                                "graphs={graphs} step={step} pos={position} token={token} head={head} dim={i}: {actual} vs {expected}");
                        }
                    }
                }
            }
        }
    }
}
