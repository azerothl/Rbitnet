//! The native hot GEMM kernels checked against CPU GGUF dequantization + F64.
use std::ffi::c_void;

type Check = unsafe extern "C" fn(
    u32,
    *const c_void,
    usize,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut f32,
    *mut f32,
) -> i32;

#[test]
fn opt_in_compensated_gemm_matches_f64_all_formats_and_dynamic_ranges() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let lib = crate::ggml::load_cuda_quant_library().expect("CUDA DLL required");
    let check = unsafe {
        *lib.get::<Check>(b"rbitnet_cuda_quant_gemm_check\0")
            .expect("GEMM oracle ABI required")
    };
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        for tokens in [7, 16, 33, 128] {
            let rows = 33;
            let cols = 512;
            let rb = crate::ggml::ggml_row_size(ty, cols as u64).unwrap();
            let mut payload: Vec<u8> = (0..rows * rb).map(|i| (i * 37 + 19) as u8).collect();
            if ty == 0 {
                for (i, b) in payload.chunks_exact_mut(4).enumerate() {
                    b.copy_from_slice(&((i as f32 * 0.031).cos() * 0.11).to_le_bytes());
                }
            } else {
                let bytes = match ty {
                    2 => 18,
                    6 => 22,
                    8 => 34,
                    12 => 144,
                    13 => 176,
                    14 => 210,
                    39 => 17,
                    _ => unreachable!(),
                };
                for b in payload.chunks_exact_mut(bytes) {
                    if ty == 39 {
                        b[0] = 119;
                    } else {
                        let offset = if ty == 14 { 208 } else { 0 };
                        b[offset..offset + 2]
                            .copy_from_slice(&half::f16::from_f32(0.003).to_bits().to_le_bytes());
                        if ty == 12 || ty == 13 {
                            b[2..4].copy_from_slice(
                                &half::f16::from_f32(0.001).to_bits().to_le_bytes(),
                            );
                        }
                    }
                }
            }
            let w = crate::ggml::tensor_to_f32(&payload, ty, &[cols as u64, rows as u64]).unwrap();
            let x: Vec<f32> = (0..cols * tokens)
                .map(|i| (i as f32 * 0.71).sin())
                .collect();
            compare(check, ty, &payload, rb, &w, &x, cols, rows, tokens);
        }
    }
    // Irregular F32 K and edges, cancellation, tiny/large finite products.
    let (cols, rows, tokens) = (513, 65, 17);
    let w: Vec<f32> = (0..cols * rows)
        .map(|i| {
            let scale = 2.0f32.powi((i / cols % 7) as i32 * 15 - 45);
            (i as f32 * 0.031).cos() * scale
        })
        .collect();
    let x: Vec<f32> = (0..cols * tokens)
        .map(|i| {
            let scale = 2.0f32.powi((i / cols % 7) as i32 * 15 - 45);
            (i as f32 * 0.73).sin() * scale
        })
        .collect();
    let payload: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    compare(check, 0, &payload, cols * 4, &w, &x, cols, rows, tokens);
    // Largest finite F32 must not turn into infinity while rounding to TF32.
    let (cols, rows, tokens) = (33, 3, 2);
    let w: Vec<f32> = (0..cols * rows)
        .map(|i| {
            if i / cols == 0 {
                f32::MAX
            } else if i / cols == 1 {
                -f32::MAX
            } else {
                f32::MAX * 0.49
            }
        })
        .collect();
    let x: Vec<f32> = (0..cols * tokens)
        .map(|i| (i as f32 * 0.17).cos() * 2.0f32.powi(-80 + (i / cols) as i32 * 10))
        .collect();
    let payload: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    compare(check, 0, &payload, cols * 4, &w, &x, cols, rows, tokens);
}

#[allow(clippy::too_many_arguments)]
fn compare(
    check: Check,
    ty: u32,
    payload: &[u8],
    rb: usize,
    w: &[f32],
    x: &[f32],
    cols: usize,
    rows: usize,
    tokens: usize,
) {
    for mode in [0, 1] {
        let mut output = vec![f32::NAN; tokens * rows];
        let mut elapsed = 0.0;
        let status = unsafe {
            check(
                ty,
                payload.as_ptr().cast(),
                rb,
                x.as_ptr(),
                cols as u32,
                rows as u32,
                tokens as u32,
                mode,
                2,
                output.as_mut_ptr(),
                &mut elapsed,
            )
        };
        assert_eq!(status, 0, "GEMM type={ty} tokens={tokens} mode={mode}");
        let mut worst: f64 = 0.0;
        for token in 0..tokens {
            for row in 0..rows {
                let mut expected = 0.0;
                let mut l1 = 0.0;
                for k in 0..cols {
                    let product = w[row * cols + k] as f64 * x[token * cols + k] as f64;
                    expected += product;
                    l1 += product.abs();
                }
                let value = output[token * rows + row] as f64;
                let error = (value - expected).abs();
                worst = worst.max(error / l1.max(1e-37));
                assert!(value.is_finite() && error<=8e-7*l1+1e-37,"type={ty} mode={mode} tokens={tokens} row={row} expected={expected} actual={value} error={error} l1={l1}");
            }
        }
        eprintln!("GGUF type={ty} K={cols} M={rows} N={tokens} mode={mode} event_ms={elapsed:.6} worst_error_over_l1={worst:.3e}");
    }
}

#[test]
fn optional_real_gguf_gemm_accuracy_and_cuda_event_cost() {
    let Ok(report_path) = std::env::var("RBITNET_CUDA_GEMM_BENCH_JSON") else {
        return;
    };
    let gguf = std::env::var("RBITNET_TEST_GGUF").expect("real GGUF required");
    let archive = crate::gguf::GgufArchive::mmap_path(std::path::Path::new(&gguf)).unwrap();
    let lib = crate::ggml::load_cuda_quant_library().expect("CUDA required");
    let check = unsafe {
        *lib.get::<Check>(b"rbitnet_cuda_quant_gemm_check\0")
            .unwrap()
    };
    let mut results = Vec::new();
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
    ] {
        let tensor = archive
            .tensor_by_name(name)
            .expect("benchmark matrix required");
        let cols = tensor.dimensions[0] as usize;
        let rows = tensor.dimensions[1] as usize;
        let ty = tensor.ggml_type;
        let rb = crate::ggml::ggml_row_size(ty, cols as u64).unwrap();
        let payload = archive.tensor_payload(tensor).unwrap();
        let w = crate::ggml::tensor_to_f32(payload, ty, &tensor.dimensions).unwrap();
        for tokens in [16, 64, 128] {
            let x: Vec<f32> = (0..cols * tokens)
                .map(|i| (i as f32 * 0.37).sin() * 1.21)
                .collect();
            for cycle in 0..3 {
                for mode in [0, 1] {
                    let mut output = vec![f32::NAN; tokens * rows];
                    let mut elapsed = 0.0;
                    let status = unsafe {
                        check(
                            ty,
                            payload.as_ptr().cast(),
                            rb,
                            x.as_ptr(),
                            cols as u32,
                            rows as u32,
                            tokens as u32,
                            mode,
                            30,
                            output.as_mut_ptr(),
                            &mut elapsed,
                        )
                    };
                    assert_eq!(status, 0, "matrix {name} mode={mode}");
                    let mut worst: f64 = 0.0;
                    for index in 0..64 {
                        let row = (index * 713 + 7) % rows;
                        let token = (index * 11 + 3) % tokens;
                        let mut expected = 0.0;
                        let mut l1 = 0.0;
                        for k in 0..cols {
                            let product = w[row * cols + k] as f64 * x[token * cols + k] as f64;
                            expected += product;
                            l1 += product.abs();
                        }
                        let value = output[token * rows + row] as f64;
                        let error = (value - expected).abs();
                        worst = worst.max(error / l1.max(1e-37));
                        assert!(
                            value.is_finite() && error <= 8e-7 * l1 + 1e-37,
                            "{name} mode={mode} error={error} l1={l1}"
                        );
                    }
                    results.push(serde_json::json!({"matrix":name,"ggml_type":ty,"rows":rows,"cols":cols,"tokens":tokens,"mode":mode,"cycle":cycle,"event_ms":elapsed,"worst_error_over_l1":worst}));
                    eprintln!("matrix={name} M={rows} K={cols} N={tokens} mode={mode} event_ms={elapsed:.6}");
                }
            }
        }
    }
    let report = serde_json::json!({"gguf":gguf,"timing":"CUDA events; 2 warmup + 30 kernels per call; 3 cycles; forced SIMT/TF32x3 kernels","f64_samples_per_matrix":64,"rows":results});
    std::fs::write(
        report_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}
