//! Gate E of the GPU native acceptance slice ([docs/GPU_NATIVE_ROADMAP.md](../../../docs/GPU_NATIVE_ROADMAP.md)):
//! device-resident **quantized** matvec API with CPU golden parity.
//!
//! CUDA hardware is **not** required — [`CudaDeviceQuantMatrix`] keeps a host payload and falls
//! back to `matvec_payload_quant` when `librbitnet_cuda_quant` device symbols are absent.

use bitnet_core::backend::{CudaDeviceQuantMatrix, CudaRuntimeMetrics};
use bitnet_core::ggml::{ggml_type_supports_cuda_quant, matvec_payload_quant, tensor_to_f32};

fn q4_0_two_row_fixture() -> (Vec<u8>, Vec<f32>, usize, usize) {
    // ggml Q4_0: 32 elements / 18 bytes per row. Two output rows, ne0=32.
    let mut payload = vec![0u8; 36];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(9);
    }
    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.8).collect();
    (payload, x, 2, 32)
}

#[test]
fn cuda_quant_types_match_optional_kernel_abi() {
    assert!(ggml_type_supports_cuda_quant(2));
    assert!(ggml_type_supports_cuda_quant(8));
    assert!(ggml_type_supports_cuda_quant(12));
    assert!(ggml_type_supports_cuda_quant(14));
    for ty in [0, 6, 13, 39] {
        assert!(ggml_type_supports_cuda_quant(ty));
    }
    assert!(!ggml_type_supports_cuda_quant(1));
}

#[test]
fn device_quant_matrix_cpu_fallback_matches_payload_golden() {
    let (payload, x, out_rows, in_cols) = q4_0_two_row_fixture();
    let expected = matvec_payload_quant(2, &payload, &x, in_cols, out_rows).expect("cpu quant");
    let matrix = CudaDeviceQuantMatrix::from_payload(None, 2, payload, out_rows, in_cols)
        .expect("build quant matrix without CUDA");
    assert!(!matrix.is_device_resident());
    let got = matrix.matvec(&x).expect("cpu fallback matvec");
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() <= 1e-6, "mismatch got={a} expected={b}");
    }
}

#[test]
fn opt_in_device_resident_quant_kernel_when_lib_present() {
    // Hardware opt-in: set RBITNET_CUDA_QUANT_SMOKE=1 after building native/cuda_quant.
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").ok().as_deref() != Some("1") {
        return;
    }
    let rt = bitnet_core::CudaRuntime::try_load().expect("CUDA runtime required for smoke");
    assert!(
        bitnet_core::ggml::cuda_quant_library_available(),
        "set RBITNET_CUDA_QUANT_LIB or PATH to rbitnet_cuda_quant64.dll"
    );
    let (payload, x, out_rows, in_cols) = q4_0_two_row_fixture();
    let expected = matvec_payload_quant(2, &payload, &x, in_cols, out_rows).expect("cpu");
    let matrix = CudaDeviceQuantMatrix::from_payload(Some(&rt), 2, payload, out_rows, in_cols)
        .expect("device quant matrix");
    assert!(matrix.is_device_resident());
    let before = rt.metrics_snapshot().device_resident_quant_gemv_calls;
    let got = matrix.matvec(&x).expect("device quant matvec");
    let after = rt.metrics_snapshot().device_resident_quant_gemv_calls;
    assert!(
        after > before,
        "device_resident_quant_gemv_calls did not rise ({before} -> {after})"
    );
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() <= 1e-3,
            "GPU vs CPU mismatch got={a} expected={b}"
        );
    }
}

#[test]
fn device_quant_matrix_matches_full_dequant_row_dots() {
    let (payload, x, out_rows, in_cols) = q4_0_two_row_fixture();
    let dims = vec![in_cols as u64, out_rows as u64];
    let dense = tensor_to_f32(&payload, 2, &dims).expect("dequant");
    let mut expected = vec![0.0f32; out_rows];
    for o in 0..out_rows {
        let mut acc = 0.0f32;
        for i in 0..in_cols {
            acc += dense[i + o * in_cols] * x[i];
        }
        expected[o] = acc;
    }
    let matrix = CudaDeviceQuantMatrix::from_payload(None, 2, payload, out_rows, in_cols)
        .expect("build quant matrix");
    let got = matrix.matvec(&x).expect("matvec");
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "dequant golden mismatch got={a} expected={b}"
        );
    }
}

#[test]
fn device_resident_quant_metric_defaults_zero_without_cuda() {
    let m = CudaRuntimeMetrics::default();
    assert_eq!(m.device_resident_quant_gemv_calls, 0);
    assert_eq!(m.device_resident_gemv_calls, 0);
    assert_eq!(m.gemv_calls, 0);
}

#[test]
fn opt_in_all_formats_views_batches_and_concurrent_calls() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let rt = bitnet_core::CudaRuntime::try_load().expect("CUDA required");
    let cols = 256;
    let rows = 12;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.47).sin()).collect();
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        let row_bytes = bitnet_core::ggml::ggml_row_size(ty, cols as u64).unwrap();
        let mut payload = vec![0u8; row_bytes * rows];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(19);
        }
        if ty == 0 {
            for (i, bytes) in payload.chunks_exact_mut(4).enumerate() {
                bytes.copy_from_slice(&(i as f32 * 0.31).cos().to_le_bytes());
            }
        } else {
            let (block_len, elements) = match ty {
                2 => (18, 32),
                6 => (22, 32),
                8 => (34, 32),
                12 => (144, 256),
                13 => (176, 256),
                14 => (210, 256),
                39 => (17, 32),
                _ => unreachable!(),
            };
            for block in payload.chunks_exact_mut(block_len) {
                if ty == 39 {
                    block[0] = 126;
                } else {
                    let offset = if ty == 14 { 208 } else { 0 };
                    block[offset..offset + 2]
                        .copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
                    if ty == 12 || ty == 13 {
                        block[2..4]
                            .copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                    }
                }
            }
            assert_eq!(row_bytes % block_len, 0);
            assert_eq!(cols % elements, 0);
        }
        let expected = matvec_payload_quant(ty, &payload, &x, cols, rows).unwrap();
        let matrix = std::sync::Arc::new(
            CudaDeviceQuantMatrix::from_payload(Some(&rt), ty, payload, rows, cols).unwrap(),
        );
        let before = rt.metrics_snapshot().device_resident_quant_gemv_calls;
        let output = matrix.matvec(&x).unwrap();
        assert!(
            rt.metrics_snapshot().device_resident_quant_gemv_calls > before,
            "format {ty} fell back to CPU"
        );
        for (got, expected) in output.iter().zip(&expected) {
            assert!(
                (got - expected).abs() <= 3e-4 * (1.0 + expected.abs()),
                "format {ty}: {got} vs {expected}"
            );
        }
        let view = matrix.matvec_rows(&x, 3, 3).unwrap();
        for (got, expected) in view.iter().zip(&expected[3..6]) {
            assert!((got - expected).abs() <= 3e-4 * (1.0 + expected.abs()));
        }
        let inputs: Vec<f32> = (0..4)
            .flat_map(|batch| x.iter().map(move |v| v * (batch as f32 + 1.0)))
            .collect();
        let batches = matrix.matvec_batch(&inputs, 3).unwrap();
        for (i, got) in batches.iter().enumerate() {
            let expected = expected[i] * (i / 3 + 1) as f32;
            assert!(
                (got - expected).abs() <= 3e-4 * (1.0 + expected.abs()),
                "batch format {ty}"
            );
        }
        std::thread::scope(|scope| {
            for batch in 0..4 {
                let matrix = std::sync::Arc::clone(&matrix);
                let x = &x;
                let expected = &expected;
                scope.spawn(move || {
                    for _ in 0..3 {
                        let input: Vec<f32> = x.iter().map(|v| v * (batch as f32 + 1.0)).collect();
                        let output = matrix.matvec(&input).unwrap();
                        for (got, expected) in output.iter().zip(expected) {
                            let expected = expected * (batch as f32 + 1.0);
                            assert!((got - expected).abs() <= 3e-4 * (1.0 + expected.abs()));
                        }
                    }
                });
            }
        });
    }
}
