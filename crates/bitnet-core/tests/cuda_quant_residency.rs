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
    assert!(!ggml_type_supports_cuda_quant(0));
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
    let matrix =
        CudaDeviceQuantMatrix::from_payload(Some(&rt), 2, payload, out_rows, in_cols)
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
        assert!((a - b).abs() < 1e-3, "dequant golden mismatch got={a} expected={b}");
    }
}

#[test]
fn device_resident_quant_metric_defaults_zero_without_cuda() {
    let m = CudaRuntimeMetrics::default();
    assert_eq!(m.device_resident_quant_gemv_calls, 0);
    assert_eq!(m.device_resident_gemv_calls, 0);
    assert_eq!(m.gemv_calls, 0);
}
