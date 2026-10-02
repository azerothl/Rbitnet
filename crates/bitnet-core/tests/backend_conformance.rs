//! Backend factory + numeric parity.
//!
//! Gate A of the GPU native acceptance slice ([docs/GPU_NATIVE_ROADMAP.md](../../../docs/GPU_NATIVE_ROADMAP.md)):
//! a fixed CPU golden matvec that every backend must match. CUDA hardware is **not** required —
//! `CudaBackend` falls back to CPU when the runtime is absent, so default CI stays green offline.

use bitnet_core::backend::{
    make_backend, BackendKind, ComputeBackend, CpuBackend, CudaBackend, CudaRuntimeMetrics,
    MetalBackend, RocmBackend, VulkanBackend,
};

/// Fixed 3×4 f32 golden used as the CPU reference for stub / CUDA-fallback parity (#22).
fn golden_matvec_fixture() -> (Vec<f32>, Vec<f32>, [f32; 3]) {
    let w = vec![
        0.25f32, -0.5, 0.75, 0.1, 0.6, -0.2, 0.0, 0.9, -0.4, 0.8, 0.3, -0.7,
    ];
    let x = vec![0.4f32, -1.1, 0.2, 0.7];
    // Row-major: y[i] = sum_j w[i*4+j] * x[j]
    let expected = [
        0.25 * 0.4 + (-0.5) * (-1.1) + 0.75 * 0.2 + 0.1 * 0.7,
        0.6 * 0.4 + (-0.2) * (-1.1) + 0.0 * 0.2 + 0.9 * 0.7,
        (-0.4) * 0.4 + 0.8 * (-1.1) + 0.3 * 0.2 + (-0.7) * 0.7,
    ];
    (w, x, expected)
}

fn check_backend_parity(cpu: &dyn ComputeBackend, other: &dyn ComputeBackend) {
    let (w, x, _) = golden_matvec_fixture();
    let y_cpu = cpu.matvec(&w, &x, 3, 4).expect("cpu matvec");
    let y_other = other.matvec(&w, &x, 3, 4).expect("backend matvec");
    for (a, b) in y_cpu.iter().zip(y_other.iter()) {
        assert!((a - b).abs() <= 1e-6, "mismatch cpu={a} other={b}");
    }
}

#[test]
fn cpu_matvec_matches_golden_vector() {
    let (w, x, expected) = golden_matvec_fixture();
    let y = CpuBackend.matvec(&w, &x, 3, 4).expect("cpu matvec");
    assert_eq!(y.len(), 3);
    for (got, exp) in y.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() <= 1e-6,
            "CPU golden mismatch got={got} expected={exp}"
        );
    }
}

#[test]
fn device_resident_gemv_metric_defaults_zero_without_cuda() {
    // Documents the residency counter shape for #22; no CUDA load required.
    let m = CudaRuntimeMetrics::default();
    assert_eq!(m.gemv_calls, 0);
    assert_eq!(m.device_resident_gemv_calls, 0);
    assert_eq!(m.device_resident_quant_gemv_calls, 0);
    assert_eq!(m.upload_bytes, 0);
    assert_eq!(m.download_bytes, 0);
}

#[test]
fn backend_factory_resolves_all_targets() {
    for kind in [
        BackendKind::Cpu,
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Vulkan,
        BackendKind::Metal,
    ] {
        let backend = make_backend(kind);
        assert_eq!(backend.kind(), kind);
    }
}

#[test]
fn backend_numeric_parity_cpu_vs_stubs() {
    let cpu = CpuBackend;
    let cuda = CudaBackend::default();
    let rocm = RocmBackend::default();
    let vulkan = VulkanBackend::default();
    let metal = MetalBackend::default();
    check_backend_parity(&cpu, &cuda);
    check_backend_parity(&cpu, &rocm);
    check_backend_parity(&cpu, &vulkan);
    check_backend_parity(&cpu, &metal);
}

#[test]
fn backend_auto_detect_resolves_without_panic() {
    // Does not assert a specific GPU — only that auto selection is a known kind.
    let kind = BackendKind::detect_best();
    assert!(matches!(
        kind,
        BackendKind::Cpu
            | BackendKind::Cuda
            | BackendKind::Rocm
            | BackendKind::Vulkan
            | BackendKind::Metal
    ));
    let backend = make_backend(kind);
    assert_eq!(backend.kind(), kind);
}

#[test]
fn backend_intel_alias_maps_to_vulkan() {
    // Safety: process-local env for this unit test only.
    std::env::set_var("RBITNET_BACKEND", "intel");
    assert_eq!(BackendKind::from_env(), BackendKind::Vulkan);
    std::env::set_var("RBITNET_BACKEND", "cpu");
    assert_eq!(BackendKind::from_env(), BackendKind::Cpu);
    std::env::remove_var("RBITNET_BACKEND");
}

#[test]
fn backend_default_is_auto_detect() {
    // Safety: process-local env for this unit test only.
    std::env::remove_var("RBITNET_BACKEND");
    assert_eq!(BackendKind::from_env(), BackendKind::detect_best());
    std::env::set_var("RBITNET_BACKEND", "auto");
    assert_eq!(BackendKind::from_env(), BackendKind::detect_best());
    std::env::remove_var("RBITNET_BACKEND");
}
