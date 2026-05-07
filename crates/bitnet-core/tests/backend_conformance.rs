use bitnet_core::backend::{
    make_backend, BackendKind, ComputeBackend, CpuBackend, CudaBackend, MetalBackend, RocmBackend,
    VulkanBackend,
};

fn check_backend_parity(cpu: &dyn ComputeBackend, other: &dyn ComputeBackend) {
    let w = vec![
        0.25f32, -0.5, 0.75, 0.1, 0.6, -0.2, 0.0, 0.9, -0.4, 0.8, 0.3, -0.7,
    ];
    let x = vec![0.4f32, -1.1, 0.2, 0.7];
    let y_cpu = cpu.matvec(&w, &x, 3, 4).expect("cpu matvec");
    let y_other = other.matvec(&w, &x, 3, 4).expect("backend matvec");
    for (a, b) in y_cpu.iter().zip(y_other.iter()) {
        assert!((a - b).abs() <= 1e-6, "mismatch cpu={a} other={b}");
    }
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
    let cpu = CpuBackend::default();
    let cuda = CudaBackend::default();
    let rocm = RocmBackend::default();
    let vulkan = VulkanBackend::default();
    let metal = MetalBackend::default();
    check_backend_parity(&cpu, &cuda);
    check_backend_parity(&cpu, &rocm);
    check_backend_parity(&cpu, &vulkan);
    check_backend_parity(&cpu, &metal);
}
