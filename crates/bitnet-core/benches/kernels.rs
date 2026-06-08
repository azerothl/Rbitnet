//! Micro-benchmarks for hot kernels (baseline before SIMD / packed BitNet layouts).

use bitnet_core::backend::{ComputeBackend, CudaBackend};
use bitnet_core::kernels::matvec_ternary_i8;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_matvec_medium(c: &mut Criterion) {
    let n = 512usize;
    let k = 4096usize;
    let w = vec![0i8; n * k];
    let x = vec![0.0f32; k];
    let mut y = vec![0.0f32; n];
    c.bench_function("matvec_ternary_i8 512x4096", |b| {
        b.iter(|| {
            matvec_ternary_i8(
                black_box(&w),
                black_box(&x),
                black_box(&mut y),
                black_box(n),
                black_box(k),
            )
        })
    });
}

fn bench_cuda_backend_matvec(c: &mut Criterion) {
    if std::env::var("RBITNET_BENCH_CUDA").as_deref() != Ok("1") {
        return;
    }
    let backend = CudaBackend::default();
    if !backend.is_native_accelerated() {
        return;
    }
    let n = 512usize;
    let k = 4096usize;
    let w = vec![0.0f32; n * k];
    let x = vec![0.0f32; k];
    c.bench_function("cuda_backend_matvec_f32 512x4096", |b| {
        b.iter(|| {
            black_box(
                backend
                    .matvec(black_box(&w), black_box(&x), black_box(n), black_box(k))
                    .expect("cuda backend matvec"),
            )
        })
    });
}

criterion_group!(benches, bench_matvec_medium, bench_cuda_backend_matvec);
criterion_main!(benches);
