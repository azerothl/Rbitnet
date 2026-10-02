//! Micro-benchmarks for hot ternary kernels (I2_S / TL2-style, NATIVE_FIRST).

use bitnet_core::backend::{ComputeBackend, CudaBackend, CudaDeviceMatrix, CudaDeviceQuantMatrix};
use bitnet_core::fused_batch::{dense_matvec_multi_seq, dense_matvec_sequential_reference};
use bitnet_core::ggml::matvec_payload_quant;
use bitnet_core::kernels::{
    matvec_ternary_auto, matvec_ternary_i2s, matvec_ternary_i8, matvec_ternary_tl2_lut,
    pack_ternary_i2s,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_matvec_medium(c: &mut Criterion) {
    let n = 512usize;
    let k = 4096usize;
    let mut w = vec![0i8; n * k];
    for (i, slot) in w.iter_mut().enumerate() {
        *slot = match i % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }
    let packed = pack_ternary_i2s(&w);
    let x = vec![0.01f32; k];
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
    c.bench_function("matvec_ternary_i2s 512x4096", |b| {
        b.iter(|| {
            matvec_ternary_i2s(
                black_box(&packed),
                black_box(&x),
                black_box(&mut y),
                black_box(n),
                black_box(k),
            )
        })
    });
    c.bench_function("matvec_ternary_tl2_lut 512x4096", |b| {
        b.iter(|| {
            matvec_ternary_tl2_lut(
                black_box(&packed),
                black_box(&x),
                black_box(&mut y),
                black_box(n),
                black_box(k),
            )
        })
    });
    c.bench_function("matvec_ternary_auto 512x4096", |b| {
        b.iter(|| {
            matvec_ternary_auto(
                black_box(&packed),
                black_box(&x),
                black_box(&mut y),
                black_box(n),
                black_box(k),
            )
        })
    });
}

fn bench_fused_multi_seq(c: &mut Criterion) {
    // Weight-stationary vs N independent matvecs — building block for #46 (not e2e).
    let n = 512usize;
    let k = 2048usize;
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32) * 0.01).collect();
    for batch in [4usize, 8usize] {
        let x: Vec<f32> = (0..batch * k)
            .map(|i| ((i % 13) as f32) * 0.02)
            .collect();
        let mut y = vec![0.0f32; batch * n];
        c.bench_function(&format!("fused_multi_seq dense {n}x{k} batch={batch}"), |b| {
            b.iter(|| {
                dense_matvec_multi_seq(
                    black_box(&w),
                    black_box(&x),
                    black_box(&mut y),
                    black_box(n),
                    black_box(k),
                    black_box(batch),
                )
            })
        });
        c.bench_function(
            &format!("fused_multi_seq sequential_ref {n}x{k} batch={batch}"),
            |b| {
                b.iter(|| {
                    dense_matvec_sequential_reference(
                        black_box(&w),
                        black_box(&x),
                        black_box(&mut y),
                        black_box(n),
                        black_box(k),
                        black_box(batch),
                    )
                })
            },
        );
    }
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

    // Device-resident f32 + quant Q4_0 rows (#22 Gate B / E) when CUDA is present.
    if let Some(rt) = bitnet_core::CudaRuntime::try_load() {
        if let Some(matrix) = CudaDeviceMatrix::upload(rt.clone(), &w, n, k) {
            c.bench_function("cuda_device_resident_f32 512x4096", |b| {
                b.iter(|| black_box(matrix.matvec(black_box(&x)).expect("resident f32")))
            });
        }

        let row_bytes = 4096 / 32 * 18;
        let mut payload = vec![0u8; row_bytes * n];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(3).wrapping_add(1);
        }
        if let Ok(qmatrix) = CudaDeviceQuantMatrix::from_payload(Some(&rt), 2, payload.clone(), n, k)
        {
            c.bench_function("cuda_device_quant_q4_0 512x4096", |b| {
                b.iter(|| {
                    black_box(
                        qmatrix
                            .matvec(black_box(&x))
                            .expect("resident quant matvec"),
                    )
                })
            });
            let _ = matvec_payload_quant(2, &payload, &x, k, n);
            let _ = qmatrix.is_device_resident();
        }
    }
}

criterion_group!(
    benches,
    bench_matvec_medium,
    bench_fused_multi_seq,
    bench_cuda_backend_matvec
);
criterion_main!(benches);
