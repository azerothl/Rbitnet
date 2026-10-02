//! Isolate f32::mul_add code generation, without changing the production kernels.
//! rustc -O -C target-feature=+crt-static --emit=asm,link scripts/fma_diagnostic.rs --out-dir target/fma-diagnostic
use std::{hint::black_box, time::Instant};

#[inline(never)]
fn generic_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        sum = a.mul_add(b, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline(never)]
unsafe fn hardware_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        sum = a.mul_add(b, sum);
    }
    sum
}

fn measure(dot: impl Fn(&[f32], &[f32]) -> f32, a: &[f32], b: &[f32]) -> (f64, u32) {
    let started = Instant::now();
    let mut result = 0.0;
    for _ in 0..20000 {
        result = black_box(dot(black_box(a), black_box(b)));
    }
    (started.elapsed().as_secs_f64() * 1000.0, result.to_bits())
}

fn main() {
    let a: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.031).sin()).collect();
    let b: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.019).cos()).collect();
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            std::is_x86_feature_detected!("fma"),
            "FMA-capable CPU required"
        );
        for repetition in 0..3 {
            let (generic_ms, generic_bits) = measure(generic_dot, &a, &b);
            let (fma_ms, fma_bits) = measure(|a, b| unsafe { hardware_dot(a, b) }, &a, &b);
            assert_eq!(generic_bits, fma_bits, "fused-result mismatch");
            println!("{{\"repetition\":{repetition},\"elements\":2048,\"iterations\":20000,\"generic_ms\":{generic_ms},\"hardware_fma_ms\":{fma_ms},\"result_bits\":{generic_bits}}}");
        }
    }
}
