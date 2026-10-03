//! Portable dispatch for floating point dot products. GGUF blocks stay quantized in memory.
use std::sync::OnceLock;

type Dot = unsafe fn(&[f32], &[f32]) -> f32;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    static KERNEL: OnceLock<Dot> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return dot_avx2;
        }
        dot_scalar
    });
    // The dispatch checks the CPU features before selecting the specialized function.
    unsafe { kernel(a, b) }
}

unsafe fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        s0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            s0,
        );
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            s1,
        );
        i += 16;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(s0, s1));
    let mut sum: f32 = lanes.into_iter().sum();
    while i < a.len() {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    #[test]
    fn dispatch_matches_reference_with_tail_and_cancellation() {
        for n in [0, 1, 15, 16, 17, 32, 256, 2049] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.61).cos()).collect();
            let reference: f64 = a.iter().zip(&b).map(|(&a, &b)| a as f64 * b as f64).sum();
            assert!((super::dot(&a, &b) as f64 - reference).abs() < 2e-4);
        }
    }
}
