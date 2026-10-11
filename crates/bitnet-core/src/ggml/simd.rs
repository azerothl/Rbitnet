//! Portable dispatch for floating point dot products. GGUF blocks stay quantized in memory.
use std::sync::OnceLock;

type Dot = unsafe fn(&[f32], &[f32]) -> f32;
type ScaleAdd = unsafe fn(&mut [f32], &[f32], f32, f32);

unsafe fn scale_add_scalar(dst: &mut [f32], src: &[f32], scale_dst: f32, scale_src: f32) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst = dst.mul_add(scale_dst, src * scale_src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_add_avx2(dst: &mut [f32], src: &[f32], scale_dst: f32, scale_src: f32) {
    use std::arch::x86_64::*;
    let scale_dst_v = _mm256_set1_ps(scale_dst);
    let scale_src_v = _mm256_set1_ps(scale_src);
    let mut i = 0;
    while i + 8 <= dst.len() {
        let mixed = _mm256_fmadd_ps(
            _mm256_loadu_ps(dst.as_ptr().add(i)),
            scale_dst_v,
            _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), scale_src_v),
        );
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), mixed);
        i += 8;
    }
    while i < dst.len() {
        dst[i] = dst[i].mul_add(scale_dst, src[i] * scale_src);
        i += 1;
    }
}

/// `dst[i] = dst[i] * scale_dst + src[i] * scale_src`.
pub fn scale_add(dst: &mut [f32], src: &[f32], scale_dst: f32, scale_src: f32) {
    assert_eq!(dst.len(), src.len());
    static KERNEL: OnceLock<ScaleAdd> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return scale_add_avx2;
        }
        scale_add_scalar
    });
    unsafe { kernel(dst, src, scale_dst, scale_src) }
}

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

    #[test]
    fn scale_add_matches_reference() {
        let mut dst = vec![0.25f32, -1.5, 3.0, 0.0, 8.0, -0.125, 1.0, 2.0, 4.0];
        let src = vec![1.0f32, 2.0, -1.0, 0.5, 0.25, 4.0, -2.0, 0.0, 1.5];
        let mut reference = dst.clone();
        for (dst, src) in reference.iter_mut().zip(&src) {
            *dst = *dst * 0.5 + src * -1.25;
        }
        super::scale_add(&mut dst, &src, 0.5, -1.25);
        for (got, want) in dst.iter().zip(&reference) {
            assert!((got - want).abs() < 1e-6);
        }
    }
}
