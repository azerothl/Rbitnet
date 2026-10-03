//! One native CUDA launch for all GQA heads, with a resident per-sequence KV cache.
use crate::error::{BitNetError, Result};
use std::ffi::c_void;
type Create = unsafe extern "C" fn(usize, usize, usize, usize, usize) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Reset = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const f32,
    *const f32,
    usize,
    usize,
    f32,
    *const f32,
    *mut f32,
) -> i32;
pub(crate) struct CudaAttention {
    context: usize,
    destroy: Destroy,
    reset: Reset,
    step: Step,
    capacity: usize,
    kv_heads: usize,
    key: usize,
    value: usize,
    heads: usize,
    filled: usize,
}
impl CudaAttention {
    pub fn new(
        capacity: usize,
        kv_heads: usize,
        key: usize,
        value: usize,
        heads: usize,
    ) -> Option<Self> {
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { lib.get::<Create>(b"rbitnet_cuda_attention_create\0").ok()? };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_attention_destroy\0")
                .ok()?
        };
        let reset = unsafe { *lib.get::<Reset>(b"rbitnet_cuda_attention_reset\0").ok()? };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_attention_step\0").ok()? };
        let context = unsafe { create(capacity, kv_heads, key, value, heads) } as usize;
        if context == 0 {
            return None;
        }
        Some(Self {
            context,
            destroy,
            reset,
            step,
            capacity,
            kv_heads,
            key,
            value,
            heads,
            filled: 0,
        })
    }
    pub fn clear(&mut self) {
        unsafe {
            (self.reset)(self.context as *mut c_void);
        }
        self.filled = 0;
    }
    pub fn run(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        pos: usize,
        first: usize,
        scale: f32,
        sinks: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        if pos >= self.capacity
            || first > pos
            || q.len() != self.heads * self.key
            || out.len() != self.heads * self.value
            || k.len() < (pos + 1) * self.kv_heads * self.key
            || v.len() < (pos + 1) * self.kv_heads * self.value
            || sinks.is_some_and(|s| s.len() != self.heads)
        {
            return Err(BitNetError::Inference(
                "CUDA attention shape mismatch".into(),
            ));
        }
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                pos,
                first,
                scale,
                sinks.map_or(std::ptr::null(), |s| s.as_ptr()),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA attention failed: {status}"
            )));
        }
        let begin = if pos >= self.filled { self.filled } else { pos };
        let upload = ((pos + 1 - begin) * self.kv_heads * (self.key + self.value)
            + q.len()
            + sinks.map_or(0, |s| s.len()))
            * 4;
        crate::perf::record_gpu_transfer(upload as u64, (out.len() * 4) as u64, 0);
        crate::perf::record_gpu_attention();
        self.filled = pos + 1;
        Ok(())
    }
}
impl Drop for CudaAttention {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn opt_in_attention_matches_gqa_sinks_window_and_restored_prefix() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        check_attention(2, 64, 64);
        // GLM's absorbed MLA keys and compressed values have different widths.
        check_attention(1, 576, 512);
    }

    fn check_attention(kv_heads: usize, dim: usize, value_dim: usize) {
        let heads = 4;
        let max_seq = 2048;
        let mut attention = super::CudaAttention::new(max_seq, kv_heads, dim, value_dim, heads)
            .expect("native CUDA attention required");
        let q: Vec<f32> = (0..heads * dim).map(|i| (i as f32 * 0.73).sin()).collect();
        let mut k: Vec<f32> = (0..max_seq * kv_heads * dim)
            .map(|i| (i as f32 * 0.19).sin())
            .collect();
        let v: Vec<f32> = (0..max_seq * kv_heads * value_dim)
            .map(|i| (i as f32 * 0.43).cos())
            .collect();
        let sinks = [-0.1, 0.5, 1.0, 0.0];
        for (step, &pos) in [0, 18, 256, 2047, 31].iter().enumerate() {
            if step == 4 {
                attention.clear();
                for x in &mut k {
                    *x *= -0.25;
                }
            }
            let first = if step % 2 == 0 { 0 } else { pos / 2 };
            let mut output = vec![0.0; heads * value_dim];
            attention
                .run(&q, &k, &v, pos, first, 0.125, Some(&sinks), &mut output)
                .unwrap();
            for h in 0..heads {
                let kh = h / (heads / kv_heads);
                let mut scores: Vec<f64> = (first..=pos)
                    .map(|p| {
                        (0..dim)
                            .map(|i| {
                                q[h * dim + i] as f64 * k[(p * kv_heads + kh) * dim + i] as f64
                            })
                            .sum::<f64>()
                            * 0.125
                    })
                    .collect();
                let max = scores
                    .iter()
                    .copied()
                    .chain([sinks[h] as f64])
                    .fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let total = scores.iter().sum::<f64>() + (sinks[h] as f64 - max).exp();
                for i in 0..value_dim {
                    let expected = (first..=pos)
                        .zip(&scores)
                        .map(|(p, s)| s / total * v[(p * kv_heads + kh) * value_dim + i] as f64)
                        .sum::<f64>();
                    assert!(
                        (output[h * value_dim + i] as f64 - expected).abs() < 2e-5,
                        "step {step}, head {h}, dim {i}"
                    );
                }
            }
        }
    }
}
