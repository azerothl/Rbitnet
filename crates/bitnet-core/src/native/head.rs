//! Shared GPU output projection and greedy reduction for native architectures.
use super::weights::Weights;
use crate::backend::CudaDeviceQuantMatrix;
use crate::error::{BitNetError, Result};
use std::ffi::c_void;

#[repr(C)]
struct Matrix {
    weights: *const c_void,
    row_bytes: usize,
    ty: u32,
    cols: u32,
    rows: u32,
}
type Create = unsafe extern "C" fn(*const Matrix, *const f32, f32) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, *mut f32, *mut u32) -> i32;
pub(crate) struct GpuHead {
    context: usize,
    destroy: Destroy,
    step: Step,
    _weight: CudaDeviceQuantMatrix,
    embd: usize,
    vocab: usize,
}
impl GpuHead {
    pub fn new(weights: &Weights, name: &str, epsilon: f32) -> Option<Self> {
        // The resident head reduces transfers, but current single-request Windows
        // measurements do not improve latency. Keep it opt-in until this is resolved.
        if std::env::var("RBITNET_CUDA_HEAD").as_deref() != Ok("1") {
            return None;
        }
        let owned = weights.device_matrix(name)?;
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_head_create\0").ok()? };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_head_destroy\0").ok()? };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_head_step\0").ok()? };
        let matrix = Matrix {
            weights: owned.device_address()? as *const c_void,
            row_bytes: owned.bytes().checked_div(owned.out_rows())?,
            ty: owned.ggml_type(),
            cols: owned.in_cols().try_into().ok()?,
            rows: owned.out_rows().try_into().ok()?,
        };
        let norm = weights.dense("output_norm.weight").ok()?;
        if norm.len() != owned.in_cols() {
            return None;
        }
        let context = unsafe { create(&matrix, norm.as_ptr(), epsilon) } as usize;
        if context == 0 {
            return None;
        }
        Some(Self {
            context,
            destroy,
            step,
            embd: owned.in_cols(),
            vocab: owned.out_rows(),
            _weight: owned,
        })
    }
    pub fn run(&mut self, input: &[f32], greedy: bool) -> Result<(Vec<f32>, Option<u32>)> {
        if input.len() != self.embd {
            return Err(BitNetError::Inference(
                "resident output head input shape mismatch".into(),
            ));
        }
        let mut logits = if greedy {
            Vec::new()
        } else {
            vec![0.0; self.vocab]
        };
        let mut token = 0;
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                input.as_ptr(),
                u32::from(greedy),
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "resident output head failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            (self.embd * 4) as u64,
            if greedy { 4 } else { (self.vocab * 4) as u64 },
            1,
        );
        crate::perf::record_cuda_graph_replay();
        Ok((logits, greedy.then_some(token)))
    }
}
impl Drop for GpuHead {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn opt_in_head_preserves_f64_logits_large_vocab_nan_ties_and_input_changes() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let rt = crate::backend::CudaRuntime::try_load().expect("CUDA required");
        let lib = crate::ggml::load_cuda_quant_library().expect("CUDA DLL required");
        let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_head_create\0").unwrap() };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_head_destroy\0").unwrap() };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_head_step\0").unwrap() };
        let cols = 32;
        let vocab = 131079;
        let mut weights: Vec<f32> = (0..cols * vocab)
            .map(|i| (i as f32 * 0.13).cos() * 0.003)
            .collect();
        for (id, first) in [
            (vocab - 2, 20.0),
            (vocab - 1, 20.0),
            (3, -20.0),
            (vocab / 2 + 3, -20.0),
            (1, f32::NAN),
        ] {
            weights[id * cols..(id + 1) * cols].fill(0.0);
            weights[id * cols] = first;
        }
        let payload = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let owned =
            CudaDeviceQuantMatrix::from_payload(Some(&rt), 0, payload, vocab, cols).unwrap();
        let m = Matrix {
            weights: owned.device_address().unwrap() as *const c_void,
            row_bytes: cols * 4,
            ty: 0,
            cols: cols as u32,
            rows: vocab as u32,
        };
        let norm: Vec<_> = (0..cols)
            .map(|i| 1.0 + (i as f32 * 0.17).sin() * 0.01)
            .collect();
        let context = unsafe { create(&m, norm.as_ptr(), 1e-5) } as usize;
        assert_ne!(context, 0);
        let mut head = GpuHead {
            context,
            destroy,
            step,
            _weight: owned,
            embd: cols,
            vocab,
        };
        for sign in [1.0, -1.0, 1.0] {
            let mut input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.7).sin()).collect();
            input[0] = sign;
            let inv = (input.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / cols as f64 + 1e-5)
                .sqrt()
                .recip();
            let x: Vec<f64> = input
                .iter()
                .zip(&norm)
                .map(|(&x, &w)| x as f64 * w as f64 * inv)
                .collect();
            let (logits, id) = head.run(&input, false).unwrap();
            assert!(id.is_none());
            for (row, &got) in weights.chunks_exact(cols).zip(&logits) {
                let expected = row.iter().zip(&x).map(|(&w, &x)| w as f64 * x).sum::<f64>();
                if expected.is_nan() {
                    assert!(got.is_nan());
                } else {
                    assert!((got as f64 - expected).abs() < 2e-5 * (1.0 + expected.abs()));
                }
            }
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            let expected = crate::sampling::sample_token(
                &logits,
                &crate::sampling::SamplingOptions::from_temperature(0.0),
                &[],
                &mut rng,
            );
            assert_eq!(
                expected as usize,
                if sign > 0.0 { vocab - 1 } else { vocab / 2 + 3 }
            );
            let (empty, id) = head.run(&input, true).unwrap();
            assert!(empty.is_empty());
            assert_eq!(id, Some(expected));
        }
    }
}
