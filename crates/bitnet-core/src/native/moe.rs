//! Keep routed expert activations on CUDA; one synchronization per FFN layer.
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
#[repr(C)]
struct Config {
    embd: u32,
    ffn: u32,
    experts: u32,
    used: u32,
    oai: u32,
}
type Create = unsafe extern "C" fn(
    *const Config,
    *const Matrix,
    *const Matrix,
    *const Matrix,
    *const f32,
    *const f32,
    *const f32,
) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(*mut c_void, *const f32, *const u32, *const f32, *mut f32) -> i32;

pub(super) struct GpuMoe {
    context: usize,
    destroy: Destroy,
    step: Step,
    _weights: [CudaDeviceQuantMatrix; 3],
    embd: usize,
    used: usize,
    experts: usize,
}
impl GpuMoe {
    pub fn new(
        weights: &Weights,
        layer: usize,
        experts: usize,
        used: usize,
        oai: bool,
    ) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_MOE").as_deref() == Ok("0") {
            return None;
        }
        let name = |suffix: &str| format!("blk.{layer}.ffn_{suffix}_exps");
        let owned = [
            weights.device_matrix(&(name("gate") + ".weight"))?,
            weights.device_matrix(&(name("up") + ".weight"))?,
            weights.device_matrix(&(name("down") + ".weight"))?,
        ];
        let matrix = |m: &CudaDeviceQuantMatrix| -> Option<Matrix> {
            Some(Matrix {
                weights: m.device_address()? as *const c_void,
                row_bytes: m.bytes() / m.out_rows(),
                ty: m.ggml_type(),
                cols: m.in_cols().try_into().ok()?,
                rows: m.out_rows().try_into().ok()?,
            })
        };
        let (gate, up, down) = (matrix(&owned[0])?, matrix(&owned[1])?, matrix(&owned[2])?);
        if experts == 0 || used == 0 || used > experts || owned[0].out_rows() % experts != 0 {
            return None;
        }
        let bias = |suffix: &str, rows: usize| -> Option<*const f32> {
            match weights.dense(&(name(suffix) + ".bias")) {
                Ok(b) if b.len() == rows => Some(b.as_ptr()),
                Ok(_) => None,
                Err(_) => Some(std::ptr::null()),
            }
        };
        let cfg = Config {
            embd: gate.cols,
            ffn: (owned[0].out_rows() / experts).try_into().ok()?,
            experts: experts.try_into().ok()?,
            used: used.try_into().ok()?,
            oai: u32::from(oai),
        };
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { lib.get::<Create>(b"rbitnet_cuda_moe_create\0").ok()? };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").ok()? };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_moe_step\0").ok()? };
        let context = unsafe {
            create(
                &cfg,
                &gate,
                &up,
                &down,
                bias("gate", gate.rows as usize)?,
                bias("up", up.rows as usize)?,
                bias("down", down.rows as usize)?,
            )
        } as usize;
        if context == 0 {
            return None;
        }
        Some(Self {
            context,
            destroy,
            step,
            _weights: owned,
            embd: cfg.embd as usize,
            used,
            experts,
        })
    }
    pub fn run(
        &mut self,
        input: &[f32],
        selected: &[usize],
        probabilities: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != self.embd
            || selected.len() != self.used
            || probabilities.len() != self.used
            || selected.iter().any(|&i| i >= self.experts)
        {
            return Err(BitNetError::Inference(
                "CUDA routed expert shape mismatch".into(),
            ));
        }
        let indices: Vec<u32> = selected.iter().map(|&i| i as u32).collect();
        let mut output = vec![0.0; self.embd];
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                input.as_ptr(),
                indices.as_ptr(),
                probabilities.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA routed experts failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            ((input.len() + 2 * self.used) * 4) as u64,
            (output.len() * 4) as u64,
            3,
        );
        crate::perf::record_cuda_graph_replay();
        Ok(output)
    }
}
impl Drop for GpuMoe {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_routed_graph_matches_f64_experts_biases_and_selection_changes() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let rt = crate::backend::CudaRuntime::try_load().expect("CUDA required");
        let lib = crate::ggml::load_cuda_quant_library().expect("native CUDA DLL required");
        let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_moe_create\0").unwrap() };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").unwrap() };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_moe_step\0").unwrap() };
        const N: usize = 256;
        const EXPERTS: usize = 5;
        for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
            let row_bytes = crate::ggml::ggml_row_size(ty, N as u64).unwrap();
            let mut dense = Vec::new();
            let mut owned = Vec::new();
            let mut biases = Vec::new();
            for projection in 0..3 {
                let mut payload: Vec<u8> = (0..row_bytes * N * EXPERTS)
                    .map(|i| (i * 37 + projection * 53 + 19) as u8)
                    .collect();
                if ty == 0 {
                    for (i, slot) in payload.chunks_exact_mut(4).enumerate() {
                        slot.copy_from_slice(
                            &((i as f32 * 0.031 + projection as f32).cos() * 0.1).to_le_bytes(),
                        );
                    }
                } else {
                    let bytes = match ty {
                        2 => 18,
                        6 => 22,
                        8 => 34,
                        12 => 144,
                        13 => 176,
                        14 => 210,
                        39 => 17,
                        _ => unreachable!(),
                    };
                    for b in payload.chunks_exact_mut(bytes) {
                        if ty == 39 {
                            b[0] = 119;
                        } else {
                            let offset = if ty == 14 { 208 } else { 0 };
                            b[offset..offset + 2].copy_from_slice(
                                &half::f16::from_f32(0.0007).to_bits().to_le_bytes(),
                            );
                            if ty == 12 || ty == 13 {
                                b[2..4].copy_from_slice(
                                    &half::f16::from_f32(0.0003).to_bits().to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                dense.push(
                    crate::ggml::tensor_to_f32(&payload, ty, &[N as u64, N as u64, EXPERTS as u64])
                        .unwrap(),
                );
                owned.push(
                    CudaDeviceQuantMatrix::from_payload(Some(&rt), ty, payload, N * EXPERTS, N)
                        .unwrap(),
                );
                biases.push(
                    (0..N * EXPERTS)
                        .map(|i| (i as f32 * 0.17 + projection as f32).sin() * 0.2)
                        .collect::<Vec<_>>(),
                );
            }
            let matrices: Vec<_> = owned
                .iter()
                .map(|m| Matrix {
                    weights: m.device_address().unwrap() as *const c_void,
                    row_bytes,
                    ty,
                    cols: N as u32,
                    rows: (N * EXPERTS) as u32,
                })
                .collect();
            for oai in [0, 1] {
                let cfg = Config {
                    embd: N as u32,
                    ffn: N as u32,
                    experts: EXPERTS as u32,
                    used: 3,
                    oai,
                };
                let context = unsafe {
                    create(
                        &cfg,
                        &matrices[0],
                        &matrices[1],
                        &matrices[2],
                        biases[0].as_ptr(),
                        biases[1].as_ptr(),
                        biases[2].as_ptr(),
                    )
                };
                assert!(!context.is_null());
                let probabilities = [0.5, 0.3, 0.2];
                for selected in [[3, 1, 4], [4, 0, 2]] {
                    let x: Vec<f32> = (0..N).map(|i| (i as f32 * 0.23).sin() * 4.0).collect();
                    let mut expected = vec![0.0f64; N];
                    let dot = |p: usize, e: usize, row: usize, input: &[f64]| -> f64 {
                        dense[p][(e * N + row) * N..(e * N + row + 1) * N]
                            .iter()
                            .zip(input)
                            .map(|(&w, &v)| w as f64 * v)
                            .sum::<f64>()
                            + biases[p][e * N + row] as f64
                    };
                    let input: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                    for (slot, &e) in selected.iter().enumerate() {
                        let hidden: Vec<f64> = (0..N)
                            .map(|r| {
                                let mut g = dot(0, e as usize, r, &input);
                                let mut u = dot(1, e as usize, r, &input);
                                if oai == 1 {
                                    g = g.min(7.0);
                                    u = u.clamp(-7.0, 7.0) + 1.0;
                                }
                                g / (1.0
                                    + (-(if oai == 1 { 1.702f32 as f64 } else { 1.0 }) * g).exp())
                                    * u
                            })
                            .collect();
                        for (r, value) in expected.iter_mut().enumerate() {
                            *value += probabilities[slot] as f64 * dot(2, e as usize, r, &hidden);
                        }
                    }
                    let mut output = vec![0.0; N];
                    assert_eq!(
                        unsafe {
                            step(
                                context,
                                x.as_ptr(),
                                selected.as_ptr(),
                                probabilities.as_ptr(),
                                output.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    for (i, (&got, &expected)) in output.iter().zip(&expected).enumerate() {
                        assert!(
                            (got as f64 - expected).abs() < 5e-4 * (1.0 + expected.abs()),
                            "format {ty}, oai {oai}, row {i}: {got} vs {expected}"
                        );
                    }
                }
                unsafe {
                    destroy(context);
                }
            }
        }
    }
}
