//! Keep routed expert activations on CUDA; one synchronization per FFN layer.
use super::expert_cache::{ExpertGroup, SharedCache, Upload};
use super::weights::Weights;
use crate::backend::CudaDeviceQuantMatrix;
use crate::error::{BitNetError, Result};
use std::ffi::c_void;
use std::sync::Arc;

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
type DynamicStep = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const u32,
    *const f32,
    *mut f32,
    *const *const c_void,
) -> i32;

enum Storage {
    Fixed {
        _weights: [CudaDeviceQuantMatrix; 3],
    },
    Cached {
        cache: SharedCache,
        layer: usize,
        step: DynamicStep,
        selected_bytes: usize,
    },
}

/// Active/pending native FFN reads must finish before these lease owners drop.
pub(super) struct SelectedExperts {
    _leases: Vec<Arc<ExpertGroup>>,
    pointers: Vec<*const c_void>,
    upload: Upload,
}
impl SelectedExperts {
    pub(super) fn upload(&self) -> Upload {
        self.upload
    }

    pub(super) fn pointers(&self) -> Option<&[*const c_void]> {
        (!self.pointers.is_empty()).then_some(self.pointers.as_slice())
    }
}
pub(super) struct GpuMoe {
    context: usize,
    destroy: Destroy,
    step: Step,
    storage: Storage,
    embd: usize,
    used: usize,
    experts: usize,
    fused: bool,
}
impl GpuMoe {
    pub(super) fn is_fused(&self)->bool {self.fused}
    pub(super) fn context_address(&self) -> usize {
        self.context
    }
    pub(super) fn lease_selected(&self, selected: &[usize]) -> Result<Option<SelectedExperts>> {
        if selected.len() != self.used || selected.iter().any(|&e| e >= self.experts) {
            return Err(BitNetError::Inference(
                "CUDA expert selection shape mismatch".into(),
            ));
        }
        let mut leases = Vec::new();
        let mut upload = Upload::default();
        let mut pointers = Vec::new();
        if let Storage::Cached {
            cache,
            layer,
            selected_bytes,
            ..
        } = &self.storage
        {
            let waiting = std::time::Instant::now();
            let mut cache = cache
                .lock()
                .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?;
            cache.record_lock_wait(
                *layer,
                waiting.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            );
            cache.trace_route(*layer, selected, *selected_bytes / self.used);
            if !cache.fits(*selected_bytes) {
                cache.record_capacity_refusal(*layer);
                return Ok(None);
            }
            let Some((selected_leases, selected_upload)) =
                cache.acquire_selected_with_upload(*layer, selected)?
            else {
                return Ok(None);
            };
            leases = selected_leases;
            upload = selected_upload;
            drop(cache);
            for projection in 0..3 {
                for group in &leases {
                    pointers.push(group.matrices[projection].device_address().ok_or_else(|| {
                        BitNetError::Inference("cached expert missing device buffer".into())
                    })? as *const c_void);
                }
            }
        }
        Ok(Some(SelectedExperts {
            _leases: leases,
            pointers,
            upload,
        }))
    }
    /// Estimate without admission: looking at locality cannot upload or evict.
    pub(super) fn estimate_selected(&self, selected: &[usize]) -> Result<(bool, usize)> {
        if selected.len() != self.used || selected.iter().any(|&e| e >= self.experts) {
            return Err(BitNetError::Inference(
                "CUDA expert selection shape mismatch".into(),
            ));
        }
        match &self.storage {
            Storage::Fixed { .. } => Ok((true, 0)),
            Storage::Cached {
                cache,
                layer,
                selected_bytes,
                ..
            } => {
                let waiting = std::time::Instant::now();
                let cache = cache
                    .lock()
                    .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?;
                cache.record_lock_wait(
                    *layer,
                    waiting.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                Ok((
                    cache.fits(*selected_bytes),
                    cache.missing_selected_bytes(*layer, selected, *selected_bytes / self.used)?,
                ))
            }
        }
    }
    // The whole-token graph borrows only immutable, fully resident expert banks.
    // Cached banks require lease acquisition between router and expert execution.
    pub(super) fn fixed_context_address(&self) -> Option<usize> {
        matches!(self.storage, Storage::Fixed { .. }).then_some(self.context)
    }
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
        let cached = weights.expert_cache.clone();
        let owned = if cached.is_none() {
            Some([
                weights.device_matrix(&(name("gate") + ".weight"))?,
                weights.device_matrix(&(name("up") + ".weight"))?,
                weights.device_matrix(&(name("down") + ".weight"))?,
            ])
        } else {
            None
        };
        let matrix = |m: &CudaDeviceQuantMatrix| -> Option<Matrix> {
            Some(Matrix {
                weights: m.device_address()? as *const c_void,
                row_bytes: m.bytes() / m.out_rows(),
                ty: m.ggml_type(),
                cols: m.in_cols().try_into().ok()?,
                rows: m.out_rows().try_into().ok()?,
            })
        };
        let descriptor = |projection: &str| -> Option<Matrix> {
            let t = weights.tensor(&(name(projection) + ".weight")).ok()?;
            if t.dimensions.len() != 3
                || t.dimensions[2] as usize != experts
                || !crate::ggml::ggml_type_supports_cuda_quant(t.ggml_type)
            {
                return None;
            }
            Some(Matrix {
                weights: std::ptr::null(),
                row_bytes: crate::ggml::ggml_row_size(t.ggml_type, t.dimensions[0]).ok()?,
                ty: t.ggml_type,
                cols: t.dimensions[0].try_into().ok()?,
                rows: t.dimensions[1]
                    .checked_mul(t.dimensions[2])?
                    .try_into()
                    .ok()?,
            })
        };
        let (gate, up, down) = if let Some(ref owned) = owned {
            (matrix(&owned[0])?, matrix(&owned[1])?, matrix(&owned[2])?)
        } else {
            (descriptor("gate")?, descriptor("up")?, descriptor("down")?)
        };
        if experts == 0 || used == 0 || used > experts || gate.rows as usize % experts != 0 {
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
            ffn: (gate.rows as usize / experts).try_into().ok()?,
            experts: experts.try_into().ok()?,
            used: used.try_into().ok()?,
            oai: u32::from(oai),
        };
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe {
            lib.get::<Create>(if cached.is_some() {
                b"rbitnet_cuda_moe_dynamic_create\0"
            } else {
                b"rbitnet_cuda_moe_create\0"
            })
            .ok()?
        };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").ok()? };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_moe_step\0").ok()? };
        let storage = if let Some(cache) = cached {
            Storage::Cached {
                cache,
                layer,
                step: unsafe {
                    *lib.get::<DynamicStep>(b"rbitnet_cuda_moe_dynamic_step\0")
                        .ok()?
                },
                selected_bytes: [(&gate, cfg.ffn), (&up, cfg.ffn), (&down, cfg.embd)]
                    .iter()
                    .try_fold(0usize, |n, (m, rows)| {
                        n.checked_add(m.row_bytes.checked_mul(*rows as usize)?)
                    })?
                    .checked_mul(used)?,
            }
        } else {
            Storage::Fixed { _weights: owned? }
        };
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
        let requested=std::env::var("RBITNET_CUDA_MOE_FUSED").as_deref()==Ok("1");
        let fused=if requested {
            type Configure=unsafe extern "C" fn(*mut c_void,u32)->i32;
            let configure=unsafe{lib.get::<Configure>(b"rbitnet_cuda_moe_configure_fused\0").ok()};
            configure.is_some_and(|f|unsafe{f(context as *mut c_void,1)}==0)
        } else {false};
        Some(Self {
            fused,
            context,
            destroy,
            step,
            storage,
            embd: cfg.embd as usize,
            used,
            experts,
        })
    }
    pub fn run_with_upload(
        &mut self,
        input: &[f32],
        selected: &[usize],
        probabilities: &[f32],
    ) -> Result<Option<(Vec<f32>, Upload)>> {
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
        let Some(leased) = self.lease_selected(selected)? else {
            return Ok(None);
        };
        let status = if let Storage::Cached { step, .. } = &self.storage {
            unsafe {
                step(
                    self.context as *mut c_void,
                    input.as_ptr(),
                    indices.as_ptr(),
                    probabilities.as_ptr(),
                    output.as_mut_ptr(),
                    leased.pointers.as_ptr(),
                )
            }
        } else {
            unsafe {
                (self.step)(
                    self.context as *mut c_void,
                    input.as_ptr(),
                    indices.as_ptr(),
                    probabilities.as_ptr(),
                    output.as_mut_ptr(),
                )
            }
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA routed experts failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            ((input.len() + 2 * self.used) * 4
                + if matches!(self.storage, Storage::Cached { .. }) {
                    3 * self.used * std::mem::size_of::<usize>()
                } else {
                    0
                }) as u64,
            (output.len() * 4) as u64,
            3,
        );
        crate::perf::record_cuda_graph_replay();
        Ok(Some((output, leased.upload)))
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
        let dynamic_create = unsafe {
            *lib.get::<Create>(b"rbitnet_cuda_moe_dynamic_create\0")
                .unwrap()
        };
        let dynamic_step = unsafe {
            *lib.get::<DynamicStep>(b"rbitnet_cuda_moe_dynamic_step\0")
                .unwrap()
        };
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
            if std::env::var("RBITNET_CUDA_PREFILL_TEST").as_deref() == Ok("1") {
                type Gemm = unsafe extern "C" fn(
                    u32,
                    *const c_void,
                    usize,
                    *const f32,
                    u32,
                    u32,
                    u32,
                    *mut f32,
                ) -> i32;
                let gemm = unsafe {
                    *lib.get::<Gemm>(b"rbitnet_cuda_quant_gemm_device\0")
                        .unwrap()
                };
                for tokens in [1usize, 2, 17] {
                    let input: Vec<f32> =
                        (0..tokens * N).map(|i| (i as f32 * 0.23).sin()).collect();
                    let x = rt.upload_f32(&input).unwrap();
                    let y = rt.upload_f32(&vec![0.0; tokens * N * EXPERTS]).unwrap();
                    for (p, matrix) in owned.iter().enumerate() {
                        assert_eq!(
                            unsafe {
                                gemm(
                                    ty,
                                    matrix.device_address().unwrap() as *const c_void,
                                    row_bytes,
                                    x.as_device_ptr().cast(),
                                    N as u32,
                                    (N * EXPERTS) as u32,
                                    tokens as u32,
                                    y.as_device_ptr().cast(),
                                )
                            },
                            0
                        );
                        let actual = y.download_f32().unwrap();
                        for token in 0..tokens {
                            for row in 0..N * EXPERTS {
                                let expected: f64 = dense[p][row * N..(row + 1) * N]
                                    .iter()
                                    .zip(&input[token * N..(token + 1) * N])
                                    .map(|(&w, &v)| w as f64 * v as f64)
                                    .sum();
                                let got = actual[token * N * EXPERTS + row] as f64;
                                assert!((got-expected).abs() < 2e-5*(1.0+expected.abs()), "GEMM format {ty} tokens {tokens} projection {p} row {row}: {got} vs {expected}");
                            }
                        }
                    }
                }
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
            for (oai, dynamic) in [(0, false), (1, false), (0, true), (1, true)] {
                let cfg = Config {
                    embd: N as u32,
                    ffn: N as u32,
                    experts: EXPERTS as u32,
                    used: 3,
                    oai,
                };
                let context = unsafe {
                    (if dynamic { dynamic_create } else { create })(
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
                let mut slots: Vec<CudaDeviceQuantMatrix> = Vec::new();
                for (iteration, selected) in
                    [[3, 1, 4], [4, 0, 2], [1, 4, 3]].into_iter().enumerate()
                {
                    // Replay both after compatible slot refills and after all
                    // selected allocations are freed/replaced (eviction).
                    if iteration == 2 {
                        slots.clear();
                    }
                    if dynamic {
                        for (p, bank) in owned.iter().enumerate() {
                            for (s, &e) in selected.iter().enumerate() {
                                let slab = row_bytes * N;
                                let payload = bank.host_payload()
                                    [e as usize * slab..(e as usize + 1) * slab]
                                    .to_vec();
                                if iteration == 1 {
                                    assert!(slots[p * 3 + s].refill(payload).unwrap());
                                } else {
                                    slots.push(
                                        CudaDeviceQuantMatrix::from_payload(
                                            Some(&rt),
                                            ty,
                                            payload,
                                            N,
                                            N,
                                        )
                                        .unwrap(),
                                    );
                                }
                            }
                        }
                    }
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
                            if dynamic {
                                let pointers: Vec<_> = slots
                                    .iter()
                                    .map(|m| m.device_address().unwrap() as *const c_void)
                                    .collect();
                                dynamic_step(
                                    context,
                                    x.as_ptr(),
                                    selected.as_ptr(),
                                    probabilities.as_ptr(),
                                    output.as_mut_ptr(),
                                    pointers.as_ptr(),
                                )
                            } else {
                                step(
                                    context,
                                    x.as_ptr(),
                                    selected.as_ptr(),
                                    probabilities.as_ptr(),
                                    output.as_mut_ptr(),
                                )
                            }
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

#[cfg(test)]
#[path="fused_tests.rs"]
mod fused_tests;

#[cfg(test)]
#[path="grouped_tests.rs"]
mod grouped_tests;
