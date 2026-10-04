//! Dense Qwen token pipeline. The owning runtime drops this before recurrent contexts.
use super::{qwen_recurrent::GpuRecurrent, weights::Weights};
use crate::backend::{BackendKind, CudaDeviceQuantMatrix};
use crate::error::{BitNetError, Result};
use crate::qwen35::config::Qwen35Config;
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Matrix {
    weights: *const c_void,
    row_bytes: usize,
    ty: u32,
    cols: u32,
    rows: u32,
}
impl Matrix {
    fn from_device(m: &CudaDeviceQuantMatrix) -> Option<Self> {
        Some(Self {
            weights: m.device_address()? as *const c_void,
            row_bytes: m.bytes().checked_div(m.out_rows())?,
            ty: m.ggml_type(),
            cols: m.in_cols().try_into().ok()?,
            rows: m.out_rows().try_into().ok()?,
        })
    }
}
#[repr(C)]
struct Config {
    embd: u32,
    ffn: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary: u32,
    capacity: u32,
    gated: u32,
    graphs: u32,
    epsilon: f32,
    scale: f32,
}
#[repr(C)]
struct Layer {
    kind: u32,
    context: *mut c_void,
}
type Destroy = unsafe extern "C" fn(*mut c_void);
type AttentionCreate = unsafe extern "C" fn(
    *const Config,
    *const Matrix,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
) -> *mut c_void;
type Create = unsafe extern "C" fn(
    u32,
    u32,
    u32,
    u32,
    u32,
    *const Layer,
    *const Matrix,
    *const f32,
    f32,
) -> *mut c_void;
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32, *mut u32) -> i32;
type Prefill =
    unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, u32, *mut f32, *mut u32) -> i32;
type Counter = unsafe extern "C" fn(*mut c_void) -> u32;
struct PrefillApi {
    run: Prefill,
    tensor_calls: Counter,
    capacity: usize,
}
type Snapshot = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;
type Restored = unsafe extern "C" fn(*mut c_void, u32) -> i32;
pub(crate) struct SavedAttention {
    context: usize,
    destroy: Destroy,
}
impl Drop for SavedAttention {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}
struct GpuAttention {
    context: usize,
    destroy: Destroy,
    _weights: Vec<CudaDeviceQuantMatrix>,
}
impl Drop for GpuAttention {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}
pub(crate) struct GpuFull {
    context: usize,
    destroy: Destroy,
    step: Step,
    snapshot: Snapshot,
    restore: Restore,
    snapshot_destroy: Destroy,
    restored: Restored,
    attention: Vec<Option<GpuAttention>>,
    _head: CudaDeviceQuantMatrix,
    embd: usize,
    vocab: usize,
    capacity: usize,
    kv_stride: usize,
    matrices: u64,
    graphs: bool,
    split_layers: u64,
    prefill: Option<PrefillApi>,
}
impl GpuFull {
    pub fn new(
        weights: &Weights,
        cfg: &Qwen35Config,
        recurrent: &[Option<GpuRecurrent>],
        head_name: &str,
        backend: BackendKind,
    ) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_QWEN_FULL").as_deref() != Ok("1")
            || backend != BackendKind::Cuda
            || cfg.n_expert != 0
            || cfg.max_seq > 8192
        {
            return None;
        }
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_qwen_full_create\0").ok()? };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_full_destroy\0")
                .ok()?
        };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_qwen_full_step\0").ok()? };
        let restored = unsafe {
            *lib.get::<Restored>(b"rbitnet_cuda_qwen_full_restored\0")
                .ok()?
        };
        let ac = unsafe {
            *lib.get::<AttentionCreate>(b"rbitnet_cuda_qwen_full_attention_create\0")
                .ok()?
        };
        let ad = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_full_attention_destroy\0")
                .ok()?
        };
        let snapshot = unsafe {
            *lib.get::<Snapshot>(b"rbitnet_cuda_qwen_full_attention_snapshot\0")
                .ok()?
        };
        let restore = unsafe {
            *lib.get::<Restore>(b"rbitnet_cuda_qwen_full_attention_restore\0")
                .ok()?
        };
        let snapshot_destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_full_attention_snapshot_destroy\0")
                .ok()?
        };
        let graphs = std::env::var("RBITNET_CUDA_QWEN_FULL_GRAPH").as_deref() != Ok("0");
        let mut attention = Vec::with_capacity(cfg.n_layer);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut matrices = 0;
        for il in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(il) {
                layers.push(Layer {
                    kind: 0,
                    context: recurrent.get(il)?.as_ref()?.context_address() as *mut c_void,
                });
                attention.push(None);
                matrices += 8;
                continue;
            }
            let name = |s: &str| format!("blk.{il}.{s}");
            let owned: Vec<_> = [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            .iter()
            .map(|s| weights.device_matrix(&name(s)))
            .collect::<Option<_>>()?;
            let m: Vec<_> = owned
                .iter()
                .map(Matrix::from_device)
                .collect::<Option<_>>()?;
            let qs = cfg.n_head.checked_mul(cfg.head_dim)?;
            let gated = if owned[0].out_rows() == qs {
                0
            } else if owned[0].out_rows() == qs.checked_mul(2)? {
                1
            } else {
                return None;
            };
            let rotary = cfg.rope_dim_pairs.min(cfg.head_dim / 2 * 2);
            let c = Config {
                embd: cfg.n_embd.try_into().ok()?,
                ffn: owned[4].out_rows().try_into().ok()?,
                heads: cfg.n_head.try_into().ok()?,
                kv_heads: cfg.n_head_kv.try_into().ok()?,
                head_dim: cfg.head_dim.try_into().ok()?,
                rotary: rotary.try_into().ok()?,
                capacity: cfg.max_seq.try_into().ok()?,
                gated,
                graphs: u32::from(graphs),
                epsilon: cfg.norm_eps,
                scale: cfg.attn_scale,
            };
            let an = weights.dense(&name("attn_norm.weight")).ok()?;
            let fnorm = weights.dense(&name("post_attention_norm.weight")).ok()?;
            let qn = weights.dense(&name("attn_q_norm.weight")).ok()?;
            let kn = weights.dense(&name("attn_k_norm.weight")).ok()?;
            if an.len() != cfg.n_embd
                || fnorm.len() != cfg.n_embd
                || qn.len() != cfg.head_dim
                || kn.len() != cfg.head_dim
            {
                return None;
            }
            let freq: Vec<f32> = (0..rotary / 2)
                .map(|i| cfg.rope_freq_base.powf(-2.0 * i as f32 / rotary as f32))
                .collect();
            let context = unsafe {
                ac(
                    &c,
                    m.as_ptr(),
                    an.as_ptr(),
                    fnorm.as_ptr(),
                    qn.as_ptr(),
                    kn.as_ptr(),
                    freq.as_ptr(),
                )
            } as usize;
            if context == 0 {
                return None;
            }
            attention.push(Some(GpuAttention {
                context,
                destroy: ad,
                _weights: owned,
            }));
            layers.push(Layer {
                kind: 1,
                context: context as *mut c_void,
            });
            matrices += 7;
        }
        let head = weights.device_matrix(head_name)?;
        let hm = Matrix::from_device(&head)?;
        let norm = weights.dense("output_norm.weight").ok()?;
        if norm.len() != cfg.n_embd {
            return None;
        }
        let context = unsafe {
            create(
                cfg.n_embd.try_into().ok()?,
                cfg.n_vocab.try_into().ok()?,
                cfg.max_seq.try_into().ok()?,
                cfg.n_layer.try_into().ok()?,
                u32::from(graphs),
                layers.as_ptr(),
                &hm,
                norm.as_ptr(),
                cfg.norm_eps,
            )
        } as usize;
        if context == 0 {
            return None;
        }
        type SplitLayers = unsafe extern "C" fn(*mut c_void) -> u32;
        let split_layers = unsafe {
            lib.get::<SplitLayers>(b"rbitnet_cuda_qwen_split_attention_layers\0")
                .map_or(0, |query| query(context as *mut c_void) as u64)
        };
        let prefill = if std::env::var("RBITNET_CUDA_QWEN_PREFILL").as_deref() == Ok("1") {
            type Configure = unsafe extern "C" fn(*mut c_void, u32, u32) -> i32;
            unsafe {
                (|| {
                    let configure = lib
                        .get::<Configure>(b"rbitnet_cuda_qwen_configure_prefill\0")
                        .ok()?;
                    let capacity = lib
                        .get::<Counter>(b"rbitnet_cuda_qwen_prefill_capacity\0")
                        .ok()?;
                    let tensor_calls = *lib
                        .get::<Counter>(b"rbitnet_cuda_qwen_tensor_gemm_calls\0")
                        .ok()?;
                    let run = *lib
                        .get::<Prefill>(b"rbitnet_cuda_qwen_full_prefill\0")
                        .ok()?;
                    let tensor = u32::from(
                        std::env::var("RBITNET_CUDA_PREFILL_TF32X3").as_deref() == Ok("1"),
                    );
                    let status = configure(context as *mut c_void, 1, tensor);
                    if status != 0 {
                        tracing::warn!(
                            status,
                            "Qwen block workspace unavailable; using serial GPU pipeline"
                        );
                        return None;
                    }
                    if std::env::var("RBITNET_QWEN_ORDERED_BLOCK_TEST").as_deref() == Ok("1") {
                        type Ordered = unsafe extern "C" fn(*mut c_void, u32) -> i32;
                        let configure = lib
                            .get::<Ordered>(b"rbitnet_cuda_qwen_configure_ordered_prefill\0")
                            .expect("ordered native ABI required");
                        assert_eq!(
                            configure(context as *mut c_void, 1),
                            0,
                            "ordered block explicitly requested"
                        );
                    }
                    let capacity = capacity(context as *mut c_void) as usize;
                    (capacity > 0 && capacity <= 128).then_some(PrefillApi {
                        run,
                        tensor_calls,
                        capacity,
                    })
                })()
            }
        } else {
            None
        };
        Some(Self {
            context,
            destroy,
            step,
            snapshot,
            restore,
            snapshot_destroy,
            restored,
            attention,
            _head: head,
            embd: cfg.n_embd,
            vocab: cfg.n_vocab,
            capacity: cfg.max_seq,
            kv_stride: cfg.n_head_kv * cfg.head_dim,
            matrices,
            graphs,
            split_layers,
            prefill,
        })
    }
    // Private typed wrapper, every native call owns the runtime exclusively.
    #[cfg(test)]
    pub(crate) fn spec_state(&self) -> Vec<f32> {
        type Count = unsafe extern "C" fn(*mut c_void) -> usize;
        type Read = unsafe extern "C" fn(*mut c_void, *mut f32, usize) -> i32;
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        unsafe {
            let count = lib
                .get::<Count>(b"rbitnet_cuda_qwen_spec_state_elements\0")
                .unwrap();
            let read = lib
                .get::<Read>(b"rbitnet_cuda_qwen_spec_state_read\0")
                .unwrap();
            let mut values = vec![0.0; count(self.context as *mut c_void)];
            assert_eq!(
                read(
                    self.context as *mut c_void,
                    values.as_mut_ptr(),
                    values.len()
                ),
                0
            );
            values
        }
    }
    pub(crate) fn spec_configure(&mut self) -> Result<usize> {
        type Configure = unsafe extern "C" fn(*mut c_void, u32) -> i32;
        type Bytes = unsafe extern "C" fn(*mut c_void) -> u64;
        let q8_tile = match std::env::var("RBITNET_QWEN_SPEC_Q8_TILE") {
            Err(std::env::VarError::NotPresent) => 0,
            Ok(value) if value == "0" => 0,
            Ok(value) if value == "4" => 4,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_QWEN_SPEC_Q8_TILE must be 0 or 4".into(),
                ))
            }
        };
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("speculative DLL unavailable".into()))?;
        unsafe {
            let configure = lib
                .get::<Configure>(b"rbitnet_cuda_qwen_spec_configure\0")
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            let bytes = lib
                .get::<Bytes>(b"rbitnet_cuda_qwen_spec_bytes\0")
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            let status = configure(self.context as *mut c_void, 1);
            if status != 0 {
                return Err(BitNetError::Inference(format!(
                    "Qwen spec configure failed: {status}"
                )));
            }
            if q8_tile != 0 {
                let tile = lib
                    .get::<Configure>(b"rbitnet_cuda_qwen_spec_q8_tile\0")
                    .map_err(|e| {
                        BitNetError::Inference(format!("Qwen Q8 tile ABI unavailable: {e}"))
                    })?;
                let status = tile(self.context as *mut c_void, q8_tile);
                if status != 0 {
                    return Err(BitNetError::Inference(format!(
                        "Qwen Q8 tile configure failed: {status}"
                    )));
                }
            }
            usize::try_from(bytes(self.context as *mut c_void))
                .map_err(|_| BitNetError::Inference("speculative byte count overflow".into()))
        }
    }
    pub(crate) fn spec_save(&mut self) -> Result<u64> {
        type Save = unsafe extern "C" fn(*mut c_void, *mut u64) -> i32;
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("speculative DLL unavailable".into()))?;
        let mut nonce = 0;
        let status = unsafe {
            let call = lib
                .get::<Save>(b"rbitnet_cuda_qwen_spec_save\0")
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            call(self.context as *mut c_void, &mut nonce)
        };
        if status != 0 || nonce == 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen spec save failed: {status}"
            )));
        }
        Ok(nonce)
    }
    pub(crate) fn spec_finish(&mut self, nonce: u64, restore: bool) -> Result<()> {
        type Finish = unsafe extern "C" fn(*mut c_void, u64) -> i32;
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("speculative DLL unavailable".into()))?;
        let symbol: &[u8] = if restore {
            b"rbitnet_cuda_qwen_spec_restore\0"
        } else {
            b"rbitnet_cuda_qwen_spec_discard\0"
        };
        let status = unsafe {
            let call = lib
                .get::<Finish>(symbol)
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            call(self.context as *mut c_void, nonce)
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen spec finish failed: {status}"
            )));
        }
        Ok(())
    }
    pub(crate) fn spec_replay(
        &mut self,
        input: &[f32],
        position: usize,
        count: usize,
    ) -> Result<()> {
        type Verify = unsafe extern "C" fn(
            *mut c_void,
            *const f32,
            usize,
            u32,
            u32,
            u32,
            *mut f32,
            usize,
            *mut u32,
            usize,
        ) -> i32;
        if count == 0
            || count > 9
            || position >= self.capacity
            || count > self.capacity - position
            || input.len() != count * self.embd
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(BitNetError::Inference(
                "Qwen speculative replay shape/finite mismatch".into(),
            ));
        }
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("speculative DLL unavailable".into()))?;
        let status = unsafe {
            let call = lib
                .get::<Verify>(b"rbitnet_cuda_qwen_spec_verify\0")
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            call(
                self.context as *mut c_void,
                input.as_ptr(),
                input.len(),
                position as u32,
                count as u32,
                0,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen spec replay failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer((input.len() * 4 + 4) as u64, 0, 0);
        crate::perf::record_qwen_full_tokens(count);
        crate::perf::record_split_attention(self.split_layers * count as u64);
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok(())
    }
    pub(crate) fn spec_verify(
        &mut self,
        input: &[f32],
        position: usize,
        count: usize,
        greedy: bool,
    ) -> Result<(Vec<f32>, Vec<u32>)> {
        type Verify = unsafe extern "C" fn(
            *mut c_void,
            *const f32,
            usize,
            u32,
            u32,
            u32,
            *mut f32,
            usize,
            *mut u32,
            usize,
        ) -> i32;
        if count == 0
            || count > 9
            || position >= self.capacity
            || count > self.capacity - position
            || input.len() != count * self.embd
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(BitNetError::Inference(
                "Qwen speculative verify shape/finite mismatch".into(),
            ));
        }
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| BitNetError::Inference("speculative DLL unavailable".into()))?;
        let mut logits = if greedy {
            Vec::new()
        } else {
            vec![0.0; count * self.vocab]
        };
        let mut ids = if greedy { vec![0; count] } else { Vec::new() };
        let status = unsafe {
            let call = lib
                .get::<Verify>(b"rbitnet_cuda_qwen_spec_verify\0")
                .map_err(|e| BitNetError::Inference(e.to_string()))?;
            call(
                self.context as *mut c_void,
                input.as_ptr(),
                input.len(),
                position as u32,
                count as u32,
                if greedy { 2 } else { 1 },
                if greedy {
                    std::ptr::null_mut()
                } else {
                    logits.as_mut_ptr()
                },
                logits.len(),
                if greedy {
                    ids.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                ids.len(),
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen spec verify failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            (input.len() * 4 + 4) as u64,
            (logits.len() * 4 + ids.len() * 4) as u64,
            0,
        );
        crate::perf::record_qwen_full_tokens(count);
        crate::perf::record_split_attention(self.split_layers * count as u64);
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok((logits, ids))
    }

    pub fn prefill_capacity(&self) -> usize {
        self.prefill.as_ref().map_or(1, |p| p.capacity)
    }
    pub fn prefill(
        &mut self,
        input: &[f32],
        position: usize,
        count: usize,
        output: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        let api = self
            .prefill
            .as_ref()
            .ok_or_else(|| BitNetError::Inference("Qwen GPU block prefill unavailable".into()))?;
        if count == 0
            || count > api.capacity
            || position >= self.capacity
            || count > self.capacity - position
            || input.len() != count * self.embd
        {
            return Err(BitNetError::Inference(
                "Qwen prefill shape/position mismatch".into(),
            ));
        }
        let mode = if !output {
            0
        } else if greedy {
            2
        } else {
            1
        };
        let mut logits = if mode == 1 {
            vec![0.0; self.vocab]
        } else {
            Vec::new()
        };
        let mut token = 0;
        let status = unsafe {
            (api.run)(
                self.context as *mut c_void,
                input.as_ptr(),
                position as u32,
                count as u32,
                mode,
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen GPU block prefill failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            (input.len() * 4 + 4) as u64,
            if mode == 1 {
                (self.vocab * 4) as u64
            } else if mode == 2 {
                4
            } else {
                0
            },
            u64::from(output) + if count == 1 { self.matrices } else { 0 },
        );
        crate::perf::record_gpu_prefill(count, if count > 1 { self.matrices } else { 0 });
        crate::perf::record_qwen_full_tokens(count);
        crate::perf::record_split_attention(self.split_layers * count as u64);
        crate::perf::record_gpu_tensor_gemm(unsafe {
            (api.tensor_calls)(self.context as *mut c_void)
        } as u64);
        for _ in 0..count {
            for _ in self.attention.iter().flatten() {
                crate::perf::record_gpu_attention();
            }
        }
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok((logits, (mode == 2).then_some(token)))
    }
    pub fn prefix_bytes(&self, length: usize) -> usize {
        self.attention
            .iter()
            .flatten()
            .count()
            .saturating_mul(length)
            .saturating_mul(self.kv_stride)
            .saturating_mul(8)
    }
    pub fn snapshots(&self) -> Option<Vec<Option<SavedAttention>>> {
        self.attention
            .iter()
            .map(|a| match a {
                None => Some(None),
                Some(a) => {
                    let context = unsafe { (self.snapshot)(a.context as *mut c_void) } as usize;
                    (context != 0).then_some(Some(SavedAttention {
                        context,
                        destroy: self.snapshot_destroy,
                    }))
                }
            })
            .collect()
    }
    pub fn restore_attention(
        &mut self,
        saved: &[Option<SavedAttention>],
        length: usize,
    ) -> Result<()> {
        if saved.len() != self.attention.len() || length > self.capacity {
            return Err(BitNetError::Inference(
                "Qwen attention snapshot shape mismatch".into(),
            ));
        }
        for (a, s) in self.attention.iter().zip(saved) {
            match (a, s) {
                (Some(a), Some(s)) => {
                    let status = unsafe {
                        (self.restore)(
                            a.context as *mut c_void,
                            s.context as *const c_void,
                            length as u32,
                        )
                    };
                    if status != 0 {
                        return Err(BitNetError::Inference(format!(
                            "Qwen attention restore failed: {status}"
                        )));
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(BitNetError::Inference(
                        "Qwen attention snapshot layer mismatch".into(),
                    ))
                }
            }
        }
        let status = unsafe { (self.restored)(self.context as *mut c_void, length as u32) };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen pipeline restore failed: {status}"
            )));
        }
        Ok(())
    }
    pub fn run(
        &mut self,
        input: &[f32],
        position: usize,
        output: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        if input.len() != self.embd || position >= self.capacity {
            return Err(BitNetError::Inference(
                "Qwen pipeline input shape/position mismatch".into(),
            ));
        }
        let mode = if !output {
            0
        } else if greedy {
            2
        } else {
            1
        };
        let mut logits = if mode == 1 {
            vec![0.0; self.vocab]
        } else {
            Vec::new()
        };
        let mut token = 0;
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                input.as_ptr(),
                position as u32,
                mode,
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen full pipeline failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            (self.embd * 4 + 4) as u64,
            if mode == 1 {
                (self.vocab * 4) as u64
            } else if mode == 2 {
                4
            } else {
                0
            },
            self.matrices + u64::from(output),
        );
        for _ in self.attention.iter().flatten() {
            crate::perf::record_gpu_attention();
        }
        crate::perf::record_qwen_full_token();
        crate::perf::record_split_attention(self.split_layers);
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok((logits, (mode == 2).then_some(token)))
    }
}
impl Drop for GpuFull {
    fn drop(&mut self) {
        // Graphs contain borrowed layer/weight addresses. Destroy them before fields.
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type AttentionStep = unsafe extern "C" fn(*mut c_void, *const f32, u32, *mut f32) -> i32;
    fn norm(x: &[f64], w: &[f32]) -> Vec<f64> {
        let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-5)
            .sqrt()
            .recip();
        x.iter().zip(w).map(|(&x, &w)| x * w as f64 * inv).collect()
    }
    fn projection(w: &[f32], cols: usize, x: &[f64]) -> Vec<f64> {
        w.chunks_exact(cols)
            .map(|r| r.iter().zip(x).map(|(&w, &x)| w as f64 * x).sum())
            .collect()
    }
    fn rotate(x: &mut [f64], rotary: usize, pos: usize, freq: &[f32]) {
        for i in 0..rotary / 2 {
            let angle = pos as f64 * freq[i] as f64;
            let (s, c) = angle.sin_cos();
            let (a, b) = (x[i], x[i + rotary / 2]);
            x[i] = a * c - b * s;
            x[i + rotary / 2] = a * s + b * c;
        }
    }
    struct Oracle {
        weights: Vec<Vec<f32>>,
        cols: Vec<usize>,
        an: Vec<f32>,
        fnorm: Vec<f32>,
        qn: Vec<f32>,
        kn: Vec<f32>,
        freq: Vec<f32>,
        heads: usize,
        kh: usize,
        dim: usize,
        rotary: usize,
        gated: bool,
        scale: f64,
        keys: Vec<Vec<f64>>,
        values: Vec<Vec<f64>>,
    }
    impl Oracle {
        fn layer(&mut self, input: &[f32], position: usize) -> Vec<f64> {
            if position == 0 {
                self.keys.clear();
                self.values.clear();
            }
            self.keys.truncate(position);
            self.values.truncate(position);
            let mut x: Vec<f64> = input.iter().map(|&v| v as f64).collect();
            let h = norm(&x, &self.an);
            let qraw = projection(&self.weights[0], self.cols[0], &h);
            let kraw = projection(&self.weights[1], self.cols[1], &h);
            let v = projection(&self.weights[2], self.cols[2], &h);
            let mut q = Vec::new();
            let mut k = Vec::new();
            for head in 0..self.heads {
                let start = head * self.dim * (1 + usize::from(self.gated));
                let mut head = norm(&qraw[start..start + self.dim], &self.qn);
                rotate(&mut head, self.rotary, position, &self.freq);
                q.extend(head);
            }
            for head in kraw.chunks_exact(self.dim) {
                let mut head = norm(head, &self.kn);
                rotate(&mut head, self.rotary, position, &self.freq);
                k.extend(head);
            }
            self.keys.push(k);
            self.values.push(v);
            let mut attn = vec![0.0; self.heads * self.dim];
            for head in 0..self.heads {
                let kh = head / (self.heads / self.kh);
                let query = &q[head * self.dim..(head + 1) * self.dim];
                let mut scores: Vec<f64> = self
                    .keys
                    .iter()
                    .map(|k| {
                        query
                            .iter()
                            .zip(&k[kh * self.dim..(kh + 1) * self.dim])
                            .map(|(&q, &k)| q * k)
                            .sum::<f64>()
                            * self.scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for (p, s) in scores.iter().enumerate() {
                    for i in 0..self.dim {
                        attn[head * self.dim + i] += s / sum * self.values[p][kh * self.dim + i];
                    }
                }
                if self.gated {
                    for i in 0..self.dim {
                        attn[head * self.dim + i] /=
                            1.0 + (-qraw[head * 2 * self.dim + self.dim + i]).exp();
                    }
                }
            }
            let a = projection(&self.weights[3], self.cols[3], &attn);
            for (x, a) in x.iter_mut().zip(a) {
                *x += a;
            }
            let h = norm(&x, &self.fnorm);
            let g = projection(&self.weights[4], self.cols[4], &h);
            let u = projection(&self.weights[5], self.cols[5], &h);
            let hidden: Vec<f64> = g
                .iter()
                .zip(u)
                .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let f = projection(&self.weights[6], self.cols[6], &hidden);
            for (x, f) in x.iter_mut().zip(f) {
                *x += f;
            }
            x
        }
    }
    #[test]
    fn opt_in_full_attention_matches_f64_for_eight_formats_gating_rope_gqa_and_restore() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let rt = crate::backend::CudaRuntime::try_load().expect("CUDA required");
        let lib = crate::ggml::load_cuda_quant_library().expect("CUDA DLL required");
        let create = unsafe {
            *lib.get::<AttentionCreate>(b"rbitnet_cuda_qwen_full_attention_create\0")
                .unwrap()
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_full_attention_destroy\0")
                .unwrap()
        };
        let step = unsafe {
            *lib.get::<AttentionStep>(b"rbitnet_cuda_qwen_full_attention_step\0")
                .unwrap()
        };
        let snapshot = unsafe {
            *lib.get::<Snapshot>(b"rbitnet_cuda_qwen_full_attention_snapshot\0")
                .unwrap()
        };
        let restore = unsafe {
            *lib.get::<Restore>(b"rbitnet_cuda_qwen_full_attention_restore\0")
                .unwrap()
        };
        let sd = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_full_attention_snapshot_destroy\0")
                .unwrap()
        };
        type Block = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32) -> i32;
        let block = unsafe {
            *lib.get::<Block>(b"rbitnet_cuda_qwen_attention_prefill_check\0")
                .unwrap()
        };
        for graphs in [0, 1] {
            for (case, &ty) in [0, 2, 6, 8, 12, 13, 14, 39].iter().enumerate() {
                let embd = 256;
                let ffn = 256;
                let heads = 4;
                let kh = 2;
                let dim = if case % 2 == 0 { 64 } else { 128 };
                let gated = case % 3 != 0;
                let rotary = if case % 3 == 0 { 0 } else { dim / 2 };
                let cols = vec![embd, embd, embd, heads * dim, embd, embd, ffn];
                let rows = [
                    heads * dim * (1 + usize::from(gated)),
                    kh * dim,
                    kh * dim,
                    embd,
                    ffn,
                    ffn,
                    embd,
                ];
                let mut owned = Vec::new();
                let mut dense = Vec::new();
                let mut m = Vec::new();
                for p in 0..7 {
                    let rb = crate::ggml::ggml_row_size(ty, cols[p] as u64).unwrap();
                    let mut payload: Vec<u8> = (0..rb * rows[p])
                        .map(|i| (i * 37 + p * 53 + 19) as u8)
                        .collect();
                    if ty == 0 {
                        for (i, v) in payload.chunks_exact_mut(4).enumerate() {
                            v.copy_from_slice(
                                &((i as f32 * 0.031 + p as f32).cos() * 0.03).to_le_bytes(),
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
                                    // Q6_K has signed per-group scales up to 127;
                                    // keep the synthetic layer well conditioned so
                                    // the F64 oracle measures the implementation,
                                    // rather than amplified FP32 rounding in the FFN.
                                    &half::f16::from_f32(if ty == 14 { 0.00003 } else { 0.00007 })
                                        .to_bits()
                                        .to_le_bytes(),
                                );
                                if ty == 12 || ty == 13 {
                                    b[2..4].copy_from_slice(
                                        &half::f16::from_f32(0.00003).to_bits().to_le_bytes(),
                                    );
                                }
                            }
                        }
                    }
                    dense.push(
                        crate::ggml::tensor_to_f32(&payload, ty, &[cols[p] as u64, rows[p] as u64])
                            .unwrap(),
                    );
                    let d = CudaDeviceQuantMatrix::from_payload(
                        Some(&rt),
                        ty,
                        payload,
                        rows[p],
                        cols[p],
                    )
                    .unwrap();
                    m.push(Matrix::from_device(&d).unwrap());
                    owned.push(d);
                }
                let w = |n| {
                    (0..n)
                        .map(|i| 1.0 + (i as f32 * 0.13).sin() * 0.07)
                        .collect::<Vec<_>>()
                };
                let freq: Vec<f32> = (0..rotary / 2)
                    .map(|i| 10000.0f32.powf(-2.0 * i as f32 / rotary as f32))
                    .collect();
                let mut oracle = Oracle {
                    weights: dense,
                    cols,
                    an: w(embd),
                    fnorm: w(embd),
                    qn: w(dim),
                    kn: w(dim),
                    freq,
                    heads,
                    kh,
                    dim,
                    rotary,
                    gated,
                    scale: 0.17,
                    keys: Vec::new(),
                    values: Vec::new(),
                };
                let cfg = Config {
                    embd: embd as u32,
                    ffn: ffn as u32,
                    heads: heads as u32,
                    kv_heads: kh as u32,
                    head_dim: dim as u32,
                    rotary: rotary as u32,
                    capacity: 512,
                    gated: u32::from(gated),
                    graphs,
                    epsilon: 1e-5,
                    scale: oracle.scale as f32,
                };
                let context = unsafe {
                    create(
                        &cfg,
                        m.as_ptr(),
                        oracle.an.as_ptr(),
                        oracle.fnorm.as_ptr(),
                        oracle.qn.as_ptr(),
                        oracle.kn.as_ptr(),
                        oracle.freq.as_ptr(),
                    )
                } as usize;
                assert_ne!(context, 0);
                let a = GpuAttention {
                    context,
                    destroy,
                    _weights: owned,
                };
                let mut output = vec![0.0; embd];
                let mut saved = None;
                // A different tail, a truncated immutable checkpoint and a fresh reset
                // exercise graph arguments and stale KV exclusion together.
                for pos in (0..13).chain(3..9).chain(0..5) {
                    if pos == 3 && saved.is_some() {
                        assert_eq!(
                            unsafe {
                                restore(
                                    a.context as *mut c_void,
                                    saved.as_ref().map(|s: &SavedAttention| s.context).unwrap()
                                        as *const c_void,
                                    3,
                                )
                            },
                            0
                        );
                    }
                    let input: Vec<f32> = (0..embd)
                        .map(|i| (i as f32 * 0.7 + pos as f32 * 0.29).sin())
                        .collect();
                    assert_eq!(
                        unsafe {
                            step(
                                a.context as *mut c_void,
                                input.as_ptr(),
                                pos as u32,
                                output.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    let expected = oracle.layer(&input, pos);
                    for (i, (&got, &e)) in output.iter().zip(&expected).enumerate() {
                        assert!(
                            (got as f64 - e).abs() < 5e-5 * (1.0 + e.abs()),
                            "type={ty} graphs={graphs} pos={pos} element={i}: {got} vs {e}"
                        );
                    }
                    if pos == 4 && saved.is_none() {
                        let context = unsafe { snapshot(a.context as *mut c_void) } as usize;
                        assert_ne!(context, 0);
                        saved = Some(SavedAttention {
                            context,
                            destroy: sd,
                        });
                    }
                }
                for (pos, count) in [(0, 7), (7, 16), (23, 33), (56, 64), (120, 128), (0, 17)] {
                    let input: Vec<_> = (0..embd * count)
                        .map(|i| (i as f32 * 0.037 + pos as f32 * 0.21).sin() * 1.7)
                        .collect();
                    let mut output = vec![0.0; input.len()];
                    assert_eq!(
                        unsafe {
                            block(
                                a.context as *mut c_void,
                                input.as_ptr(),
                                pos as u32,
                                count as u32,
                                output.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    for (t, (input, output)) in input
                        .chunks_exact(embd)
                        .zip(output.chunks_exact(embd))
                        .enumerate()
                    {
                        let expected = oracle.layer(input, pos + t);
                        for (i, (&got, &e)) in output.iter().zip(&expected).enumerate() {
                            assert!((got as f64-e).abs()<5e-5*(1.0+e.abs()),"type={ty} graphs={graphs} block={count} pos={} element={i}: {got} vs {e}",pos+t);
                        }
                    }
                }
            }
        }
    }
}
