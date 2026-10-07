//! Resident compressed MLA attention, with host admission of leased experts.
use super::{Config, Family};
use crate::backend::{BackendKind, CudaDeviceQuantMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use crate::native::{moe::GpuMoe, prefix::PrefixStore, weights::Weights};
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
impl Default for Matrix {
    fn default() -> Self {
        Self {
            weights: std::ptr::null(),
            row_bytes: 0,
            ty: 0,
            cols: 0,
            rows: 0,
        }
    }
}
impl Matrix {
    fn device(m: &CudaDeviceQuantMatrix) -> Option<Self> {
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
struct NativeConfig {
    embd: u32,
    vocab: u32,
    layers: u32,
    heads: u32,
    head: u32,
    value: u32,
    rotary: u32,
    rank: u32,
    capacity: u32,
    experts: u32,
    used: u32,
    groups: u32,
    groups_used: u32,
    sigmoid: u32,
    weight_norm: u32,
    ordered: u32,
    graphs: u32,
    split: u32,
    dense: u32,
    epsilon: f32,
    rope_magnitude: f32,
    weight_scale: f32,
}
#[repr(C)]
struct Layer {
    qa: Matrix,
    qb: Matrix,
    kva: Matrix,
    kb: Matrix,
    vb: Matrix,
    out: Matrix,
    router: Matrix,
    shared_gate: Matrix,
    shared_up: Matrix,
    shared_down: Matrix,
    attn_norm: *const f32,
    qa_norm: *const f32,
    kv_norm: *const f32,
    ffn_norm: *const f32,
    selection_bias: *const f32,
    moe: *mut c_void,
}
type Create = unsafe extern "C" fn(
    *const NativeConfig,
    *const Layer,
    *const Matrix,
    *const f32,
    *const f32,
) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Begin = unsafe extern "C" fn(*mut c_void, *const f32, u32) -> i32;
type Prepare = unsafe extern "C" fn(*mut c_void, u32, *mut u32, *mut f32) -> i32;
type FfnInput = unsafe extern "C" fn(*mut c_void, *mut f32) -> i32;
type Finish = unsafe extern "C" fn(*mut c_void, u32, *const *const c_void, *const f32) -> i32;
type End = unsafe extern "C" fn(*mut c_void, u32, *mut f32, *mut u32) -> i32;
type Snapshot = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;
type ConfigureBlock = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type BlockCapacity = unsafe extern "C" fn(*mut c_void) -> u32;
type BlockStep = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    u32,
    u32,
    u32,
    *mut f32,
    *mut u32,
) -> i32;
struct BlockApi {
    prefill: BlockStep,
    capacity: usize,
}
struct Checkpoint {
    context: usize,
    destroy: Destroy,
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

pub(super) struct GpuMla {
    block: Option<BlockApi>,
    context: usize,
    destroy: Destroy,
    begin: Begin,
    prepare: Prepare,
    input: FfnInput,
    finish: Finish,
    end: End,
    snapshot: Snapshot,
    snapshot_destroy: Destroy,
    restore: Restore,
    _weights: Vec<CudaDeviceQuantMatrix>,
    prefix: PrefixStore<Checkpoint>,
    embd: usize,
    vocab: usize,
    layers: usize,
    used: usize,
    dense: usize,
    capacity: usize,
    prepare_calls: Vec<u64>,
    pub graphs: bool,
    pub split: bool,
    kv_bytes_per_token: usize,
    pub extra_weights_bytes: usize,
}
impl GpuMla {
    pub(super) fn new(
        weights: &Weights,
        cfg: &Config,
        moe: &[Option<GpuMoe>],
        kind: BackendKind,
    ) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_MLA_FULL").as_deref() != Ok("1")
            || cfg.family != Family::Mla
            || !matches!(kind, BackendKind::Cuda | BackendKind::Hybrid)
            || cfg.max_seq > 8192
            || cfg.experts >= 128
            || matches!(
                std::env::var("RBITNET_CUDA_ATTENTION").as_deref(),
                Ok("0" | "false" | "no")
            )
        {
            return None;
        }
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_mla_full_create\0").ok()? };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_mla_full_destroy\0")
                .ok()?
        };
        let begin = unsafe { *lib.get::<Begin>(b"rbitnet_cuda_mla_full_begin\0").ok()? };
        let prepare = unsafe {
            *lib.get::<Prepare>(b"rbitnet_cuda_mla_full_prepare\0")
                .ok()?
        };
        let input = unsafe {
            *lib.get::<FfnInput>(b"rbitnet_cuda_mla_full_ffn_input\0")
                .ok()?
        };
        let finish = unsafe { *lib.get::<Finish>(b"rbitnet_cuda_mla_full_finish\0").ok()? };
        let end = unsafe { *lib.get::<End>(b"rbitnet_cuda_mla_full_end\0").ok()? };
        let snapshot = unsafe { *lib.get::<Snapshot>(b"rbitnet_cuda_mla_snapshot\0").ok()? };
        let snapshot_destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_mla_snapshot_destroy\0")
                .ok()?
        };
        let restore = unsafe { *lib.get::<Restore>(b"rbitnet_cuda_mla_restore\0").ok()? };
        let lanes = crate::ggml::f32_accumulator_lanes()?;
        let rt = CudaRuntime::try_load()?;
        let cache_budget = weights
            .expert_cache
            .as_ref()
            .map(|c| c.lock().ok().map(|c| c.budget_bytes()))
            .unwrap_or(Some(0))?;
        let extra_budget = weights
            .residency_budget_bytes
            .saturating_sub(weights.resident_bytes)
            .saturating_sub(cache_budget);
        let mut extra_weights_bytes = 0usize;
        let mut owned = Vec::new();
        let mut layers = Vec::new();
        for il in 0..cfg.layers {
            let name = |s: &str| format!("blk.{il}.{s}");
            // This ABI does not model biased MLA/dense/shared projections.
            // Keep such a model on the existing reference path.
            if [
                "attn_q_a.bias",
                "attn_q_b.bias",
                "attn_kv_a_mqa.bias",
                "attn_output.bias",
                "ffn_gate_inp.bias",
                "ffn_gate.bias",
                "ffn_up.bias",
                "ffn_down.bias",
                "ffn_gate_shexp.bias",
                "ffn_up_shexp.bias",
                "ffn_down_shexp.bias",
            ]
            .iter()
            .any(|s| weights.archive.tensor_by_name(&name(s)).is_some())
            {
                return None;
            }
            let mut matrix = |s: &str| -> Option<Matrix> {
                let m = if let Some(m) = weights.device_matrix(&name(s)) {
                    m
                } else if s == "ffn_gate_inp.weight" {
                    // Small routers stay CPU-backed in the reference graph; only
                    // this pipeline borrows a separate immutable device copy.
                    let t = weights.tensor(&name(s)).ok()?;
                    if t.dimensions.len() != 2 {
                        return None;
                    }
                    let payload = weights.archive.tensor_payload(t).ok()?;
                    extra_weights_bytes = extra_weights_bytes.checked_add(payload.len())?;
                    if extra_weights_bytes > extra_budget {
                        return None;
                    }
                    let m = CudaDeviceQuantMatrix::from_payload(
                        Some(&rt),
                        t.ggml_type,
                        payload.to_vec(),
                        t.dimensions[1] as usize,
                        t.dimensions[0] as usize,
                    )
                    .ok()?;
                    if !m.is_device_resident() {
                        return None;
                    }
                    m
                } else {
                    return None;
                };
                let descriptor = Matrix::device(&m)?;
                owned.push(m);
                Some(descriptor)
            };
            let qa = matrix("attn_q_a.weight")?;
            let qb = matrix("attn_q_b.weight")?;
            let kva = matrix("attn_kv_a_mqa.weight")?;
            let kb = matrix("attn_k_b.weight")?;
            let vb = matrix("attn_v_b.weight")?;
            let out = matrix("attn_output.weight")?;
            let router = if il >= cfg.dense_layers {
                matrix("ffn_gate_inp.weight")?
            } else {
                Matrix::default()
            };
            let suffix = if il < cfg.dense_layers { "" } else { "_shexp" };
            let shared = weights
                .archive
                .tensor_by_name(&name(&format!("ffn_gate{suffix}.weight")))
                .is_some();
            let (shared_gate, shared_up, shared_down) = if shared {
                (
                    matrix(&format!("ffn_gate{suffix}.weight"))?,
                    matrix(&format!("ffn_up{suffix}.weight"))?,
                    matrix(&format!("ffn_down{suffix}.weight"))?,
                )
            } else {
                (Matrix::default(), Matrix::default(), Matrix::default())
            };
            let dense = |s: &str, n: usize| -> Option<*const f32> {
                let v = weights.dense(&name(s)).ok()?;
                (v.len() == n).then_some(v.as_ptr())
            };
            let selection_bias = if weights
                .archive
                .tensor_by_name(&name("exp_probs_b.bias"))
                .is_some()
            {
                dense("exp_probs_b.bias", cfg.experts)?
            } else {
                std::ptr::null()
            };
            layers.push(Layer {
                qa,
                qb,
                kva,
                kb,
                vb,
                out,
                router,
                shared_gate,
                shared_up,
                shared_down,
                attn_norm: dense("attn_norm.weight", cfg.embd)?,
                qa_norm: dense("attn_q_a_norm.weight", qa.rows as usize)?,
                kv_norm: dense("attn_kv_a_norm.weight", cfg.kv_rank)?,
                ffn_norm: dense("ffn_norm.weight", cfg.embd)?,
                selection_bias,
                moe: moe
                    .get(il)?
                    .as_ref()
                    .map(|m| m.context_address() as *mut c_void)
                    .unwrap_or(std::ptr::null_mut()),
            });
        }
        let head = weights.device_matrix("output.weight")?;
        let descriptor = Matrix::device(&head)?;
        owned.push(head);
        let norm = weights.dense("output_norm.weight").ok()?;
        if norm.len() != cfg.embd {
            return None;
        }
        let graphs = std::env::var("RBITNET_CUDA_MLA_FULL_GRAPH").as_deref() != Ok("0");
        let split = std::env::var("RBITNET_CUDA_SPLIT_KV").as_deref() == Ok("1");
        let magnitude = if cfg.yarn_factor > 1.0 {
            1.0 + 0.1 * cfg.yarn_factor.ln()
        } else {
            1.0
        };
        let corr = |rot: f32| {
            cfg.rotary as f32 * (cfg.yarn_orig as f32 / (rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * cfg.theta.ln())
        };
        let low = corr(cfg.yarn_beta_fast).floor().max(0.0);
        let high = corr(cfg.yarn_beta_slow).ceil().min(cfg.rotary as f32 - 1.0);
        let mut phases = Vec::with_capacity(cfg.max_seq * cfg.rotary);
        for pos in 0..cfg.max_seq {
            for i in 0..cfg.rotary / 2 {
                let f = cfg.theta.powf(-2.0 * i as f32 / cfg.rotary as f32);
                let frequency = if cfg.yarn_factor > 1.0 {
                    let ramp = 1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                    f / cfg.yarn_factor * (1.0 - ramp) + f * ramp
                } else {
                    f
                };
                let (s, c) = (pos as f32 * frequency).sin_cos();
                phases.extend([s, c]);
            }
        }
        let c = NativeConfig {
            embd: cfg.embd.try_into().ok()?,
            vocab: cfg.vocab.try_into().ok()?,
            layers: cfg.layers.try_into().ok()?,
            heads: cfg.heads.try_into().ok()?,
            head: cfg.head.try_into().ok()?,
            value: cfg.value.try_into().ok()?,
            rotary: cfg.rotary.try_into().ok()?,
            rank: cfg.kv_rank.try_into().ok()?,
            capacity: cfg.max_seq.try_into().ok()?,
            experts: cfg.experts.try_into().ok()?,
            used: cfg.used.try_into().ok()?,
            groups: cfg.groups.try_into().ok()?,
            groups_used: cfg.groups_used.try_into().ok()?,
            sigmoid: u32::from(cfg.sigmoid),
            weight_norm: u32::from(cfg.weight_norm),
            ordered: lanes.try_into().ok()?,
            graphs: u32::from(graphs),
            split: u32::from(split),
            dense: cfg.dense_layers.try_into().ok()?,
            epsilon: cfg.eps,
            rope_magnitude: magnitude,
            weight_scale: cfg.weight_scale,
        };
        let prepare_calls = layers
            .iter()
            .enumerate()
            .map(|(il, l)| {
                6 + u64::from(il >= cfg.dense_layers)
                    + if l.shared_gate.weights.is_null() {
                        0
                    } else {
                        3
                    }
            })
            .collect();
        let context = unsafe {
            create(
                &c,
                layers.as_ptr(),
                &descriptor,
                norm.as_ptr(),
                phases.as_ptr(),
            )
        } as usize;
        if context == 0 {
            return None;
        }
        let block = if std::env::var("RBITNET_CUDA_MLA_PREFILL").as_deref() == Ok("1") {
            (|| {
                let configure = unsafe {
                    *lib.get::<ConfigureBlock>(b"rbitnet_cuda_mla_configure_prefill\0").ok()?
                };
                let capacity_fn = unsafe {
                    *lib.get::<BlockCapacity>(b"rbitnet_cuda_mla_prefill_capacity\0").ok()?
                };
                let prefill = unsafe {
                    *lib.get::<BlockStep>(b"rbitnet_cuda_mla_full_prefill\0").ok()?
                };
                let count = std::env::var("RBITNET_CUDA_MLA_PREFILL_TOKENS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16)
                    .clamp(1, 32);
                if unsafe { configure(context as *mut c_void, count as u32) } != 0 {
                    return None;
                }
                let actual = unsafe { capacity_fn(context as *mut c_void) } as usize;
                if actual != count {
                    return None;
                }
                Some(BlockApi {
                    prefill,
                    capacity: actual,
                })
            })()
        } else {
            None
        };
        Some(Self {
            block,
            context,
            destroy,
            begin,
            prepare,
            input,
            finish,
            end,
            snapshot,
            snapshot_destroy,
            restore,
            _weights: owned,
            prefix: PrefixStore::from_env(),
            embd: cfg.embd,
            vocab: cfg.vocab,
            layers: cfg.layers,
            used: cfg.used,
            dense: cfg.dense_layers,
            capacity: cfg.max_seq,
            graphs,
            split,
            extra_weights_bytes,
            prepare_calls,
            kv_bytes_per_token: cfg.layers * (cfg.kv_rank + cfg.rotary) * 4,
        })
    }
    fn check(status: i32, operation: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(BitNetError::Inference(format!(
                "resident MLA {operation} failed (status {status})"
            )))
        }
    }
    pub(super) fn prefill_capacity(&self) -> usize {
        self.block.as_ref().map_or(0, |b| b.capacity)
    }
    pub(super) fn prefill(
        &mut self,
        input: &[f32],
        pos: usize,
        count: usize,
        output: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        let api = self
            .block
            .as_ref()
            .ok_or_else(|| BitNetError::Inference("MLA block ABI not configured".into()))?;
        if count == 0
            || count > api.capacity
            || count.checked_mul(self.embd) != Some(input.len())
            || pos.checked_add(count).is_none_or(|end| end > self.capacity)
        {
            return Err(BitNetError::Inference(
                "MLA block input/context mismatch".into(),
            ));
        }
        if crate::cancel::inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
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
            (api.prefill)(
                self.context as *mut c_void,
                input.as_ptr(),
                pos as u32,
                count as u32,
                mode,
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        Self::check(status, "block prefill")?;
        crate::perf::record_gpu_transfer(
            (input.len() * 4 + 4) as u64,
            if mode == 2 {
                4
            } else {
                (logits.len() * 4) as u64
            },
            self.prepare_calls.iter().sum::<u64>() * count as u64,
        );
        crate::perf::record_gpu_prefill(count, self.layers as u64 * 8 + u64::from(output));
        for _ in 0..count {
            crate::perf::record_gpu_mla_full_tokens(1);
            crate::perf::record_kv_write(self.kv_bytes_per_token);
            crate::perf::record_gpu_attention();
        }
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        if self.split {
            crate::perf::record_split_attention((self.layers * count) as u64);
        }
        if crate::cancel::inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        Ok((logits, (mode == 2).then_some(token)))
    }
    pub(super) fn begin(&mut self, embedding: &[f32], position: usize) -> Result<()> {
        if embedding.len() != self.embd || position >= self.capacity {
            return Err(BitNetError::Inference(
                "resident MLA embedding/context mismatch".into(),
            ));
        }
        Self::check(
            unsafe {
                (self.begin)(
                    self.context as *mut c_void,
                    embedding.as_ptr(),
                    position as u32,
                )
            },
            "begin",
        )?;
        crate::perf::record_gpu_transfer((embedding.len() * 4 + 4) as u64, 0, 0);
        crate::perf::record_kv_write(self.kv_bytes_per_token);
        Ok(())
    }
    pub(super) fn prepare(&mut self, layer: usize) -> Result<(Vec<usize>, Vec<f32>)> {
        if layer >= self.layers {
            return Err(BitNetError::Inference("resident MLA layer index".into()));
        }
        let mut ids = vec![0u32; self.used];
        let mut probabilities = vec![0.0; self.used];
        Self::check(
            unsafe {
                (self.prepare)(
                    self.context as *mut c_void,
                    layer as u32,
                    ids.as_mut_ptr(),
                    probabilities.as_mut_ptr(),
                )
            },
            "attention/router",
        )?;
        let routed = layer >= self.dense;
        crate::perf::record_gpu_transfer(
            0,
            if routed { (self.used * 8) as u64 } else { 0 },
            self.prepare_calls[layer],
        );
        crate::perf::record_gpu_attention();
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        if self.split {
            crate::perf::record_split_attention(1);
        }
        Ok((ids.into_iter().map(|i| i as usize).collect(), probabilities))
    }
    pub(super) fn ffn_input(&mut self) -> Result<Vec<f32>> {
        let mut input = vec![0.0; self.embd];
        Self::check(
            unsafe { (self.input)(self.context as *mut c_void, input.as_mut_ptr()) },
            "fallback input",
        )?;
        crate::perf::record_gpu_transfer(0, (input.len() * 4) as u64, 0);
        Ok(input)
    }
    pub(super) fn finish(
        &mut self,
        layer: usize,
        pointers: Option<&[*const c_void]>,
        cpu_routed: Option<&[f32]>,
    ) -> Result<()> {
        if layer >= self.layers
            || pointers.is_some_and(|p| p.len() != 3 * self.used)
            || cpu_routed.is_some_and(|p| p.len() != self.embd)
        {
            return Err(BitNetError::Inference(
                "resident MLA expert shape mismatch".into(),
            ));
        }
        Self::check(
            unsafe {
                (self.finish)(
                    self.context as *mut c_void,
                    layer as u32,
                    pointers.map_or(std::ptr::null(), |p| p.as_ptr()),
                    cpu_routed.map_or(std::ptr::null(), |p| p.as_ptr()),
                )
            },
            "FFN/residual",
        )?;
        crate::perf::record_gpu_transfer(
            (pointers.map_or(0, std::mem::size_of_val)
                + cpu_routed.map_or(0, std::mem::size_of_val)) as u64,
            0,
            if layer >= self.dense && cpu_routed.is_none() {
                3
            } else {
                0
            },
        );
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok(())
    }
    pub(super) fn end(&mut self, logits: bool, greedy: bool) -> Result<(Vec<f32>, Option<u32>)> {
        let mode = if !logits {
            0
        } else if greedy {
            2
        } else {
            1
        };
        let mut out = if mode == 1 {
            vec![0.0; self.vocab]
        } else {
            Vec::new()
        };
        let mut id = 0;
        Self::check(
            unsafe { (self.end)(self.context as *mut c_void, mode, out.as_mut_ptr(), &mut id) },
            "output",
        )?;
        if mode == 2 && id as usize >= self.vocab {
            return Err(BitNetError::Inference(
                "resident MLA argmax outside vocabulary".into(),
            ));
        }
        crate::perf::record_gpu_transfer(
            0,
            if mode == 2 { 4 } else { (out.len() * 4) as u64 },
            u64::from(logits),
        );
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        crate::perf::record_gpu_mla_full_tokens(1);
        Ok((out, (mode == 2).then_some(id)))
    }
    pub(super) fn restore_prefix(&mut self, tokens: &[u32]) -> Result<usize> {
        if !crate::native::prefix::enabled() {
            return Ok(0);
        }
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        if let Some((saved, length)) =
            self.prefix
                .lookup(reusable, true, crate::native::prefix::minimum_tokens())
        {
            Self::check(
                unsafe {
                    (self.restore)(
                        self.context as *mut c_void,
                        saved.context as *const c_void,
                        length as u32,
                    )
                },
                "prefix restore",
            )?;
            crate::perf::record_prefix_hit(length * self.kv_bytes_per_token);
            Ok(length)
        } else {
            crate::perf::record_prefix_cache_miss();
            Ok(0)
        }
    }
    pub(super) fn save_prefix(&mut self, tokens: &[u32]) -> Result<()> {
        if !crate::native::prefix::enabled()
            || tokens.len() < crate::native::prefix::minimum_tokens()
            || self.prefix.contains(tokens)
        {
            return Ok(());
        }
        let bytes = tokens
            .len()
            .checked_mul(self.kv_bytes_per_token)
            .ok_or_else(|| BitNetError::Inference("MLA prefix bytes overflow".into()))?;
        if !self.prefix.reserve(tokens, bytes) {
            return Ok(());
        }
        let context =
            unsafe { (self.snapshot)(self.context as *mut c_void, tokens.len() as u32) } as usize;
        // Optional snapshots yield to current states/experts under the shared cap.
        if context != 0 {
            self.prefix.insert(
                tokens.to_vec(),
                Checkpoint {
                    context,
                    destroy: self.snapshot_destroy,
                },
                bytes,
            );
        }
        Ok(())
    }
}
impl Drop for GpuMla {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

#[cfg(test)]
#[path = "mla_full_tests.rs"]
mod tests;
