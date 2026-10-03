//! GPT-OSS whole-token graph or segmented expert admission; dropped before borrowed MoE contexts.
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
struct NativeConfig {
    embd: u32,
    vocab: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary: u32,
    capacity: u32,
    window: u32,
    experts: u32,
    used: u32,
    graphs: u32,
    split: u32,
    ordered: u32,
    epsilon: f32,
    rope_magnitude: f32,
    weight_scale: f32,
}
#[repr(C)]
struct Layer {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    out: Matrix,
    router: Matrix,
    attn_norm: *const f32,
    ffn_norm: *const f32,
    q_bias: *const f32,
    k_bias: *const f32,
    v_bias: *const f32,
    out_bias: *const f32,
    router_bias: *const f32,
    selection_bias: *const f32,
    sinks: *const f32,
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
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32, *mut u32) -> i32;

type Begin = unsafe extern "C" fn(*mut c_void, *const f32, u32) -> i32;
type Prepare = unsafe extern "C" fn(*mut c_void, u32, *mut u32, *mut f32) -> i32;
type FfnInput = unsafe extern "C" fn(*mut c_void, *mut f32) -> i32;
type Finish = unsafe extern "C" fn(*mut c_void, u32, *const *const c_void, *const f32) -> i32;
type End = unsafe extern "C" fn(*mut c_void, u32, *mut f32, *mut u32) -> i32;
type Snapshot = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;
struct Checkpoint {
    context: usize,
    destroy: Destroy,
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

struct Segmented {
    begin: Begin,
    prepare: Prepare,
    input: FfnInput,
    finish: Finish,
    end: End,
}
struct PrefixApi {
    snapshot: Snapshot,
    destroy: Destroy,
    restore: Restore,
}
pub(super) struct GpuFull {
    context: usize,
    destroy: Destroy,
    step: Step,
    segmented: Option<Segmented>,
    prefix_api: Option<PrefixApi>,
    prefix: PrefixStore<Checkpoint>,
    used: usize,
    _weights: Vec<CudaDeviceQuantMatrix>,
    pub extra_weights_bytes: usize,
    pub graphs: bool,
    pub split: bool,
    embd: usize,
    vocab: usize,
    layers: usize,
    capacity: usize,
    kv_bytes_per_token: usize,
}
impl GpuFull {
    #[cfg(test)]
    pub(super) fn trace(
        &mut self,
        input: &[f32],
        pos: usize,
        experts: usize,
        used: usize,
    ) -> Vec<f32> {
        type Trace = unsafe extern "C" fn(*mut c_void, *const f32, u32, *mut f32) -> i32;
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        let run = unsafe {
            *lib.get::<Trace>(b"rbitnet_cuda_gpt_full_layers_check\0")
                .unwrap()
        };
        let mut out = vec![0.0; self.layers * (self.embd + experts + used)];
        assert_eq!(
            unsafe {
                run(
                    self.context as *mut c_void,
                    input.as_ptr(),
                    pos as u32,
                    out.as_mut_ptr(),
                )
            },
            0
        );
        out
    }
    pub(super) fn new(
        weights: &Weights,
        cfg: &Config,
        moe: &[Option<GpuMoe>],
        backend: BackendKind,
    ) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_GPT_FULL").as_deref() != Ok("1")
            || !matches!(backend, BackendKind::Cuda | BackendKind::Hybrid)
            || cfg.family != Family::GptOss
            || cfg.dense_layers != 0
            || cfg.groups != 1
            || cfg.head != cfg.value
            || cfg.max_seq > 8192
            || matches!(
                std::env::var("RBITNET_CUDA_ATTENTION").as_deref(),
                Ok("0" | "false" | "no")
            )
        {
            return None;
        }
        let lib = crate::ggml::load_cuda_quant_library()?;
        let router_lanes = crate::ggml::f32_accumulator_lanes()?;
        let segmented = weights.expert_cache.is_some()
            || moe
                .iter()
                .any(|m| m.as_ref().and_then(GpuMoe::fixed_context_address).is_none())
            || std::env::var("RBITNET_CUDA_GPT_SEGMENTED").as_deref() == Ok("1");
        let create = unsafe {
            *lib.get::<Create>(if segmented {
                b"rbitnet_cuda_gpt_segmented_create\0"
            } else {
                b"rbitnet_cuda_gpt_full_create\0"
            })
            .ok()?
        };
        let segment_api = if segmented {
            Some(Segmented {
                begin: unsafe {
                    *lib.get::<Begin>(b"rbitnet_cuda_gpt_segmented_begin\0")
                        .ok()?
                },
                prepare: unsafe {
                    *lib.get::<Prepare>(b"rbitnet_cuda_gpt_segmented_prepare\0")
                        .ok()?
                },
                input: unsafe {
                    *lib.get::<FfnInput>(b"rbitnet_cuda_gpt_segmented_ffn_input\0")
                        .ok()?
                },
                finish: unsafe {
                    *lib.get::<Finish>(b"rbitnet_cuda_gpt_segmented_finish\0")
                        .ok()?
                },
                end: unsafe { *lib.get::<End>(b"rbitnet_cuda_gpt_segmented_end\0").ok()? },
            })
        } else {
            None
        };
        // Legacy fixed libraries retain their token path; prefixes require the
        // complete optional snapshot ABI before any checkpoint is allocated.
        let prefix_api = (|| {
            Some(PrefixApi {
                snapshot: unsafe { *lib.get::<Snapshot>(b"rbitnet_cuda_gpt_snapshot\0").ok()? },
                destroy: unsafe {
                    *lib.get::<Destroy>(b"rbitnet_cuda_gpt_snapshot_destroy\0")
                        .ok()?
                },
                restore: unsafe { *lib.get::<Restore>(b"rbitnet_cuda_gpt_restore\0").ok()? },
            })
        })();
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_gpt_full_destroy\0")
                .ok()?
        };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_gpt_full_step\0").ok()? };
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
            if weights
                .archive
                .tensor_by_name(&name("ffn_gate_shexp.weight"))
                .is_some()
            {
                return None;
            }
            let mut matrices = Vec::new();
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate_inp.weight",
            ] {
                let n = name(suffix);
                let m = if let Some(m) = weights.device_matrix(&n) {
                    m
                } else if suffix == "ffn_gate_inp.weight" {
                    // The shared placement planner skips <128-row routers.
                    // Copy this small matrix only within its remaining weight budget.
                    let t = weights.tensor(&n).ok()?;
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
                matrices.push(Matrix::from_device(&m)?);
                owned.push(m);
            }
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
                q: matrices[0],
                k: matrices[1],
                v: matrices[2],
                out: matrices[3],
                router: matrices[4],
                attn_norm: dense("attn_norm.weight", cfg.embd)?,
                ffn_norm: dense("post_attention_norm.weight", cfg.embd)?,
                q_bias: dense("attn_q.bias", cfg.heads * cfg.head)?,
                k_bias: dense("attn_k.bias", cfg.kv_heads * cfg.head)?,
                v_bias: dense("attn_v.bias", cfg.kv_heads * cfg.head)?,
                out_bias: dense("attn_output.bias", cfg.embd)?,
                router_bias: dense("ffn_gate_inp.bias", cfg.experts)?,
                selection_bias,
                sinks: dense("attn_sinks.weight", cfg.heads)?,
                moe: if segmented {
                    moe.get(il)?.as_ref().map_or(0, GpuMoe::context_address)
                } else {
                    moe.get(il)?.as_ref()?.fixed_context_address()?
                } as *mut c_void,
            });
        }
        let output = weights.device_matrix("output.weight")?;
        let head = Matrix::from_device(&output)?;
        owned.push(output);
        let norm = weights.dense("output_norm.weight").ok()?;
        if norm.len() != cfg.embd {
            return None;
        }
        let graphs = std::env::var("RBITNET_CUDA_GPT_FULL_GRAPH").as_deref() != Ok("0");
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
        let frequency: Vec<f32> = (0..cfg.rotary / 2)
            .map(|i| {
                let f = cfg.theta.powf(-2.0 * i as f32 / cfg.rotary as f32);
                if cfg.yarn_factor > 1.0 {
                    let ramp = 1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                    f / cfg.yarn_factor * (1.0 - ramp) + f * ramp
                } else {
                    f
                }
            })
            .collect();
        let c = NativeConfig {
            embd: cfg.embd.try_into().ok()?,
            vocab: cfg.vocab.try_into().ok()?,
            layers: cfg.layers.try_into().ok()?,
            heads: cfg.heads.try_into().ok()?,
            kv_heads: cfg.kv_heads.try_into().ok()?,
            head_dim: cfg.head.try_into().ok()?,
            rotary: cfg.rotary.try_into().ok()?,
            capacity: cfg.max_seq.try_into().ok()?,
            window: cfg.window.try_into().ok()?,
            experts: cfg.experts.try_into().ok()?,
            used: cfg.used.try_into().ok()?,
            graphs: u32::from(graphs),
            split: u32::from(split),
            ordered: router_lanes.try_into().ok()?,
            epsilon: cfg.eps,
            rope_magnitude: magnitude,
            weight_scale: cfg.weight_scale,
        };
        let context = unsafe {
            create(
                &c,
                layers.as_ptr(),
                &head,
                norm.as_ptr(),
                frequency.as_ptr(),
            )
        } as usize;
        if context == 0 {
            return None;
        }
        Some(Self {
            context,
            destroy,
            step,
            segmented: segment_api,
            prefix_api,
            prefix: PrefixStore::from_env(),
            used: cfg.used,
            _weights: owned,
            extra_weights_bytes,
            graphs,
            split,
            embd: cfg.embd,
            vocab: cfg.vocab,
            layers: cfg.layers,
            capacity: cfg.max_seq,
            kv_bytes_per_token: cfg.layers * cfg.kv_heads * cfg.head * 2 * 4,
        })
    }
    pub(super) fn run(
        &mut self,
        input: &[f32],
        pos: usize,
        output: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        if self.segmented.is_some() || input.len() != self.embd || pos >= self.capacity {
            return Err(BitNetError::Inference(
                "GPT resident input/context mismatch".into(),
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
                pos as u32,
                mode,
                logits.as_mut_ptr(),
                &mut token,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "GPT resident token failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer(
            (self.embd * 4 + 4) as u64,
            if mode == 2 {
                4
            } else {
                (logits.len() * 4) as u64
            },
            self.layers as u64 * 8 + u64::from(output),
        );
        crate::perf::record_gpt_full_token();
        crate::perf::record_native_moe(true, self.layers as u64, 0);
        crate::perf::record_kv_write(self.kv_bytes_per_token);
        for _ in 0..self.layers {
            crate::perf::record_gpu_attention();
        }
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        if self.split {
            crate::perf::record_split_attention(self.layers as u64);
        }
        Ok((logits, (mode == 2).then_some(token)))
    }
    pub(super) fn is_segmented(&self) -> bool {
        self.segmented.is_some()
    }
    pub(super) fn supports_prefix(&self) -> bool {
        self.prefix_api.is_some()
    }
    fn check(status: i32, operation: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(BitNetError::Inference(format!(
                "resident GPT {operation} failed (status {status})"
            )))
        }
    }
    pub(super) fn begin(&mut self, embedding: &[f32], position: usize) -> Result<()> {
        if embedding.len() != self.embd || position >= self.capacity {
            return Err(BitNetError::Inference(
                "resident GPT embedding/context mismatch".into(),
            ));
        }
        Self::check(
            unsafe {
                (self
                    .segmented
                    .as_ref()
                    .ok_or_else(|| BitNetError::Inference("GPT segmented unavailable".into()))?
                    .begin)(
                    self.context as *mut c_void,
                    embedding.as_ptr(),
                    position as u32,
                )
            },
            "begin",
        )?;
        crate::perf::record_gpu_transfer((embedding.len() * 4 + 4) as u64, 0, 0);
        Ok(())
    }
    pub(super) fn prepare(&mut self, layer: usize) -> Result<(Vec<usize>, Vec<f32>)> {
        if layer >= self.layers {
            return Err(BitNetError::Inference("resident GPT layer index".into()));
        }
        let mut ids = vec![0u32; self.used];
        let mut probabilities = vec![0.0; self.used];
        Self::check(
            unsafe {
                (self
                    .segmented
                    .as_ref()
                    .ok_or_else(|| BitNetError::Inference("GPT segmented unavailable".into()))?
                    .prepare)(
                    self.context as *mut c_void,
                    layer as u32,
                    ids.as_mut_ptr(),
                    probabilities.as_mut_ptr(),
                )
            },
            "attention/router",
        )?;
        crate::perf::record_gpu_transfer(0, (self.used * 8) as u64, 5);
        crate::perf::record_kv_write(self.kv_bytes_per_token / self.layers);
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
            unsafe {
                (self
                    .segmented
                    .as_ref()
                    .ok_or_else(|| BitNetError::Inference("GPT segmented unavailable".into()))?
                    .input)(self.context as *mut c_void, input.as_mut_ptr())
            },
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
                "resident GPT expert shape mismatch".into(),
            ));
        }
        Self::check(
            unsafe {
                (self
                    .segmented
                    .as_ref()
                    .ok_or_else(|| BitNetError::Inference("GPT segmented unavailable".into()))?
                    .finish)(
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
            if cpu_routed.is_none() { 3 } else { 0 },
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
            unsafe {
                (self
                    .segmented
                    .as_ref()
                    .ok_or_else(|| BitNetError::Inference("GPT segmented unavailable".into()))?
                    .end)(
                    self.context as *mut c_void, mode, out.as_mut_ptr(), &mut id
                )
            },
            "output",
        )?;
        if mode == 2 && id as usize >= self.vocab {
            return Err(BitNetError::Inference(
                "resident GPT argmax outside vocabulary".into(),
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
        crate::perf::record_gpt_full_token();
        Ok((out, (mode == 2).then_some(id)))
    }
    pub(super) fn restore_prefix(&mut self, tokens: &[u32]) -> Result<usize> {
        if self.prefix_api.is_none() || !crate::native::prefix::enabled() {
            return Ok(0);
        }
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        if let Some((saved, length)) =
            self.prefix
                .lookup(reusable, true, crate::native::prefix::minimum_tokens())
        {
            Self::check(
                unsafe {
                    (self.prefix_api.as_ref().unwrap().restore)(
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
        if self.prefix_api.is_none()
            || !crate::native::prefix::enabled()
            || tokens.len() < crate::native::prefix::minimum_tokens()
            || self.prefix.contains(tokens)
        {
            return Ok(());
        }
        let bytes = tokens
            .len()
            .checked_mul(self.kv_bytes_per_token)
            .ok_or_else(|| BitNetError::Inference("GPT prefix bytes overflow".into()))?;
        if !self.prefix.reserve(tokens, bytes) {
            return Ok(());
        }
        let context = unsafe {
            (self.prefix_api.as_ref().unwrap().snapshot)(
                self.context as *mut c_void,
                tokens.len() as u32,
            )
        } as usize;
        // Optional snapshots yield to current states/experts under the shared cap.
        if context != 0 {
            self.prefix.insert(
                tokens.to_vec(),
                Checkpoint {
                    context,
                    destroy: self.prefix_api.as_ref().unwrap().destroy,
                },
                bytes,
            );
        }
        Ok(())
    }
}
impl Drop for GpuFull {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Router =
        unsafe extern "C" fn(*const f32, *const f32, u32, u32, f32, *mut u32, *mut f32) -> i32;
    type Hidden = unsafe extern "C" fn(*mut c_void, *const f32, u32, *mut f32) -> i32;
    #[repr(C)]
    struct MoeConfig {
        embd: u32,
        ffn: u32,
        experts: u32,
        used: u32,
        oai: u32,
    }
    type MoeCreate = unsafe extern "C" fn(
        *const MoeConfig,
        *const Matrix,
        *const Matrix,
        *const Matrix,
        *const f32,
        *const f32,
        *const f32,
    ) -> *mut c_void;
    struct Handle {
        ptr: *mut c_void,
        destroy: Destroy,
    }
    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { (self.destroy)(self.ptr) }
        }
    }
    fn enabled() -> bool {
        std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() == Ok("1")
    }
    #[test]
    fn router_matches_total_cmp_ties_nan_signed_zero_and_selected_softmax() {
        if !enabled() {
            return;
        }
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        let run = unsafe {
            *lib.get::<Router>(b"rbitnet_cuda_gpt_router_check\0")
                .unwrap()
        };
        let cases = [
            vec![3.0, 3.0, 1.0, 2.0, -2.0],
            vec![-0.0, 0.0, -0.0, 0.0, -1.0],
            vec![
                f32::from_bits(0x7fc00003),
                f32::from_bits(0xffc00002),
                4.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            vec![
                f32::from_bits(0x7fc00001),
                f32::from_bits(0x7fc00003),
                -4.0,
                2.0,
                2.0,
            ],
        ];
        for raw in cases {
            for biased in [false, true] {
                for used in [1, 3, 5] {
                    let bias = [-1.0, 2.0, -3.0, 0.0, 2.0];
                    let selection: Vec<_> = raw
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| p + if biased { bias[i] } else { 0.0 })
                        .collect();
                    let mut expected: Vec<_> = (0..raw.len()).collect();
                    expected
                        .sort_by(|&a, &b| selection[b].total_cmp(&selection[a]).then(a.cmp(&b)));
                    expected.truncate(used);
                    let mut probabilities: Vec<_> = expected.iter().map(|&i| raw[i]).collect();
                    super::super::softmax(&mut probabilities, None);
                    for p in &mut probabilities {
                        *p *= 0.7;
                    }
                    let mut got = vec![0; used];
                    let mut weights = vec![0.0; used];
                    assert_eq!(
                        unsafe {
                            run(
                                raw.as_ptr(),
                                if biased {
                                    bias.as_ptr()
                                } else {
                                    std::ptr::null()
                                },
                                raw.len() as u32,
                                used as u32,
                                0.7,
                                got.as_mut_ptr(),
                                weights.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    assert_eq!(got, expected.iter().map(|&i| i as u32).collect::<Vec<_>>());
                    for (a, b) in weights.iter().zip(&probabilities) {
                        assert!(
                            (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-6,
                            "{raw:?} {got:?}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
    fn dot(w: &[f32], cols: usize, x: &[f64], bias: &[f32]) -> Vec<f64> {
        w.chunks_exact(cols)
            .enumerate()
            .map(|(r, w)| {
                w.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum::<f64>()
                    + bias.get(r).copied().unwrap_or(0.0) as f64
            })
            .collect()
    }
    fn rms(x: &[f64], w: &[f32]) -> Vec<f64> {
        let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-5f32 as f64)
            .sqrt()
            .recip();
        x.iter().zip(w).map(|(&x, &w)| x * inv * w as f64).collect()
    }
    fn payload(ty: u32, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
        let rb = crate::ggml::ggml_row_size(ty, cols as u64).unwrap();
        let mut p: Vec<_> = (0..rb * rows)
            .map(|i| (i * 37 + seed * 53 + 19) as u8)
            .collect();
        if ty == 0 {
            for (i, b) in p.chunks_exact_mut(4).enumerate() {
                b.copy_from_slice(&((i as f32 * 0.031 + seed as f32).cos() * 0.015).to_le_bytes());
            }
        } else {
            let size = match ty {
                2 => 18,
                6 => 22,
                8 => 34,
                12 => 144,
                13 => 176,
                14 => 210,
                39 => 17,
                _ => unreachable!(),
            };
            for b in p.chunks_exact_mut(size) {
                if ty == 39 {
                    b[0] = 119;
                } else {
                    let offset = if ty == 14 { 208 } else { 0 };
                    b[offset..offset + 2].copy_from_slice(
                        &half::f16::from_f32(if ty == 14 { 0.00003 } else { 0.00007 })
                            .to_bits()
                            .to_le_bytes(),
                    );
                    if ty == 12 || ty == 13 {
                        b[2..4]
                            .copy_from_slice(&half::f16::from_f32(0.00003).to_bits().to_le_bytes());
                    }
                }
            }
        }
        p
    }
    struct OracleLayer {
        weights: Vec<Vec<f32>>,
        biases: Vec<Vec<f32>>,
        an: Vec<f32>,
        fnorm: Vec<f32>,
        sinks: Vec<f32>,
        selection: Vec<f32>,
        keys: Vec<f64>,
        values: Vec<f64>,
    }
    impl OracleLayer {
        fn forward(
            &mut self,
            x: &mut [f64],
            pos: usize,
            window: usize,
            freq: &[f32],
            magnitude: f32,
        ) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
            const HEADS: usize = 6;
            const KV: usize = 2;
            const DIM: usize = 64;
            const USED: usize = 3;
            const EXP: usize = 5;
            const N: usize = 256;
            let h = rms(x, &self.an);
            let mut q = dot(&self.weights[0], N, &h, &self.biases[0]);
            let mut k = dot(&self.weights[1], N, &h, &self.biases[1]);
            let v = dot(&self.weights[2], N, &h, &self.biases[2]);
            for row in q.chunks_exact_mut(DIM).chain(k.chunks_exact_mut(DIM)) {
                for (i, &f) in freq.iter().enumerate() {
                    let (s, c) = (pos as f64 * f as f64).sin_cos();
                    let a = row[i];
                    let b = row[i + freq.len()];
                    row[i] = (a * c - b * s) * magnitude as f64;
                    row[i + freq.len()] = (a * s + b * c) * magnitude as f64;
                }
            }
            if pos == 0 {
                self.keys.clear();
                self.values.clear();
            }
            self.keys.extend(k);
            self.values.extend(v);
            let mut attended = vec![0.0; HEADS * DIM];
            let first = if window > 0 {
                (pos + 1).saturating_sub(window)
            } else {
                0
            };
            for head in 0..HEADS {
                let kh = head / (HEADS / KV);
                let scores: Vec<_> = (first..=pos)
                    .map(|p| {
                        (0..DIM)
                            .map(|i| q[head * DIM + i] * self.keys[(p * KV + kh) * DIM + i])
                            .sum::<f64>()
                            / (DIM as f64).sqrt()
                    })
                    .collect();
                let max = scores
                    .iter()
                    .copied()
                    .fold(self.sinks[head] as f64, f64::max);
                let probs: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum = (self.sinks[head] as f64 - max).exp() + probs.iter().sum::<f64>();
                for i in 0..DIM {
                    attended[head * DIM + i] = probs
                        .iter()
                        .enumerate()
                        .map(|(p, &s)| s / sum * self.values[((p + first) * KV + kh) * DIM + i])
                        .sum();
                }
            }
            let projected = dot(&self.weights[3], HEADS * DIM, &attended, &self.biases[3]);
            for (x, p) in x.iter_mut().zip(projected) {
                *x += p;
            }
            let h = rms(x, &self.fnorm);
            let raw = dot(&self.weights[4], N, &h, &self.biases[4]);
            let mut ids: Vec<_> = (0..EXP).collect();
            ids.sort_by(|&a, &b| {
                (raw[b] + self.selection[b] as f64)
                    .total_cmp(&(raw[a] + self.selection[a] as f64))
                    .then(a.cmp(&b))
            });
            ids.truncate(USED);
            let max = ids
                .iter()
                .map(|&i| raw[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut probs: Vec<_> = ids.iter().map(|&i| (raw[i] - max).exp()).collect();
            let sum = probs.iter().sum::<f64>();
            for p in &mut probs {
                *p = *p / sum * 0.7f32 as f64;
            }
            for (&e, &p) in ids.iter().zip(&probs) {
                let start = e * N * N;
                let end = start + N * N;
                let gate = dot(
                    &self.weights[5][start..end],
                    N,
                    &h,
                    &self.biases[5][e * N..(e + 1) * N],
                );
                let up = dot(
                    &self.weights[6][start..end],
                    N,
                    &h,
                    &self.biases[6][e * N..(e + 1) * N],
                );
                let hidden: Vec<_> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| {
                        let g = g.min(7.0);
                        g / (1.0 + (-1.702f32 as f64 * g).exp()) * (u.clamp(-7.0, 7.0) + 1.0)
                    })
                    .collect();
                let down = dot(
                    &self.weights[7][start..end],
                    N,
                    &hidden,
                    &self.biases[7][e * N..(e + 1) * N],
                );
                for (x, d) in x.iter_mut().zip(down) {
                    *x += p * d;
                }
            }
            (ids, probs, h)
        }
    }
    #[test]
    fn full_graph_matches_f64_biases_sinks_windows_rope_gqa_reset_and_output_modes() {
        if !enabled() {
            return;
        }
        let rt = CudaRuntime::try_load().unwrap();
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        let create = unsafe {
            *lib.get::<Create>(b"rbitnet_cuda_gpt_full_create\0")
                .unwrap()
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_gpt_full_destroy\0")
                .unwrap()
        };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_gpt_full_step\0").unwrap() };
        let hidden = unsafe {
            *lib.get::<Hidden>(b"rbitnet_cuda_gpt_full_hidden_check\0")
                .unwrap()
        };
        let moe_create = unsafe { *lib.get::<MoeCreate>(b"rbitnet_cuda_moe_create\0").unwrap() };
        let moe_destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").unwrap() };
        const N: usize = 256;
        const EXP: usize = 5;
        let frequency: Vec<_> = (0..16)
            .map(|i| 10000f32.powf(-2.0 * i as f32 / 32.0) / 1.3)
            .collect();
        for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
            let mut owned = Vec::new();
            let mut oracles = Vec::new();
            let mut descriptor = Vec::new();
            let mut moes = Vec::new();
            for il in 0..2 {
                let cols = [N, N, N, 384, N, N, N, N];
                let rows = [384, 128, 128, N, EXP, N * EXP, N * EXP, N * EXP];
                let mut weights = Vec::new();
                let mut matrices = Vec::new();
                let mut biases = Vec::new();
                for p in 0..8 {
                    // Out projection width 384 and the five-row router use F32;
                    // GGUF block formats require aligned quantization rows.
                    let format = if p == 3 || p == 4 { 0 } else { ty };
                    let payload = payload(format, cols[p], rows[p], p + il * 11);
                    weights.push(
                        crate::ggml::tensor_to_f32(
                            &payload,
                            format,
                            &[cols[p] as u64, rows[p] as u64],
                        )
                        .unwrap(),
                    );
                    let m = CudaDeviceQuantMatrix::from_payload(
                        Some(&rt),
                        format,
                        payload,
                        rows[p],
                        cols[p],
                    )
                    .unwrap();
                    matrices.push(Matrix::from_device(&m).unwrap());
                    owned.push(m);
                    // Experts exercise gate/up clamps and biased down projections.
                    biases.push(
                        (0..rows[p])
                            .map(|i| {
                                if p == 4 {
                                    i as f32 * 0.41
                                } else {
                                    (i as f32 * 0.17 + p as f32).sin()
                                        * if p == 5 || p == 6 { 8.0 } else { 0.07 }
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                let norm = |phase: f32| {
                    (0..N)
                        .map(|i| 1.0 + (i as f32 * 0.13 + phase).sin() * 0.04)
                        .collect::<Vec<_>>()
                };
                let oracle = OracleLayer {
                    weights,
                    biases,
                    an: norm(il as f32),
                    fnorm: norm(il as f32 + 2.0),
                    sinks: (0..6).map(|i| i as f32 * 0.9 - 2.0).collect(),
                    selection: vec![0.1, -0.2, 0.3, 0.0, -0.1],
                    keys: Vec::new(),
                    values: Vec::new(),
                };
                let mc = MoeConfig {
                    embd: N as u32,
                    ffn: N as u32,
                    experts: EXP as u32,
                    used: 3,
                    oai: 1,
                };
                let ptr = unsafe {
                    moe_create(
                        &mc,
                        &matrices[5],
                        &matrices[6],
                        &matrices[7],
                        oracle.biases[5].as_ptr(),
                        oracle.biases[6].as_ptr(),
                        oracle.biases[7].as_ptr(),
                    )
                };
                assert!(!ptr.is_null());
                moes.push(Handle {
                    ptr,
                    destroy: moe_destroy,
                });
                descriptor.push(Layer {
                    q: matrices[0],
                    k: matrices[1],
                    v: matrices[2],
                    out: matrices[3],
                    router: matrices[4],
                    attn_norm: oracle.an.as_ptr(),
                    ffn_norm: oracle.fnorm.as_ptr(),
                    q_bias: oracle.biases[0].as_ptr(),
                    k_bias: oracle.biases[1].as_ptr(),
                    v_bias: oracle.biases[2].as_ptr(),
                    out_bias: oracle.biases[3].as_ptr(),
                    router_bias: oracle.biases[4].as_ptr(),
                    selection_bias: oracle.selection.as_ptr(),
                    sinks: oracle.sinks.as_ptr(),
                    moe: ptr,
                });
                oracles.push(oracle);
            }
            let head_payload = payload(ty, N, 257, 29);
            let head_dense =
                crate::ggml::tensor_to_f32(&head_payload, ty, &[N as u64, 257]).unwrap();
            let output =
                CudaDeviceQuantMatrix::from_payload(Some(&rt), ty, head_payload, 257, N).unwrap();
            let head = Matrix::from_device(&output).unwrap();
            let norm = vec![1.0; N];
            for graphs in [0, 1] {
                for split in [0, 1] {
                    let c = NativeConfig {
                        embd: N as u32,
                        vocab: 257,
                        layers: 2,
                        heads: 6,
                        kv_heads: 2,
                        head_dim: 64,
                        rotary: 32,
                        capacity: 545,
                        window: 12,
                        experts: EXP as u32,
                        used: 3,
                        graphs,
                        split,
                        ordered: 16,
                        epsilon: 1e-5,
                        rope_magnitude: 1.13,
                        weight_scale: 0.7,
                    };
                    let ptr = unsafe {
                        create(
                            &c,
                            descriptor.as_ptr(),
                            &head,
                            norm.as_ptr(),
                            frequency.as_ptr(),
                        )
                    };
                    assert!(!ptr.is_null());
                    let context = Handle { ptr, destroy };
                    let positions = if ty == 0 { 270 } else { 35 };
                    for repeat in 0..2 {
                        for pos in 0..positions {
                            let input: Vec<_> = (0..N)
                                .map(|i| (i as f32 * 0.23 + pos as f32 * 0.31).sin() * 0.9)
                                .collect();
                            let mut expected: Vec<_> = input.iter().map(|&v| v as f64).collect();
                            for (il, oracle) in oracles.iter_mut().enumerate() {
                                oracle.forward(
                                    &mut expected,
                                    pos,
                                    if il % 2 == 0 { 12 } else { 0 },
                                    &frequency,
                                    1.13,
                                );
                            }
                            let mut got = vec![0.0; if repeat == 0 { N } else { 257 }];
                            let mut id = 0;
                            let rc = if repeat == 0 {
                                unsafe { hidden(ptr, input.as_ptr(), pos as u32, got.as_mut_ptr()) }
                            } else {
                                unsafe {
                                    step(
                                        ptr,
                                        input.as_ptr(),
                                        pos as u32,
                                        1,
                                        got.as_mut_ptr(),
                                        &mut id,
                                    )
                                }
                            };
                            assert_eq!(rc, 0);
                            if repeat == 1 {
                                expected = dot(&head_dense, N, &rms(&expected, &norm), &[]);
                            }
                            for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                                assert!((a as f64-b).abs()<5e-5*(1.0+b.abs()),"format={ty} graphs={graphs} split={split} repeat={repeat} pos={pos} row={i}: {a} vs {b}");
                            }
                        }
                    }
                    let input = vec![0.37; N];
                    let mut logits = vec![0.0; 257];
                    let mut id = 0;
                    assert_eq!(
                        unsafe { step(ptr, input.as_ptr(), 0, 1, logits.as_mut_ptr(), &mut id) },
                        0
                    );
                    let expected = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    assert_eq!(
                        unsafe { step(ptr, input.as_ptr(), 0, 2, std::ptr::null_mut(), &mut id) },
                        0
                    );
                    assert_eq!(id, expected);
                    // Invalid/out-of-order positions must not advance KV validity.
                    assert_ne!(
                        unsafe { step(ptr, input.as_ptr(), 2, 0, std::ptr::null_mut(), &mut id) },
                        0
                    );
                    assert_eq!(
                        unsafe { step(ptr, input.as_ptr(), 1, 0, std::ptr::null_mut(), &mut id) },
                        0
                    );
                    drop(context);
                }
            }
            drop(moes);
            drop(owned);
            eprintln!("GPT-OSS full F64 oracle passed format {ty}, graphs/eager, ordinary/split attention");
        }
    }
    include!("gpt_segmented_tests.rs");
}
