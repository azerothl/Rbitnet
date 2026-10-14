//! Reference dense Qwen3 transformer runtime.

use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::backend::{CudaDeviceQuantMatrix, CudaRuntime};
use crate::ggml::{
    embedding_row_mmap, ggml_type_supports_cuda_quant, matvec_batch_mmap, matvec_embd_out_mmap,
    matvec_ff_mmap, tensor_to_f32,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::timings::PhaseTimings;

use super::config::Qwen3Config;

#[derive(Clone)]
struct LayerTensors {
    attn_norm: GgufTensorInfo,
    q: GgufTensorInfo,
    q_norm: GgufTensorInfo,
    k: GgufTensorInfo,
    k_norm: GgufTensorInfo,
    v: GgufTensorInfo,
    o: GgufTensorInfo,
    ffn_norm: GgufTensorInfo,
    ffn_gate: GgufTensorInfo,
    ffn_up: GgufTensorInfo,
    ffn_down: GgufTensorInfo,
    q_gpu: Option<CudaDeviceQuantMatrix>,
    k_gpu: Option<CudaDeviceQuantMatrix>,
    v_gpu: Option<CudaDeviceQuantMatrix>,
    o_gpu: Option<CudaDeviceQuantMatrix>,
    ffn_gate_gpu: Option<CudaDeviceQuantMatrix>,
    ffn_up_gpu: Option<CudaDeviceQuantMatrix>,
    ffn_down_gpu: Option<CudaDeviceQuantMatrix>,
}

pub struct Qwen3Runtime {
    cfg: Qwen3Config,
    archive: Arc<GgufArchive>,
    pub(super) tokenizer: Arc<LoadedPromptTokenizer>,
    tok_embd: GgufTensorInfo,
    out_norm: GgufTensorInfo,
    out_head: GgufTensorInfo,
    layers: Vec<LayerTensors>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
}

fn must_tensor(archive: &GgufArchive, name: &str) -> Result<GgufTensorInfo> {
    archive
        .tensor_by_name(name)
        .cloned()
        .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
}

fn tensor_f32_flat(archive: &GgufArchive, t: &GgufTensorInfo) -> Result<Vec<f32>> {
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != w.len() {
        return Err(BitNetError::Inference(format!(
            "rmsnorm shape mismatch: x={} w={}",
            x.len(),
            w.len()
        )));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = 1.0 / (s + eps).sqrt();
    Ok(x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * scale)
        .collect())
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) -> Result<()> {
    if x.len() != w.len() {
        return Err(BitNetError::Inference(format!(
            "head rmsnorm shape mismatch: x={} w={}",
            x.len(),
            w.len()
        )));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = 1.0 / (s + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi *= scale * *wi;
    }
    Ok(())
}

fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// Apply a precomputed `(cos, sin)` pair per rotary channel.
fn rope_apply(slice: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = slice.len() / 2;
    for i in 0..half {
        let (c, s) = (cos[i], sin[i]);
        let x0 = slice[i];
        let x1 = slice[i + half];
        slice[i] = x0 * c - x1 * s;
        slice[i + half] = x0 * s + x1 * c;
    }
}

/// GPT-NeoX style RoPE. `inv_freq` is `rope_inv_freq(head_dim, theta)`.
fn rope_inplace(slice: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = slice.len() / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for i in 0..half {
        let angle = pos as f32 * inv_freq[i];
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[i];
        let x1 = slice[i + half];
        slice[i] = x0 * c - x1 * s;
        slice[i + half] = x0 * s + x1 * c;
    }
}

fn softmax_inplace(scores: &mut [f32]) {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

fn silu_inplace(v: &mut [f32]) {
    for x in v {
        *x = *x / (1.0 + (-*x).exp());
    }
}

fn matvec_out(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    matvec_embd_out_mmap(archive, tensor, x, ne0, ne1)
}

fn matvec_ff(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    x: &[f32],
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    matvec_ff_mmap(archive, tensor, x, n_ff, n_embd)
}

fn attend_one_head(
    k_cache: &[f32],
    v_cache: &[f32],
    q_slice: &[f32],
    pos: usize,
    kv_h: usize,
    head_dim: usize,
    kv_stride: usize,
    scale: f32,
    comb: &mut [f32],
) {
    let mut scores = vec![0.0f32; pos + 1];
    for p in 0..=pos {
        let off = p * kv_stride + kv_h * head_dim;
        let k_slice = &k_cache[off..off + head_dim];
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q_slice[i] * k_slice[i];
        }
        scores[p] = dot * scale;
    }
    softmax_inplace(&mut scores);
    comb.fill(0.0);
    for p in 0..=pos {
        let off = p * kv_stride + kv_h * head_dim;
        crate::ggml::simd::scale_add(comb, &v_cache[off..off + head_dim], 1.0, scores[p]);
    }
}

fn attend_cached(
    k_cache: &[f32],
    v_cache: &[f32],
    q: &[f32],
    pos: usize,
    n_head: usize,
    head_dim: usize,
    n_rep: usize,
    kv_stride: usize,
    scale: f32,
    attn_out: &mut [f32],
) {
    use rayon::prelude::*;
    debug_assert_eq!(attn_out.len(), n_head * head_dim);
    attn_out
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(qh, comb)| {
            let kv_h = qh / n_rep;
            let q_slice = &q[qh * head_dim..(qh + 1) * head_dim];
            if pos < 64 {
                attend_one_head(
                    k_cache, v_cache, q_slice, pos, kv_h, head_dim, kv_stride, scale, comb,
                );
            } else {
                attend_one_head_simd(
                    k_cache, v_cache, q_slice, pos, kv_h, head_dim, kv_stride, scale, comb,
                );
            }
        });
}

fn attend_one_head_simd(
    k_cache: &[f32],
    v_cache: &[f32],
    q_slice: &[f32],
    pos: usize,
    kv_h: usize,
    head_dim: usize,
    kv_stride: usize,
    scale: f32,
    comb: &mut [f32],
) {
    let mut scores = vec![0.0f32; pos + 1];
    for p in 0..=pos {
        let off = p * kv_stride + kv_h * head_dim;
        scores[p] = crate::ggml::simd::dot(q_slice, &k_cache[off..off + head_dim]) * scale;
    }
    softmax_inplace(&mut scores);
    comb.fill(0.0);
    for p in 0..=pos {
        let off = p * kv_stride + kv_h * head_dim;
        crate::ggml::simd::scale_add(comb, &v_cache[off..off + head_dim], 1.0, scores[p]);
    }
}

fn online_mix(acc: &mut [f32], sum: &mut f32, maxv: &mut f32, score: f32, value: &[f32]) {
    if score <= *maxv {
        let weight = (score - *maxv).exp();
        *sum += weight;
        crate::ggml::simd::scale_add(acc, value, 1.0, weight);
        return;
    }
    let alpha = if maxv.is_finite() {
        (*maxv - score).exp()
    } else {
        0.0
    };
    *sum = *sum * alpha + 1.0;
    *maxv = score;
    crate::ggml::simd::scale_add(acc, value, alpha, 1.0);
}

type GqaPrefill = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    *mut f32,
) -> i32;

fn gqa_prefill_fn() -> Option<GqaPrefill> {
    static SLOT: OnceLock<Option<GqaPrefill>> = OnceLock::new();
    *SLOT.get_or_init(|| {
        let lib = crate::ggml::load_cuda_quant_library()?;
        unsafe {
            lib.get::<GqaPrefill>(b"rbitnet_cuda_gqa_prefill\0")
                .ok()
                .map(|symbol| *symbol)
        }
    })
}

#[repr(C)]
struct Qwen3CudaLayer {
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    o: *const c_void,
    gate: *const c_void,
    up: *const c_void,
    down: *const c_void,
    q_bytes: usize,
    k_bytes: usize,
    v_bytes: usize,
    o_bytes: usize,
    gate_bytes: usize,
    up_bytes: usize,
    down_bytes: usize,
    q_type: u32,
    k_type: u32,
    v_type: u32,
    o_type: u32,
    gate_type: u32,
    up_type: u32,
    down_type: u32,
    attn_norm: *const f32,
    q_norm: *const f32,
    k_norm: *const f32,
    ffn_norm: *const f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
}

type Qwen3Chunk = unsafe extern "C" fn(
    *mut f32,
    *mut Qwen3CudaLayer,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
) -> i32;

fn qwen3_chunk_fn() -> Option<Qwen3Chunk> {
    static SLOT: OnceLock<Option<Qwen3Chunk>> = OnceLock::new();
    *SLOT.get_or_init(|| {
        let lib = crate::ggml::load_cuda_quant_library()?;
        unsafe {
            lib.get::<Qwen3Chunk>(b"rbitnet_cuda_qwen3_chunk\0")
                .ok()
                .map(|symbol| *symbol)
        }
    })
}

fn attend_chunk(
    k_cache: &[f32],
    v_cache: &[f32],
    q: &[f32],
    base_pos: usize,
    n_tokens: usize,
    n_head: usize,
    head_dim: usize,
    n_rep: usize,
    kv_stride: usize,
    scale: f32,
    attn: &mut [f32],
) {
    let n_q = n_head * head_dim;
    debug_assert_eq!(attn.len(), n_tokens * n_q);
    let n_kv = n_head / n_rep;
    if kv_stride == n_kv * head_dim {
        if let Some(fun) = gqa_prefill_fn() {
            let end = base_pos.saturating_add(n_tokens);
            if kv_stride > 0
                && end <= k_cache.len() / kv_stride
                && end <= v_cache.len() / kv_stride
            {
                let status = unsafe {
                    fun(
                        k_cache.as_ptr(),
                        v_cache.as_ptr(),
                        q.as_ptr(),
                        base_pos as u32,
                        n_tokens as u32,
                        n_kv as u32,
                        n_head as u32,
                        head_dim as u32,
                        scale,
                        attn.as_mut_ptr(),
                    )
                };
                if status == 0 {
                    return;
                }
            }
        }
    }
    let tile = 32usize;
    use rayon::prelude::*;
    attn.par_chunks_mut(tile * n_q)
        .enumerate()
        .for_each(|(tile_i, attn_tile)| {
        let tlen = attn_tile.len() / n_q;
        let t0 = tile_i * tile;
        let max_pos = base_pos + t0 + tlen - 1;
        let mut acc = vec![0.0f32; tlen * n_q];
        let mut maxs = vec![f32::NEG_INFINITY; tlen * n_head];
        let mut sums = vec![0.0f32; tlen * n_head];
        for p in 0..=max_pos {
            for kv_h in 0..n_kv {
                let off = p * kv_stride + kv_h * head_dim;
                let k_slice = &k_cache[off..off + head_dim];
                let v_slice = &v_cache[off..off + head_dim];
                for local in 0..tlen {
                    let pos = base_pos + t0 + local;
                    if p > pos {
                        continue;
                    }
                    let q_tok = &q[(t0 + local) * n_q..(t0 + local + 1) * n_q];
                    for rep in 0..n_rep {
                        let qh = kv_h * n_rep + rep;
                        let score = crate::ggml::simd::dot(
                            &q_tok[qh * head_dim..(qh + 1) * head_dim],
                            k_slice,
                        ) * scale;
                        let state = local * n_head + qh;
                        online_mix(
                            &mut acc[qh * head_dim + local * n_q..(qh + 1) * head_dim + local * n_q],
                            &mut sums[state],
                            &mut maxs[state],
                            score,
                            v_slice,
                        );
                    }
                }
            }
        }
        for local in 0..tlen {
            for qh in 0..n_head {
                let state = local * n_head + qh;
                let inv = 1.0 / sums[state];
                let src = &acc[local * n_q + qh * head_dim..local * n_q + (qh + 1) * head_dim];
                let dst = &mut attn_tile[local * n_q + qh * head_dim..local * n_q + (qh + 1) * head_dim];
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst = src * inv;
                }
            }
        }
    });
}

fn qwen3_cuda_runtime() -> Option<Arc<CudaRuntime>> {
    let raw = std::env::var("RBITNET_BACKEND").unwrap_or_else(|_| "auto".into());
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" | "rocm" | "vulkan" | "intel" | "level-zero" | "oneapi" | "metal" => None,
        _ => {
            if crate::ggml::cuda_quant_library_available() {
                CudaRuntime::try_load()
            } else {
                None
            }
        }
    }
}

fn resident_matrix(
    rt: &Arc<CudaRuntime>,
    archive: &Arc<GgufArchive>,
    tensor: &GgufTensorInfo,
) -> Option<CudaDeviceQuantMatrix> {
    if tensor.dimensions.len() < 2 || !ggml_type_supports_cuda_quant(tensor.ggml_type) {
        return None;
    }
    let cols = tensor.dimensions[0] as usize;
    let rows = tensor.dimensions[1] as usize;
    CudaDeviceQuantMatrix::from_archive_range(
        Some(rt),
        Arc::clone(archive),
        tensor,
        0,
        rows,
        cols,
        false,
    )
    .ok()
    .filter(|matrix| matrix.is_device_resident())
}

fn project_tokens(
    archive: &GgufArchive,
    tensor: &GgufTensorInfo,
    gpu: &Option<CudaDeviceQuantMatrix>,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    if let Some(matrix) = gpu {
        if matrix.in_cols() == ne0 && matrix.out_rows() == ne1 {
            match matrix.gemm_tokens(xs, n_tokens) {
                Ok(y) => return Ok(y),
                Err(err) => note_gemm_fallback(&err),
            }
        }
    }
    matvec_batch_mmap(archive, tensor, xs, n_tokens, ne0, ne1)
}

fn note_gemm_fallback(err: &BitNetError) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        eprintln!("qwen3 prefill GEMM fell back to CPU: {err}");
    }
}

fn prefill_chunk_tokens(cuda: bool) -> usize {
    let default_chunk = if cuda { 1024 } else { 128 };
    std::env::var("RBITNET_PREFILL_CHUNK_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_chunk)
}

fn resolve_lm_head(archive: &GgufArchive, tok_embd: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for name in ["output.weight", "lm_head.weight"] {
        if let Some(t) = archive.tensor_by_name(name) {
            return Ok(t.clone());
        }
    }
    Ok(tok_embd.clone())
}

impl Qwen3Runtime {
    pub(crate) fn context_capacity(&self) -> usize {
        self.cfg.max_seq
    }

    pub fn load(archive: Arc<GgufArchive>, tokenizer_path: &Path) -> Result<Self> {
        if crate::context_native::enabled() {
            return Err(BitNetError::NotImplemented(
                "context tiers support Native F32 Llama and dense Qwen only",
            ));
        }

        let cfg = Qwen3Config::from_gguf(archive.as_ref())?;
        let tok_embd = must_tensor(archive.as_ref(), "token_embd.weight")?;
        let out_norm = must_tensor(archive.as_ref(), "output_norm.weight")?;
        let out_head = resolve_lm_head(archive.as_ref(), &tok_embd)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("blk.{il}");
            layers.push(LayerTensors {
                attn_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_norm.weight"))?,
                q: must_tensor(archive.as_ref(), &format!("{p}.attn_q.weight"))?,
                q_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_q_norm.weight"))?,
                k: must_tensor(archive.as_ref(), &format!("{p}.attn_k.weight"))?,
                k_norm: must_tensor(archive.as_ref(), &format!("{p}.attn_k_norm.weight"))?,
                v: must_tensor(archive.as_ref(), &format!("{p}.attn_v.weight"))?,
                o: must_tensor(archive.as_ref(), &format!("{p}.attn_output.weight"))?,
                ffn_norm: must_tensor(archive.as_ref(), &format!("{p}.ffn_norm.weight"))?,
                ffn_gate: must_tensor(archive.as_ref(), &format!("{p}.ffn_gate.weight"))?,
                ffn_up: must_tensor(archive.as_ref(), &format!("{p}.ffn_up.weight"))?,
                ffn_down: must_tensor(archive.as_ref(), &format!("{p}.ffn_down.weight"))?,
                q_gpu: None,
                k_gpu: None,
                v_gpu: None,
                o_gpu: None,
                ffn_gate_gpu: None,
                ffn_up_gpu: None,
                ffn_down_gpu: None,
            });
        }
        if let Some(rt) = qwen3_cuda_runtime() {
            for layer in &mut layers {
                layer.q_gpu = resident_matrix(&rt, &archive, &layer.q);
                layer.k_gpu = resident_matrix(&rt, &archive, &layer.k);
                layer.v_gpu = resident_matrix(&rt, &archive, &layer.v);
                layer.o_gpu = resident_matrix(&rt, &archive, &layer.o);
                layer.ffn_gate_gpu = resident_matrix(&rt, &archive, &layer.ffn_gate);
                layer.ffn_up_gpu = resident_matrix(&rt, &archive, &layer.ffn_up);
                layer.ffn_down_gpu = resident_matrix(&rt, &archive, &layer.ffn_down);
            }
        }
        let tokenizer = Arc::new(LoadedPromptTokenizer::from_path_for_gguf(
            tokenizer_path,
            &archive,
        )?);

        let stride = cfg.n_kv * cfg.head_dim;
        cfg.n_layer
            .checked_mul(cfg.max_seq)
            .and_then(|v| v.checked_mul(stride))
            .ok_or_else(|| BitNetError::Inference("qwen3 KV cache size overflow".into()))?;
        let max_seq = cfg.max_seq;
        let n_layer = cfg.n_layer;
        Ok(Self {
            cfg,
            archive,
            tokenizer,
            tok_embd,
            out_norm,
            out_head,
            layers,
            k_cache: vec![vec![0f32; max_seq * stride]; n_layer],
            v_cache: vec![vec![0f32; max_seq * stride]; n_layer],
        })
    }

    pub fn generate_with_timings(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        sampling.validate_structured_output()?;
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        for row in &mut self.k_cache {
            row.fill(0.0);
        }
        for row in &mut self.v_cache {
            row.fill(0.0);
        }

        let t_enc = Instant::now();
        let prompt_ids = self.tokenizer.encode_ids(prompt, true)?;
        let encode_ms = t_enc.elapsed().as_millis() as u64;
        crate::context_capacity::check_request(prompt_ids.len(), max_tokens, self.cfg.max_seq)?;
        if prompt_ids.is_empty() {
            return Ok((
                String::new(),
                PhaseTimings {
                    encode_ms,
                    ..Default::default()
                },
            ));
        }

        let t_pf = Instant::now();
        let mut logits = Vec::new();
        let chunk_sz = prefill_chunk_tokens(self.layers.iter().any(|layer| layer.q_gpu.is_some()));
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            logits = self.prefill_chunk(chunk, chunk_idx * chunk_sz)?;
        }
        let prefill_ms = t_pf.elapsed().as_millis() as u64;

        let eos_id = self.tokenizer.eos_token_id();
        let t_dec = Instant::now();
        let mut generated = Vec::new();
        let mut rng = seeded_rng(sampling.seed);
        let mut pos = prompt_ids.len();
        let mut finish_reason = crate::timings::GenerationFinishReason::Length;
        for _ in 0..max_tokens {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next_id = sample_token(&logits, &sampling, &generated, &mut rng);
            if Some(next_id) == eos_id {
                finish_reason = crate::timings::GenerationFinishReason::Stop;
                break;
            }
            generated.push(next_id);
            logits = self.decode_one(next_id, pos)?;
            pos += 1;
        }
        let decode_ms = t_dec.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode_ids(&generated, true)?;

        Ok((
            text,
            PhaseTimings {
                encode_ms,
                prefill_ms,
                decode_ms,
                prompt_tokens: prompt_ids.len() as u32,
                completion_tokens: generated.len() as u32,
                finish_reason,
            },
        ))
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32], base_pos: usize) -> Result<Vec<f32>> {
        if tokens.len() > 1 {
            return self.forward_chunk(tokens, base_pos);
        }
        let mut logits = Vec::new();
        for (idx, &tid) in tokens.iter().enumerate() {
            logits = self.decode_one(tid, base_pos + idx)?;
        }
        Ok(logits)
    }

    fn qwen3_cuda_layers(
        &mut self,
        archive: &GgufArchive,
    ) -> Option<(Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>, Vec<Qwen3CudaLayer>)> {
        let n = self.cfg.n_embd;
        let n_q = self.cfg.n_head * self.cfg.head_dim;
        let n_kv = self.cfg.n_kv * self.cfg.head_dim;
        let n_ff = self.cfg.n_ff;
        let fits = |matrix: &CudaDeviceQuantMatrix, rows: usize, cols: usize| {
            matrix.is_device_resident() && matrix.out_rows() == rows && matrix.in_cols() == cols
        };
        let mut norms = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (Some(q), Some(k), Some(v), Some(o), Some(gate), Some(up), Some(down)) = (
                layer.q_gpu.as_ref(),
                layer.k_gpu.as_ref(),
                layer.v_gpu.as_ref(),
                layer.o_gpu.as_ref(),
                layer.ffn_gate_gpu.as_ref(),
                layer.ffn_up_gpu.as_ref(),
                layer.ffn_down_gpu.as_ref(),
            ) else {
                return None;
            };
            if !fits(q, n_q, n)
                || !fits(k, n_kv, n)
                || !fits(v, n_kv, n)
                || !fits(o, n, n_q)
                || !fits(gate, n_ff, n)
                || !fits(up, n_ff, n)
                || !fits(down, n, n_ff)
            {
                return None;
            }
            norms.push((
                tensor_f32_flat(archive, &layer.attn_norm).ok()?,
                tensor_f32_flat(archive, &layer.q_norm).ok()?,
                tensor_f32_flat(archive, &layer.k_norm).ok()?,
                tensor_f32_flat(archive, &layer.ffn_norm).ok()?,
            ));
        }
        let mut desc = Vec::with_capacity(self.layers.len());
        for il in 0..self.layers.len() {
            let (q_ptr, q_bytes, q_type, k_ptr, k_bytes, k_type, v_ptr, v_bytes, v_type, o_ptr, o_bytes, o_type, gate_ptr, gate_bytes, gate_type, up_ptr, up_bytes, up_type, down_ptr, down_bytes, down_type) = {
                let layer = &self.layers[il];
                let q = layer.q_gpu.as_ref()?;
                let k = layer.k_gpu.as_ref()?;
                let v = layer.v_gpu.as_ref()?;
                let o = layer.o_gpu.as_ref()?;
                let gate = layer.ffn_gate_gpu.as_ref()?;
                let up = layer.ffn_up_gpu.as_ref()?;
                let down = layer.ffn_down_gpu.as_ref()?;
                (
                    q.device_address()? as *const c_void,
                    q.row_bytes(),
                    q.ggml_type(),
                    k.device_address()? as *const c_void,
                    k.row_bytes(),
                    k.ggml_type(),
                    v.device_address()? as *const c_void,
                    v.row_bytes(),
                    v.ggml_type(),
                    o.device_address()? as *const c_void,
                    o.row_bytes(),
                    o.ggml_type(),
                    gate.device_address()? as *const c_void,
                    gate.row_bytes(),
                    gate.ggml_type(),
                    up.device_address()? as *const c_void,
                    up.row_bytes(),
                    up.ggml_type(),
                    down.device_address()? as *const c_void,
                    down.row_bytes(),
                    down.ggml_type(),
                )
            };
            desc.push(Qwen3CudaLayer {
                q: q_ptr,
                k: k_ptr,
                v: v_ptr,
                o: o_ptr,
                gate: gate_ptr,
                up: up_ptr,
                down: down_ptr,
                q_bytes,
                k_bytes,
                v_bytes,
                o_bytes,
                gate_bytes,
                up_bytes,
                down_bytes,
                q_type,
                k_type,
                v_type,
                o_type,
                gate_type,
                up_type,
                down_type,
                attn_norm: norms[il].0.as_ptr(),
                q_norm: norms[il].1.as_ptr(),
                k_norm: norms[il].2.as_ptr(),
                ffn_norm: norms[il].3.as_ptr(),
                k_cache: self.k_cache[il].as_mut_ptr(),
                v_cache: self.v_cache[il].as_mut_ptr(),
            });
        }
        Some((norms, desc))
    }

    /// Prefill every token in `tokens` through each layer together.
    /// Weight rows are dequantized once per chunk instead of once per token.
    fn forward_chunk(&mut self, tokens: &[u32], base_pos: usize) -> Result<Vec<f32>> {
        let archive = Arc::clone(&self.archive);
        let archive = archive.as_ref();
        let c = &self.cfg;
        let n_layer = c.n_layer;
        let n = c.n_embd;
        let n_head = c.n_head;
        let n_kv_heads = c.n_kv;
        let head_dim = c.head_dim;
        let n_ff = c.n_ff;
        let n_vocab = c.n_vocab;
        let norm_eps = c.norm_eps;
        let rope_theta = c.rope_theta;
        let rope_inv = rope_inv_freq(head_dim, rope_theta);
        let max_seq = c.max_seq;
        let n_tokens = tokens.len();
        let end_pos = base_pos + n_tokens;
        if end_pos > max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let n_q = n_head * head_dim;
        let n_kv = n_kv_heads * head_dim;
        let n_rep = n_head / n_kv_heads;
        let kv_stride = n_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut xs = vec![0.0f32; n_tokens * n];
        for (i, &tid) in tokens.iter().enumerate() {
            let tok = tid as usize;
            if tok >= n_vocab {
                return Err(BitNetError::Inference("token id out of range".into()));
            }
            embedding_row_mmap(
                archive,
                &self.tok_embd,
                tok,
                n,
                n_vocab,
                &mut xs[i * n..(i + 1) * n],
            )?;
        }

        let split = std::env::var("RBITNET_PREFILL_TIMING").ok().as_deref() == Some("1");
        let t_chunk = Instant::now();
        let mut gemm_ns = 0u128;
        let mut attn_ns = 0u128;
        let mut device_ok = false;
        if let Some(run) = qwen3_chunk_fn() {
            if let Some((norms, mut desc)) = self.qwen3_cuda_layers(archive) {
                let status = unsafe {
                    run(
                        xs.as_mut_ptr(),
                        desc.as_mut_ptr(),
                        n_layer as u32,
                        n_tokens as u32,
                        base_pos as u32,
                        n as u32,
                        n_head as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                        n_ff as u32,
                        norm_eps,
                        rope_theta,
                    )
                };
                let _keep_norms = norms;
                if status == 0 {
                    device_ok = true;
                    if split {
                        eprintln!(
                            "qwen3 device chunk tokens={n_tokens} ms={}",
                            t_chunk.elapsed().as_millis()
                        );
                    }
                } else {
                    eprintln!("qwen3 device chunk status={status}");
                }
            }
        }
        if !device_ok {
        let half = head_dim / 2;
        let mut rope_cos = vec![0.0f32; n_tokens * half];
        let mut rope_sin = vec![0.0f32; n_tokens * half];
        for token in 0..n_tokens {
            let pos = (base_pos + token) as f32;
            for i in 0..half {
                let angle = pos * rope_inv[i];
                rope_cos[token * half + i] = angle.cos();
                rope_sin[token * half + i] = angle.sin();
            }
        }

        for il in 0..n_layer {
            let residual = xs.clone();
            let attn_norm_w = tensor_f32_flat(archive, &self.layers[il].attn_norm)?;
            let mut h = vec![0.0f32; n_tokens * n];
            for i in 0..n_tokens {
                let normed = rmsnorm(&xs[i * n..(i + 1) * n], &attn_norm_w, norm_eps)?;
                h[i * n..(i + 1) * n].copy_from_slice(&normed);
            }
            let q_info = self.layers[il].q.clone();
            let k_info = self.layers[il].k.clone();
            let v_info = self.layers[il].v.clone();
            let q_gpu = self.layers[il].q_gpu.clone();
            let k_gpu = self.layers[il].k_gpu.clone();
            let v_gpu = self.layers[il].v_gpu.clone();
            let t_gemm = Instant::now();
            let mut q = project_tokens(archive, &q_info, &q_gpu, &h, n_tokens, n, n_q)?;
            let mut k = project_tokens(archive, &k_info, &k_gpu, &h, n_tokens, n, n_kv)?;
            let v = project_tokens(archive, &v_info, &v_gpu, &h, n_tokens, n, n_kv)?;
            gemm_ns += t_gemm.elapsed().as_nanos();
            let q_norm_w = tensor_f32_flat(archive, &self.layers[il].q_norm)?;
            let k_norm_w = tensor_f32_flat(archive, &self.layers[il].k_norm)?;
            for i in 0..n_tokens {
                let pos = base_pos + i;
                let cos = &rope_cos[i * half..(i + 1) * half];
                let sin = &rope_sin[i * half..(i + 1) * half];
                let q_row = &mut q[i * n_q..(i + 1) * n_q];
                let k_row = &mut k[i * n_kv..(i + 1) * n_kv];
                for hidx in 0..n_head {
                    let s = &mut q_row[hidx * head_dim..(hidx + 1) * head_dim];
                    rmsnorm_inplace(s, &q_norm_w, norm_eps)?;
                    rope_apply(s, cos, sin);
                }
                for hidx in 0..n_kv_heads {
                    let s = &mut k_row[hidx * head_dim..(hidx + 1) * head_dim];
                    rmsnorm_inplace(s, &k_norm_w, norm_eps)?;
                    rope_apply(s, cos, sin);
                }
                let kv_off = pos * kv_stride;
                self.k_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(k_row);
                self.v_cache[il][kv_off..kv_off + kv_stride]
                    .copy_from_slice(&v[i * n_kv..(i + 1) * n_kv]);
            }
            let mut attn = vec![0.0f32; n_tokens * n_q];
            let t_attn = Instant::now();
            attend_chunk(
                &self.k_cache[il],
                &self.v_cache[il],
                &q,
                base_pos,
                n_tokens,
                n_head,
                head_dim,
                n_rep,
                kv_stride,
                scale,
                &mut attn,
            );
            attn_ns += t_attn.elapsed().as_nanos();
            let o_info = self.layers[il].o.clone();
            let o_gpu = self.layers[il].o_gpu.clone();
            let t_o = Instant::now();
            let y = project_tokens(archive, &o_info, &o_gpu, &attn, n_tokens, n_q, n)?;
            gemm_ns += t_o.elapsed().as_nanos();
            for i in 0..n_tokens * n {
                xs[i] = residual[i] + y[i];
            }

            let ffn_residual = xs.clone();
            let ffn_norm_w = tensor_f32_flat(archive, &self.layers[il].ffn_norm)?;
            let mut h2 = vec![0.0f32; n_tokens * n];
            for i in 0..n_tokens {
                let normed = rmsnorm(&xs[i * n..(i + 1) * n], &ffn_norm_w, norm_eps)?;
                h2[i * n..(i + 1) * n].copy_from_slice(&normed);
            }
            let gate_info = self.layers[il].ffn_gate.clone();
            let up_info = self.layers[il].ffn_up.clone();
            let down_info = self.layers[il].ffn_down.clone();
            let gate_gpu = self.layers[il].ffn_gate_gpu.clone();
            let up_gpu = self.layers[il].ffn_up_gpu.clone();
            let down_gpu = self.layers[il].ffn_down_gpu.clone();
            let t_ffn = Instant::now();
            let mut gate = project_tokens(archive, &gate_info, &gate_gpu, &h2, n_tokens, n, n_ff)?;
            for i in 0..n_tokens {
                silu_inplace(&mut gate[i * n_ff..(i + 1) * n_ff]);
            }
            let up = project_tokens(archive, &up_info, &up_gpu, &h2, n_tokens, n, n_ff)?;
            for i in 0..n_tokens * n_ff {
                gate[i] *= up[i];
            }
            let y2 = project_tokens(archive, &down_info, &down_gpu, &gate, n_tokens, n_ff, n)?;
            gemm_ns += t_ffn.elapsed().as_nanos();
            for i in 0..n_tokens * n {
                xs[i] = ffn_residual[i] + y2[i];
            }
        }
        }

        if split && !device_ok {
            eprintln!(
                "qwen3 chunk tokens={n_tokens} total_ms={} gemm_ms={} attn_ms={}",
                t_chunk.elapsed().as_millis(),
                gemm_ns / 1_000_000,
                attn_ns / 1_000_000
            );
        }
        let last = n_tokens - 1;
        let out_norm_w = tensor_f32_flat(archive, &self.out_norm)?;
        let xn = rmsnorm(&xs[last * n..(last + 1) * n], &out_norm_w, norm_eps)?;
        matvec_out(archive, &self.out_head, &xn, n, n_vocab)
    }

    pub fn decode_one(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let archive = Arc::clone(&self.archive);
        self.forward_one(token, pos, archive.as_ref())
    }

    fn forward_one(&mut self, token: u32, pos: usize, archive: &GgufArchive) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let mut x = vec![0f32; cfg.n_embd];
        embedding_row_mmap(
            archive,
            &self.tok_embd,
            tok,
            cfg.n_embd,
            cfg.n_vocab,
            &mut x,
        )?;

        let n_q = cfg.n_head * cfg.head_dim;
        let n_kv = cfg.n_kv * cfg.head_dim;
        let n_rep = cfg.n_head / cfg.n_kv;
        let kv_stride = n_kv;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let rope_inv = rope_inv_freq(cfg.head_dim, cfg.rope_theta);

        for il in 0..cfg.n_layer {
            let layer = &self.layers[il];
            let residual = x.clone();
            let attn_norm_w = tensor_f32_flat(archive, &layer.attn_norm)?;
            let h = rmsnorm(&x, &attn_norm_w, cfg.norm_eps)?;

            let mut q = matvec_out(archive, &layer.q, &h, cfg.n_embd, n_q)?;
            let mut k = matvec_out(archive, &layer.k, &h, cfg.n_embd, n_kv)?;
            let v = matvec_out(archive, &layer.v, &h, cfg.n_embd, n_kv)?;
            let q_norm_w = tensor_f32_flat(archive, &layer.q_norm)?;
            let k_norm_w = tensor_f32_flat(archive, &layer.k_norm)?;
            for hidx in 0..cfg.n_head {
                let s = &mut q[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rmsnorm_inplace(s, &q_norm_w, cfg.norm_eps)?;
                rope_inplace(s, pos, &rope_inv);
            }
            for hidx in 0..cfg.n_kv {
                let s = &mut k[hidx * cfg.head_dim..(hidx + 1) * cfg.head_dim];
                rmsnorm_inplace(s, &k_norm_w, cfg.norm_eps)?;
                rope_inplace(s, pos, &rope_inv);
            }

            let kv_off = pos * kv_stride;
            self.k_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&k);
            self.v_cache[il][kv_off..kv_off + kv_stride].copy_from_slice(&v);

            let mut attn_out = vec![0f32; n_q];
            attend_cached(
                &self.k_cache[il],
                &self.v_cache[il],
                &q,
                pos,
                cfg.n_head,
                cfg.head_dim,
                n_rep,
                kv_stride,
                scale,
                &mut attn_out,
            );

            let y = matvec_out(archive, &layer.o, &attn_out, n_q, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = residual[i] + y[i];
            }

            let ffn_residual = x.clone();
            let ffn_norm_w = tensor_f32_flat(archive, &layer.ffn_norm)?;
            let h2 = rmsnorm(&x, &ffn_norm_w, cfg.norm_eps)?;
            let mut gate = matvec_out(archive, &layer.ffn_gate, &h2, cfg.n_embd, cfg.n_ff)?;
            silu_inplace(&mut gate);
            let up = matvec_out(archive, &layer.ffn_up, &h2, cfg.n_embd, cfg.n_ff)?;
            for i in 0..cfg.n_ff {
                gate[i] *= up[i];
            }
            let y2 = matvec_ff(archive, &layer.ffn_down, &gate, cfg.n_ff, cfg.n_embd)?;
            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + y2[i];
            }
        }

        let out_norm_w = tensor_f32_flat(archive, &self.out_norm)?;
        let xn = rmsnorm(&x, &out_norm_w, cfg.norm_eps)?;
        matvec_out(archive, &self.out_head, &xn, cfg.n_embd, cfg.n_vocab)
    }

    /// After a full prompt, return the **greedy** next token id (temperature 0, no penalties).
    /// Used by golden / doctor checks; not a full chat template.
    pub fn greedy_next_token_id_after_prompt(&mut self, prompt: &str) -> Result<u32> {
        for row in &mut self.k_cache {
            row.fill(0.0);
        }
        for row in &mut self.v_cache {
            row.fill(0.0);
        }
        let prompt_ids = self.tokenizer.encode_ids(prompt, true)?;
        if prompt_ids.is_empty() {
            return Err(BitNetError::Inference(
                "greedy_next_token: empty prompt encoding".into(),
            ));
        }
        let mut logits = Vec::new();
        let chunk_sz = prefill_chunk_tokens(self.layers.iter().any(|layer| layer.q_gpu.is_some()));
        for (chunk_idx, chunk) in prompt_ids.chunks(chunk_sz).enumerate() {
            logits = self.prefill_chunk(chunk, chunk_idx * chunk_sz)?;
        }
        let mut rng = seeded_rng(Some(0));
        Ok(sample_token(
            &logits,
            &SamplingOptions {
                temperature: 0.0,
                top_p: None,
                seed: Some(0),
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                structured_json: false,
            },
            &[],
            &mut rng,
        ))
    }
}

fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn u32_le(w: &mut File, x: u32) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn u64_le(w: &mut File, x: u64) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn s(w: &mut File, v: &str) -> std::io::Result<()> {
        u64_le(w, v.len() as u64)?;
        w.write_all(v.as_bytes())
    }

    fn kv_s(w: &mut File, k: &str, v: &str) -> std::io::Result<()> {
        s(w, k)?;
        u32_le(w, 8)?;
        s(w, v)
    }

    fn kv_u(w: &mut File, k: &str, v: u32) -> std::io::Result<()> {
        s(w, k)?;
        u32_le(w, 4)?;
        u32_le(w, v)
    }

    fn tensor(w: &mut File, name: &str, dims: &[u64]) -> std::io::Result<()> {
        s(w, name)?;
        u32_le(w, dims.len() as u32)?;
        for &d in dims {
            u64_le(w, d)?;
        }
        u32_le(w, 0)?;
        u64_le(w, 0)
    }

    #[test]
    fn blocked_attention_matches_scalar_heads() {
        let head_dim = 16;
        let n_head = 4;
        let n_rep = 2;
        let n_kv = n_head / n_rep;
        let kv_stride = n_kv * head_dim;
        let base_pos = 3;
        let n_tokens = 40;
        let seq = base_pos + n_tokens;
        let n_q = n_head * head_dim;
        let mut k_cache = vec![0.0f32; seq * kv_stride];
        let mut v_cache = vec![0.0f32; seq * kv_stride];
        let mut q = vec![0.0f32; n_tokens * n_q];
        for (i, value) in k_cache.iter_mut().enumerate() {
            *value = ((i * 17) % 23) as f32 * 0.01 - 0.1;
        }
        for (i, value) in v_cache.iter_mut().enumerate() {
            *value = ((i * 13) % 19) as f32 * 0.02;
        }
        for (i, value) in q.iter_mut().enumerate() {
            *value = ((i * 11) % 29) as f32 * 0.015 - 0.05;
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut got = vec![0.0f32; n_tokens * n_q];
        super::attend_chunk(
            &k_cache, &v_cache, &q, base_pos, n_tokens, n_head, head_dim, n_rep, kv_stride,
            scale, &mut got,
        );
        for token in 0..n_tokens {
            let pos = base_pos + token;
            let q_tok = &q[token * n_q..(token + 1) * n_q];
            for qh in 0..n_head {
                let mut reference = vec![0.0f32; head_dim];
                super::attend_one_head(
                    &k_cache,
                    &v_cache,
                    &q_tok[qh * head_dim..(qh + 1) * head_dim],
                    pos,
                    qh / n_rep,
                    head_dim,
                    kv_stride,
                    scale,
                    &mut reference,
                );
                let start = token * n_q + qh * head_dim;
                for (got, want) in got[start..start + head_dim].iter().zip(&reference) {
                    assert!(
                        (got - want).abs() < 2e-3,
                        "token {token} head {qh}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn qwen3_runtime_reports_missing_q_norm_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen3_missing_q_norm.gguf");
        let mut f = File::create(&path).unwrap();
        let tensor_names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        f.write_all(b"GGUF").unwrap();
        u32_le(&mut f, 3).unwrap();
        u64_le(&mut f, tensor_names.len() as u64).unwrap();
        u64_le(&mut f, 10).unwrap();
        kv_s(&mut f, "general.architecture", "qwen3").unwrap();
        kv_u(&mut f, "qwen3.embedding_length", 16).unwrap();
        kv_u(&mut f, "qwen3.vocab_size", 32).unwrap();
        kv_u(&mut f, "qwen3.block_count", 1).unwrap();
        kv_u(&mut f, "qwen3.attention.head_count", 2).unwrap();
        kv_u(&mut f, "qwen3.attention.head_count_kv", 1).unwrap();
        kv_u(&mut f, "qwen3.feed_forward_length", 24).unwrap();
        kv_u(&mut f, "qwen3.context_length", 128).unwrap();
        kv_u(&mut f, "general.alignment", 32).unwrap();
        kv_u(&mut f, "qwen3.rope.dimension_count", 16).unwrap();
        for name in tensor_names {
            let dims: &[u64] = match name {
                "token_embd.weight" => &[16, 32],
                "output_norm.weight" | "blk.0.attn_norm.weight" | "blk.0.ffn_norm.weight" => &[16],
                "blk.0.attn_k_norm.weight" => &[16],
                "blk.0.attn_q.weight" => &[16, 32],
                "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => &[16, 16],
                "blk.0.attn_output.weight" => &[32, 16],
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => &[16, 24],
                "blk.0.ffn_down.weight" => &[24, 16],
                _ => unreachable!(),
            };
            tensor(&mut f, name, dims).unwrap();
        }
        let pos = f.metadata().unwrap().len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad]).unwrap();
        drop(f);

        let archive = Arc::new(GgufArchive::mmap_path(&path).unwrap());
        let err = match Qwen3Runtime::load(archive, dir.path().join("tokenizer.json").as_path()) {
            Ok(_) => panic!("expected missing q norm before tokenizer loading"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("attn_q_norm"));
    }
}
