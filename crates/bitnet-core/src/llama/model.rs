//! Llama forward: dense `f32` weights or mmap-backed quantized tensors (GEMV without full dequant).

use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::{BackendKind, ComputeBackend, CpuBackend, CudaDeviceMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use crate::ggml::{
    embedding_row_mmap, ggml_type_supported_mmap_matvec, matvec_embd_out_mmap, matvec_ff_mmap,
    tensor_to_f32,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::scratch::ScratchArena;

use super::blas_runtime;
use super::config::LlamaConfig;
use super::ggml_bridge;
use super::kv_storage::{KvCache, KvStorage};

/// How Llama matrices are stored / executed (`RBITNET_LLAMA_WEIGHT_MODE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlamaWeightMode {
    /// Legacy: full `tensor_to_f32` at load (high RAM).
    Dense,
    /// mmap GGUF + row-wise quant GEMV (`mmap_quant`).
    MmapQuant,
    /// Use mmap quant when all weight tensors use supported GGML types; else dense.
    Auto,
}

pub fn llama_weight_mode_from_env() -> LlamaWeightMode {
    match std::env::var("RBITNET_LLAMA_WEIGHT_MODE").as_deref() {
        Ok(s) if s.eq_ignore_ascii_case("dense") => LlamaWeightMode::Dense,
        Ok(s) if s.eq_ignore_ascii_case("mmap_quant") => LlamaWeightMode::MmapQuant,
        Ok(s) if s.eq_ignore_ascii_case("auto") => LlamaWeightMode::Auto,
        Ok(other) => {
            tracing::warn!("unknown RBITNET_LLAMA_WEIGHT_MODE={other}, using auto");
            LlamaWeightMode::Auto
        }
        Err(_) => LlamaWeightMode::Auto,
    }
}

/// Large linear weights: either dense `f32` or a mmap tensor view.
#[derive(Clone)]
pub enum MatrixWeights {
    Dense(Vec<f32>),
    CudaDense {
        host: Vec<f32>,
        device: CudaDeviceMatrix,
        label: String,
    },
    Quant {
        archive: Arc<GgufArchive>,
        tensor: GgufTensorInfo,
    },
}

impl MatrixWeights {
    fn matvec_embd_out(&self, x: &[f32], ne0: usize, ne1: usize) -> Result<Vec<f32>> {
        match self {
            Self::Dense(w) => Ok(matvec_embd_out_dense(w, x, ne0, ne1)),
            Self::CudaDense {
                host,
                device,
                label,
            } => match device.matvec(x) {
                Some(out) => Ok(out),
                None => {
                    tracing::warn!(tensor = label.as_str(), "hybrid matvec fallback to CPU");
                    Ok(matvec_embd_out_dense(host, x, ne0, ne1))
                }
            },
            Self::Quant { archive, tensor } => {
                matvec_embd_out_mmap(archive.as_ref(), tensor, x, ne0, ne1)
            }
        }
    }

    fn matvec_ff(&self, x: &[f32], n_ff: usize, n_embd: usize) -> Result<Vec<f32>> {
        match self {
            Self::Dense(w) => Ok(matvec_ff_embd_dense(w, x, n_ff, n_embd)),
            Self::CudaDense {
                host,
                device,
                label,
            } => match device.matvec(x) {
                Some(out) => Ok(out),
                None => {
                    tracing::warn!(tensor = label.as_str(), "hybrid ffn_down fallback to CPU");
                    Ok(matvec_ff_embd_dense(host, x, n_ff, n_embd))
                }
            },
            Self::Quant { archive, tensor } => {
                matvec_ff_mmap(archive.as_ref(), tensor, x, n_ff, n_embd)
            }
        }
    }

    fn embed_row(&self, tok: usize, n_embd: usize, n_vocab: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::Dense(v) => {
                for j in 0..n_embd {
                    out[j] = v[j + tok * n_embd];
                }
                Ok(())
            }
            Self::CudaDense { host, .. } => {
                for j in 0..n_embd {
                    out[j] = host[j + tok * n_embd];
                }
                Ok(())
            }
            Self::Quant { archive, tensor } => {
                embedding_row_mmap(archive.as_ref(), tensor, tok, n_embd, n_vocab, out)
            }
        }
    }
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: MatrixWeights,
    pub wk: MatrixWeights,
    pub wv: MatrixWeights,
    pub wo: MatrixWeights,
    /// Optional per-head RMSNorm on Q (length `head_dim`), applied before RoPE when present.
    pub attn_q_norm: Option<Vec<f32>>,
    /// Optional per-head RMSNorm on K before RoPE.
    pub attn_k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: MatrixWeights,
    pub ffn_up: MatrixWeights,
    pub ffn_down: MatrixWeights,
}

pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub token_embd: MatrixWeights,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: MatrixWeights,
    /// Optional per-dimension inverse frequencies from GGUF `rope_freqs.weight` (Llama 3+).
    /// Length `head_dim / 2`. Used only when values match the analytic inv-freq from `theta`
    /// (otherwise the tensor may include Yarn/NTK scaling we do not apply yet — fall back to metadata).
    pub rope_inv_freq: Option<Vec<f32>>,
}

fn load_tensor_dense(archive: &GgufArchive, names: &[&str]) -> Result<Vec<f32>> {
    let t = archive
        .tensor_first_of(names)
        .ok_or_else(|| BitNetError::Inference(format!("missing tensor (tried {:?})", names)))?;
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn load_tensor_strings_dense(archive: &GgufArchive, names: &[String]) -> Result<Vec<f32>> {
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    load_tensor_dense(archive, &refs)
}

fn tensor_info_first(archive: &GgufArchive, names: &[&str]) -> Result<GgufTensorInfo> {
    let t = archive
        .tensor_first_of(names)
        .ok_or_else(|| BitNetError::Inference(format!("missing tensor (tried {:?})", names)))?;
    Ok(t.clone())
}

fn tensor_info_strings(archive: &GgufArchive, names: &[String]) -> Result<GgufTensorInfo> {
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    tensor_info_first(archive, &refs)
}

fn matrix_mmap_supported(t: &GgufTensorInfo) -> bool {
    ggml_type_supported_mmap_matvec(t.ggml_type)
}

/// Newer Llama-3 GGUFs may ship `rope_freqs.weight` `[rope_rot_dims/2]`. If it matches the analytic
/// inv-frequencies from `theta`, use it; otherwise it may encode Yarn/NTK scaling — ignore it.
fn try_load_rope_inv_freq(
    archive: &GgufArchive,
    rope_rot_dims: usize,
    theta: f32,
) -> Option<Vec<f32>> {
    let half = rope_rot_dims.checked_div(2)?;
    let t = archive.tensor_first_of(&["rope_freqs.weight"])?;
    if t.dimensions.len() != 1 || t.dimensions[0] as usize != half {
        tracing::warn!(
            got_dims = ?t.dimensions,
            expected_len = half,
            "rope_freqs.weight: unexpected shape; using analytic RoPE from metadata"
        );
        return None;
    }
    let payload = archive.tensor_payload(t).ok()?;
    let v = tensor_to_f32(payload, t.ggml_type, &t.dimensions).ok()?;
    if v.len() != half {
        return None;
    }
    let h = rope_rot_dims as f32;
    let analytical: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * (i as f32) / h))
        .collect();
    let tol = 1e-3_f32;
    let close = v
        .iter()
        .zip(analytical.iter())
        .take(half.min(8))
        .all(|(a, b)| (a - b).abs() <= tol * b.abs().max(1e-6));
    if close {
        tracing::info!(
            len = half,
            "llama: using rope_freqs.weight (matches analytic inv_freq)"
        );
        Some(v)
    } else {
        tracing::warn!(
            "rope_freqs.weight differs from analytic inv_freq (likely scaled RoPE); using metadata theta only"
        );
        None
    }
}

/// Returns `Ok(())` if every Llama weight matrix uses a GGML type we can mmap-GEMV.
pub fn llama_mmap_quant_supported(archive: &GgufArchive) -> Result<()> {
    let cfg = LlamaConfig::from_gguf(archive)?;
    let check = |names: &[&str]| -> Result<()> {
        let t = archive.tensor_first_of(names).ok_or_else(|| {
            BitNetError::Inference(format!("mmap check: missing tensor {:?}", names))
        })?;
        if !matrix_mmap_supported(t) {
            return Err(BitNetError::UnsupportedGgmlType(t.ggml_type));
        }
        Ok(())
    };

    check(&["token_embd.weight", "token_embd"])?;
    if archive
        .tensor_first_of(&["output.weight", "lm_head.weight"])
        .is_some()
    {
        check(&["output.weight", "lm_head.weight"])?;
    }

    for i in 0..cfg.n_layer {
        let p = format!("blk.{i}");
        check(&[&format!("{p}.attn_q.weight")])?;
        check(&[&format!("{p}.attn_k.weight")])?;
        check(&[&format!("{p}.attn_v.weight")])?;
        check(&[
            &format!("{p}.attn_output.weight"),
            &format!("{p}.attn_out.weight"),
        ])?;
        check(&[&format!("{p}.ffn_gate.weight")])?;
        check(&[&format!("{p}.ffn_up.weight")])?;
        check(&[&format!("{p}.ffn_down.weight")])?;
    }
    Ok(())
}

fn llama_mmap_quant_supported_ok(archive: &GgufArchive) -> bool {
    llama_mmap_quant_supported(archive).is_ok()
}

#[derive(Clone, Debug)]
pub struct LlamaOffloadPlan {
    enabled: bool,
    layers: Vec<bool>,
    min_rows: usize,
    output: bool,
    estimated_weight_bytes: usize,
    reason: String,
}

impl LlamaOffloadPlan {
    pub fn disabled(reason: impl Into<String>, n_layer: usize) -> Self {
        Self {
            enabled: false,
            layers: vec![false; n_layer],
            min_rows: usize::MAX,
            output: false,
            estimated_weight_bytes: 0,
            reason: reason.into(),
        }
    }

    pub fn from_env(kind: BackendKind, cfg: &LlamaConfig) -> Self {
        if kind != BackendKind::Hybrid {
            return Self::disabled("backend is not hybrid", cfg.n_layer);
        }
        let min_rows = std::env::var("RBITNET_HYBRID_MIN_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(512);
        let layer_bytes = llama_layer_f32_bytes(cfg);
        let max_bytes = std::env::var("RBITNET_HYBRID_MAX_VRAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(512)
            .saturating_mul(1024 * 1024);
        let policy = std::env::var("RBITNET_HYBRID_POLICY")
            .unwrap_or_else(|_| "layers".into())
            .trim()
            .to_ascii_lowercase();
        let layers = hybrid_layer_policy(&policy, cfg, layer_bytes, max_bytes);
        let layer_count = layers.iter().filter(|&&v| v).count();
        let output = matches!(
            std::env::var("RBITNET_HYBRID_OUTPUT").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let estimated_weight_bytes =
            layer_count
                .saturating_mul(layer_bytes)
                .saturating_add(if output {
                    cfg.n_embd
                        .saturating_mul(cfg.n_vocab)
                        .saturating_mul(std::mem::size_of::<f32>())
                } else {
                    0
                });
        let enabled = layer_count > 0 || output;
        Self {
            enabled,
            layers,
            min_rows,
            output,
            estimated_weight_bytes,
            reason: if enabled {
                format!("hybrid offload policy={policy} selected {layer_count} layers")
            } else {
                format!("hybrid backend selected but policy={policy} selected no layers")
            },
        }
    }

    pub fn layer_enabled(&self, layer: usize) -> bool {
        self.enabled && self.layers.get(layer).copied().unwrap_or(false)
    }

    pub fn summary(&self) -> String {
        format!(
            "enabled={} layers={} output={} estimated_weight_mb={} reason={}",
            self.enabled,
            self.layers.iter().filter(|&&v| v).count(),
            self.output,
            self.estimated_weight_bytes / (1024 * 1024),
            self.reason
        )
    }
}

fn hybrid_layer_policy(
    policy: &str,
    cfg: &LlamaConfig,
    layer_bytes: usize,
    max_bytes: usize,
) -> Vec<bool> {
    if let Ok(spec) = std::env::var("RBITNET_HYBRID_LAYERS") {
        if policy == "layers" || policy == "auto" {
            return parse_layer_spec(&spec, cfg.n_layer);
        }
    }
    let mut selected = vec![false; cfg.n_layer];
    let max_layers = if layer_bytes == 0 {
        0
    } else {
        (max_bytes / layer_bytes).max(1).min(cfg.n_layer)
    };
    match policy {
        // Keep the first N layers for compatibility with the initial hybrid implementation.
        "layers" => {
            for enabled in selected.iter_mut().take(max_layers) {
                *enabled = true;
            }
        }
        // A simple PowerInfer-style proxy until per-tensor activation telemetry is persisted:
        // retain deeper layers first because they dominate decode reuse and are hit every token.
        "hotcold" | "auto" => {
            for idx in (0..cfg.n_layer).rev().take(max_layers) {
                selected[idx] = true;
            }
        }
        _ => {
            for enabled in selected.iter_mut().take(max_layers) {
                *enabled = true;
            }
        }
    }
    selected
}

fn llama_layer_f32_bytes(cfg: &LlamaConfig) -> usize {
    let n_embd = cfg.n_embd;
    let n_kv = cfg.n_kv * cfg.head_dim;
    let n_ff = cfg.n_ff;
    let elems = n_embd
        .saturating_mul(n_embd) // wq
        .saturating_add(n_embd.saturating_mul(n_kv)) // wk
        .saturating_add(n_embd.saturating_mul(n_kv)) // wv
        .saturating_add(n_embd.saturating_mul(n_embd)) // wo
        .saturating_add(n_embd.saturating_mul(n_ff)) // gate
        .saturating_add(n_embd.saturating_mul(n_ff)) // up
        .saturating_add(n_ff.saturating_mul(n_embd)); // down
    elems.saturating_mul(std::mem::size_of::<f32>())
}

fn parse_layer_spec(spec: &str, n_layer: usize) -> Vec<bool> {
    let mut layers = vec![false; n_layer];
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let Some(start) = a.trim().parse::<usize>().ok() else {
                continue;
            };
            let Some(end) = b.trim().parse::<usize>().ok() else {
                continue;
            };
            for idx in start.min(end)..=start.max(end) {
                if let Some(slot) = layers.get_mut(idx) {
                    *slot = true;
                }
            }
        } else if let Ok(idx) = part.parse::<usize>() {
            if let Some(slot) = layers.get_mut(idx) {
                *slot = true;
            }
        }
    }
    layers
}

fn maybe_cuda_dense(
    rt: Option<&Arc<CudaRuntime>>,
    plan: &LlamaOffloadPlan,
    label: String,
    host: Vec<f32>,
    out_rows: usize,
    in_cols: usize,
) -> MatrixWeights {
    let Some(rt) = rt else {
        return MatrixWeights::Dense(host);
    };
    if out_rows < plan.min_rows {
        return MatrixWeights::Dense(host);
    }
    match CudaDeviceMatrix::upload(Arc::clone(rt), &host, out_rows, in_cols) {
        Some(device) => MatrixWeights::CudaDense {
            host,
            device,
            label,
        },
        None => {
            tracing::warn!(
                tensor = label.as_str(),
                "hybrid upload failed; using CPU dense"
            );
            MatrixWeights::Dense(host)
        }
    }
}

/// `y[out] = sum_i W[i + out * n_embd] * x[i]` — GGUF layout `ne[0]=n_embd`, `ne[1]=out`.
fn matvec_embd_out_dense(w: &[f32], x: &[f32], n_embd: usize, n_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_out];
    if blas_runtime::blas_attention_enabled()
        && blas_runtime::blas_ready()
        && blas_runtime::sgemv_row_major_notrans(w, n_out, n_embd, n_embd, 1.0, x, &mut y, 0.0)
            .is_ok()
    {
        return y;
    }
    for o in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n_embd {
            acc += w[i + o * n_embd] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// `ffn_down`: `ne[0]=n_ff`, `ne[1]=n_embd` — `y[out] = sum_i W[i + out * n_ff] * x[i]`.
fn matvec_ff_embd_dense(w: &[f32], x: &[f32], n_ff: usize, n_embd: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_embd];
    if blas_runtime::blas_attention_enabled()
        && blas_runtime::blas_ready()
        && blas_runtime::sgemv_row_major_notrans(w, n_embd, n_ff, n_ff, 1.0, x, &mut y, 0.0)
            .is_ok()
    {
        return y;
    }
    for o in 0..n_embd {
        let mut acc = 0.0f32;
        for i in 0..n_ff {
            acc += w[i + o * n_ff] * x[i];
        }
        y[o] = acc;
    }
    y
}

fn rmsnorm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let scale = 1.0 / (s + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * w[i] * scale;
    }
}

fn softmax_inplace(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for z in s.iter_mut() {
        *z = (*z - m).exp();
        sum += *z;
    }
    if sum > 0.0 {
        for z in s.iter_mut() {
            *z /= sum;
        }
    }
}

/// Llama-family RoPE as in llama.cpp `LLAMA_ROPE_TYPE_NORM` (`LLM_ARCH_LLAMA`, …): rotate **adjacent**
/// dimension pairs `(2i, 2i+1)` with `inv_freq[i] = 1/θ^(2i/head_dim)`.
///
/// This differs from `LLAMA_ROPE_TYPE_NEOX` (pairs offset by `head_dim/2`) used by Qwen2/3, Phi, Gemma, …
/// — see `qwen3/runtime.rs` and `qwen35/attention.rs`.
///
/// `inv_freq_flat`: when `Some`, length ≥ `head_dim/2`; entry `i` replaces the analytic `inv_freq` for band `i`.
fn rope_inplace(slice: &mut [f32], pos: usize, theta: f32, inv_freq_flat: Option<&[f32]>) {
    let h = slice.len();
    assert!(h % 2 == 0);
    let half = h / 2;
    for i in 0..half {
        let inv_freq = inv_freq_flat
            .and_then(|v| v.get(i).copied())
            .unwrap_or_else(|| 1.0 / theta.powf(2.0 * (i as f32) / (h as f32)));
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[2 * i];
        let x1 = slice[2 * i + 1];
        slice[2 * i] = x0 * c - x1 * s;
        slice[2 * i + 1] = x0 * s + x1 * c;
    }
}

fn rope_heads_inplace(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    rope_rot_dims: usize,
    pos: usize,
    theta: f32,
    inv_freq: Option<&[f32]>,
) {
    assert!(rope_rot_dims <= head_dim);
    assert!(rope_rot_dims % 2 == 0);
    for h in 0..n_head {
        let s = &mut x[h * head_dim..(h + 1) * head_dim];
        rope_inplace(&mut s[..rope_rot_dims], pos, theta, inv_freq);
    }
}

fn silu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        let g = gate[i];
        out[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

fn load_optional_head_rmsnorm(
    archive: &GgufArchive,
    tensor_name: &str,
    head_dim: usize,
) -> Option<Vec<f32>> {
    let t = archive.tensor_first_of(&[tensor_name])?;
    if t.dimensions.len() != 1 || t.dimensions[0] as usize != head_dim {
        tracing::warn!(
            tensor = tensor_name,
            dims = ?t.dimensions,
            expected = head_dim,
            "attn q/k norm: unexpected shape; skipping"
        );
        return None;
    }
    let payload = archive.tensor_payload(t).ok()?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions).ok()
}

fn apply_optional_head_rmsnorm_inplace(
    heads: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    w: &[f32],
    eps: f32,
    scratch: &mut ScratchArena,
) {
    for h in 0..n_heads {
        let lo = h * head_dim;
        let hi = lo + head_dim;
        let mut t = scratch.take(head_dim);
        rmsnorm_into(&heads[lo..hi], w, eps, &mut t);
        heads[lo..hi].copy_from_slice(&t);
        scratch.recycle(t);
    }
}

fn mask_sliding_window_scores(scores: &mut [f32], window_start: usize) {
    for p in 0..scores.len().min(window_start) {
        scores[p] = f32::NEG_INFINITY;
    }
}

/// One query head on the CPU / hybrid attention path (scores → softmax → V combination).
///
/// # Safety contract for callers
///
/// `kv` must only be **read** for layer `il` and positions `0..=pos` (no concurrent writers).
fn llama_cpu_attention_one_head(
    kv: &KvStorage,
    il: usize,
    pos: usize,
    qh: usize,
    n_rep: usize,
    head_dim: usize,
    stride: usize,
    q_heads: &[f32],
    scale: f32,
    sw_start: usize,
    use_blas_scores: bool,
) -> (usize, Vec<f32>) {
    let kv_h = qh / n_rep;
    let q_slice = &q_heads[qh * head_dim..(qh + 1) * head_dim];
    let mut scores = vec![0.0f32; pos + 1];
    if use_blas_scores {
        let mut k_mat = vec![0.0f32; (pos + 1) * head_dim];
        kv.fill_k_rows_gpu(il, pos, kv_h, head_dim, stride, &mut k_mat);
        let blas_ok = blas_runtime::sgemv_row_major_notrans(
            &k_mat,
            pos + 1,
            head_dim,
            head_dim,
            scale,
            q_slice,
            &mut scores,
            0.0,
        )
        .is_ok();
        if blas_ok {
            mask_sliding_window_scores(&mut scores, sw_start);
        } else {
            kv.attention_scores_cpu(
                il,
                pos,
                kv_h,
                head_dim,
                stride,
                q_slice,
                scale,
                &mut scores,
            );
            mask_sliding_window_scores(&mut scores, sw_start);
        }
    } else {
        kv.attention_scores_cpu(
            il,
            pos,
            kv_h,
            head_dim,
            stride,
            q_slice,
            scale,
            &mut scores,
        );
        mask_sliding_window_scores(&mut scores, sw_start);
    }
    softmax_inplace(&mut scores);
    let mut comb = vec![0.0f32; head_dim];
    let mut v_values = vec![0.0f32; head_dim];
    for p in 0..=pos {
        kv.fill_v_head_values(il, p, kv_h, head_dim, stride, &mut v_values);
        let sp = scores[p];
        for i in 0..head_dim {
            comb[i] += sp * v_values[i];
        }
    }
    (qh, comb)
}

fn add_residual_inplace(x: &mut [f32], y: &[f32]) {
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

impl LlamaModel {
    /// Load from mmap archive using `RBITNET_LLAMA_WEIGHT_MODE`.
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        Self::from_gguf_arc(Arc::new(archive.clone()))
    }

    /// Preferred entry: keeps a single `Arc` to the mmap-backed archive for quant weights.
    pub fn from_gguf_arc(archive: Arc<GgufArchive>) -> Result<Self> {
        Self::from_gguf_arc_for_backend(archive, BackendKind::Cpu)
    }

    pub fn from_gguf_arc_for_backend(
        archive: Arc<GgufArchive>,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        if backend_kind == BackendKind::Hybrid {
            return Self::from_gguf_hybrid_internal(archive);
        }
        let mode = llama_weight_mode_from_env();
        match mode {
            LlamaWeightMode::Dense => Self::from_gguf_dense_internal(archive),
            LlamaWeightMode::MmapQuant => {
                if llama_mmap_quant_supported(archive.as_ref()).is_err() {
                    return Err(BitNetError::Inference(
                        "mmap_quant: unsupported ggml_type on one or more Llama matrices \
                         (see ggml::ggml_type_supported_mmap_matvec)"
                            .into(),
                    ));
                }
                Self::from_gguf_mmap_internal(archive)
            }
            LlamaWeightMode::Auto => {
                if llama_mmap_quant_supported_ok(archive.as_ref()) {
                    Self::from_gguf_mmap_internal(archive)
                } else {
                    Self::from_gguf_dense_internal(archive)
                }
            }
        }
    }

    fn from_gguf_hybrid_internal(archive: Arc<GgufArchive>) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        let plan = LlamaOffloadPlan::from_env(BackendKind::Hybrid, &cfg);
        let cuda = if plan.enabled {
            CudaRuntime::try_load()
        } else {
            None
        };
        tracing::info!(
            summary = plan.summary(),
            cuda = cuda.is_some(),
            "llama hybrid offload plan"
        );

        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = MatrixWeights::Quant {
            archive: Arc::clone(&archive),
            tensor: tensor_info_first(archive.as_ref(), &["token_embd.weight", "token_embd"])?,
        };

        let output_norm = load_tensor_dense(archive.as_ref(), &["output_norm.weight"])?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = if plan.output {
            let output_host = if archive
                .tensor_first_of(&["output.weight", "lm_head.weight"])
                .is_some()
            {
                load_tensor_dense(archive.as_ref(), &["output.weight", "lm_head.weight"])?
            } else {
                load_tensor_dense(archive.as_ref(), &["token_embd.weight", "token_embd"])?
            };
            maybe_cuda_dense(
                cuda.as_ref(),
                &plan,
                "output.weight".into(),
                output_host,
                n_vocab,
                n_embd,
            )
        } else if archive
            .tensor_first_of(&["output.weight", "lm_head.weight"])
            .is_some()
        {
            MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_first(archive.as_ref(), &["output.weight", "lm_head.weight"])?,
            }
        } else {
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.attn_norm.weight")])?;
            let ffn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_norm.weight")])?;
            let offload = plan.layer_enabled(i) && cuda.is_some();
            let make_embd_out =
                |names: Vec<String>, out_rows: usize, in_cols: usize| -> Result<MatrixWeights> {
                    if offload {
                        let label = names.first().cloned().unwrap_or_default();
                        Ok(maybe_cuda_dense(
                            cuda.as_ref(),
                            &plan,
                            label,
                            load_tensor_strings_dense(archive.as_ref(), &names)?,
                            out_rows,
                            in_cols,
                        ))
                    } else {
                        Ok(MatrixWeights::Quant {
                            archive: Arc::clone(&archive),
                            tensor: tensor_info_strings(archive.as_ref(), &names)?,
                        })
                    }
                };
            let wq = make_embd_out(vec![format!("{p}.attn_q.weight")], n_embd, n_embd)?;
            let wk = make_embd_out(vec![format!("{p}.attn_k.weight")], n_embd_kv, n_embd)?;
            let wv = make_embd_out(vec![format!("{p}.attn_v.weight")], n_embd_kv, n_embd)?;
            let attn_q_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_q_norm.weight"),
                cfg.head_dim,
            );
            let attn_k_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_k_norm.weight"),
                cfg.head_dim,
            );
            let wo = make_embd_out(
                vec![
                    format!("{p}.attn_output.weight"),
                    format!("{p}.attn_out.weight"),
                ],
                n_embd,
                n_embd,
            )?;
            let ffn_gate = make_embd_out(vec![format!("{p}.ffn_gate.weight")], n_ff, n_embd)?;
            let ffn_up = make_embd_out(vec![format!("{p}.ffn_up.weight")], n_ff, n_embd)?;
            let ffn_down = if offload {
                maybe_cuda_dense(
                    cuda.as_ref(),
                    &plan,
                    format!("{p}.ffn_down.weight"),
                    load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_down.weight")])?,
                    n_embd,
                    n_ff,
                )
            } else {
                MatrixWeights::Quant {
                    archive: Arc::clone(&archive),
                    tensor: tensor_info_strings(
                        archive.as_ref(),
                        &[format!("{p}.ffn_down.weight")],
                    )?,
                }
            };

            Self::validate_layer_mmap(
                archive.as_ref(),
                i,
                &attn_norm,
                &wq,
                &wk,
                &wv,
                &wo,
                &ffn_norm,
                &ffn_gate,
                &ffn_up,
                &ffn_down,
                n_embd,
                n_embd_kv,
                n_ff,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                attn_q_norm,
                attn_k_norm,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            rope_inv_freq,
        })
    }

    fn from_gguf_dense_internal(archive: Arc<GgufArchive>) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = MatrixWeights::Dense(load_tensor_dense(
            archive.as_ref(),
            &["token_embd.weight", "token_embd"],
        )?);
        if let MatrixWeights::Dense(ref v) = token_embd {
            if v.len() != n_embd * n_vocab {
                return Err(BitNetError::Inference(
                    "token_embd.weight element count mismatch".into(),
                ));
            }
        }

        let output_norm = load_tensor_dense(archive.as_ref(), &["output_norm.weight"])?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = if archive
            .tensor_first_of(&["output.weight", "lm_head.weight"])
            .is_some()
        {
            MatrixWeights::Dense(load_tensor_dense(
                archive.as_ref(),
                &["output.weight", "lm_head.weight"],
            )?)
        } else {
            // Some Llama-family GGUFs tie the LM head to token embeddings.
            token_embd.clone()
        };
        if let MatrixWeights::Dense(ref v) = output {
            if v.len() != n_embd * n_vocab {
                return Err(BitNetError::Inference(
                    "output.weight shape mismatch".into(),
                ));
            }
        }

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.attn_norm.weight")])?;
            let wq = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_q.weight")],
            )?);
            let wk = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_k.weight")],
            )?);
            let wv = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_v.weight")],
            )?);
            let attn_q_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_q_norm.weight"),
                cfg.head_dim,
            );
            let attn_k_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_k_norm.weight"),
                cfg.head_dim,
            );
            let wo = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[
                    format!("{p}.attn_output.weight"),
                    format!("{p}.attn_out.weight"),
                ],
            )?);
            let ffn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_norm.weight")])?;
            let ffn_gate = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_gate.weight")],
            )?);
            let ffn_up = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_up.weight")],
            )?);
            let ffn_down = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_down.weight")],
            )?);

            Self::validate_layer_dense(
                i, &attn_norm, &wq, &wk, &wv, &wo, &ffn_norm, &ffn_gate, &ffn_up, &ffn_down,
                n_embd, n_embd_kv, n_ff,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                attn_q_norm,
                attn_k_norm,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            rope_inv_freq,
        })
    }

    fn validate_layer_dense(
        i: usize,
        attn_norm: &[f32],
        wq: &MatrixWeights,
        wk: &MatrixWeights,
        wv: &MatrixWeights,
        wo: &MatrixWeights,
        ffn_norm: &[f32],
        ffn_gate: &MatrixWeights,
        ffn_up: &MatrixWeights,
        ffn_down: &MatrixWeights,
        n_embd: usize,
        n_embd_kv: usize,
        n_ff: usize,
    ) -> Result<()> {
        let len = |m: &MatrixWeights| -> Result<usize> {
            match m {
                MatrixWeights::Dense(v) => Ok(v.len()),
                MatrixWeights::CudaDense { host, .. } => Ok(host.len()),
                MatrixWeights::Quant { .. } => Err(BitNetError::Inference(
                    "validate_layer_dense: expected dense matrix".into(),
                )),
            }
        };
        if attn_norm.len() != n_embd
            || len(wq)? != n_embd * n_embd
            || len(wk)? != n_embd * n_embd_kv
            || len(wv)? != n_embd * n_embd_kv
            || len(wo)? != n_embd * n_embd
            || ffn_norm.len() != n_embd
            || len(ffn_gate)? != n_embd * n_ff
            || len(ffn_up)? != n_embd * n_ff
            || len(ffn_down)? != n_ff * n_embd
        {
            return Err(BitNetError::Inference(format!(
                "layer {i} weight shape mismatch"
            )));
        }
        Ok(())
    }

    fn from_gguf_mmap_internal(archive: Arc<GgufArchive>) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = MatrixWeights::Quant {
            archive: Arc::clone(&archive),
            tensor: tensor_info_first(archive.as_ref(), &["token_embd.weight", "token_embd"])?,
        };

        let output_norm = load_tensor_dense(archive.as_ref(), &["output_norm.weight"])?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = if archive
            .tensor_first_of(&["output.weight", "lm_head.weight"])
            .is_some()
        {
            MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_first(archive.as_ref(), &["output.weight", "lm_head.weight"])?,
            }
        } else {
            // Some Llama-family GGUFs tie the LM head to token embeddings.
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.attn_norm.weight")])?;
            let wq = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_q.weight")])?,
            };
            let wk = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_k.weight")])?,
            };
            let wv = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_v.weight")])?,
            };
            let attn_q_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_q_norm.weight"),
                cfg.head_dim,
            );
            let attn_k_norm = load_optional_head_rmsnorm(
                archive.as_ref(),
                &format!("{p}.attn_k_norm.weight"),
                cfg.head_dim,
            );
            let wo = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(
                    archive.as_ref(),
                    &[
                        format!("{p}.attn_output.weight"),
                        format!("{p}.attn_out.weight"),
                    ],
                )?,
            };
            let ffn_norm =
                load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_norm.weight")])?;
            let ffn_gate = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_gate.weight")])?,
            };
            let ffn_up = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_up.weight")])?,
            };
            let ffn_down = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_down.weight")])?,
            };

            Self::validate_layer_mmap(
                archive.as_ref(),
                i,
                &attn_norm,
                &wq,
                &wk,
                &wv,
                &wo,
                &ffn_norm,
                &ffn_gate,
                &ffn_up,
                &ffn_down,
                n_embd,
                n_embd_kv,
                n_ff,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                attn_q_norm,
                attn_k_norm,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            rope_inv_freq,
        })
    }

    fn validate_layer_mmap(
        _archive: &GgufArchive,
        i: usize,
        attn_norm: &[f32],
        wq: &MatrixWeights,
        wk: &MatrixWeights,
        wv: &MatrixWeights,
        wo: &MatrixWeights,
        ffn_norm: &[f32],
        ffn_gate: &MatrixWeights,
        ffn_up: &MatrixWeights,
        ffn_down: &MatrixWeights,
        n_embd: usize,
        n_embd_kv: usize,
        n_ff: usize,
    ) -> Result<()> {
        let nelements = |m: &MatrixWeights| -> Result<usize> {
            match m {
                MatrixWeights::Dense(v) => Ok(v.len()),
                MatrixWeights::CudaDense { host, .. } => Ok(host.len()),
                MatrixWeights::Quant { tensor, .. } => Ok(tensor
                    .dimensions
                    .iter()
                    .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
                    .ok_or_else(|| BitNetError::InvalidGguf("tensor dims".into()))?),
            }
        };
        if attn_norm.len() != n_embd
            || nelements(wq)? != n_embd * n_embd
            || nelements(wk)? != n_embd * n_embd_kv
            || nelements(wv)? != n_embd * n_embd_kv
            || nelements(wo)? != n_embd * n_embd
            || ffn_norm.len() != n_embd
            || nelements(ffn_gate)? != n_embd * n_ff
            || nelements(ffn_up)? != n_embd * n_ff
            || nelements(ffn_down)? != n_ff * n_embd
        {
            return Err(BitNetError::Inference(format!(
                "layer {i} weight shape mismatch (mmap)"
            )));
        }
        // Touch payload bounds once per matrix
        let touch = |m: &MatrixWeights| -> Result<()> {
            if let MatrixWeights::Quant { archive, tensor } = m {
                archive.tensor_payload(tensor)?;
            }
            Ok(())
        };
        touch(wq)?;
        touch(wk)?;
        touch(wv)?;
        touch(wo)?;
        touch(ffn_gate)?;
        touch(ffn_up)?;
        touch(ffn_down)?;
        Ok(())
    }

    /// Run one forward step: token embedding + all layers + output matmul. Returns logits `[n_vocab]`.
    #[allow(dead_code)]
    pub fn forward(&self, kv: &mut KvCache, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cpu = CpuBackend;
        let mut wrap = KvStorage::Dense(std::mem::replace(kv, KvCache::new(&self.cfg)));
        let out = self.forward_with_backend(&mut wrap, token, pos, &cpu);
        if let KvStorage::Dense(d) = wrap {
            *kv = d;
        }
        out
    }

    pub fn forward_with_backend(
        &self,
        kv: &mut KvStorage,
        token: u32,
        pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Vec<f32>> {
        let mut scratch = ScratchArena::default();
        self.forward_with_backend_and_scratch(kv, token, pos, backend, &mut scratch)
    }

    pub fn forward_with_backend_and_scratch(
        &self,
        kv: &mut KvStorage,
        token: u32,
        pos: usize,
        backend: &dyn ComputeBackend,
        scratch: &mut ScratchArena,
    ) -> Result<Vec<f32>> {
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

        ggml_bridge::warn_if_ggml_env_without_bridge();

        let n_embd = cfg.n_embd;
        let mut x = scratch.take(n_embd);
        self.token_embd
            .embed_row(tok, n_embd, cfg.n_vocab, &mut x)?;

        let n_rep = cfg.n_head / cfg.n_kv;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = scratch.take(n_embd);
            rmsnorm_into(&x, &layer.attn_norm, cfg.norm_eps, &mut h);
            let q = layer.wq.matvec_embd_out(&h, n_embd, n_embd)?;
            let k = layer
                .wk
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;
            let v = layer
                .wv
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;
            scratch.recycle(h);

            let mut q_heads = q;
            if let Some(w) = &layer.attn_q_norm {
                apply_optional_head_rmsnorm_inplace(
                    &mut q_heads,
                    cfg.n_head,
                    cfg.head_dim,
                    w,
                    cfg.norm_eps,
                    scratch,
                );
            }
            rope_heads_inplace(
                &mut q_heads,
                cfg.n_head,
                cfg.head_dim,
                cfg.rope_rot_dims,
                pos,
                cfg.rope_theta,
                self.rope_inv_freq.as_deref(),
            );

            let mut k_heads = k;
            if let Some(w) = &layer.attn_k_norm {
                apply_optional_head_rmsnorm_inplace(
                    &mut k_heads,
                    cfg.n_kv,
                    cfg.head_dim,
                    w,
                    cfg.norm_eps,
                    scratch,
                );
            }
            rope_heads_inplace(
                &mut k_heads,
                cfg.n_kv,
                cfg.head_dim,
                cfg.rope_rot_dims,
                pos,
                cfg.rope_theta,
                self.rope_inv_freq.as_deref(),
            );

            let stride = cfg.n_kv * cfg.head_dim;
            kv.write_layer_kv(il, pos, &k_heads, &v, stride)?;

            let mut attn_out = scratch.take(n_embd);
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            let sw_start = cfg.sliding_window_key_start(pos);

            if matches!(
                backend.kind(),
                BackendKind::Cpu | BackendKind::Hybrid
            ) {
                let use_blas_scores =
                    blas_runtime::blas_attention_enabled() && blas_runtime::blas_ready();
                // SAFETY: `write_layer_kv` for this layer/position finished above; attention only
                // reads KV for `il` and positions `0..=pos` until the next layer iteration.
                let kv_ro: &KvStorage = unsafe { &*(kv as *mut KvStorage as *const KvStorage) };
                let mut head_parts: Vec<(usize, Vec<f32>)> = (0..cfg.n_head)
                    .into_par_iter()
                    .map(|qh| {
                        llama_cpu_attention_one_head(
                            kv_ro,
                            il,
                            pos,
                            qh,
                            n_rep,
                            cfg.head_dim,
                            stride,
                            &q_heads,
                            scale,
                            sw_start,
                            use_blas_scores,
                        )
                    })
                    .collect();
                head_parts.sort_by_key(|(qh, _)| *qh);
                for (qh, comb) in head_parts {
                    let dst = qh * cfg.head_dim;
                    attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
                }
            } else {
                for qh in 0..cfg.n_head {
                    let kv_h = qh / n_rep;
                    let q_slice = &q_heads[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                    let mut scores: Vec<f32> = {
                        let mut k_mat = scratch.take((pos + 1) * cfg.head_dim);
                        kv.fill_k_rows_gpu(il, pos, kv_h, cfg.head_dim, stride, &mut k_mat);
                        let mut s = backend.matvec(&k_mat, q_slice, pos + 1, cfg.head_dim)?;
                        scratch.recycle(k_mat);
                        for v in &mut s {
                            *v *= scale;
                        }
                        mask_sliding_window_scores(&mut s, sw_start);
                        s
                    };
                    softmax_inplace(&mut scores);
                    let mut comb = scratch.take(cfg.head_dim);
                    let mut v_values = scratch.take(cfg.head_dim);
                    for p in 0..=pos {
                        kv.fill_v_head_values(il, p, kv_h, cfg.head_dim, stride, &mut v_values);
                        let sp = scores[p];
                        for i in 0..cfg.head_dim {
                            comb[i] += sp * v_values[i];
                        }
                    }
                    let dst = qh * cfg.head_dim;
                    attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
                    scratch.recycle(scores);
                    scratch.recycle(comb);
                    scratch.recycle(v_values);
                }
            }

            let y = layer.wo.matvec_embd_out(&attn_out, n_embd, n_embd)?;
            scratch.recycle(attn_out);
            add_residual_inplace(&mut x, &y);

            let mut h2 = scratch.take(n_embd);
            rmsnorm_into(&x, &layer.ffn_norm, cfg.norm_eps, &mut h2);
            let gate = layer.ffn_gate.matvec_embd_out(&h2, n_embd, cfg.n_ff)?;
            let up = layer.ffn_up.matvec_embd_out(&h2, n_embd, cfg.n_ff)?;
            let mut tmp = scratch.take(cfg.n_ff);
            silu_mul_into(&gate, &up, &mut tmp);
            scratch.recycle(h2);
            let y2 = layer.ffn_down.matvec_ff(&tmp, cfg.n_ff, n_embd)?;
            scratch.recycle(tmp);
            add_residual_inplace(&mut x, &y2);
        }

        let mut xn = scratch.take(n_embd);
        rmsnorm_into(&x, &self.output_norm, cfg.norm_eps, &mut xn);
        let logits = self.output.matvec_embd_out(&xn, n_embd, cfg.n_vocab);
        scratch.recycle(xn);
        scratch.recycle(x);
        logits
    }
}

#[cfg(test)]
mod rope_norm_tests {
    use super::rope_inplace;

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        rope_inplace(&mut v, 0, 10_000.0, None);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_norm_rotates_adjacent_pairs() {
        let theta = 10_000.0_f32;
        let h = 4_usize;
        let pos = 1_usize;
        let mut expected = vec![1.0_f32, 2.0, 3.0, 4.0];
        for i in 0..2 {
            let inv_freq = 1.0 / theta.powf(2.0 * (i as f32) / (h as f32));
            let angle = pos as f32 * inv_freq;
            let c = angle.cos();
            let s = angle.sin();
            let x0 = expected[2 * i];
            let x1 = expected[2 * i + 1];
            expected[2 * i] = x0 * c - x1 * s;
            expected[2 * i + 1] = x0 * s + x1 * c;
        }
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        rope_inplace(&mut v, pos, theta, None);
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {v:?} expected {expected:?}");
        }
    }
}
