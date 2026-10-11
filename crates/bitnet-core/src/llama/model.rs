//! Llama forward: dense `f32` weights or mmap-backed quantized tensors (GEMV without full dequant).

use std::cell::RefCell;
use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::{
    BackendKind, ComputeBackend, CpuBackend, CudaDeviceMatrix, CudaDeviceQuantMatrix, CudaRuntime,
};
use crate::error::{BitNetError, Result};
use crate::ggml::{
    embedding_row_mmap, ggml_type_supported_mmap_matvec, ggml_type_supports_cuda_quant,
    matvec_batch_mmap, matvec_embd_out_mmap, matvec_ff_mmap, matvec_q8_0_rows,
    quantize_f16_rows_to_q8_0, tensor_to_f32,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use crate::scratch::ScratchArena;

use super::blas_runtime;
use super::config::LlamaConfig;
use super::ggml_bridge;
use super::kv_storage::{KvCache, KvStorage};
use super::slim_attention;

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
    /// Device-resident quantized payload (#22 CUDA vertical) with host fallback.
    CudaQuant {
        device: CudaDeviceQuantMatrix,
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
            Self::CudaQuant { device, label } => device.matvec(x).map_err(|e| {
                tracing::warn!(
                    tensor = label.as_str(),
                    error = %e,
                    "cuda quant matvec failed"
                );
                e
            }),
            Self::Quant { archive, tensor } => {
                matvec_embd_out_mmap(archive.as_ref(), tensor, x, ne0, ne1)
            }
        }
    }

    /// Token-major batch: `xs` is `[n_tokens, ne0]`, result is `[n_tokens, ne1]`.
    fn matvec_batch(&self, xs: &[f32], n_tokens: usize, ne0: usize, ne1: usize) -> Result<Vec<f32>> {
        match self {
            Self::Quant { archive, tensor } => {
                matvec_batch_mmap(archive.as_ref(), tensor, xs, n_tokens, ne0, ne1)
            }
            _ => {
                let mut out = vec![0.0f32; n_tokens * ne1];
                for t in 0..n_tokens {
                    let y = self.matvec_embd_out(&xs[t * ne0..(t + 1) * ne0], ne0, ne1)?;
                    out[t * ne1..(t + 1) * ne1].copy_from_slice(&y);
                }
                Ok(out)
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
            Self::CudaQuant { device, label } => device.matvec(x).map_err(|e| {
                tracing::warn!(
                    tensor = label.as_str(),
                    error = %e,
                    "cuda quant ffn_down failed"
                );
                e
            }),
            Self::Quant { archive, tensor } => {
                matvec_ff_mmap(archive.as_ref(), tensor, x, n_ff, n_embd)
            }
        }
    }

    pub(super) fn embed_row(
        &self,
        tok: usize,
        n_embd: usize,
        n_vocab: usize,
        out: &mut [f32],
    ) -> Result<()> {
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
            Self::CudaQuant { .. } => Err(BitNetError::Inference(
                "token embedding does not use CudaQuant residency".into(),
            )),
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
    /// BitNet b1.58 only: RMSNorm on the attention output, width `n_embd`, before `attn_output`.
    pub attn_sub_norm: Option<Vec<f32>>,
    /// BitNet b1.58 only: RMSNorm on SiLU(gate)×up, width `n_ff`, before `ffn_down`.
    pub ffn_sub_norm: Option<Vec<f32>>,
}

pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub token_embd: MatrixWeights,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: MatrixWeights,
    /// Q8_0 copy of a large f16 output head. Token embedding stays on the original tensor.
    output_q8: Option<Vec<u8>>,
    /// Effective inverse frequencies computed from GGUF `rope_freqs.weight` factors (Llama 3+).
    /// Length `rope_rot_dims / 2`; each analytic frequency is divided by its GGUF factor.
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

/// GGML's RoPE uses `angle = position * analytic_inv_freq / freq_factor`.
/// `rope_freqs.weight` contains divisors, not inverse frequencies.
fn try_load_rope_inv_freq(
    archive: &GgufArchive,
    rope_rot_dims: usize,
    theta: f32,
) -> Result<Option<Vec<f32>>> {
    let half = rope_rot_dims / 2;
    let Some(t) = archive.tensor_first_of(&["rope_freqs.weight"]) else {
        return Ok(None);
    };
    if t.dimensions.len() != 1 || t.dimensions[0] as usize != half {
        return Err(BitNetError::Inference(format!(
            "rope_freqs.weight shape {:?}, expected [{half}]",
            t.dimensions
        )));
    }
    let payload = archive.tensor_payload(t)?;
    let factors = tensor_to_f32(payload, t.ggml_type, &t.dimensions)?;
    Ok(Some(rope_inv_freq_from_factors(
        &factors,
        rope_rot_dims,
        theta,
    )?))
}

fn rope_inv_freq_from_factors(
    factors: &[f32],
    rope_rot_dims: usize,
    theta: f32,
) -> Result<Vec<f32>> {
    if factors.len() != rope_rot_dims / 2
        || !theta.is_finite()
        || theta <= 0.0
        || factors.iter().any(|f| !f.is_finite() || *f <= 0.0)
    {
        return Err(BitNetError::Inference(
            "invalid rope_freqs.weight factors or frequency base".into(),
        ));
    }
    let h = rope_rot_dims as f32;
    Ok(factors
        .iter()
        .enumerate()
        .map(|(i, factor)| 1.0 / theta.powf(2.0 * i as f32 / h) / factor)
        .collect())
}

/// Returns `Ok(())` if every Llama weight matrix uses a GGML type we can mmap-GEMV.
pub fn llama_mmap_quant_supported(archive: &GgufArchive) -> Result<()> {
    let cfg = LlamaConfig::from_gguf(archive)?;
    llama_mmap_quant_supported_with_config(archive, &cfg)
}
fn llama_mmap_quant_supported_with_config(archive: &GgufArchive, cfg: &LlamaConfig) -> Result<()> {
    let check = |names: &[&str]| -> Result<()> {
        let t = archive.tensor_first_of(names).ok_or_else(|| {
            BitNetError::Inference(format!("mmap check: missing tensor {:?}", names))
        })?;
        if !matrix_mmap_supported(t) {
            return Err(BitNetError::Inference(format!(
                "unsupported GGML tensor type {} ({}) for {:?}",
                t.ggml_type,
                crate::ggml::ggml_type_name(t.ggml_type),
                names
            )));
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
        if kind != BackendKind::Hybrid && kind != BackendKind::Cuda {
            return Self::disabled("backend is not cuda/hybrid", cfg.n_layer);
        }
        let min_rows = std::env::var("RBITNET_HYBRID_MIN_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(512);
        let layer_bytes = llama_layer_f32_bytes(cfg);
        // Soft planning budget: hybrid defaults 512 MiB; cuda defaults higher so a single
        // GPU box can stage several layers without forcing densify of the whole model.
        // 4096 MiB left the tail of an 8B Q4 on CPU and disabled the resident graph.
        let default_vram_mb = if kind == BackendKind::Cuda { 12288 } else { 512 };
        let max_bytes = std::env::var("RBITNET_HYBRID_MAX_VRAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(default_vram_mb)
            .saturating_mul(1024 * 1024);
        let policy = std::env::var("RBITNET_HYBRID_POLICY")
            .unwrap_or_else(|_| {
                if kind == BackendKind::Cuda {
                    "auto".into()
                } else {
                    "layers".into()
                }
            })
            .trim()
            .to_ascii_lowercase();
        // Reserve the vocabulary projection first: leaving it on CPU dominated CUDA decode.
        let output_bytes = cfg.n_embd.saturating_mul(cfg.n_vocab).saturating_mul(4);
        let output_requested = match std::env::var("RBITNET_HYBRID_OUTPUT").as_deref() {
            Ok("0" | "false" | "no") => false,
            Ok("1" | "true" | "yes") => true,
            _ => kind == BackendKind::Cuda,
        };
        let output = output_requested && output_bytes <= max_bytes;
        let layers = hybrid_layer_policy(
            &policy,
            cfg,
            layer_bytes,
            max_bytes.saturating_sub(if output { output_bytes } else { 0 }),
        );
        let layer_count = layers.iter().filter(|&&v| v).count();
        let estimated_weight_bytes = layer_count
            .saturating_mul(layer_bytes)
            .saturating_add(if output { output_bytes } else { 0 });
        let enabled = layer_count > 0 || output;
        let kind_label = kind.as_str();
        Self {
            enabled,
            layers,
            min_rows,
            output,
            estimated_weight_bytes,
            reason: if enabled {
                format!("{kind_label} offload policy={policy} selected {layer_count} layers")
            } else {
                format!("{kind_label} backend selected but policy={policy} selected no layers")
            },
        }
    }

    pub fn layer_enabled(&self, layer: usize) -> bool {
        self.enabled && self.layers.get(layer).copied().unwrap_or(false)
    }

    fn for_quant_archive(
        kind: BackendKind,
        cfg: &LlamaConfig,
        archive: &GgufArchive,
    ) -> Result<Self> {
        let mut plan = Self::from_env(kind, cfg);
        let policy = std::env::var("RBITNET_HYBRID_POLICY").unwrap_or_else(|_| {
            if kind == BackendKind::Cuda {
                "auto".into()
            } else {
                "layers".into()
            }
        });
        if !matches!(kind, BackendKind::Cuda | BackendKind::Hybrid)
            || !policy.trim().eq_ignore_ascii_case("auto")
            || std::env::var("RBITNET_HYBRID_LAYERS").is_ok()
        {
            return Ok(plan);
        }
        #[cfg(feature = "profile-llama")]
        if std::env::var("RBITNET_PROFILE_CUDA_DENSE").as_deref() == Ok("1") {
            return Ok(plan);
        }
        let budget = std::env::var("RBITNET_HYBRID_MAX_VRAM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if kind == BackendKind::Cuda { 12288 } else { 512 })
            .saturating_mul(1024 * 1024);
        let cost = |t: &GgufTensorInfo| -> Result<usize> {
            if t.dimensions.get(1).copied().unwrap_or(0) < plan.min_rows as u64 {
                return Ok(0);
            }
            if ggml_type_supports_cuda_quant(t.ggml_type) {
                Ok(archive.tensor_payload(t)?.len())
            } else {
                t.dimensions
                    .iter()
                    .try_fold(4usize, |n, &d| n.checked_mul(d as usize))
                    .ok_or_else(|| BitNetError::Inference("offload size overflow".into()))
            }
        };
        let output = archive
            .tensor_first_of(&["output.weight", "token_embd.weight"])
            .ok_or_else(|| BitNetError::Inference("missing output weights".into()))?;
        let requested = match std::env::var("RBITNET_HYBRID_OUTPUT").as_deref() {
            Ok("0" | "false" | "no") => false,
            Ok("1" | "true" | "yes") => true,
            _ => kind == BackendKind::Cuda,
        };
        let output_bytes = cost(output)?;
        // An f16 vocabulary (BitNet TQ2) stays on the CPU Q8 head. Uploading it
        // as dense f32 adds about 1.3 GiB and changes the logits that already
        // match llama.cpp. The ternary weights are the CUDA payload.
        plan.output = requested && output.ggml_type != 1 && output_bytes <= budget;
        let mut used = if plan.output { output_bytes } else { 0 };
        for il in 0..cfg.n_layer {
            let prefix = format!("blk.{il}.");
            let mut bytes = 0usize;
            for t in archive
                .tensors
                .iter()
                .filter(|t| t.name.starts_with(&prefix) && t.dimensions.len() == 2)
            {
                bytes = bytes.saturating_add(cost(t)?);
            }
            plan.layers[il] = bytes > 0 && used.saturating_add(bytes) <= budget;
            if plan.layers[il] {
                used += bytes;
            }
        }
        plan.estimated_weight_bytes = used;
        plan.enabled = plan.output || plan.layers.iter().any(|&v| v);
        plan.reason = format!(
            "{} auto offload uses actual quantized payload sizes",
            kind.as_str()
        );
        Ok(plan)
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
        (max_bytes / layer_bytes).min(cfg.n_layer)
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

/// Prefer device-resident quantized weights when the GGML type has a CUDA quant ABI;
/// otherwise densify to `f32` and reuse the existing [`CudaDeviceMatrix`] path.
fn maybe_cuda_quant_or_dense(
    archive: &Arc<GgufArchive>,
    rt: Option<&Arc<CudaRuntime>>,
    plan: &LlamaOffloadPlan,
    label: String,
    names: &[String],
    out_rows: usize,
    in_cols: usize,
) -> Result<MatrixWeights> {
    let tensor = tensor_info_strings(archive.as_ref(), names)?;
    if out_rows < plan.min_rows {
        return Ok(MatrixWeights::Quant {
            archive: Arc::clone(archive),
            tensor,
        });
    }
    // Diagnostic ablation only: compare resident cuBLAS F32 with the native quant kernel.
    #[cfg(feature = "profile-llama")]
    let force_dense = matches!(
        std::env::var("RBITNET_PROFILE_CUDA_DENSE").as_deref(),
        Ok("1")
    );
    #[cfg(not(feature = "profile-llama"))]
    let force_dense = false;
    if !force_dense && ggml_type_supports_cuda_quant(tensor.ggml_type) {
        let payload = archive.tensor_payload(&tensor)?.to_vec();
        match CudaDeviceQuantMatrix::from_payload(rt, tensor.ggml_type, payload, out_rows, in_cols)
        {
            Ok(device) => {
                tracing::debug!(
                    tensor = label.as_str(),
                    ggml_type = tensor.ggml_type,
                    device_resident = device.is_device_resident(),
                    "llama cuda quant residency"
                );
                return Ok(MatrixWeights::CudaQuant { device, label });
            }
            Err(e) => {
                tracing::warn!(
                    tensor = label.as_str(),
                    error = %e,
                    "cuda quant residency build failed; densifying"
                );
            }
        }
    }
    let host = load_tensor_strings_dense(archive.as_ref(), names)?;
    Ok(maybe_cuda_dense(rt, plan, label, host, out_rows, in_cols))
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
        && blas_runtime::sgemv_row_major_notrans(w, n_embd, n_ff, n_ff, 1.0, x, &mut y, 0.0).is_ok()
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
    // ggml_compute_forward_rms_norm_f32 squares in f32 and accumulates in f64,
    // then uses an f32 mean and `sqrtf`. The scale multiply is `(x * scale) * w`.
    let sum_sq = sum_f32_squares(x);
    let mean = (sum_sq / x.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    apply_rms_scale(x, w, scale, out);
}

fn sum_f32_squares(x: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        return unsafe { sum_f32_squares_avx(x) };
    }
    let mut sum_sq = 0.0f64;
    for value in x {
        sum_sq += f64::from(value * value);
    }
    sum_sq
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_f32_squares_avx(x: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let p = x.as_ptr();
    let mut acc = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 4 <= x.len() {
        let v = _mm_loadu_ps(p.add(i));
        let sq = _mm_mul_ps(v, v);
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(sq));
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum_sq = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < x.len() {
        sum_sq += f64::from(*p.add(i) * *p.add(i));
        i += 1;
    }
    sum_sq
}

fn apply_rms_scale(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        unsafe { apply_rms_scale_avx(x, w, scale, out) }
        return;
    }
    for i in 0..x.len() {
        out[i] = x[i] * scale * w[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn apply_rms_scale_avx(x: &[f32], w: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let s = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= x.len() {
        let y = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), s), _mm256_loadu_ps(w.as_ptr().add(i)));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), y);
        i += 8;
    }
    while i < x.len() {
        *out.as_mut_ptr().add(i) = *x.as_ptr().add(i) * scale * *w.as_ptr().add(i);
        i += 1;
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

fn rope_neox_inplace(slice: &mut [f32], pos: usize, theta: f32, inv_freq_flat: Option<&[f32]>) {
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
        let x0 = slice[i];
        let x1 = slice[i + half];
        slice[i] = x0 * c - x1 * s;
        slice[i + half] = x0 * s + x1 * c;
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
    neox: bool,
) {
    assert!(rope_rot_dims <= head_dim);
    assert!(rope_rot_dims % 2 == 0);
    for h in 0..n_head {
        let s = &mut x[h * head_dim..(h + 1) * head_dim];
        if neox {
            rope_neox_inplace(&mut s[..rope_rot_dims], pos, theta, inv_freq);
        } else {
            rope_inplace(&mut s[..rope_rot_dims], pos, theta, inv_freq);
        }
    }
}

fn silu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        let g = gate[i];
        out[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

fn bitnet_subnorms(
    archive: &GgufArchive,
    prefix: &str,
    n_embd: usize,
    n_ff: usize,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    if !matches!(
        archive.normalized_architecture().as_deref(),
        Some("bitnet") | Some("bitnet-b1.58")
    ) {
        return (None, None);
    }
    (
        load_optional_head_rmsnorm(archive, &format!("{prefix}.attn_sub_norm.weight"), n_embd),
        load_optional_head_rmsnorm(archive, &format!("{prefix}.ffn_sub_norm.weight"), n_ff),
    )
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

/// Dense F32 attention for every query head at one position.
/// Queries that share a KV head reuse that head's K and V.
fn attention_dense_grouped(
    k_layer: &[f32],
    v_layer: &[f32],
    q_heads: &[f32],
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    sw_start: usize,
    attn_out: &mut [f32],
) {
    let n_rep = n_head / n_kv;
    let seq = pos + 1;
    let width = n_head * head_dim;
    attn_out[..width].fill(0.0);
    let mut scores = vec![0.0f32; n_rep * seq];
    for kv_h in 0..n_kv {
        for r in 0..n_rep {
            scores[r * seq..r * seq + sw_start.min(seq)].fill(f32::NEG_INFINITY);
        }
        for p in sw_start.min(seq)..seq {
            let k = &k_layer[p * stride + kv_h * head_dim..p * stride + (kv_h + 1) * head_dim];
            for r in 0..n_rep {
                let qh = kv_h * n_rep + r;
                let q = &q_heads[qh * head_dim..(qh + 1) * head_dim];
                scores[r * seq + p] = crate::ggml::simd::dot(q, k) * scale;
            }
        }
        for r in 0..n_rep {
            softmax_inplace(&mut scores[r * seq..(r + 1) * seq]);
            let dst_h = kv_h * n_rep + r;
            let dst = &mut attn_out[dst_h * head_dim..(dst_h + 1) * head_dim];
            for p in sw_start.min(seq)..seq {
                let sp = scores[r * seq + p];
                let v = &v_layer[p * stride + kv_h * head_dim..p * stride + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    dst[i] += sp * v[i];
                }
            }
        }
    }
}

fn pack_kv_head(layer: &[f32], kv_h: usize, head_dim: usize, stride: usize, seq: usize) -> Vec<f32> {
    let mut packed = vec![0.0f32; seq * head_dim];
    for p in 0..seq {
        let src = p * stride + kv_h * head_dim;
        packed[p * head_dim..(p + 1) * head_dim]
            .copy_from_slice(&layer[src..src + head_dim]);
    }
    packed
}

/// One query against dense K laid out as `pos * stride + kv_head * head_dim`.
///
/// AVX2 uses the same two-accumulator FMA order as `simd::dot`.
fn score_strided_query(
    q: &[f32],
    k_layer: &[f32],
    kv_h: usize,
    head_dim: usize,
    stride: usize,
    seq: usize,
    scale: f32,
    scores: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if head_dim % 16 == 0
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        unsafe {
            score_strided_query_avx2(q, k_layer, kv_h, head_dim, stride, seq, scale, scores);
        }
        return;
    }
    for p in 0..seq {
        let off = p * stride + kv_h * head_dim;
        scores[p] = crate::ggml::simd::dot(q, &k_layer[off..off + head_dim]) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn score_strided_query_avx2(
    q: &[f32],
    k_layer: &[f32],
    kv_h: usize,
    head_dim: usize,
    stride: usize,
    seq: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use std::arch::x86_64::*;
    let qp = q.as_ptr();
    let base = k_layer.as_ptr().add(kv_h * head_dim);
    for p in 0..seq {
        let kp = base.add(p * stride);
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= head_dim {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(kp.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(qp.add(i + 8)),
                _mm256_loadu_ps(kp.add(i + 8)),
                s1,
            );
            i += 16;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(s0, s1));
        scores[p] = lanes.into_iter().sum::<f32>() * scale;
    }
}

fn gqa4_online_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("RBITNET_ATTN_EXACT").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
    })
}

fn write_dense_grouped_attention(
    kv: &KvStorage,
    il: usize,
    pos: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    _stride: usize,
    q_heads: &[f32],
    scale: f32,
    sw_start: usize,
    attn_out: &mut [f32],
) {
    let KvStorage::Dense(cache) = kv else {
        return;
    };
    let seq = pos + 1;
    let n_rep = n_head / n_kv;
    // Long GQA decode was rereading each KV head once per query. Four queries share
    // one K/V stream, scored online over sequence tiles.
    // Around 900 tokens the f16 online scores miss a ~0.005-logit tie (BitNet long-4).
    // The plain f32 score/softmax path matches there. The tiled f32 path does not.
    // Shorter and longer turns match on the online kernel.
    let f32_window = (768..1024).contains(&seq);
    #[cfg(target_arch = "x86_64")]
    if !f32_window
        && seq >= 256
        && n_rep == 4
        && head_dim % 16 == 0
        && gqa4_online_enabled()
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        write_gqa4_online(
            cache, il, seq, n_kv, head_dim, q_heads, scale, sw_start, attn_out,
        );
        return;
    }
    if !f32_window && seq >= 256 && n_rep > 1 && gqa4_online_enabled() {
        write_dense_grouped_attention_tiled(
            cache, il, seq, n_head, n_kv, n_rep, head_dim, q_heads, scale, sw_start, attn_out,
        );
        return;
    }
    let k_major = &cache.k_major[il];
    let v_major = &cache.v_major[il];
    let max_seq = cache.max_seq;
    attn_out[..n_head * head_dim]
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(qh, dst)| {
            let kv_h = qh / n_rep;
            let q = &q_heads[qh * head_dim..(qh + 1) * head_dim];
            let head_base = kv_h * max_seq * head_dim;
            let k_head = &k_major[head_base..head_base + seq * head_dim];
            let v_head = &v_major[head_base..head_base + seq * head_dim];
            ATTN_SCORE_SCRATCH.with(|cell| {
                let mut scores = cell.borrow_mut();
                if scores.len() < seq {
                    scores.resize(seq, 0.0);
                }
                let scores = &mut scores[..seq];
                score_strided_query(q, k_head, 0, head_dim, head_dim, seq, scale, scores);
                mask_sliding_window_scores(scores, sw_start);
                softmax_inplace(scores);
                dst.fill(0.0);
                mix_scores_into(dst, v_head, scores, head_dim);
            });
        });
}

/// Sequence tiles × KV heads. Each job owns a disjoint score span and a disjoint partial.
fn write_dense_grouped_attention_tiled(
    cache: &KvCache,
    il: usize,
    seq: usize,
    n_head: usize,
    n_kv: usize,
    n_rep: usize,
    head_dim: usize,
    q_heads: &[f32],
    scale: f32,
    sw_start: usize,
    attn_out: &mut [f32],
) {
    let tiles = if seq >= 1024 { 8 } else { 4 };
    let tile_len = seq.div_ceil(tiles);
    let k_major = &cache.k_major[il];
    let v_major = &cache.v_major[il];
    let max_seq = cache.max_seq;
    let mut scores = vec![0.0f32; n_head * seq];
    let scores_addr = scores.as_mut_ptr() as usize;
    (0..n_kv * tiles).into_par_iter().for_each(|job| {
        let kv_h = job / tiles;
        let tile = job % tiles;
        let start = tile * tile_len;
        if start >= seq {
            return;
        }
        let end = (start + tile_len).min(seq);
        let width = end - start;
        let head_base = kv_h * max_seq * head_dim;
        let k = &k_major[head_base + start * head_dim..head_base + end * head_dim];
        for r in 0..n_rep {
            let qh = kv_h * n_rep + r;
            let q = &q_heads[qh * head_dim..(qh + 1) * head_dim];
            unsafe {
                // Job (kv_h, tile, query) writes scores[qh * seq + start..end] and nothing else.
                let out = std::slice::from_raw_parts_mut(
                    (scores_addr as *mut f32).add(qh * seq + start),
                    width,
                );
                score_strided_query(q, k, 0, head_dim, head_dim, width, scale, out);
            }
        }
    });
    scores.par_chunks_mut(seq).for_each(|row| {
        mask_sliding_window_scores(row, sw_start);
        softmax_inplace(row);
    });
    let mut partials = vec![0.0f32; n_kv * tiles * n_rep * head_dim];
    let partial_addr = partials.as_mut_ptr() as usize;
    (0..n_kv * tiles).into_par_iter().for_each(|job| {
        let kv_h = job / tiles;
        let tile = job % tiles;
        let start = tile * tile_len;
        if start >= seq {
            return;
        }
        let end = (start + tile_len).min(seq);
        let head_base = kv_h * max_seq * head_dim;
        let v = &v_major[head_base + start * head_dim..head_base + end * head_dim];
        for r in 0..n_rep {
            let qh = kv_h * n_rep + r;
            let row = &scores[qh * seq + start..qh * seq + end];
            unsafe {
                let dst = std::slice::from_raw_parts_mut(
                    (partial_addr as *mut f32).add((job * n_rep + r) * head_dim),
                    head_dim,
                );
                mix_scores_into(dst, v, row, head_dim);
            }
        }
    });
    attn_out[..n_head * head_dim].fill(0.0);
    for kv_h in 0..n_kv {
        for tile in 0..tiles {
            let job = kv_h * tiles + tile;
            for r in 0..n_rep {
                let qh = kv_h * n_rep + r;
                let dst = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
                let src_off = (job * n_rep + r) * head_dim;
                for i in 0..head_dim {
                    dst[i] += partials[src_off + i];
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    static GQA_ONLINE_SCRATCH: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> =
        RefCell::new((Vec::new(), Vec::new(), Vec::new()));
}

fn write_gqa4_online(
    cache: &KvCache,
    il: usize,
    seq: usize,
    n_kv: usize,
    head_dim: usize,
    q_heads: &[f32],
    scale: f32,
    sw_start: usize,
    attn_out: &mut [f32],
) {
    let n_rep = 4usize;
    let tiles = if seq >= 1024 { 8 } else { 4 };
    let tile_len = seq.div_ceil(tiles);
    let jobs = n_kv * tiles;
    let k_f16 = &cache.k_f16[il];
    let v_f16 = &cache.v_f16[il];
    let max_seq = cache.max_seq;
    let (mut m, mut l, mut acc) = GQA_ONLINE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        (
            std::mem::take(&mut slot.0),
            std::mem::take(&mut slot.1),
            std::mem::take(&mut slot.2),
        )
    });
    m.clear();
    m.resize(jobs * n_rep, f32::NEG_INFINITY);
    l.clear();
    l.resize(jobs * n_rep, 0.0);
    acc.clear();
    acc.resize(jobs * n_rep * head_dim, 0.0);
    let m_addr = m.as_mut_ptr() as usize;
    let l_addr = l.as_mut_ptr() as usize;
    let acc_addr = acc.as_mut_ptr() as usize;
    (0..jobs).into_par_iter().for_each(|job| {
        let kv_h = job / tiles;
        let tile = job % tiles;
        let start = tile * tile_len;
        if start >= seq {
            return;
        }
        let end = (start + tile_len).min(seq);
        let width = end - start;
        let head_base = kv_h * max_seq * head_dim;
        let k = &k_f16[head_base + start * head_dim..head_base + end * head_dim];
        let v = &v_f16[head_base + start * head_dim..head_base + end * head_dim];
        let q = &q_heads[kv_h * n_rep * head_dim..(kv_h + 1) * n_rep * head_dim];
        ATTN_SCORE_SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            let need = width * n_rep;
            if buf.len() < need {
                buf.resize(need, 0.0);
            }
            unsafe {
                gqa4_online_tile_avx2(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    head_dim,
                    width,
                    start,
                    sw_start,
                    scale,
                    (m_addr as *mut f32).add(job * n_rep),
                    (l_addr as *mut f32).add(job * n_rep),
                    (acc_addr as *mut f32).add(job * n_rep * head_dim),
                    buf.as_mut_ptr(),
                );
            }
        });
    });
    let n_head = n_kv * n_rep;
    attn_out[..n_head * head_dim].fill(0.0);
    for kv_h in 0..n_kv {
        for r in 0..n_rep {
            let qh = kv_h * n_rep + r;
            let dst = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
            let mut mm = f32::NEG_INFINITY;
            let mut ll = 0.0f32;
            for tile in 0..tiles {
                let job = kv_h * tiles + tile;
                let m2 = m[job * n_rep + r];
                let l2 = l[job * n_rep + r];
                if !m2.is_finite() || l2 == 0.0 {
                    continue;
                }
                let src = (job * n_rep + r) * head_dim;
                let m_new = if mm.is_finite() { mm.max(m2) } else { m2 };
                let a1 = if mm.is_finite() {
                    (mm - m_new).exp()
                } else {
                    0.0
                };
                let a2 = (m2 - m_new).exp();
                for i in 0..head_dim {
                    dst[i] = dst[i] * a1 + acc[src + i] * a2;
                }
                ll = ll * a1 + l2 * a2;
                mm = m_new;
            }
            if ll > 0.0 {
                let inv = 1.0 / ll;
                for z in dst.iter_mut() {
                    *z *= inv;
                }
            }
        }
    }
    GQA_ONLINE_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        slot.0 = m;
        slot.1 = l;
        slot.2 = acc;
    });
}

/// Range-reduced exp for the online-softmax update. Libm `exp` inside the
/// position loop was several milliseconds per token at long context.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    const LN2: f32 = 0.69314718246459961;
    const INV_LN2: f32 = 1.4426950408889634;
    let x = x.clamp(-80.0, 80.0);
    let n = (x * INV_LN2).round();
    let f = x - n * LN2;
    let p = (1.0f32 / 5040.0)
        .mul_add(f, 1.0 / 720.0)
        .mul_add(f, 1.0 / 120.0)
        .mul_add(f, 1.0 / 24.0)
        .mul_add(f, 1.0 / 6.0)
        .mul_add(f, 0.5);
    let exp_f = f.mul_add(f.mul_add(p, 1.0), 1.0);
    let bits = (n as i32 + 127) as u32;
    exp_f * f32::from_bits(bits << 23)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gqa4_online_tile_avx2(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    head_dim: usize,
    width: usize,
    global_start: usize,
    sw_start: usize,
    scale: f32,
    m: *mut f32,
    l: *mut f32,
    acc: *mut f32,
    scores: *mut f32,
) {
    use std::arch::x86_64::*;
    macro_rules! hsum {
        ($v:expr) => {{
            let mut v = $v;
            v = _mm256_add_ps(v, _mm256_permute2f128_ps(v, v, 1));
            v = _mm256_add_ps(v, _mm256_shuffle_ps(v, v, 0x4E));
            v = _mm256_add_ps(v, _mm256_shuffle_ps(v, v, 0xB1));
            _mm256_cvtss_f32(v)
        }};
    }
    let q0 = q;
    let q1 = q.add(head_dim);
    let q2 = q.add(head_dim * 2);
    let q3 = q.add(head_dim * 3);
    for p in 0..width {
        let out = scores.add(p * 4);
        if global_start + p < sw_start {
            *out = f32::NEG_INFINITY;
            *out.add(1) = f32::NEG_INFINITY;
            *out.add(2) = f32::NEG_INFINITY;
            *out.add(3) = f32::NEG_INFINITY;
            continue;
        }
        let kp = k.add(p * head_dim);
        if p + 1 < width {
            _mm_prefetch(
                k.add((p + 1) * head_dim) as *const i8,
                _MM_HINT_T0,
            );
        }
        let mut s00 = _mm256_setzero_ps();
        let mut s01 = _mm256_setzero_ps();
        let mut s10 = _mm256_setzero_ps();
        let mut s11 = _mm256_setzero_ps();
        let mut s20 = _mm256_setzero_ps();
        let mut s21 = _mm256_setzero_ps();
        let mut s30 = _mm256_setzero_ps();
        let mut s31 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= head_dim {
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i) as *const __m128i));
            let k1 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(i + 8) as *const __m128i));
            s00 = _mm256_fmadd_ps(_mm256_loadu_ps(q0.add(i)), k0, s00);
            s01 = _mm256_fmadd_ps(_mm256_loadu_ps(q0.add(i + 8)), k1, s01);
            s10 = _mm256_fmadd_ps(_mm256_loadu_ps(q1.add(i)), k0, s10);
            s11 = _mm256_fmadd_ps(_mm256_loadu_ps(q1.add(i + 8)), k1, s11);
            s20 = _mm256_fmadd_ps(_mm256_loadu_ps(q2.add(i)), k0, s20);
            s21 = _mm256_fmadd_ps(_mm256_loadu_ps(q2.add(i + 8)), k1, s21);
            s30 = _mm256_fmadd_ps(_mm256_loadu_ps(q3.add(i)), k0, s30);
            s31 = _mm256_fmadd_ps(_mm256_loadu_ps(q3.add(i + 8)), k1, s31);
            i += 16;
        }
        *out = hsum!(_mm256_add_ps(s00, s01)) * scale;
        *out.add(1) = hsum!(_mm256_add_ps(s10, s11)) * scale;
        *out.add(2) = hsum!(_mm256_add_ps(s20, s21)) * scale;
        *out.add(3) = hsum!(_mm256_add_ps(s30, s31)) * scale;
    }
    let p0 = acc;
    let p1 = acc.add(head_dim);
    let p2 = acc.add(head_dim * 2);
    let p3 = acc.add(head_dim * 3);
    for p in 0..width {
        let s0 = *scores.add(p * 4);
        if !s0.is_finite() && !(*scores.add(p * 4 + 1)).is_finite() {
            continue;
        }
        let row = [s0, *scores.add(p * 4 + 1), *scores.add(p * 4 + 2), *scores.add(p * 4 + 3)];
        let mut alpha = [0.0f32; 4];
        let mut ev = [0.0f32; 4];
        for r in 0..4 {
            let s = row[r];
            if !s.is_finite() {
                alpha[r] = 1.0;
                ev[r] = 0.0;
                continue;
            }
            let m_old = *m.add(r);
            let m_new = if m_old.is_finite() { m_old.max(s) } else { s };
            alpha[r] = if m_old.is_finite() {
                exp_fast(m_old - m_new)
            } else {
                0.0
            };
            ev[r] = exp_fast(s - m_new);
            *l.add(r) = *l.add(r) * alpha[r] + ev[r];
            *m.add(r) = m_new;
        }
        let vp = v.add(p * head_dim);
        if p + 1 < width {
            _mm_prefetch(
                v.add((p + 1) * head_dim) as *const i8,
                _MM_HINT_T0,
            );
        }
        let a0 = _mm256_set1_ps(alpha[0]);
        let a1 = _mm256_set1_ps(alpha[1]);
        let a2 = _mm256_set1_ps(alpha[2]);
        let a3 = _mm256_set1_ps(alpha[3]);
        let e0 = _mm256_set1_ps(ev[0]);
        let e1 = _mm256_set1_ps(ev[1]);
        let e2 = _mm256_set1_ps(ev[2]);
        let e3 = _mm256_set1_ps(ev[3]);
        let mut j = 0;
        while j + 8 <= head_dim {
            let vv = _mm256_cvtph_ps(_mm_loadu_si128(vp.add(j) as *const __m128i));
            _mm256_storeu_ps(
                p0.add(j),
                _mm256_fmadd_ps(vv, e0, _mm256_mul_ps(_mm256_loadu_ps(p0.add(j)), a0)),
            );
            _mm256_storeu_ps(
                p1.add(j),
                _mm256_fmadd_ps(vv, e1, _mm256_mul_ps(_mm256_loadu_ps(p1.add(j)), a1)),
            );
            _mm256_storeu_ps(
                p2.add(j),
                _mm256_fmadd_ps(vv, e2, _mm256_mul_ps(_mm256_loadu_ps(p2.add(j)), a2)),
            );
            _mm256_storeu_ps(
                p3.add(j),
                _mm256_fmadd_ps(vv, e3, _mm256_mul_ps(_mm256_loadu_ps(p3.add(j)), a3)),
            );
            j += 8;
        }
    }
}
thread_local! {
    static ATTN_SCORE_SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn mix_scores_into(dst: &mut [f32], v_head: &[f32], scores: &[f32], head_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    if head_dim % 8 == 0
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        unsafe { mix_scores_into_avx2(dst, v_head, scores, head_dim) }
        return;
    }
    for (p, &sp) in scores.iter().enumerate() {
        let v = &v_head[p * head_dim..(p + 1) * head_dim];
        for i in 0..head_dim {
            dst[i] += sp * v[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mix_scores_into_avx2(dst: &mut [f32], v_head: &[f32], scores: &[f32], head_dim: usize) {
    use std::arch::x86_64::*;
    for (p, &sp) in scores.iter().enumerate() {
        let scale = _mm256_set1_ps(sp);
        let v = v_head.as_ptr().add(p * head_dim);
        let mut i = 0;
        while i + 8 <= head_dim {
            let acc = _mm256_fmadd_ps(
                scale,
                _mm256_loadu_ps(v.add(i)),
                _mm256_loadu_ps(dst.as_ptr().add(i)),
            );
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), acc);
            i += 8;
        }
    }
}

fn write_cpu_attention(
    kv: &KvStorage,
    il: usize,
    pos: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    stride: usize,
    q_heads: &[f32],
    scale: f32,
    sw_start: usize,
    attn_out: &mut [f32],
) {
    if matches!(kv, KvStorage::Dense(_))
        && head_dim % 16 == 0
        && n_kv > 0
        && n_head % n_kv == 0
    {
        write_dense_grouped_attention(
            kv, il, pos, n_head, n_kv, head_dim, stride, q_heads, scale, sw_start, attn_out,
        );
        return;
    }
    let use_blas_scores = blas_runtime::blas_attention_enabled() && blas_runtime::blas_ready();
    let n_rep = n_head / n_kv;
    let mut head_parts: Vec<(usize, Vec<f32>)> = (0..n_head)
        .into_par_iter()
        .map(|qh| {
            llama_cpu_attention_one_head(
                kv,
                il,
                pos,
                qh,
                n_rep,
                head_dim,
                stride,
                q_heads,
                scale,
                sw_start,
                use_blas_scores,
            )
        })
        .collect();
    head_parts.sort_by_key(|(qh, _)| *qh);
    for (qh, comb) in head_parts {
        let dst = qh * head_dim;
        attn_out[dst..dst + head_dim].copy_from_slice(&comb);
    }
}

/// One query head on the CPU / hybrid attention path (scores → softmax → V combination).
///
/// When `RBITNET_SLIM_ATTENTION` is on, materializes the sliding-window KV slice and runs
/// SlimAttention 1D tiled online-softmax ([`slim_attention::attention_tiled`]) instead of the
/// contiguous scores→softmax→V loop. Default remains the contiguous path.
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

    if slim_attention::slim_attention_enabled() {
        let seq_start = sw_start.min(pos + 1);
        let seq = (pos + 1).saturating_sub(seq_start);
        let mut comb = vec![0.0f32; head_dim];
        if seq == 0 {
            return (qh, comb);
        }
        let mut k_mat = vec![0.0f32; seq * head_dim];
        let mut v_mat = vec![0.0f32; seq * head_dim];
        for (local, p) in (seq_start..=pos).enumerate() {
            let row = local * head_dim;
            kv.fill_k_head_values(
                il,
                p,
                kv_h,
                head_dim,
                stride,
                &mut k_mat[row..row + head_dim],
            );
            kv.fill_v_head_values(
                il,
                p,
                kv_h,
                head_dim,
                stride,
                &mut v_mat[row..row + head_dim],
            );
        }
        slim_attention::attention_tiled(
            q_slice,
            &k_mat,
            &v_mat,
            seq,
            head_dim,
            scale,
            slim_attention::tile_tokens_from_env(),
            &mut comb,
        );
        return (qh, comb);
    }

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
            kv.attention_scores_cpu(il, pos, kv_h, head_dim, stride, q_slice, scale, &mut scores);
            mask_sliding_window_scores(&mut scores, sw_start);
        }
    } else {
        kv.attention_scores_cpu(il, pos, kv_h, head_dim, stride, q_slice, scale, &mut scores);
        mask_sliding_window_scores(&mut scores, sw_start);
    }
    softmax_inplace(&mut scores);
    let mut comb = vec![0.0f32; head_dim];
    let mut v_values = vec![0.0f32; head_dim];
    for p in 0..=pos {
        let values = if matches!(kv, KvStorage::Dense(_)) {
            kv.v_head_slice(il, p, kv_h, head_dim, stride)
        } else {
            kv.fill_v_head_values(il, p, kv_h, head_dim, stride, &mut v_values);
            &v_values
        };
        let sp = scores[p];
        for i in 0..head_dim {
            comb[i] += sp * values[i];
        }
    }
    (qh, comb)
}

fn add_residual_inplace(x: &mut [f32], y: &[f32]) {
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

/// Q8_0 copy of a large f16 vocabulary projection. Smaller matrices stay exact.
fn pack_output_q8(output: &MatrixWeights, ne0: usize, ne1: usize) -> Option<Vec<u8>> {
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (output, ne0, ne1);
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return None;
        }
        let MatrixWeights::Quant { archive, tensor } = output else {
            return None;
        };
        if tensor.ggml_type != 1 || ne0 % 32 != 0 || ne1 < 32_768 {
            return None;
        }
        if tensor.dimensions.len() < 2
            || tensor.dimensions[0] as usize != ne0
            || tensor.dimensions[1] as usize != ne1
        {
            return None;
        }
        let payload = archive.tensor_payload(tensor).ok()?;
        quantize_f16_rows_to_q8_0(payload, ne0, ne1).ok()
    }
}

impl LlamaModel {
    fn output_logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(packed) = &self.output_q8 {
            let row_bytes = (self.cfg.n_embd / 32) * 34;
            return matvec_q8_0_rows(packed, row_bytes, x, self.cfg.n_vocab);
        }
        self.output
            .matvec_embd_out(x, self.cfg.n_embd, self.cfg.n_vocab)
    }

    /// Load from mmap archive using `RBITNET_LLAMA_WEIGHT_MODE`.
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
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        Self::from_gguf_arc_with_config(archive, backend_kind, cfg)
    }
    pub(crate) fn from_gguf_arc_with_config(
        archive: Arc<GgufArchive>,
        backend_kind: BackendKind,
        cfg: LlamaConfig,
    ) -> Result<Self> {
        if backend_kind == BackendKind::Hybrid || backend_kind == BackendKind::Cuda {
            return Self::from_gguf_device_offload_internal(archive, backend_kind, cfg);
        }
        let mode = llama_weight_mode_from_env();
        match mode {
            LlamaWeightMode::Dense => Self::from_gguf_dense_internal(archive, cfg),
            LlamaWeightMode::MmapQuant => {
                llama_mmap_quant_supported_with_config(archive.as_ref(), &cfg)?;
                Self::from_gguf_mmap_internal(archive, cfg)
            }
            LlamaWeightMode::Auto => {
                if llama_mmap_quant_supported_with_config(archive.as_ref(), &cfg).is_ok() {
                    Self::from_gguf_mmap_internal(archive, cfg)
                } else {
                    Self::from_gguf_dense_internal(archive, cfg)
                }
            }
        }
    }

    fn from_gguf_device_offload_internal(
        archive: Arc<GgufArchive>,
        backend_kind: BackendKind,
        cfg: LlamaConfig,
    ) -> Result<Self> {
        let plan = LlamaOffloadPlan::for_quant_archive(backend_kind, &cfg, archive.as_ref())?;
        let cuda = if plan.enabled {
            CudaRuntime::try_load()
        } else {
            None
        };
        tracing::info!(
            summary = plan.summary(),
            cuda = cuda.is_some(),
            backend = backend_kind.as_str(),
            "llama device offload plan"
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
            let names: Vec<String> = if archive
                .tensor_first_of(&["output.weight", "lm_head.weight"])
                .is_some()
            {
                vec!["output.weight".into(), "lm_head.weight".into()]
            } else {
                vec!["token_embd.weight".into(), "token_embd".into()]
            };
            let label = names
                .first()
                .cloned()
                .unwrap_or_else(|| "output.weight".into());
            maybe_cuda_quant_or_dense(
                &archive,
                cuda.as_ref(),
                &plan,
                label,
                &names,
                n_vocab,
                n_embd,
            )?
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
                        maybe_cuda_quant_or_dense(
                            &archive,
                            cuda.as_ref(),
                            &plan,
                            label,
                            &names,
                            out_rows,
                            in_cols,
                        )
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
                maybe_cuda_quant_or_dense(
                    &archive,
                    cuda.as_ref(),
                    &plan,
                    format!("{p}.ffn_down.weight"),
                    &[format!("{p}.ffn_down.weight")],
                    n_embd,
                    n_ff,
                )?
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

            let (attn_sub_norm, ffn_sub_norm) =
                bitnet_subnorms(archive.as_ref(), &p, n_embd, n_ff);
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
                attn_sub_norm,
                ffn_sub_norm,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta)?;
        let output_q8 = pack_output_q8(&output, cfg.n_embd, cfg.n_vocab);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            output_q8,
            rope_inv_freq,
        })
    }

    fn from_gguf_dense_internal(archive: Arc<GgufArchive>, cfg: LlamaConfig) -> Result<Self> {
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

            let (attn_sub_norm, ffn_sub_norm) =
                bitnet_subnorms(archive.as_ref(), &p, n_embd, n_ff);
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
                attn_sub_norm,
                ffn_sub_norm,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta)?;
        let output_q8 = pack_output_q8(&output, cfg.n_embd, cfg.n_vocab);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            output_q8,
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
                MatrixWeights::CudaQuant { device, .. } => Ok(device
                    .out_rows()
                    .checked_mul(device.in_cols())
                    .ok_or_else(|| BitNetError::Inference("cuda quant dims overflow".into()))?),
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

    fn from_gguf_mmap_internal(archive: Arc<GgufArchive>, cfg: LlamaConfig) -> Result<Self> {
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

            let (attn_sub_norm, ffn_sub_norm) =
                bitnet_subnorms(archive.as_ref(), &p, n_embd, n_ff);
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
                attn_sub_norm,
                ffn_sub_norm,
            });
        }

        let rope_inv_freq =
            try_load_rope_inv_freq(archive.as_ref(), cfg.rope_rot_dims, cfg.rope_theta)?;
        let output_q8 = pack_output_q8(&output, cfg.n_embd, cfg.n_vocab);

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
            output_q8,
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
                MatrixWeights::CudaQuant { device, .. } => Ok(device
                    .out_rows()
                    .checked_mul(device.in_cols())
                    .ok_or_else(|| BitNetError::Inference("cuda quant dims overflow".into()))?),
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
            match m {
                MatrixWeights::Quant { archive, tensor } => {
                    archive.tensor_payload(tensor)?;
                }
                MatrixWeights::CudaQuant { device, .. } => {
                    if device.host_payload().is_empty() {
                        return Err(BitNetError::Inference("cuda quant empty payload".into()));
                    }
                }
                _ => {}
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
        self.forward_step(kv, token, pos, backend, scratch, true)
    }

    /// Updates all layer caches, optionally omitting the unused vocabulary projection.
    pub(crate) fn forward_step(
        &self,
        kv: &mut KvStorage,
        token: u32,
        pos: usize,
        backend: &dyn ComputeBackend,
        scratch: &mut ScratchArena,
        logits_required: bool,
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

        let n_embd = cfg.n_embd;
        let mut x = scratch.take(n_embd);
        self.token_embd
            .embed_row(tok, n_embd, cfg.n_vocab, &mut x)?;
        self.forward_from_embedding(kv, x, pos, backend, scratch, logits_required)
    }

    /// Prefill `tokens` starting at `base_pos`. Projections for the chunk share one read of each weight matrix.
    /// Logits are returned for the last token only. KV writes match the single-token forward.
    pub(crate) fn prefill_tokens(
        &self,
        kv: &mut KvStorage,
        tokens: &[u32],
        base_pos: usize,
        scratch: &mut ScratchArena,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let cfg = &self.cfg;
        let last_pos = base_pos + n - 1;
        if last_pos >= cfg.max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let n_embd = cfg.n_embd;
        let mut hidden = vec![0.0f32; n * n_embd];
        for (t, &token) in tokens.iter().enumerate() {
            let tok = token as usize;
            if tok >= cfg.n_vocab {
                return Err(BitNetError::Inference("token id out of range".into()));
            }
            self.token_embd.embed_row(
                tok,
                n_embd,
                cfg.n_vocab,
                &mut hidden[t * n_embd..(t + 1) * n_embd],
            )?;
        }

        let stride = cfg.n_kv * cfg.head_dim;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let kv_dim = stride;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut normed = vec![0.0f32; n * n_embd];
            for t in 0..n {
                rmsnorm_into(
                    &hidden[t * n_embd..(t + 1) * n_embd],
                    &layer.attn_norm,
                    cfg.norm_eps,
                    &mut normed[t * n_embd..(t + 1) * n_embd],
                );
            }
            let q = layer.wq.matvec_batch(&normed, n, n_embd, n_embd)?;
            let k = layer.wk.matvec_batch(&normed, n, n_embd, kv_dim)?;
            let v = layer.wv.matvec_batch(&normed, n, n_embd, kv_dim)?;
            let mut q_bank = vec![0.0f32; n * n_embd];
            for t in 0..n {
                let pos = base_pos + t;
                let mut q_heads = q[t * n_embd..(t + 1) * n_embd].to_vec();
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
                    cfg.rope_neox,
                );
                q_bank[t * n_embd..(t + 1) * n_embd].copy_from_slice(&q_heads);
                let mut k_heads = k[t * kv_dim..(t + 1) * kv_dim].to_vec();
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
                    cfg.rope_neox,
                );
                kv.write_layer_kv(il, pos, &k_heads, &v[t * kv_dim..(t + 1) * kv_dim], stride)?;
            }
            let mut attn = vec![0.0f32; n * n_embd];
            let kv_ro: &KvStorage = kv;
            if let KvStorage::Dense(cache) = kv_ro {
                let n_rep = cfg.n_head / cfg.n_kv;
                let total = base_pos + n;
                for kv_h in 0..cfg.n_kv {
                    let k_pack =
                        pack_kv_head(&cache.k[il], kv_h, cfg.head_dim, stride, total);
                    let v_pack =
                        pack_kv_head(&cache.v[il], kv_h, cfg.head_dim, stride, total);
                    let qh0 = kv_h * n_rep;
                    let group = n_rep * cfg.head_dim;
                    attn.par_chunks_mut(n_embd).enumerate().for_each(|(t, dst)| {
                        let pos = base_pos + t;
                        let q = &q_bank[t * n_embd + qh0 * cfg.head_dim..][..group];
                        attention_dense_grouped(
                            &k_pack,
                            &v_pack,
                            q,
                            n_rep,
                            1,
                            cfg.head_dim,
                            cfg.head_dim,
                            pos,
                            scale,
                            cfg.sliding_window_key_start(pos),
                            &mut dst[qh0 * cfg.head_dim..][..group],
                        );
                    });
                }
            } else {
                attn.par_chunks_mut(n_embd)
                    .enumerate()
                    .for_each(|(t, dst)| {
                        let pos = base_pos + t;
                        write_cpu_attention(
                            kv_ro,
                            il,
                            pos,
                            cfg.n_head,
                            cfg.n_kv,
                            cfg.head_dim,
                            stride,
                            &q_bank[t * n_embd..(t + 1) * n_embd],
                            scale,
                            cfg.sliding_window_key_start(pos),
                            dst,
                        );
                    });
            }
            if let Some(w) = &layer.attn_sub_norm {
                for t in 0..n {
                    let mut tmp = vec![0.0f32; n_embd];
                    rmsnorm_into(
                        &attn[t * n_embd..(t + 1) * n_embd],
                        w,
                        cfg.norm_eps,
                        &mut tmp,
                    );
                    attn[t * n_embd..(t + 1) * n_embd].copy_from_slice(&tmp);
                }
            }
            let yo = layer.wo.matvec_batch(&attn, n, n_embd, n_embd)?;
            for t in 0..n {
                add_residual_inplace(
                    &mut hidden[t * n_embd..(t + 1) * n_embd],
                    &yo[t * n_embd..(t + 1) * n_embd],
                );
            }

            let mut ffn_in = vec![0.0f32; n * n_embd];
            for t in 0..n {
                rmsnorm_into(
                    &hidden[t * n_embd..(t + 1) * n_embd],
                    &layer.ffn_norm,
                    cfg.norm_eps,
                    &mut ffn_in[t * n_embd..(t + 1) * n_embd],
                );
            }
            let gate = layer.ffn_gate.matvec_batch(&ffn_in, n, n_embd, cfg.n_ff)?;
            let up = layer.ffn_up.matvec_batch(&ffn_in, n, n_embd, cfg.n_ff)?;
            let mut mid = vec![0.0f32; n * cfg.n_ff];
            for t in 0..n {
                silu_mul_into(
                    &gate[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                    &up[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                    &mut mid[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                );
            }
            if let Some(w) = &layer.ffn_sub_norm {
                for t in 0..n {
                    let mut tmp = vec![0.0f32; cfg.n_ff];
                    rmsnorm_into(
                        &mid[t * cfg.n_ff..(t + 1) * cfg.n_ff],
                        w,
                        cfg.norm_eps,
                        &mut tmp,
                    );
                    mid[t * cfg.n_ff..(t + 1) * cfg.n_ff].copy_from_slice(&tmp);
                }
            }
            let y2 = layer.ffn_down.matvec_batch(&mid, n, cfg.n_ff, n_embd)?;
            for t in 0..n {
                add_residual_inplace(
                    &mut hidden[t * n_embd..(t + 1) * n_embd],
                    &y2[t * n_embd..(t + 1) * n_embd],
                );
            }
        }

        let last = n - 1;
        let mut xn = vec![0.0f32; n_embd];
        rmsnorm_into(
            &hidden[last * n_embd..(last + 1) * n_embd],
            &self.output_norm,
            cfg.norm_eps,
            &mut xn,
        );
        self.output_logits(&xn)
    }

    /// `RBITNET_CUDA_ATTENTION` is read once. BitNet (`rope_neox`) stays on the CPU
    /// f32 window unless the variable forces CUDA attention on.
    fn cuda_attention_requested(rope_neox: bool) -> bool {
        static PREF: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        match PREF.get_or_init(|| {
            match std::env::var("RBITNET_CUDA_ATTENTION").ok().as_deref() {
                Some("0" | "false" | "no") => Some(false),
                Some("1" | "true" | "yes") => Some(true),
                _ => None,
            }
        }) {
            Some(forced) => *forced,
            None => !rope_neox,
        }
    }

    /// Transformer step starting from a precomputed embedding row (`[n_embd]`).
    ///
    /// Used for LLaVA-style vision patch injection during CPU prefill.
    pub(crate) fn forward_from_embedding(
        &self,
        kv: &mut KvStorage,
        mut x: Vec<f32>,
        pos: usize,
        backend: &dyn ComputeBackend,
        scratch: &mut ScratchArena,
        logits_required: bool,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference(
                "sequence position >= max_seq".into(),
            ));
        }
        let n_embd = cfg.n_embd;
        if x.len() != n_embd {
            return Err(BitNetError::Inference(format!(
                "embedding len {} != n_embd {n_embd}",
                x.len()
            )));
        }

        ggml_bridge::warn_if_ggml_env_without_bridge();

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = scratch.take(n_embd);
            rmsnorm_into(&x, &layer.attn_norm, cfg.norm_eps, &mut h);
            #[cfg(feature = "profile-llama")]
            let qkv_span = super::profile::Span::new("qkv");
            let q = layer.wq.matvec_embd_out(&h, n_embd, n_embd)?;
            let k = layer
                .wk
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;
            let v = layer
                .wv
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;
            #[cfg(feature = "profile-llama")]
            drop(qkv_span);
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
                cfg.rope_neox,
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
                cfg.rope_neox,
            );

            let stride = cfg.n_kv * cfg.head_dim;
            kv.write_layer_kv(il, pos, &k_heads, &v, stride)?;

            #[cfg(feature = "profile-llama")]
            let attention_span = super::profile::Span::new("attention");
            let mut attn_out = scratch.take(n_embd);
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            let sw_start = cfg.sliding_window_key_start(pos);

            // BitNet long-4 is a ~0.005 logit tie that only the CPU f32 window
            // reproduces. CUDA attention stays opt-in for that architecture.
            let cuda_attention_requested = Self::cuda_attention_requested(cfg.rope_neox);
            let cuda_attention = cuda_attention_requested
                && matches!(backend.kind(), BackendKind::Cuda | BackendKind::Hybrid)
                && kv.attention_cuda(
                    il,
                    pos,
                    sw_start,
                    &q_heads,
                    cfg.n_head,
                    cfg.n_kv,
                    cfg.head_dim,
                    scale,
                    &mut attn_out,
                )?;
            if !cuda_attention {
                write_cpu_attention(
                    kv,
                    il,
                    pos,
                    cfg.n_head,
                    cfg.n_kv,
                    cfg.head_dim,
                    stride,
                    &q_heads,
                    scale,
                    sw_start,
                    &mut attn_out,
                );
            }

            #[cfg(feature = "profile-llama")]
            drop(attention_span);
            #[cfg(feature = "profile-llama")]
            let attn_output_span = super::profile::Span::new("attn_output");
            if let Some(w) = &layer.attn_sub_norm {
                let mut normed = scratch.take(n_embd);
                rmsnorm_into(&attn_out, w, cfg.norm_eps, &mut normed);
                scratch.recycle(std::mem::replace(&mut attn_out, normed));
            }
            let y = layer.wo.matvec_embd_out(&attn_out, n_embd, n_embd)?;
            #[cfg(feature = "profile-llama")]
            drop(attn_output_span);
            scratch.recycle(attn_out);
            add_residual_inplace(&mut x, &y);

            let mut h2 = scratch.take(n_embd);
            rmsnorm_into(&x, &layer.ffn_norm, cfg.norm_eps, &mut h2);
            #[cfg(feature = "profile-llama")]
            let gate_up_span = super::profile::Span::new("ffn_gate_up");
            let gate = layer.ffn_gate.matvec_embd_out(&h2, n_embd, cfg.n_ff)?;
            let up = layer.ffn_up.matvec_embd_out(&h2, n_embd, cfg.n_ff)?;
            #[cfg(feature = "profile-llama")]
            drop(gate_up_span);
            let mut tmp = scratch.take(cfg.n_ff);
            silu_mul_into(&gate, &up, &mut tmp);
            scratch.recycle(h2);
            if let Some(w) = &layer.ffn_sub_norm {
                let mut normed = scratch.take(cfg.n_ff);
                rmsnorm_into(&tmp, w, cfg.norm_eps, &mut normed);
                scratch.recycle(tmp);
                tmp = normed;
            }
            #[cfg(feature = "profile-llama")]
            let down_span = super::profile::Span::new("ffn_down");
            let y2 = layer.ffn_down.matvec_ff(&tmp, cfg.n_ff, n_embd)?;
            #[cfg(feature = "profile-llama")]
            drop(down_span);
            scratch.recycle(tmp);
            add_residual_inplace(&mut x, &y2);
        }

        if !logits_required {
            scratch.recycle(x);
            return Ok(Vec::new());
        }
        let mut xn = scratch.take(n_embd);
        rmsnorm_into(&x, &self.output_norm, cfg.norm_eps, &mut xn);
        #[cfg(feature = "profile-llama")]
        let output_span = super::profile::Span::new("output");
        let logits = self.output_logits(&xn);
        #[cfg(feature = "profile-llama")]
        drop(output_span);
        scratch.recycle(xn);
        scratch.recycle(x);
        logits
    }
}

#[cfg(test)]
mod rope_norm_tests {
    use super::{rope_inplace, rope_inv_freq_from_factors, rope_neox_inplace};

    #[test]
    fn bitnet_neox_rope_pairs_are_offset_by_half() {
        let mut row = [1.0f32, 2.0, 3.0, 4.0];
        rope_neox_inplace(&mut row, 1, 10000.0, None);
        let mut adjacent = [1.0f32, 2.0, 3.0, 4.0];
        rope_inplace(&mut adjacent, 1, 10000.0, None);
        assert_ne!(row, adjacent);
        let half = 2;
        let theta = 10000.0f32;
        let inv = 1.0 / theta.powf(0.0);
        let angle = inv;
        let (c, s) = (angle.cos(), angle.sin());
        let x0 = 1.0f32;
        let x1 = 3.0f32;
        assert!((row[0] - (x0 * c - x1 * s)).abs() < 1e-5);
        assert!((row[half] - (x0 * s + x1 * c)).abs() < 1e-5);
    }

    #[test]
    fn rope_gguf_factors_divide_analytic_frequencies() {
        let got = rope_inv_freq_from_factors(&[1.0, 2.0, 4.0, 8.0], 8, 100.0).unwrap();
        let expected = [1.0, 0.15811388, 0.025, 0.003952847];
        for (a, b) in got.iter().zip(expected) {
            assert!((a - b).abs() < 1e-7, "got {got:?}");
        }
    }

    #[test]
    fn rope_invalid_factors_are_rejected() {
        for factors in [
            [1.0, 0.0],
            [1.0, -8.0],
            [1.0, f32::NAN],
            [1.0, f32::INFINITY],
        ] {
            assert!(rope_inv_freq_from_factors(&factors, 4, 100.0).is_err());
        }
    }

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

#[cfg(test)]
mod packed_attention_tests {
    use super::score_strided_query;

    #[test]
    fn strided_scores_match_simd_dot() {
        let head_dim = 128;
        let n_kv = 5;
        let stride = n_kv * head_dim;
        let seq = 17;
        let scale = 0.125f32;
        let q: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 17) % 11) as f32 * 0.01)
            .collect();
        let k: Vec<f32> = (0..seq * stride)
            .map(|i| ((i * 13) % 7) as f32 * 0.02 - 0.05)
            .collect();
        for kv_h in 0..n_kv {
            let mut scores = vec![0.0f32; seq];
            score_strided_query(&q, &k, kv_h, head_dim, stride, seq, scale, &mut scores);
            for p in 0..seq {
                let off = p * stride + kv_h * head_dim;
                let expected = crate::ggml::simd::dot(&q, &k[off..off + head_dim]) * scale;
                let got = scores[p];
                assert!(
                    (got - expected).abs() <= 1e-5,
                    "kv={kv_h} p={p} got={got} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn fused_long_attention_matches_grouped() {
        use super::{KvCache, KvStorage, LlamaConfig};
        let n_head = 20;
        let n_kv = 5;
        let head_dim = 16;
        let seq = 300;
        let cfg = LlamaConfig {
            n_vocab: 32,
            n_embd: n_head * head_dim,
            n_layer: 1,
            n_head,
            n_kv,
            head_dim,
            rope_rot_dims: head_dim,
            n_ff: 32,
            max_seq: seq,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: true,
            sliding_window: None,
        };
        let stride = n_kv * head_dim;
        let mut cache = KvCache::new(&cfg);
        for p in 0..seq {
            for kv_h in 0..n_kv {
                for d in 0..head_dim {
                    let key = ((p * 13 + kv_h * 7 + d * 3) % 17) as f32 * 0.02 - 0.1;
                    let val = ((p * 11 + kv_h * 5 + d) % 19) as f32 * 0.03;
                    cache.k[0][p * stride + kv_h * head_dim + d] = key;
                    cache.k_major[0][(kv_h * seq + p) * head_dim + d] = key;
                    cache.v[0][p * stride + kv_h * head_dim + d] = val;
                    cache.v_major[0][(kv_h * seq + p) * head_dim + d] = val;
                }
            }
        }
        cache.rebuild_f16_major();
        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|i| ((i * 17) % 11) as f32 * 0.01)
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut got = vec![0.0f32; n_head * head_dim];
        let mut expect = vec![0.0f32; n_head * head_dim];
        super::attention_dense_grouped(
            &cache.k[0],
            &cache.v[0],
            &q,
            n_head,
            n_kv,
            head_dim,
            stride,
            seq - 1,
            scale,
            0,
            &mut expect,
        );
        let kv = KvStorage::Dense(cache);
        super::write_dense_grouped_attention(
            &kv,
            0,
            seq - 1,
            n_head,
            n_kv,
            head_dim,
            stride,
            &q,
            scale,
            0,
            &mut got,
        );
        let mut worst = 0.0f32;
        for (g, e) in got.iter().zip(&expect) {
            worst = worst.max((g - e).abs());
        }
        assert!(worst <= 2e-3, "fused attention diverged by {worst}");
    }

}
