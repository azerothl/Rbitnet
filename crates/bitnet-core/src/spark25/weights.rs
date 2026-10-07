//! Quantized weight views for Spark-X2.5 — CPU mmap and optional CUDA device residency.

use std::sync::Arc;

use crate::backend::{BackendKind, CudaDeviceQuantMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use crate::ggml::{
    embedding_row_mmap, ggml_type_supports_cuda_quant, matvec_embd_out_mmap, matvec_ff_mmap,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};

use super::config::Spark25Config;

#[derive(Clone)]
pub(crate) enum SparkMatrixWeights {
    Quant {
        archive: Arc<GgufArchive>,
        tensor: GgufTensorInfo,
    },
    CudaQuant {
        device: CudaDeviceQuantMatrix,
        label: String,
    },
}

impl SparkMatrixWeights {
    pub(crate) fn matvec_embd_out(
        &self,
        x: &[f32],
        n_embd: usize,
        n_out: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Quant { archive, tensor } => {
                matvec_embd_out_mmap(archive.as_ref(), tensor, x, n_embd, n_out)
            }
            Self::CudaQuant { device, label } => device.matvec(x).map_err(|e| {
                tracing::warn!(
                    tensor = label.as_str(),
                    error = %e,
                    "spark2_5 cuda quant matvec failed"
                );
                e
            }),
        }
    }

    pub(crate) fn matvec_ff(&self, x: &[f32], n_ff: usize, n_embd: usize) -> Result<Vec<f32>> {
        match self {
            Self::Quant { archive, tensor } => {
                matvec_ff_mmap(archive.as_ref(), tensor, x, n_ff, n_embd)
            }
            Self::CudaQuant { device, label } => device.matvec(x).map_err(|e| {
                tracing::warn!(
                    tensor = label.as_str(),
                    error = %e,
                    "spark2_5 cuda quant ffn matvec failed"
                );
                e
            }),
        }
    }

    pub(crate) fn embed_row(
        &self,
        tok: usize,
        n_embd: usize,
        n_vocab: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Quant { archive, tensor } => {
                embedding_row_mmap(archive.as_ref(), tensor, tok, n_embd, n_vocab, out)
            }
            Self::CudaQuant { .. } => Err(BitNetError::Inference(
                "token embedding does not use CudaQuant residency".into(),
            )),
        }
    }
}

pub(crate) struct SparkLayerWeights {
    pub attn_norm: Vec<f32>,
    pub qkv: SparkMatrixWeights,
    pub attn_gate: SparkMatrixWeights,
    pub o: SparkMatrixWeights,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: SparkMatrixWeights,
    pub ffn_up: SparkMatrixWeights,
    pub ffn_down: SparkMatrixWeights,
}

pub(crate) struct Spark25Weights {
    pub tok_embd: SparkMatrixWeights,
    pub out_norm: Vec<f32>,
    pub out_head: SparkMatrixWeights,
    pub layers: Vec<SparkLayerWeights>,
    pub resident_bytes: usize,
    pub resident_matrices: usize,
    _cuda_rt: Option<Arc<CudaRuntime>>,
}

impl Spark25Weights {
    pub fn gpu_matvec_resident(&self) -> bool {
        self.resident_matrices > 0
    }

    pub fn load(
        archive: Arc<GgufArchive>,
        cfg: &Spark25Config,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let cuda_rt = match backend_kind {
            BackendKind::Cpu => None,
            BackendKind::Cuda | BackendKind::Hybrid => CudaRuntime::try_load(),
            BackendKind::Rocm | BackendKind::Vulkan | BackendKind::Metal => None,
        };
        if backend_kind == BackendKind::Cuda && cuda_rt.is_none() {
            return Err(BitNetError::Inference(
                "Spark-X2.5 CUDA backend requires CUDA/cuBLAS runtime".into(),
            ));
        }

        let rt_ref = cuda_rt.as_ref();
        let mut resident_bytes = 0usize;
        let mut resident_matrices = 0usize;

        let tok_embd = must_tensor(archive.as_ref(), "token_embd.weight")?;
        let out_norm = tensor_f32_vec(archive.as_ref(), &must_tensor(archive.as_ref(), "output_norm.weight")?)?;
        let out_head_tensor = resolve_lm_head(archive.as_ref(), &tok_embd)?;

        let n_q = cfg.n_head * cfg.head_dim;
        let n_kv = cfg.n_kv * cfg.head_dim;
        let qkv_out = n_q + 2 * n_kv;

        // Token embedding stays host-mmap: row gather has no CudaQuant path yet.
        let tok_embd_w = load_matrix(
            &archive,
            None,
            &tok_embd,
            cfg.n_embd,
            cfg.n_vocab,
            "token_embd.weight",
            &mut resident_bytes,
            &mut resident_matrices,
        )?;
        let out_head = load_matrix(
            &archive,
            rt_ref,
            &out_head_tensor,
            cfg.n_embd,
            cfg.n_vocab,
            "output_head",
            &mut resident_bytes,
            &mut resident_matrices,
        )?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = format!("blk.{il}");
            layers.push(SparkLayerWeights {
                attn_norm: tensor_f32_vec(
                    archive.as_ref(),
                    &must_tensor(archive.as_ref(), &format!("{p}.attn_norm.weight"))?,
                )?,
                qkv: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.attn_qkv.weight"))?,
                    cfg.n_embd,
                    qkv_out,
                    &format!("{p}.attn_qkv.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
                attn_gate: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.attn_gate.weight"))?,
                    cfg.n_embd,
                    cfg.n_head,
                    &format!("{p}.attn_gate.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
                o: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.attn_output.weight"))?,
                    n_q,
                    cfg.n_embd,
                    &format!("{p}.attn_output.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
                ffn_norm: tensor_f32_vec(
                    archive.as_ref(),
                    &must_tensor(archive.as_ref(), &format!("{p}.ffn_norm.weight"))?,
                )?,
                ffn_gate: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.ffn_gate.weight"))?,
                    cfg.n_embd,
                    cfg.n_ff,
                    &format!("{p}.ffn_gate.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
                ffn_up: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.ffn_up.weight"))?,
                    cfg.n_embd,
                    cfg.n_ff,
                    &format!("{p}.ffn_up.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
                ffn_down: load_matrix(
                    &archive,
                    rt_ref,
                    &must_tensor(archive.as_ref(), &format!("{p}.ffn_down.weight"))?,
                    cfg.n_ff,
                    cfg.n_embd,
                    &format!("{p}.ffn_down.weight"),
                    &mut resident_bytes,
                    &mut resident_matrices,
                )?,
            });
        }

        if matches!(backend_kind, BackendKind::Cuda | BackendKind::Hybrid)
            && resident_matrices == 0
        {
            tracing::warn!(
                backend = backend_kind.as_str(),
                "Spark-X2.5 CUDA/hybrid selected but no device-resident quant matrices; \
                 build native/cuda_quant (scripts/build_cuda_quant.ps1) and use Q4_K (etc.) weights"
            );
        } else if resident_matrices > 0 {
            tracing::info!(
                backend = backend_kind.as_str(),
                resident_mb = resident_bytes / (1024 * 1024),
                matrices = resident_matrices,
                "spark2_5 quantized weight residency"
            );
        }

        Ok(Self {
            tok_embd: tok_embd_w,
            out_norm,
            out_head,
            layers,
            resident_bytes,
            resident_matrices,
            _cuda_rt: cuda_rt,
        })
    }
}

fn must_tensor(archive: &GgufArchive, name: &str) -> Result<GgufTensorInfo> {
    archive
        .tensor_by_name(name)
        .cloned()
        .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
}

fn tensor_f32_vec(archive: &GgufArchive, t: &GgufTensorInfo) -> Result<Vec<f32>> {
    let payload = archive.tensor_payload(t)?;
    crate::ggml::tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn resolve_lm_head(archive: &GgufArchive, tok_embd: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for name in ["output.weight", "lm_head.weight"] {
        if let Some(t) = archive.tensor_by_name(name) {
            return Ok(t.clone());
        }
    }
    Ok(tok_embd.clone())
}

fn load_matrix(
    archive: &Arc<GgufArchive>,
    rt: Option<&Arc<CudaRuntime>>,
    tensor: &GgufTensorInfo,
    in_cols: usize,
    out_rows: usize,
    label: &str,
    resident_bytes: &mut usize,
    resident_matrices: &mut usize,
) -> Result<SparkMatrixWeights> {
    if let Some(rt) = rt {
        if ggml_type_supports_cuda_quant(tensor.ggml_type) {
            match CudaDeviceQuantMatrix::from_archive_range(
                Some(rt),
                Arc::clone(archive),
                tensor,
                0,
                out_rows,
                in_cols,
                false,
            ) {
                Ok(device) if device.is_device_resident() => {
                    *resident_bytes = resident_bytes.saturating_add(device.bytes());
                    *resident_matrices += 1;
                    return Ok(SparkMatrixWeights::CudaQuant {
                        device,
                        label: label.to_string(),
                    });
                }
                Ok(_) => {
                    tracing::debug!(
                        tensor = label,
                        "spark2_5 cuda quant matrix not device-resident; CPU mmap fallback"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        tensor = label,
                        error = %e,
                        "spark2_5 cuda quant upload failed; CPU mmap fallback"
                    );
                }
            }
        }
    }
    Ok(SparkMatrixWeights::Quant {
        archive: Arc::clone(archive),
        tensor: tensor.clone(),
    })
}

