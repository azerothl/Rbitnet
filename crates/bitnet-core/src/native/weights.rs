use crate::backend::{BackendKind, CudaDeviceQuantMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use crate::ggml::{
    ggml_row_size, ggml_type_supports_cuda_quant, matvec_payload_quant, tensor_to_f32,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Device copies are owned by one loaded runtime and shared by all its matrix views.
pub(crate) struct Weights {
    pub archive: Arc<GgufArchive>,
    resident: BTreeMap<usize, CudaDeviceQuantMatrix>,
    small: HashMap<String, Vec<f32>>,
    pub resident_bytes: usize,
}

impl Weights {
    pub fn new(archive: Arc<GgufArchive>, kind: BackendKind) -> Result<Self> {
        let mut small = HashMap::new();
        for t in &archive.tensors {
            if t.dimensions.len() == 1 || t.name.ends_with(".bias") {
                small.insert(
                    t.name.clone(),
                    tensor_to_f32(archive.tensor_payload(t)?, t.ggml_type, &t.dimensions)?,
                );
            }
        }
        let mut result = Self {
            archive,
            resident: BTreeMap::new(),
            small,
            resident_bytes: 0,
        };
        if !matches!(kind, BackendKind::Cuda | BackendKind::Hybrid) {
            return Ok(result);
        }
        let rt = CudaRuntime::try_load()
            .ok_or_else(|| BitNetError::Inference("CUDA/cuBLAS unavailable".into()))?;
        if !crate::ggml::cuda_quant_library_available() {
            return Err(BitNetError::Inference(
                "native CUDA quant library unavailable; build scripts/build_cuda_quant.ps1".into(),
            ));
        }
        let budget = std::env::var("RBITNET_HYBRID_MAX_VRAM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if kind == BackendKind::Cuda {
                12288
            } else {
                512
            })
            .saturating_mul(1024 * 1024);
        let tied_output = result.archive.tensor_by_name("output.weight").is_none();
        let mut tensors: Vec<_> = result
            .archive
            .tensors
            .iter()
            .filter(|t| {
                t.dimensions.len() >= 2
                    && (t.name != "token_embd.weight" || tied_output)
                    && !t.name.contains("nextn")
            })
            .collect();
        // Keep logits and attention/shared projections before routed experts on limited VRAM.
        tensors.sort_by_key(|t| {
            (
                if t.name == "output.weight" || (tied_output && t.name == "token_embd.weight") {
                    0
                } else if t.name.contains("_exps.") {
                    2
                } else {
                    1
                },
                t.offset,
            )
        });
        for t in tensors {
            if !ggml_type_supports_cuda_quant(t.ggml_type) || t.dimensions[1] < 128 {
                continue;
            }
            let payload = result.archive.tensor_payload(t)?;
            if result.resident_bytes.saturating_add(payload.len()) > budget {
                continue;
            }
            let cols = t.dimensions[0] as usize;
            let rows = t.dimensions[1..]
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .ok_or_else(|| BitNetError::Inference("matrix size overflow".into()))?;
            let device = CudaDeviceQuantMatrix::from_payload(
                Some(&rt),
                t.ggml_type,
                payload.to_vec(),
                rows,
                cols,
            )?;
            if device.is_device_resident() {
                result.resident_bytes += device.bytes();
                result.resident.insert(payload.as_ptr() as usize, device);
            } else {
                break;
            }
        }
        tracing::info!(
            resident_mb = result.resident_bytes / (1024 * 1024),
            matrices = result.resident.len(),
            "native quantized weight residency"
        );
        Ok(result)
    }

    pub fn tensor(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.archive
            .tensor_by_name(name)
            .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
    }
    pub fn dense(&self, name: &str) -> Result<&[f32]> {
        self.small
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| BitNetError::Inference(format!("missing small tensor `{name}`")))
    }
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dimensions.len() != 2 {
            return Err(BitNetError::Inference(format!("{name}: expected matrix")));
        }
        self.payload(
            self.archive.tensor_payload(t)?,
            t.ggml_type,
            t.dimensions[0] as usize,
            t.dimensions[1] as usize,
            x,
        )
    }
    pub fn expert(&self, name: &str, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dimensions.len() != 3 || expert >= t.dimensions[2] as usize {
            return Err(BitNetError::Inference(format!(
                "{name}: expert out of bounds"
            )));
        }
        let cols = t.dimensions[0] as usize;
        let rows = t.dimensions[1] as usize;
        let size = ggml_row_size(t.ggml_type, cols as u64)?
            .checked_mul(rows)
            .ok_or_else(|| BitNetError::Inference("expert size overflow".into()))?;
        let payload = self.archive.tensor_payload(t)?;
        self.payload(
            &payload[expert * size..(expert + 1) * size],
            t.ggml_type,
            cols,
            rows,
            x,
        )
    }

    pub fn heads(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dimensions.len() != 3 {
            return Err(BitNetError::Inference(
                "expected independent head matrices".into(),
            ));
        }
        let cols = t.dimensions[0] as usize;
        let rows = t.dimensions[1] as usize;
        let heads = t.dimensions[2] as usize;
        if x.len() != cols.saturating_mul(heads) {
            return Err(BitNetError::Inference(
                "head projection input shape mismatch".into(),
            ));
        }
        let payload = self.archive.tensor_payload(t)?;
        if let Some(matrix) = self.resident.get(&(payload.as_ptr() as usize)) {
            return matrix.matvec_batch(x, rows);
        }
        let pieces: Vec<Result<Vec<f32>>> = (0..heads)
            .into_par_iter()
            .map(|h| self.expert(name, h, &x[h * cols..(h + 1) * cols]))
            .collect();
        let mut result = Vec::with_capacity(rows * heads);
        for piece in pieces {
            result.extend(piece?);
        }
        Ok(result)
    }
    pub fn payload(
        &self,
        payload: &[u8],
        ty: u32,
        cols: usize,
        rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let pointer = payload.as_ptr() as usize;
        if let Some((&base, matrix)) = self.resident.range(..=pointer).next_back() {
            let offset = pointer - base;
            let row_bytes = ggml_row_size(ty, cols as u64)?;
            if matrix.ggml_type() == ty
                && matrix.in_cols() == cols
                && offset % row_bytes == 0
                && offset
                    .checked_add(rows.saturating_mul(row_bytes))
                    .filter(|&end| end <= matrix.bytes())
                    .is_some()
            {
                return matrix.matvec_rows(x, offset / row_bytes, rows);
            }
        }
        matvec_payload_quant(ty, payload, x, cols, rows)
    }
}
