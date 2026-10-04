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
    pub residency_budget_bytes: usize,
    pub state_reserve_bytes: usize,
    pub(super) expert_cache: Option<super::expert_cache::SharedCache>,
    pub(super) moe_metrics: Option<Arc<super::moe_metrics::Model>>,
}

impl Weights {
    pub fn new(archive: Arc<GgufArchive>, kind: BackendKind) -> Result<Self> {
        Self::new_with_state_reserve(archive, kind, 0)
    }

    pub fn new_with_state_reserve(
        archive: Arc<GgufArchive>,
        kind: BackendKind,
        state_reserve: usize,
    ) -> Result<Self> {
        let async_requested = match std::env::var("RBITNET_MOE_ASYNC") {
            Ok(value) if value == "1" => true,
            Ok(value) if value == "0" => false,
            Err(std::env::VarError::NotPresent) => false,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_MOE_ASYNC must be 0 or 1".into(),
                ))
            }
        };
        let arena_requested = crate::backend::expert_arena::from_env()?;
        if arena_requested && !async_requested {
            return Err(BitNetError::Inference(
                "RBITNET_MOE_ARENA requires RBITNET_MOE_ASYNC=1".into(),
            ));
        }
        if async_requested
            && (!matches!(kind, BackendKind::Cuda | BackendKind::Hybrid)
                || super::moe_cost::Execution::from_env() != super::moe_cost::Execution::Cache)
        {
            return Err(BitNetError::Inference(
                "async expert cache requires CUDA/hybrid and cache execution policy".into(),
            ));
        }
        let (async_slots, async_predictor) = if async_requested {
            let slots = match std::env::var("RBITNET_MOE_PINNED_SLOTS") {
                Ok(value) => value
                    .parse::<usize>()
                    .map_err(|_| BitNetError::Inference("invalid pinned slot count".into()))?,
                Err(std::env::VarError::NotPresent) => 2,
                Err(_) => return Err(BitNetError::Inference("invalid pinned slot count".into())),
            };
            if !(1..=2).contains(&slots) {
                return Err(BitNetError::Inference(
                    "async cache needs one or two pinned slots".into(),
                ));
            }
            let predictor = match std::env::var("RBITNET_MOE_PREFETCH") {
                Ok(value) if value == "previous-pass" => true,
                Ok(value) if value == "off" => false,
                Err(std::env::VarError::NotPresent) => false,
                _ => {
                    return Err(BitNetError::Inference(
                        "RBITNET_MOE_PREFETCH must be off or previous-pass".into(),
                    ))
                }
            };
            (slots, predictor)
        } else {
            (2, false)
        };
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
            residency_budget_bytes: 0,
            state_reserve_bytes: 0,
            expert_cache: None,
            moe_metrics: None,
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
        let requested_weights = std::env::var("RBITNET_HYBRID_MAX_VRAM_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if kind == BackendKind::Cuda {
                12288
            } else {
                512
            })
            .saturating_mul(1024 * 1024);
        let budget = if let Some(stats) = rt.managed_memory_stats().filter(|s| s.limit > 0) {
            let remaining = usize::try_from(stats.limit.saturating_sub(stats.live))
                .map_err(|_| BitNetError::Inference("CUDA remaining budget overflow".into()))?;
            if state_reserve >= remaining {
                return Err(BitNetError::Inference(format!("managed CUDA memory budget cannot reserve {state_reserve} state bytes with {remaining} remaining")));
            }
            result.state_reserve_bytes = state_reserve;
            requested_weights.min(remaining - state_reserve)
        } else {
            requested_weights
        };
        result.residency_budget_bytes = budget;
        // Reserve immutable small-router copies before bank/cache placement.
        // The ordinary reference continues to use its CPU SIMD router.
        let reserve_router = match result.archive.normalized_architecture().as_deref() {
            Some("deepseek2") => std::env::var("RBITNET_CUDA_MLA_FULL").as_deref() == Ok("1"),
            Some("gpt-oss" | "gptoss") => {
                std::env::var("RBITNET_CUDA_GPT_FULL").as_deref() == Ok("1")
            }
            _ => false,
        };
        let priority = if reserve_router {
            result
                .archive
                .tensors
                .iter()
                .filter(|t| {
                    t.name.ends_with("ffn_gate_inp.weight")
                        && t.dimensions.len() == 2
                        && t.dimensions[1] < 128
                        && ggml_type_supports_cuda_quant(t.ggml_type)
                })
                .try_fold(0usize, |n, t| {
                    n.checked_add(result.archive.tensor_payload(t)?.len())
                        .ok_or_else(|| {
                            BitNetError::Inference("resident router reservation overflow".into())
                        })
                })?
        } else {
            0
        };
        let placement_budget = budget.checked_sub(priority).ok_or_else(|| {
            BitNetError::Inference("CUDA weight budget cannot reserve resident routers".into())
        })?;
        let experts_cpu = super::moe_cost::Execution::from_env() == super::moe_cost::Execution::Cpu;
        let requested_cache = std::env::var("RBITNET_MOE_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
            .saturating_mul(1024 * 1024)
            .min(budget);
        let requested_cache = if experts_cpu { 0 } else { requested_cache };
        let dynamic_api = crate::ggml::load_cuda_quant_library().is_some_and(|lib| unsafe {
            lib.get::<unsafe extern "C" fn()>(b"rbitnet_cuda_moe_dynamic_create\0")
                .is_ok()
                && lib
                    .get::<unsafe extern "C" fn()>(b"rbitnet_cuda_moe_dynamic_step\0")
                    .is_ok()
        });
        let cache_architecture = matches!(
            result.archive.normalized_architecture().as_deref(),
            Some("gpt-oss" | "gptoss" | "deepseek2")
        );
        let cache_enabled = cache_architecture
            && requested_cache > 0
            && dynamic_api
            && std::env::var("RBITNET_CUDA_MOE").as_deref() != Ok("0");
        if async_requested && !cache_enabled {
            return Err(BitNetError::Inference("async expert cache requires a positive expert budget, supported MoE architecture and native dynamic API".into()));
        }
        if requested_cache > 0 && !cache_enabled {
            tracing::warn!("dynamic expert cache unavailable; retaining static placement");
        }
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
            if (cache_enabled || experts_cpu)
                && t.name.contains("_exps.")
                && t.dimensions.len() == 3
            {
                continue;
            }
            if !ggml_type_supports_cuda_quant(t.ggml_type) || t.dimensions[1] < 128 {
                continue;
            }
            let payload = result.archive.tensor_payload(t)?;
            if result.resident_bytes.saturating_add(payload.len()) > placement_budget {
                continue;
            }
            let cols = t.dimensions[0] as usize;
            let rows = t.dimensions[1..]
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .ok_or_else(|| BitNetError::Inference("matrix size overflow".into()))?;
            let device = CudaDeviceQuantMatrix::from_archive_range(
                Some(&rt),
                Arc::clone(&result.archive),
                t,
                0,
                rows,
                cols,
                false,
            )?;
            if device.is_device_resident() {
                result.resident_bytes += device.bytes();
                result.resident.insert(payload.as_ptr() as usize, device);
            } else {
                break;
            }
        }
        if cache_enabled {
            let cache_budget =
                requested_cache.min(placement_budget.saturating_sub(result.resident_bytes));
            if async_requested && cache_budget == 0 {
                return Err(BitNetError::Inference(
                    "no managed CUDA memory remains for the requested async expert cache".into(),
                ));
            }
            if cache_budget > 0 {
                let mut cache = super::expert_cache::ExpertCache::new(
                    Arc::clone(&result.archive),
                    rt,
                    cache_budget,
                );
                if async_requested {
                    cache.enable_async(async_slots, async_predictor)?;
                }
                result.expert_cache = Some(Arc::new(std::sync::Mutex::new(cache)));
                tracing::info!(
                    cache_mb = cache_budget / (1024 * 1024),
                    "dynamic expert cache enabled"
                );
            }
        }
        tracing::info!(
            resident_mb = result.resident_bytes / (1024 * 1024),
            state_reserve_bytes = result.state_reserve_bytes,
            weight_budget_bytes = result.residency_budget_bytes,
            matrices = result.resident.len(),
            "native quantized weight residency"
        );
        Ok(result)
    }

    pub(super) fn enable_moe_metrics(
        &mut self,
        metrics: Arc<super::moe_metrics::Model>,
    ) -> Result<()> {
        if let Some(cache) = &self.expert_cache {
            cache
                .lock()
                .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?
                .set_metrics(Arc::clone(&metrics));
        }
        self.moe_metrics = Some(metrics);
        Ok(())
    }
    pub fn tensor(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.archive
            .tensor_by_name(name)
            .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
    }
    pub(super) fn device_matrix(&self, name: &str) -> Option<CudaDeviceQuantMatrix> {
        let t = self.tensor(name).ok()?;
        let p = self.archive.tensor_payload(t).ok()?.as_ptr() as usize;
        self.resident.get(&p).cloned()
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

    /// Cost calibration must bypass resident views and CUDA host kernels.
    pub(super) fn expert_cpu(&self, name: &str, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.dimensions.len() != 3 || expert >= t.dimensions[2] as usize {
            return Err(BitNetError::Inference(format!(
                "{name}: expert out of bounds"
            )));
        }
        let cols = usize::try_from(t.dimensions[0])
            .map_err(|_| BitNetError::Inference("expert columns overflow".into()))?;
        let rows = usize::try_from(t.dimensions[1])
            .map_err(|_| BitNetError::Inference("expert rows overflow".into()))?;
        let bytes = ggml_row_size(t.ggml_type, cols as u64)?
            .checked_mul(rows)
            .ok_or_else(|| BitNetError::Inference("expert size overflow".into()))?;
        let start = expert
            .checked_mul(bytes)
            .ok_or_else(|| BitNetError::Inference("expert offset overflow".into()))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| BitNetError::Inference("expert end overflow".into()))?;
        let payload = self
            .archive
            .tensor_payload(t)?
            .get(start..end)
            .ok_or_else(|| BitNetError::Inference("expert bank truncated".into()))?;
        crate::ggml::QuantMatvecKernel::cpu_parallel().matvec_payload(
            t.ggml_type,
            payload,
            x,
            cols,
            rows,
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
