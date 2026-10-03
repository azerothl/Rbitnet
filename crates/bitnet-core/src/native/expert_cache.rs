//! Runtime-local cache of unchanged GGUF expert slabs. Leases protect device slots.
use super::cache_policy::{Access, Policy};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
static TRACE_ID: AtomicU64 = AtomicU64::new(0);
use crate::backend::{CudaDeviceQuantMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(super) type SharedCache = Arc<Mutex<ExpertCache>>;
pub(super) struct ExpertGroup {
    pub matrices: [CudaDeviceQuantMatrix; 3],
    bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_expert_cache_protects_leases_reuses_slots_and_enforces_budget() {
        if std::env::var("RBITNET_MOE_CACHE_TEST").as_deref() != Ok("1") {
            return;
        }
        let path = std::env::var("RBITNET_TEST_GGUF").expect("real MoE GGUF required");
        let archive = Arc::new(GgufArchive::mmap_path(std::path::Path::new(&path)).unwrap());
        let rt = CudaRuntime::try_load().expect("CUDA required");
        let mut bytes = 0;
        for projection in ["gate", "up", "down"] {
            let tensor = archive
                .tensor_by_name(&format!("blk.0.ffn_{projection}_exps.weight"))
                .unwrap();
            bytes += crate::ggml::ggml_row_size(tensor.ggml_type, tensor.dimensions[0]).unwrap()
                * tensor.dimensions[1] as usize;
        }
        let mut cache = ExpertCache::new(archive, rt, 2 * bytes);
        let first = cache.acquire(0, 0).unwrap().unwrap();
        let first_addresses: Vec<_> = first.matrices.iter().map(|m| m.device_address()).collect();
        let second = cache.acquire(0, 1).unwrap().unwrap();
        let reused_addresses: Vec<_> = second.matrices.iter().map(|m| m.device_address()).collect();
        assert!(
            cache.acquire(0, 2).unwrap().is_none(),
            "both groups are leased"
        );
        assert_eq!(cache.bytes, 2 * bytes);
        drop(second);
        let replacement = cache.acquire(0, 2).unwrap().unwrap();
        assert_eq!(
            replacement
                .matrices
                .iter()
                .map(|m| m.device_address())
                .collect::<Vec<_>>(),
            reused_addresses
        );
        assert_eq!(
            first
                .matrices
                .iter()
                .map(|m| m.device_address())
                .collect::<Vec<_>>(),
            first_addresses
        );
        assert!(!cache.entries.contains_key(&(0, 1)));
        assert_eq!(cache.bytes, 2 * bytes);
        let hit = cache.acquire(0, 0).unwrap().unwrap();
        assert!(Arc::ptr_eq(&hit, &first));
        for (expert, group) in [(0, &first), (2, &replacement)] {
            for (projection, matrix) in ["gate", "up", "down"].into_iter().zip(&group.matrices) {
                let tensor = cache
                    .archive
                    .tensor_by_name(&format!("blk.0.ffn_{projection}_exps.weight"))
                    .unwrap();
                let size = matrix.bytes();
                assert_eq!(
                    matrix.host_payload(),
                    &cache.archive.tensor_payload(tensor).unwrap()
                        [expert * size..(expert + 1) * size]
                );
            }
        }
        drop(hit);
        drop(first);
        drop(replacement);
        for expert in [3, 4, 0, 2, 1, 4, 3] {
            assert!(cache.acquire(0, expert).unwrap().is_some());
            assert!(cache.bytes <= cache.budget);
        }
        // A miss earlier in router order must not evict a selected hit that
        // appears later. Weak ownership proves it was not externally leased.
        let mut pending =
            ExpertCache::new(Arc::clone(&cache.archive), Arc::clone(&cache.rt), 2 * bytes);
        pending.policy = Policy::Lru;
        drop(pending.acquire(0, 0).unwrap().unwrap());
        let original = Arc::downgrade(&pending.entries[&(0, 0)].group);
        drop(pending.acquire(0, 1).unwrap().unwrap());
        let before = crate::perf::snapshot();
        let selection = pending.acquire_selected(0, &[2, 0]).unwrap().unwrap();
        let after = crate::perf::snapshot();
        assert!(Arc::ptr_eq(&selection[1], &original.upgrade().unwrap()));
        assert!(pending.entries.contains_key(&(0, 0)));
        assert!(!pending.entries.contains_key(&(0, 1)));
        assert_eq!(after.expert_cache_misses - before.expert_cache_misses, 1);
        assert_eq!(after.expert_cache_hits - before.expert_cache_hits, 1);
        assert_eq!(
            after.expert_cache_upload_bytes - before.expert_cache_upload_bytes,
            bytes as u64
        );
        assert_eq!(pending.bytes, 2 * bytes);
        drop(selection);
        drop(pending);
        let mut tiny =
            ExpertCache::new(Arc::clone(&cache.archive), Arc::clone(&cache.rt), bytes - 1);
        assert!(tiny.acquire(0, 0).unwrap().is_none());
        assert_eq!(tiny.bytes, 0);
    }
}
struct Entry {
    group: Arc<ExpertGroup>,
    access: Access,
}
pub(super) struct ExpertCache {
    archive: Arc<GgufArchive>,
    rt: Arc<CudaRuntime>,
    entries: BTreeMap<(usize, usize), Entry>,
    budget: usize,
    bytes: usize,
    clock: u64,
    pass: u64,
    policy: Policy,
    trace: Option<BufWriter<std::fs::File>>,
    prompt_length: usize,
    position: usize,
}
impl ExpertCache {
    pub(super) fn budget_bytes(&self) -> usize {
        self.budget
    }
    pub fn flush_trace(&mut self) {
        if let Some(trace) = &mut self.trace {
            if let Err(error) = trace.flush() {
                tracing::warn!(%error, "expert trace flush failed");
            }
        }
    }
    pub fn begin_sequence(&mut self, prompt_length: usize) {
        self.prompt_length = prompt_length;
    }
    pub fn begin_pass(&mut self, position: usize) {
        self.pass = self.pass.saturating_add(1);
        self.position = position;
    }
    pub fn trace_route(&mut self, layer: usize, selected: &[usize], group_bytes: usize) {
        if let Some(trace) = &mut self.trace {
            let row = serde_json::json!({ "pass":self.pass, "position":self.position,
                "phase":if self.position < self.prompt_length { "prefill" } else { "decode" },
                "layer":layer, "selected":selected, "group_bytes":group_bytes, "budget_bytes":self.budget });
            if let Err(error) = writeln!(trace, "{row}") {
                tracing::warn!(%error, "expert trace write failed");
                self.trace = None;
            }
        }
    }
    pub fn fits(&self, bytes: usize) -> bool {
        bytes <= self.budget
    }
    pub fn new(archive: Arc<GgufArchive>, rt: Arc<CudaRuntime>, budget: usize) -> Self {
        let trace = std::env::var_os("RBITNET_MOE_TRACE_DIR").and_then(|dir| {
            let dir = std::path::PathBuf::from(dir);
            let arch = archive
                .normalized_architecture()
                .unwrap_or_else(|| "moe".into());
            let path = dir.join(format!(
                "{arch}-{}-{}.jsonl",
                std::process::id(),
                TRACE_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(dir)
                .and_then(|_| {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path)
                })
                .map(BufWriter::new)
                .map_err(|error| tracing::warn!(%error, "expert trace unavailable"))
                .ok()
        });
        Self {
            archive,
            rt,
            entries: BTreeMap::new(),
            budget,
            bytes: 0,
            clock: 0,
            pass: 0,
            policy: Policy::from_env(),
            trace,
            prompt_length: 0,
            position: 0,
        }
    }
    fn victim(&self) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .filter(|(_, e)| Arc::strong_count(&e.group) == 1)
            .min_by_key(|(key, e)| self.policy.rank(**key, e.access, self.pass))
            .map(|(&key, _)| key)
    }
    /// Protect every ready selected group before admitting any miss. Return
    /// leases in router order; allocation policy never changes that order.
    pub fn acquire_selected(
        &mut self,
        layer: usize,
        selected: &[usize],
    ) -> Result<Option<Vec<Arc<ExpertGroup>>>> {
        let pins: Vec<_> = selected
            .iter()
            .filter_map(|&expert| {
                self.entries
                    .get(&(layer, expert))
                    .map(|e| Arc::clone(&e.group))
            })
            .collect();
        let mut leases = Vec::with_capacity(selected.len());
        for &expert in selected {
            let Some(group) = self.acquire(layer, expert)? else {
                return Ok(None);
            };
            leases.push(group);
        }
        drop(pins);
        Ok(Some(leases))
    }
    /// Holds all selected experts through the synchronous FFN. If one misses
    /// and cannot fit, the caller executes the whole FFN on CPU in router order.
    pub fn acquire(&mut self, layer: usize, expert: usize) -> Result<Option<Arc<ExpertGroup>>> {
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&(layer, expert)) {
            entry.access.touched = self.clock;
            entry.access.frequency = entry.access.frequency.saturating_add(1);
            entry.access.pass = self.pass;
            crate::perf::record_expert_cache(true, 0, 0, 0);
            return Ok(Some(Arc::clone(&entry.group)));
        }
        let mut payloads = Vec::with_capacity(3);
        let mut shapes = Vec::with_capacity(3);
        let mut bytes = 0usize;
        for projection in ["gate", "up", "down"] {
            let name = format!("blk.{layer}.ffn_{projection}_exps.weight");
            let t = self
                .archive
                .tensor_by_name(&name)
                .ok_or_else(|| BitNetError::Inference(format!("missing expert bank {name}")))?;
            if t.dimensions.len() != 3 || expert >= t.dimensions[2] as usize {
                return Err(BitNetError::Inference(
                    "expert cache index out of bounds".into(),
                ));
            }
            if !crate::ggml::ggml_type_supports_cuda_quant(t.ggml_type) {
                return Ok(None);
            }
            let cols = usize::try_from(t.dimensions[0])
                .map_err(|_| BitNetError::Inference("expert columns overflow".into()))?;
            let rows = usize::try_from(t.dimensions[1])
                .map_err(|_| BitNetError::Inference("expert rows overflow".into()))?;
            let size = crate::ggml::ggml_row_size(t.ggml_type, cols as u64)?
                .checked_mul(rows)
                .ok_or_else(|| BitNetError::Inference("expert size overflow".into()))?;
            let start = size
                .checked_mul(expert)
                .ok_or_else(|| BitNetError::Inference("expert offset overflow".into()))?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| BitNetError::Inference("expert end overflow".into()))?;
            let payload = self
                .archive
                .tensor_payload(t)?
                .get(start..end)
                .ok_or_else(|| BitNetError::Inference("truncated expert bank".into()))?;
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| BitNetError::Inference("expert group size overflow".into()))?;
            payloads.push(payload);
            shapes.push((t.ggml_type, rows, cols));
        }
        if bytes > self.budget {
            crate::perf::record_expert_cache(false, 0, 0, 0);
            return Ok(None);
        }
        // Reuse a compatible, unleased allocation. The stable graph reads a
        // pointer table, so eviction never leaves a captured weight address.
        let mut reused = None;
        let mut evictions = 0;
        if self.bytes.saturating_add(bytes) > self.budget {
            while self.bytes.saturating_add(bytes) > self.budget {
                let Some(key) = self.victim() else {
                    crate::perf::record_expert_cache(false, evictions, 0, 0);
                    return Ok(None);
                };
                let old = self.entries.remove(&key).unwrap();
                self.bytes -= old.group.bytes;
                let compatible =
                    old.group
                        .matrices
                        .iter()
                        .zip(&shapes)
                        .all(|(m, &(ty, rows, cols))| {
                            m.ggml_type() == ty && m.out_rows() == rows && m.in_cols() == cols
                        });
                if reused.is_none() && compatible {
                    reused = Arc::try_unwrap(old.group).ok();
                }
                evictions += 1;
            }
        }
        let started = Instant::now();
        let group = if let Some(mut group) = reused {
            for (matrix, payload) in group.matrices.iter_mut().zip(payloads) {
                if !matrix.refill(payload.to_vec())? {
                    return Err(BitNetError::Inference(
                        "leased expert slot was reused".into(),
                    ));
                }
            }
            group
        } else {
            let mut matrices = Vec::with_capacity(3);
            for (&(ty, rows, cols), payload) in shapes.iter().zip(payloads) {
                let matrix = CudaDeviceQuantMatrix::from_expert_payload(
                    &self.rt,
                    ty,
                    payload.to_vec(),
                    rows,
                    cols,
                )?;
                if !matrix.is_device_resident() {
                    tracing::warn!(layer, expert, "expert allocation failed; CPU fallback");
                    crate::perf::record_expert_cache(
                        false,
                        evictions,
                        0,
                        started.elapsed().as_nanos() as u64,
                    );
                    return Ok(None);
                }
                matrices.push(matrix);
            }
            ExpertGroup {
                matrices: matrices.try_into().ok().unwrap(),
                bytes,
            }
        };
        let group = Arc::new(group);
        self.bytes += bytes;
        crate::perf::record_expert_cache(
            false,
            evictions,
            bytes as u64,
            started.elapsed().as_nanos() as u64,
        );
        // CudaRuntime records the three actual uploads (also for slot refills).
        self.entries.insert(
            (layer, expert),
            Entry {
                group: Arc::clone(&group),
                access: Access {
                    touched: self.clock,
                    frequency: 1,
                    pass: self.pass,
                },
            },
        );
        Ok(Some(group))
    }
}
