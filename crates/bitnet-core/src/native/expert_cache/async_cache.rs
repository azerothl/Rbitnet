//! Opt-in asynchronous expert cache with one physical pool and bounded pinned staging.
//! Pending copies become visible only after their completion event succeeds.
use super::*;
use crate::backend::async_upload::{CopySlot, CopyStream, PendingUpload, UnpublishedQuant};

type Key = (usize, usize);
struct Blank {
    matrices: Vec<UnpublishedQuant>,
}
struct Prediction {
    pass: u64,
    used: bool,
    bytes: usize,
}
struct InFlight {
    upload: PendingUpload,
    predicted: bool,
    wanted: bool,
    pass: u64,
}
pub(super) struct AsyncState {
    blanks: Vec<Blank>,
    slots: Vec<CopySlot>,
    pending: BTreeMap<Key, InFlight>,
    predicted: BTreeMap<Key, Prediction>,
    previous: BTreeMap<usize, Vec<usize>>,
    current: BTreeMap<usize, Vec<usize>>,
    group_bytes: usize,
    predictor: bool,
    failed: bool,
}
impl AsyncState {
    pub(super) fn create(cache: &ExpertCache, slot_count: usize, predictor: bool) -> Result<Self> {
        if !(1..=2).contains(&slot_count) {
            return Err(BitNetError::Inference(
                "async cache needs one or two pinned slots".into(),
            ));
        }
        // Rows/columns agree across layers, but formats may differ (GLM Q4/Q6).
        // Each projection slot reserves its largest physical span. Its bound
        // matrix keeps the current layer's original quantized bytes and stride.
        let first = cache
            .archive
            .tensors
            .iter()
            .find(|t| t.name.starts_with("blk.") && t.name.ends_with("ffn_gate_exps.weight"))
            .ok_or_else(|| {
                BitNetError::Inference("no routed expert banks for async cache".into())
            })?;
        let prefix = first.name.strip_suffix("ffn_gate_exps.weight").unwrap();
        let mut tensors = Vec::new();
        let mut geometry = Vec::new();
        let mut group_bytes = 0usize;
        for projection in ["gate", "up", "down"] {
            let name = format!("{prefix}ffn_{projection}_exps.weight");
            let initial = cache
                .archive
                .tensor_by_name(&name)
                .ok_or_else(|| BitNetError::Inference(format!("missing {name}")))?;
            let mut t = initial;
            if t.dimensions.len() != 3 || !crate::ggml::ggml_type_supports_cuda_quant(t.ggml_type) {
                return Err(BitNetError::Inference(
                    "async cache requires supported 3D expert banks".into(),
                ));
            }
            let cols = usize::try_from(t.dimensions[0])
                .map_err(|_| BitNetError::Inference("async columns overflow".into()))?;
            let rows = usize::try_from(t.dimensions[1])
                .map_err(|_| BitNetError::Inference("async rows overflow".into()))?;
            if cols == 0 || rows == 0 || t.dimensions[2] == 0 {
                return Err(BitNetError::Inference(
                    "async expert dimensions must be positive".into(),
                ));
            }
            let mut bytes = crate::ggml::ggml_row_size(t.ggml_type, cols as u64)?
                .checked_mul(rows)
                .ok_or_else(|| BitNetError::Inference("async group bytes overflow".into()))?;
            for other in cache.archive.tensors.iter().filter(|other| {
                other.name.starts_with("blk.")
                    && other
                        .name
                        .ends_with(&format!("ffn_{projection}_exps.weight"))
            }) {
                if !crate::ggml::ggml_type_supports_cuda_quant(other.ggml_type)
                    || other.dimensions.len() != 3
                    || other.dimensions[..2] != initial.dimensions[..2]
                    || other.dimensions[2] == 0
                {
                    return Err(BitNetError::Inference("async cache requires uniform expert rows/columns and supported original quant formats across layers".into()));
                }
                let size = crate::ggml::ggml_row_size(other.ggml_type, cols as u64)?
                    .checked_mul(rows)
                    .ok_or_else(|| {
                        BitNetError::Inference("async projection capacity overflow".into())
                    })?;
                if size > bytes {
                    bytes = size;
                    t = other;
                }
            }
            group_bytes = group_bytes
                .checked_add(bytes)
                .ok_or_else(|| BitNetError::Inference("async pool bytes overflow".into()))?;
            tensors.push(t);
            geometry.push((t.ggml_type, rows, cols));
        }
        let count = cache.budget / group_bytes;
        if count == 0 {
            return Err(BitNetError::Inference(
                "async cache budget cannot hold one expert group".into(),
            ));
        }
        let channel = CopyStream::new(Arc::clone(&cache.rt))?.ok_or_else(|| {
            BitNetError::Inference("CUDA pinned/event transfer API unavailable".into())
        })?;
        let slots = (0..slot_count)
            .map(|_| CopySlot::new(Arc::clone(&channel), group_bytes))
            .collect::<Result<Vec<_>>>()?;
        let mut blanks = Vec::with_capacity(count);
        for _ in 0..count {
            let mut matrices = Vec::with_capacity(3);
            for (t, &(_, rows, cols)) in tensors.iter().zip(&geometry) {
                let matrix = UnpublishedQuant::allocate(
                    &cache.rt,
                    Arc::clone(&cache.archive),
                    t,
                    0,
                    rows,
                    cols,
                )?
                .ok_or_else(|| {
                    BitNetError::Inference(
                        "managed CUDA budget cannot preallocate async cache pool".into(),
                    )
                })?;
                matrices.push(matrix);
            }
            blanks.push(Blank { matrices });
        }
        Ok(Self {
            blanks,
            slots,
            pending: BTreeMap::new(),
            predicted: BTreeMap::new(),
            previous: BTreeMap::new(),
            current: BTreeMap::new(),
            group_bytes,
            predictor,
            failed: false,
        })
    }
    fn metric(
        cache: &ExpertCache,
        layer: usize,
        update: impl FnOnce(&super::super::moe_metrics::Layer),
    ) {
        if let Some(l) = cache.metrics.as_ref().and_then(|m| m.layer(layer)) {
            update(l);
        }
    }
    fn pending_gauge(cache: &ExpertCache, layer: usize, bytes: usize, add: bool) {
        Self::metric(cache, layer, |l| {
            if add {
                l.pending_experts.fetch_add(1, Ordering::Relaxed);
                l.pending_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            } else {
                l.pending_experts.fetch_sub(1, Ordering::Relaxed);
                l.pending_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
            }
        });
    }
    fn unused(&mut self, cache: &ExpertCache, key: Key) {
        if let Some(p) = self.predicted.remove(&key) {
            if !p.used {
                Self::metric(cache, key.0, |l| {
                    l.prefetch_unused.fetch_add(1, Ordering::Relaxed);
                    l.prefetch_wasted_bytes
                        .fetch_add(p.bytes as u64, Ordering::Relaxed);
                });
            }
        }
    }
    fn upload_metrics(cache: &ExpertCache, layer: usize, bytes: u64, ns: u64) {
        crate::perf::record_expert_cache_background(bytes, ns, 0);
        Self::metric(cache, layer, |l| {
            l.upload_bytes.fetch_add(bytes, Ordering::Relaxed);
            l.upload_ns.fetch_add(ns, Ordering::Relaxed);
        });
    }
    pub(super) fn poison(&mut self, cache: &mut ExpertCache) {
        self.failed = true;
        self.pending_gauges_clear(cache);
        self.pending.clear(); // each owner drains before releasing its buffers
        self.predicted.clear();
        self.previous.clear();
        self.current.clear();
        cache.bytes =
            cache.entries.values().map(|e| e.group.bytes).sum::<usize>() + self.reserved();
        self.pool_gauges(cache);
    }
    fn pool_gauges(&self, cache: &ExpertCache) {
        if let Some(metrics) = &cache.metrics {
            let slots = self.slots.len() + self.pending.len();
            metrics.async_pool(
                cache.bytes,
                slots * self.group_bytes,
                slots,
                true,
                self.failed,
            );
        }
    }
    pub(super) fn attach_gauges(&self, cache: &ExpertCache) {
        self.pool_gauges(cache);
        for key in self.pending.keys() {
            Self::pending_gauge(cache, key.0, self.group_bytes, true);
        }
    }
    pub(super) fn begin_sequence(&mut self, cache: &ExpertCache) {
        self.previous.clear();
        self.current.clear();
        let keys: Vec<_> = self.predicted.keys().copied().collect();
        for key in keys {
            self.unused(cache, key);
        }
        for pending in self.pending.values_mut() {
            pending.wanted = false;
        }
    }
    pub(super) fn begin_pass(&mut self, cache: &ExpertCache) {
        self.previous = std::mem::take(&mut self.current);
        let expired: Vec<_> = self
            .predicted
            .iter()
            .filter(|(_, p)| p.pass < cache.pass)
            .map(|(&key, _)| key)
            .collect();
        for key in expired {
            self.unused(cache, key);
            if let Some(p) = self.pending.get_mut(&key) {
                p.wanted = false;
            }
        }
    }
    pub(super) fn routed(&mut self, cache: &ExpertCache, layer: usize, selected: &[usize]) {
        self.current.insert(layer, selected.to_vec());
        for &expert in selected {
            if let Some(p) = self.pending.get_mut(&(layer, expert)) {
                p.wanted = true;
            }
        }
        let unused: Vec<_> = self
            .predicted
            .keys()
            .filter(|&&(il, e)| il == layer && !selected.contains(&e))
            .copied()
            .collect();
        for key in unused {
            self.unused(cache, key);
            if let Some(p) = self.pending.get_mut(&key) {
                p.wanted = false;
            }
        }
    }
    fn publish(
        &mut self,
        cache: &mut ExpertCache,
        key: Key,
        flight: InFlight,
    ) -> Result<(Arc<ExpertGroup>, Upload)> {
        let completed = flight.upload.complete();
        Self::pending_gauge(cache, key.0, self.group_bytes, false);
        let completed = completed?;
        Self::metric(cache, key.0, |l| {
            l.copy_dma_ns.fetch_add(completed.dma_ns, Ordering::Relaxed);
            l.staging_ns
                .fetch_add(completed.stage_ns, Ordering::Relaxed);
        });
        Self::upload_metrics(
            cache,
            key.0,
            completed.bytes,
            completed.stage_ns.saturating_add(completed.dma_ns),
        );
        let upload = Upload {
            bytes: completed.bytes,
            ns: completed.wait_ns,
        };
        self.slots.push(completed.slot);
        let matrices: [CudaDeviceQuantMatrix; 3] = completed
            .matrices
            .try_into()
            .map_err(|_| BitNetError::Inference("async expert projection count mismatch".into()))?;
        let group = Arc::new(ExpertGroup {
            matrices,
            bytes: self.group_bytes,
        });
        if !flight.wanted {
            let owned = Arc::try_unwrap(group).ok().unwrap();
            self.blanks.push(Blank {
                matrices: owned
                    .matrices
                    .into_iter()
                    .map(UnpublishedQuant::recycle)
                    .collect::<Result<_>>()?,
            });
            return Err(BitNetError::Inference(
                "unused async upload cannot be published".into(),
            ));
        }
        Self::metric(cache, key.0, |l| {
            l.ready_experts.fetch_add(1, Ordering::Relaxed);
            l.ready_bytes
                .fetch_add(self.group_bytes as u64, Ordering::Relaxed);
        });
        cache.entries.insert(
            key,
            Entry {
                group: Arc::clone(&group),
                access: Access {
                    touched: cache.clock,
                    frequency: u64::from(!flight.predicted),
                    pass: flight.pass,
                },
            },
        );
        Ok((group, upload))
    }
    fn reap(&mut self, cache: &mut ExpertCache) -> Result<()> {
        let mut ready = Vec::new();
        for (&key, flight) in &self.pending {
            if flight.upload.ready()? {
                ready.push(key);
            }
        }
        for key in ready {
            let flight = self.pending.remove(&key).unwrap();
            if flight.wanted {
                drop(self.publish(cache, key, flight)?);
            } else {
                // Complete/discard without publishing private device buffers.
                let completed = flight.upload.complete();
                Self::pending_gauge(cache, key.0, self.group_bytes, false);
                let completed = completed?;
                self.slots.push(completed.slot);
                Self::metric(cache, key.0, |l| {
                    l.copy_dma_ns.fetch_add(completed.dma_ns, Ordering::Relaxed);
                    l.staging_ns
                        .fetch_add(completed.stage_ns, Ordering::Relaxed);
                });
                Self::upload_metrics(
                    cache,
                    key.0,
                    completed.bytes,
                    completed.stage_ns.saturating_add(completed.dma_ns),
                );
                self.blanks.push(Blank {
                    matrices: completed
                        .matrices
                        .into_iter()
                        .map(UnpublishedQuant::recycle)
                        .collect::<Result<_>>()?,
                });
            }
        }
        Ok(())
    }
    fn blank(&mut self, cache: &mut ExpertCache) -> Result<Option<Blank>> {
        if let Some(blank) = self.blanks.pop() {
            return Ok(Some(blank));
        }
        let Some(key) = cache.victim() else {
            return Ok(None);
        };
        self.unused(cache, key);
        let entry = cache.entries.remove(&key).unwrap();
        let owned = Arc::try_unwrap(entry.group)
            .map_err(|_| BitNetError::Inference("async victim unexpectedly leased".into()))?;
        cache.evicted(key.0, owned.bytes);
        crate::perf::record_expert_cache_background(0, 0, 1);
        Ok(Some(Blank {
            matrices: owned
                .matrices
                .into_iter()
                .map(UnpublishedQuant::recycle)
                .collect::<Result<_>>()?,
        }))
    }
    fn stage(&mut self, cache: &mut ExpertCache, key: Key, predicted: bool) -> Result<bool> {
        if self.slots.is_empty() {
            Self::metric(cache, key.0, |l| {
                l.staging_refusals.fetch_add(1, Ordering::Relaxed);
            });
            return Ok(false);
        }
        let Some(mut blank) = self.blank(cache)? else {
            cache.record_capacity_refusal(key.0);
            return Ok(false);
        };
        for (projection, matrix) in ["gate", "up", "down"].into_iter().zip(&mut blank.matrices) {
            let name = format!("blk.{}.ffn_{projection}_exps.weight", key.0);
            let t = cache
                .archive
                .tensor_by_name(&name)
                .ok_or_else(|| BitNetError::Inference(format!("missing {name}")))?;
            if t.dimensions.len() != 3 || key.1 >= t.dimensions[2] as usize {
                return Err(BitNetError::Inference(
                    "async expert index out of bounds".into(),
                ));
            }
            let logical = crate::ggml::ggml_row_size(t.ggml_type, t.dimensions[0])?
                .checked_mul(
                    usize::try_from(t.dimensions[1])
                        .map_err(|_| BitNetError::Inference("async row count overflow".into()))?,
                )
                .ok_or_else(|| {
                    BitNetError::Inference("async logical expert bytes overflow".into())
                })?;
            let start = logical
                .checked_mul(key.1)
                .ok_or_else(|| BitNetError::Inference("async expert offset overflow".into()))?;
            matrix.bind(Arc::clone(&cache.archive), t, start)?;
        }
        let logical_bytes = blank.matrices.iter().map(UnpublishedQuant::bytes).sum();
        let slot = self.slots.pop().unwrap();
        let upload = slot.submit(blank.matrices)?;
        Self::pending_gauge(cache, key.0, self.group_bytes, true);
        self.pending.insert(
            key,
            InFlight {
                upload,
                predicted,
                wanted: true,
                pass: cache.pass,
            },
        );
        if predicted {
            self.predicted.insert(
                key,
                Prediction {
                    pass: cache.pass,
                    used: false,
                    bytes: logical_bytes,
                },
            );
            Self::metric(cache, key.0, |l| {
                l.prefetch_requested.fetch_add(1, Ordering::Relaxed);
            });
        }
        Ok(true)
    }
    fn used(&mut self, cache: &ExpertCache, key: Key, late: bool, wait_ns: u64) {
        if let Some(p) = self.predicted.get_mut(&key) {
            if !p.used {
                p.used = true;
                Self::metric(cache, key.0, |l| {
                    l.prefetch_used.fetch_add(1, Ordering::Relaxed);
                    if late {
                        l.prefetch_late.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        }
        Self::metric(cache, key.0, |l| {
            l.selected_wait_ns.fetch_add(wait_ns, Ordering::Relaxed);
        });
    }
    pub(super) fn acquire(
        &mut self,
        cache: &mut ExpertCache,
        key: Key,
    ) -> Result<Option<(Arc<ExpertGroup>, Upload)>> {
        if self.failed {
            return Err(BitNetError::Inference(
                "async cache failed; reload the model before retrying".into(),
            ));
        }
        self.reap(cache)?;
        cache.clock = cache.clock.wrapping_add(1);
        if let Some(entry) = cache.entries.get_mut(&key) {
            entry.access.touched = cache.clock;
            entry.access.frequency = entry.access.frequency.saturating_add(1);
            entry.access.pass = cache.pass;
            let group = Arc::clone(&entry.group);
            self.used(cache, key, false, 0);
            cache.record(key.0, true, 0, 0, 0);
            return Ok(Some((group, Upload::default())));
        }
        cache.record(key.0, false, 0, 0, 0);
        if let Some(flight) = self.pending.get_mut(&key) {
            flight.wanted = true;
        } else if !self.stage(cache, key, false)? {
            return Ok(None);
        }
        // Query while the entry is still registered. An event error must leave
        // its pending gauge visible to poison(), which drains and removes it.
        let registered = self.pending.get(&key).unwrap();
        let late = registered.predicted && !registered.upload.ready()?;
        let flight = self.pending.remove(&key).unwrap();
        let result = self.publish(cache, key, flight)?;
        self.used(cache, key, late, result.1.ns);
        Ok(Some(result))
    }
    /// Complete selected pending entries before the caller pins ready entries.
    /// An earlier demand miss must never evict a later selected pending hit.
    pub(super) fn prepare_selected(
        &mut self,
        cache: &mut ExpertCache,
        layer: usize,
        selected: &[usize],
    ) -> Result<Upload> {
        if self.failed {
            return Err(BitNetError::Inference(
                "async cache failed; reload the model before retrying".into(),
            ));
        }
        let mut total = Upload::default();
        for &expert in selected {
            let key = (layer, expert);
            if let Some(registered) = self.pending.get(&key) {
                let late = registered.predicted && !registered.upload.ready()?;
                let mut flight = self.pending.remove(&key).unwrap();
                flight.wanted = true;
                let (group, upload) = self.publish(cache, key, flight)?;
                self.used(cache, key, late, upload.ns);
                drop(group);
                total.bytes = total.bytes.saturating_add(upload.bytes);
                total.ns = total.ns.saturating_add(upload.ns);
            }
        }
        Ok(total)
    }
    pub(super) fn prefetch_next(&mut self, cache: &mut ExpertCache, layer: usize) -> Result<()> {
        if self.failed {
            return Err(BitNetError::Inference(
                "async cache failed; reload the model before retrying".into(),
            ));
        }
        if !self.predictor {
            return Ok(());
        }
        self.reap(cache)?;
        let Some(next) = layer.checked_add(1) else {
            return Ok(());
        };
        let ids = self.previous.get(&next).cloned().unwrap_or_default();
        for expert in ids {
            let key = (next, expert);
            if cache.entries.contains_key(&key) || self.pending.contains_key(&key) {
                continue;
            }
            if !self.stage(cache, key, true)? {
                break;
            }
        }
        Ok(())
    }
    pub(super) fn reserved(&self) -> usize {
        (self.blanks.len() + self.pending.len()) * self.group_bytes
    }
    pub(super) fn pending_gauges_clear(&self, cache: &ExpertCache) {
        for key in self.pending.keys() {
            Self::pending_gauge(cache, key.0, self.group_bytes, false);
        }
    }
}
impl ExpertCache {
    pub(in crate::native) fn enable_async(&mut self, slots: usize, predictor: bool) -> Result<()> {
        if self.async_state.is_some() || !self.entries.is_empty() {
            return Err(BitNetError::Inference(
                "async pool must initialize before any acquisition".into(),
            ));
        }
        if super::super::moe_cost::Execution::from_env() != super::super::moe_cost::Execution::Cache
        {
            return Err(BitNetError::Inference(
                "async cache currently requires RBITNET_MOE_EXECUTION=cache".into(),
            ));
        }
        let state = AsyncState::create(self, slots, predictor)?;
        self.bytes = state.reserved();
        state.pool_gauges(self);
        self.async_state = Some(state);
        Ok(())
    }
    pub(in crate::native) fn prefetch_next(&mut self, layer: usize) -> Result<()> {
        let Some(mut state) = self.async_state.take() else {
            return Ok(());
        };
        let result = state.prefetch_next(self, layer);
        if result.is_err() {
            state.poison(self);
        }
        self.async_state = Some(state);
        result
    }
}

#[cfg(test)]
mod tests;
