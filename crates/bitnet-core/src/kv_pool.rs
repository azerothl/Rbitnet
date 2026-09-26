//! Global multi-sequence paged KV pool (`RBITNET_KV_POOL=1`).
//!
//! When enabled, Llama runtime KV is backed by [`SharedPhysKvStore`] pages shared across
//! requests/sequences (PagedAttention-style reclaim). Enable with:
//! - `RBITNET_KV_POOL=1` (implies paged slabs; also set `RBITNET_LLAMA_PAGED_KV=1` for clarity)
//! - optional `RBITNET_PAGED_KV_PAGE_TOKENS`, `RBITNET_PAGED_KV_MAX_PAGES`, `RBITNET_KV_POOL_MAX_SEQS`

use std::sync::{Arc, Mutex, OnceLock};

use crate::error::Result;
use crate::llama::kv_storage::{KvPoolStats, KvStorage, PagedKvPool, SharedPhysKvStore};
use crate::llama::LlamaConfig;

static GLOBAL_POOL: OnceLock<Mutex<Option<PagedKvPool>>> = OnceLock::new();

pub fn kv_pool_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_KV_POOL").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn cell() -> &'static Mutex<Option<PagedKvPool>> {
    GLOBAL_POOL.get_or_init(|| Mutex::new(None))
}

/// Ensure the process-wide pool exists (called from Llama runtime load).
pub fn ensure_global_kv_pool(cfg: &LlamaConfig) -> Result<()> {
    if !kv_pool_enabled() {
        return Ok(());
    }
    let mut g = cell()
        .lock()
        .map_err(|_| crate::error::BitNetError::Inference("kv pool lock poisoned".into()))?;
    if g.is_none() {
        *g = Some(PagedKvPool::from_env(cfg)?);
    }
    Ok(())
}

/// Build runtime KV backed by the global shared physical page store.
pub fn new_runtime_kv(cfg: &LlamaConfig) -> Result<KvStorage> {
    ensure_global_kv_pool(cfg)?;
    let g = cell()
        .lock()
        .map_err(|_| crate::error::BitNetError::Inference("kv pool lock poisoned".into()))?;
    let pool = g.as_ref().ok_or_else(|| {
        crate::error::BitNetError::Inference("RBITNET_KV_POOL set but pool not initialized".into())
    })?;
    KvStorage::new_paged_shared(
        cfg,
        pool.page_tokens(),
        pool.max_pages_per_seq(),
        pool.shared_phys(),
    )
}

pub fn open_sequence(cfg: &LlamaConfig) -> Result<u64> {
    ensure_global_kv_pool(cfg)?;
    let mut g = cell()
        .lock()
        .map_err(|_| crate::error::BitNetError::Inference("kv pool lock poisoned".into()))?;
    let pool = g.as_mut().ok_or_else(|| {
        crate::error::BitNetError::Inference("RBITNET_KV_POOL set but pool not initialized".into())
    })?;
    pool.open_sequence()
}

pub fn close_sequence(seq_id: u64) {
    if let Ok(mut g) = cell().lock() {
        if let Some(pool) = g.as_mut() {
            pool.close_sequence(seq_id);
        }
    }
}

pub fn aggregate_pool_stats() -> KvPoolStats {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(PagedKvPool::aggregate_pool_stats))
        .unwrap_or_default()
}

pub fn active_sequences() -> usize {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(PagedKvPool::active_sequences))
        .unwrap_or(0)
}

pub fn allocated_phys_pages() -> usize {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(PagedKvPool::allocated_phys_pages))
        .unwrap_or(0)
}

pub fn free_phys_pages() -> usize {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(PagedKvPool::free_phys_pages))
        .unwrap_or(0)
}

/// Free-list / allocated ratio × 1000 (prom-friendly integer gauge).
pub fn fragmentation_permille() -> u64 {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|p| (p.fragmentation_ratio() * 1000.0).round() as u64))
        .unwrap_or(0)
}

pub fn shared_phys_handle() -> Option<Arc<Mutex<SharedPhysKvStore>>> {
    cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(PagedKvPool::shared_phys))
}

/// Publish pool gauges into `bitnet-core` perf / `/metrics`.
pub fn record_pool_metrics() {
    if !kv_pool_enabled() {
        return;
    }
    crate::perf::record_kv_pool(
        active_sequences(),
        allocated_phys_pages(),
        free_phys_pages(),
        fragmentation_permille(),
    );
}

/// Drop the process-wide pool so the next `ensure_global_kv_pool` rebuilds it (tests only).
pub fn reset_global_kv_pool_for_tests() {
    if let Ok(mut g) = cell().lock() {
        *g = None;
    }
}
