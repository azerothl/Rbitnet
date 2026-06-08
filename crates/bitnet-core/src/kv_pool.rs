//! Global multi-sequence paged KV pool (`RBITNET_KV_POOL=1`).

use std::sync::{Mutex, OnceLock};

use crate::error::Result;
use crate::llama::kv_storage::{KvPoolStats, PagedKvPool};
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
    let mut g = cell().lock().map_err(|_| crate::error::BitNetError::Inference("kv pool lock poisoned".into()))?;
    if g.is_none() {
        *g = Some(PagedKvPool::from_env(cfg)?);
    }
    Ok(())
}

pub fn open_sequence(cfg: &LlamaConfig) -> Result<u64> {
    ensure_global_kv_pool(cfg)?;
    let mut g = cell().lock().map_err(|_| crate::error::BitNetError::Inference("kv pool lock poisoned".into()))?;
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
