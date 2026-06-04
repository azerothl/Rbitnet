//! Execution-time prefix KV: reuse dense KV state for shared prompt token prefixes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::llama::kv_storage::KvStorage;
use crate::prefix_kv::{PrefixKvKey, PrefixKvScope};

/// Snapshot of dense per-layer KV rows after prefill (legacy layout only).
#[derive(Debug, Clone)]
pub struct DenseKvSnapshot {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub token_count: usize,
}

/// LRU cache of dense KV snapshots keyed by scoped token prefix.
#[derive(Debug, Default)]
pub struct PrefixKvExecutionCache {
    max_entries: usize,
    inner: HashMap<PrefixKvKey, DenseKvSnapshot>,
    order: Vec<PrefixKvKey>,
}

impl PrefixKvExecutionCache {
    pub fn from_env() -> Self {
        let enabled = matches!(
            std::env::var("RBITNET_PREFIX_KV").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        let max_entries = std::env::var("RBITNET_PREFIX_KV_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(32);
        Self {
            max_entries: if enabled { max_entries } else { 0 },
            inner: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_entries > 0
    }

    pub fn lookup(&self, key: &PrefixKvKey) -> Option<&DenseKvSnapshot> {
        self.inner.get(key)
    }

    pub fn store(&mut self, key: PrefixKvKey, snap: DenseKvSnapshot) {
        if self.max_entries == 0 {
            return;
        }
        if !self.inner.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.inner.insert(key, snap);
        while self.order.len() > self.max_entries {
            if let Some(old) = self.order.first().cloned() {
                self.order.remove(0);
                self.inner.remove(&old);
            }
        }
    }
}

pub type SharedPrefixKvExecutionCache = Arc<Mutex<PrefixKvExecutionCache>>;

pub fn shared_prefix_kv_execution_cache() -> SharedPrefixKvExecutionCache {
    Arc::new(Mutex::new(PrefixKvExecutionCache::from_env()))
}

pub fn snapshot_dense_kv(kv: &KvStorage, token_count: usize) -> Option<DenseKvSnapshot> {
    match kv {
        KvStorage::Dense(d) => Some(DenseKvSnapshot {
            k: d.k.clone(),
            v: d.v.clone(),
            token_count,
        }),
        KvStorage::Paged(_) => None,
    }
}

pub fn restore_dense_kv(kv: &mut KvStorage, snap: &DenseKvSnapshot) -> bool {
    match kv {
        KvStorage::Dense(d) => {
            if d.k.len() != snap.k.len() || d.v.len() != snap.v.len() {
                return false;
            }
            for (dst, src) in d.k.iter_mut().zip(snap.k.iter()) {
                if dst.len() != src.len() {
                    return false;
                }
                dst.copy_from_slice(src);
            }
            for (dst, src) in d.v.iter_mut().zip(snap.v.iter()) {
                if dst.len() != src.len() {
                    return false;
                }
                dst.copy_from_slice(src);
            }
            true
        }
        KvStorage::Paged(_) => false,
    }
}

pub fn prefix_scope_for_runtime(
    model_id: &str,
    tokenizer_id: &str,
    template_id: &str,
    kv_format: &str,
) -> PrefixKvScope {
    PrefixKvKey::with_invalidation(model_id, tokenizer_id, template_id, kv_format, "default", &[])
        .scope()
}

pub fn snapshot_key(
    scope: &PrefixKvScope,
    token_ids: &[u32],
) -> PrefixKvKey {
    PrefixKvKey::with_invalidation(
        &scope.model_id,
        &scope.tokenizer_id,
        &scope.template_id,
        &scope.kv_format,
        &scope.rope_id,
        token_ids,
    )
}
