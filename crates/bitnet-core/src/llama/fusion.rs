//! Decode fusion op counters (Sprint 5 — fused kernels land behind these metrics).

use std::sync::atomic::{AtomicU64, Ordering};

static FUSED_NORM: AtomicU64 = AtomicU64::new(0);
static FUSED_ROPE_KV: AtomicU64 = AtomicU64::new(0);

pub fn record_fused_norm_quant() {
    FUSED_NORM.fetch_add(1, Ordering::Relaxed);
}

pub fn record_fused_rope_kv_write() {
    FUSED_ROPE_KV.fetch_add(1, Ordering::Relaxed);
}

pub fn fusion_enabled() -> bool {
    matches!(
        std::env::var("RBITNET_FUSION").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

pub fn fused_norm_count() -> u64 {
    FUSED_NORM.load(Ordering::Relaxed)
}

pub fn fused_rope_kv_count() -> u64 {
    FUSED_ROPE_KV.load(Ordering::Relaxed)
}
