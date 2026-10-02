//! Optional host memory / VRAM budget checks before loading a GGUF (Ollama-style guardrails).
//!
//! The **weights** term in estimates is the **on-disk GGUF tensor payload** size (matches
//! [`RBITNET_LLAMA_WEIGHT_MODE`](crate::llama::LlamaWeightMode)=`mmap_quant` / **`auto`** when mmap-eligible:
//! weights stay quantized in the mmap). Legacy **`dense`** mode materializes full `f32` matrices at load and
//! needs far more RAM than these caps imply.
//!
//! This is **not** akasha-os `aos-placement` (no mid-token migrate, no FFI). Budgets refuse load clearly
//! so ops can retry (`POST /v1/admin/reload`) without hanging.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

static BUDGET_CHECKS: AtomicU64 = AtomicU64::new(0);
static BUDGET_REFUSALS: AtomicU64 = AtomicU64::new(0);
static LAST_ESTIMATED_LOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_WEIGHT_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_KV_ESTIMATE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes in the GGUF tensor payload section (weights + any trailing padding in that region).
#[must_use]
pub fn gguf_tensor_payload_bytes(archive: &GgufArchive) -> u64 {
    archive.tensor_data().len() as u64
}

/// Rough peak KV cache size for Llama-shaped models (F32 K/V per layer), using GGUF `llama.*` metadata.
/// Returns `None` if required keys are missing (e.g. `qwen35moe` without `llama.block_count`).
#[must_use]
pub fn estimate_llama_kv_cache_bytes_f32(
    archive: &GgufArchive,
    max_seq_tokens: u32,
) -> Option<u64> {
    let h = archive.llama_hyper_params();
    let n_layer = u64::from(h.block_count?);
    let n_embd = u64::from(h.embedding_length?);
    let n_head = u64::from(h.head_count?);
    let n_kv = u64::from(h.head_count_kv.unwrap_or(h.head_count?));
    if n_head == 0 {
        return None;
    }
    let head_dim = n_embd / n_head;
    let max_seq = u64::from(max_seq_tokens);
    // K and V: 2 * layers * seq * n_kv * head_dim * sizeof(f32)
    let bytes = 2u64
        .saturating_mul(n_layer)
        .saturating_mul(max_seq)
        .saturating_mul(n_kv)
        .saturating_mul(head_dim)
        .saturating_mul(4);
    Some(bytes)
}

fn parse_u64_env(key: &str) -> Option<u64> {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|&v| v > 0)
}

fn budget_max_seq_from_env_or(archive: &GgufArchive) -> u32 {
    if let Some(v) = parse_u64_env("RBITNET_BUDGET_MAX_SEQ").and_then(|x| u32::try_from(x).ok()) {
        return v.max(1);
    }
    archive
        .llama_hyper_params()
        .context_length
        .unwrap_or(8192)
        .max(1)
}

/// Pure budget comparison (testable without a GGUF).
pub fn evaluate_load_budget(
    weights: u64,
    kv: u64,
    max_weight_bytes: Option<u64>,
    max_load_bytes: Option<u64>,
    max_vram_mb: Option<u64>,
) -> std::result::Result<u64, String> {
    let total = weights.saturating_add(kv);
    if let Some(cap) = max_weight_bytes {
        if weights > cap {
            return Err(format!(
                "GGUF tensor payload ({weights} bytes) exceeds RBITNET_MAX_WEIGHT_BYTES ({cap})"
            ));
        }
    }
    if let Some(cap) = max_load_bytes {
        if total > cap {
            return Err(format!(
                "estimated load footprint ({total} bytes ≈ weights {weights} + KV {kv}) exceeds RBITNET_MAX_LOAD_BYTES ({cap})"
            ));
        }
    }
    if let Some(mb) = max_vram_mb {
        let cap = mb.saturating_mul(1024 * 1024);
        if total > cap {
            return Err(format!(
                "estimated load footprint ({total} bytes) exceeds RBITNET_MAX_VRAM_MB budget ({mb} MiB)"
            ));
        }
    }
    Ok(total)
}

/// Enforce `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, and/or `RBITNET_MAX_VRAM_MB` if set.
pub fn check_load_memory_budget(archive: &GgufArchive) -> Result<()> {
    BUDGET_CHECKS.fetch_add(1, Ordering::Relaxed);
    let weights = gguf_tensor_payload_bytes(archive);
    let max_seq = budget_max_seq_from_env_or(archive);
    let kv = estimate_llama_kv_cache_bytes_f32(archive, max_seq).unwrap_or(0);
    LAST_WEIGHT_BYTES.store(weights, Ordering::Relaxed);
    LAST_KV_ESTIMATE_BYTES.store(kv, Ordering::Relaxed);

    match evaluate_load_budget(
        weights,
        kv,
        parse_u64_env("RBITNET_MAX_WEIGHT_BYTES"),
        parse_u64_env("RBITNET_MAX_LOAD_BYTES"),
        parse_u64_env("RBITNET_MAX_VRAM_MB"),
    ) {
        Ok(total) => {
            LAST_ESTIMATED_LOAD_BYTES.store(total, Ordering::Relaxed);
            tracing::info!(
                weights_bytes = weights,
                kv_estimate_bytes = kv,
                estimated_load_bytes = total,
                "memory budget check OK"
            );
            Ok(())
        }
        Err(msg) => {
            BUDGET_REFUSALS.fetch_add(1, Ordering::Relaxed);
            LAST_ESTIMATED_LOAD_BYTES.store(weights.saturating_add(kv), Ordering::Relaxed);
            tracing::warn!(%msg, "memory budget over limit — refusing load (no hang)");
            Err(BitNetError::Inference(format!(
                "{msg} — load refused (over budget); fix caps or use a smaller GGUF, then POST /v1/admin/reload"
            )))
        }
    }
}

/// Prometheus text for budget guardrails (appended by the HTTP server scrape).
#[must_use]
pub fn prometheus_text() -> String {
    let mut s = String::new();
    let checks = BUDGET_CHECKS.load(Ordering::Relaxed);
    let refusals = BUDGET_REFUSALS.load(Ordering::Relaxed);
    let est = LAST_ESTIMATED_LOAD_BYTES.load(Ordering::Relaxed);
    let w = LAST_WEIGHT_BYTES.load(Ordering::Relaxed);
    let kv = LAST_KV_ESTIMATE_BYTES.load(Ordering::Relaxed);
    s.push_str("# HELP rbitnet_core_memory_budget_checks_total Load-time memory budget evaluations\n");
    s.push_str("# TYPE rbitnet_core_memory_budget_checks_total counter\n");
    s.push_str(&format!("rbitnet_core_memory_budget_checks_total {checks}\n"));
    s.push_str("# HELP rbitnet_core_memory_budget_refusals_total Loads refused for exceeding RAM/VRAM budgets\n");
    s.push_str("# TYPE rbitnet_core_memory_budget_refusals_total counter\n");
    s.push_str(&format!("rbitnet_core_memory_budget_refusals_total {refusals}\n"));
    s.push_str("# HELP rbitnet_core_memory_budget_estimated_load_bytes Last estimated weights+KV footprint at budget check\n");
    s.push_str("# TYPE rbitnet_core_memory_budget_estimated_load_bytes gauge\n");
    s.push_str(&format!("rbitnet_core_memory_budget_estimated_load_bytes {est}\n"));
    s.push_str("# HELP rbitnet_core_memory_budget_weight_bytes Last GGUF tensor payload size at budget check\n");
    s.push_str("# TYPE rbitnet_core_memory_budget_weight_bytes gauge\n");
    s.push_str(&format!("rbitnet_core_memory_budget_weight_bytes {w}\n"));
    s.push_str("# HELP rbitnet_core_memory_budget_kv_estimate_bytes Last F32 KV estimate used in budget check\n");
    s.push_str("# TYPE rbitnet_core_memory_budget_kv_estimate_bytes gauge\n");
    s.push_str(&format!("rbitnet_core_memory_budget_kv_estimate_bytes {kv}\n"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuse_when_weights_over_cap() {
        let err = evaluate_load_budget(100, 0, Some(50), None, None).unwrap_err();
        assert!(err.contains("RBITNET_MAX_WEIGHT_BYTES"), "{err}");
    }

    #[test]
    fn refuse_when_total_over_load_cap() {
        let err = evaluate_load_budget(80, 40, None, Some(100), None).unwrap_err();
        assert!(err.contains("RBITNET_MAX_LOAD_BYTES"), "{err}");
    }

    #[test]
    fn refuse_when_over_vram_mb() {
        // 2 MiB total vs 1 MiB cap
        let err = evaluate_load_budget(1024 * 1024, 1024 * 1024, None, None, Some(1)).unwrap_err();
        assert!(err.contains("RBITNET_MAX_VRAM_MB"), "{err}");
    }

    #[test]
    fn ok_under_all_caps() {
        let total = evaluate_load_budget(10, 5, Some(100), Some(100), Some(1)).unwrap();
        assert_eq!(total, 15);
    }
}
