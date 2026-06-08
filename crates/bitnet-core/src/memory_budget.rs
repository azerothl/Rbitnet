//! Optional host memory / VRAM budget checks before loading a GGUF (Ollama-style guardrails).
//!
//! The **weights** term in estimates is the **on-disk GGUF tensor payload** size (matches
//! [`RBITNET_LLAMA_WEIGHT_MODE`](crate::llama::LlamaWeightMode)=`mmap_quant` / **`auto`** when mmap-eligible:
//! weights stay quantized in the mmap). Legacy **`dense`** mode materializes full `f32` matrices at load and
//! needs far more RAM than these caps imply.

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

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

/// Enforce `RBITNET_MAX_WEIGHT_BYTES`, `RBITNET_MAX_LOAD_BYTES`, and/or `RBITNET_MAX_VRAM_MB` if set.
pub fn check_load_memory_budget(archive: &GgufArchive) -> Result<()> {
    let weights = gguf_tensor_payload_bytes(archive);
    if let Some(cap) = parse_u64_env("RBITNET_MAX_WEIGHT_BYTES") {
        if weights > cap {
            return Err(BitNetError::Inference(format!(
                "GGUF tensor payload ({weights} bytes) exceeds RBITNET_MAX_WEIGHT_BYTES ({cap})"
            )));
        }
    }

    let max_seq = budget_max_seq_from_env_or(archive);
    let kv = estimate_llama_kv_cache_bytes_f32(archive, max_seq).unwrap_or(0);
    let total = weights.saturating_add(kv);

    if let Some(cap) = parse_u64_env("RBITNET_MAX_LOAD_BYTES") {
        if total > cap {
            return Err(BitNetError::Inference(format!(
                "estimated load footprint ({total} bytes ≈ weights {weights} + KV {kv}) exceeds RBITNET_MAX_LOAD_BYTES ({cap})"
            )));
        }
    }

    if let Some(mb) = parse_u64_env("RBITNET_MAX_VRAM_MB") {
        let cap = mb.saturating_mul(1024 * 1024);
        if total > cap {
            return Err(BitNetError::Inference(format!(
                "estimated load footprint ({total} bytes) exceeds RBITNET_MAX_VRAM_MB budget ({mb} MiB)"
            )));
        }
    }

    Ok(())
}
