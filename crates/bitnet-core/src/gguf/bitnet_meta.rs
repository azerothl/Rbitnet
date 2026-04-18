//! Llama-compatible hyperparameters extracted from GGUF metadata (BitNet uses the same layout names).

use super::parse::{GgufArchive, GgufValue};

/// Common LLaMA-like fields used to size a transformer graph.
#[derive(Debug, Clone, Default)]
pub struct LlamaHyperParams {
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub block_count: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub vocab_size: Option<u32>,
}

fn u32_from_value(v: &GgufValue) -> Option<u32> {
    match v {
        GgufValue::U32(x) => Some(*x),
        GgufValue::I32(x) if *x >= 0 => Some(*x as u32),
        GgufValue::U64(x) => u32::try_from(*x).ok(),
        GgufValue::I64(x) if *x >= 0 => u32::try_from(*x).ok(),
        _ => None,
    }
}

impl GgufArchive {
    /// Best-effort parse of `llama.*` keys (also works when architecture is BitNet-llama).
    pub fn llama_hyper_params(&self) -> LlamaHyperParams {
        let mut h = LlamaHyperParams::default();
        let m = &self.metadata;
        if let Some(v) = m.get("llama.context_length") {
            h.context_length = u32_from_value(v);
        }
        if let Some(v) = m.get("llama.embedding_length") {
            h.embedding_length = u32_from_value(v);
        }
        if let Some(v) = m.get("llama.block_count") {
            h.block_count = u32_from_value(v);
        }
        if let Some(v) = m.get("llama.attention.head_count") {
            h.head_count = u32_from_value(v);
        }
        if let Some(v) = m.get("llama.attention.head_count_kv") {
            h.head_count_kv = u32_from_value(v);
        }
        if let Some(v) = m.get("llama.vocab_size") {
            h.vocab_size = u32_from_value(v);
        }
        h
    }
}
