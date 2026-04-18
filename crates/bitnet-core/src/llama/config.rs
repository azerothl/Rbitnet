//! Hyperparameters from GGUF `llama.*` metadata plus tensor-derived checks.

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

/// Runtime shape for a Llama / Llama-compatible GGUF.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub n_ff: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub max_seq: usize,
}

fn u32_val(v: &GgufValue) -> Option<u32> {
    match v {
        GgufValue::U32(x) => Some(*x),
        GgufValue::I32(x) if *x >= 0 => Some(*x as u32),
        GgufValue::U64(x) => u32::try_from(*x).ok(),
        GgufValue::I64(x) if *x >= 0 => u32::try_from(*x).ok(),
        _ => None,
    }
}

fn f32_val(v: &GgufValue) -> Option<f32> {
    match v {
        GgufValue::F32(x) => Some(*x),
        GgufValue::F64(x) => Some(*x as f32),
        _ => None,
    }
}

impl LlamaConfig {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let m = &archive.metadata;
        let h = archive.llama_hyper_params();

        let n_embd = h
            .embedding_length
            .ok_or_else(|| BitNetError::Inference("missing llama.embedding_length".into()))?
            as usize;
        let n_layer = h
            .block_count
            .ok_or_else(|| BitNetError::Inference("missing llama.block_count".into()))?
            as usize;
        let n_head = h
            .head_count
            .ok_or_else(|| BitNetError::Inference("missing llama.attention.head_count".into()))?
            as usize;
        let n_kv = h.head_count_kv.map(|v| v as usize).unwrap_or(n_head);
        if n_head == 0 || n_kv == 0 || n_head % n_kv != 0 {
            return Err(BitNetError::Inference(
                "invalid head_count / head_count_kv".into(),
            ));
        }
        let head_dim = n_embd / n_head;
        if n_head * head_dim != n_embd {
            return Err(BitNetError::Inference(
                "embedding_length not divisible by head_count".into(),
            ));
        }

        let n_ff = m
            .get("llama.feed_forward_length")
            .and_then(u32_val)
            .ok_or_else(|| BitNetError::Inference("missing llama.feed_forward_length".into()))?
            as usize;

        let n_vocab = h.vocab_size.map(|v| v as usize).or_else(|| {
            archive
                .tensor_by_name("token_embd.weight")
                .and_then(|t| t.dimensions.get(1).copied())
                .map(|d| d as usize)
        });

        let n_vocab = n_vocab.ok_or_else(|| {
            BitNetError::Inference("missing llama.vocab_size and token_embd.weight".into())
        })?;

        let rope_theta = m
            .get("llama.rope.freq_base")
            .and_then(f32_val)
            .unwrap_or(10_000.0);

        let norm_eps = m
            .get("llama.attention.layer_norm_rms_epsilon")
            .and_then(f32_val)
            .unwrap_or(1e-5);

        let max_seq = h
            .context_length
            .map(|c| c as usize)
            .unwrap_or(2048)
            .min(8192);

        Ok(Self {
            n_embd,
            n_vocab,
            n_layer,
            n_head,
            n_kv,
            n_ff,
            head_dim,
            rope_theta,
            norm_eps,
            max_seq,
        })
    }
}
