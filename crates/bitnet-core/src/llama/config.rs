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
    /// RoPE applies to the first `rope_rot_dims` elements of each head; the tail `head_dim -
    /// rope_rot_dims` are left unchanged (matches `ggml_compute_forward_rope` when `n_dims < ne0`).
    pub rope_rot_dims: usize,
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

fn metadata_u32_any(archive: &GgufArchive, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| archive.metadata.get(*key).and_then(u32_val))
}

fn metadata_f32_any(archive: &GgufArchive, keys: &[&str]) -> Option<f32> {
    keys.iter()
        .find_map(|key| archive.metadata.get(*key).and_then(f32_val))
}

impl LlamaConfig {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let m = &archive.metadata;
        let h = archive.llama_hyper_params();

        let n_embd = h
            .embedding_length
            .or_else(|| metadata_u32_any(archive, &["bitnet.embedding_length"]))
            .ok_or_else(|| BitNetError::Inference("missing llama.embedding_length".into()))?
            as usize;
        let n_layer = h
            .block_count
            .or_else(|| metadata_u32_any(archive, &["bitnet.block_count"]))
            .ok_or_else(|| BitNetError::Inference("missing llama.block_count".into()))?
            as usize;
        let n_head = h
            .head_count
            .or_else(|| metadata_u32_any(archive, &["bitnet.attention.head_count"]))
            .ok_or_else(|| BitNetError::Inference("missing llama.attention.head_count".into()))?
            as usize;
        let n_kv = h
            .head_count_kv
            .or_else(|| metadata_u32_any(archive, &["bitnet.attention.head_count_kv"]))
            .map(|v| v as usize)
            .unwrap_or(n_head);
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
        if head_dim % 2 != 0 {
            return Err(BitNetError::Inference(
                "invalid head_dim: embedding_length / head_count must be even".into(),
            ));
        }

        let mut rope_rot_dims = metadata_u32_any(archive, &["llama.rope.dimension_count"])
            .map(|v| v as usize)
            .unwrap_or(head_dim)
            .min(head_dim);
        if rope_rot_dims == 0 {
            rope_rot_dims = head_dim;
        }
        if rope_rot_dims % 2 != 0 {
            rope_rot_dims = rope_rot_dims.saturating_sub(1);
        }
        if rope_rot_dims == 0 {
            rope_rot_dims = head_dim;
        }

        let n_ff = m
            .get("llama.feed_forward_length")
            .and_then(u32_val)
            .or_else(|| metadata_u32_any(archive, &["bitnet.feed_forward_length"]))
            .ok_or_else(|| BitNetError::Inference("missing llama.feed_forward_length".into()))?
            as usize;

        let n_vocab = h
            .vocab_size
            .or_else(|| metadata_u32_any(archive, &["bitnet.vocab_size"]))
            .map(|v| v as usize)
            .or_else(|| {
                archive
                    .tensor_first_of(&["token_embd.weight", "token_embd"])
                    .and_then(|t| t.dimensions.get(1).copied())
                    .map(|d| d as usize)
            });

        let n_vocab = n_vocab.ok_or_else(|| {
            BitNetError::Inference(
                "missing llama.vocab_size and token embedding tensor (token_embd.weight / token_embd)"
                    .into(),
            )
        })?;

        let rope_theta = m
            .get("llama.rope.freq_base")
            .and_then(f32_val)
            .or_else(|| metadata_f32_any(archive, &["bitnet.rope.freq_base"]))
            .unwrap_or(10_000.0);

        let norm_eps = m
            .get("llama.attention.layer_norm_rms_epsilon")
            .and_then(f32_val)
            .or_else(|| {
                metadata_f32_any(
                    archive,
                    &[
                        "bitnet.attention.layer_norm_rms_epsilon",
                        "bitnet.attention.layer_norm_epsilon",
                    ],
                )
            })
            .unwrap_or(1e-5);

        let max_seq = h
            .context_length
            .or_else(|| metadata_u32_any(archive, &["bitnet.context_length"]))
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
            rope_rot_dims,
            rope_theta,
            norm_eps,
            max_seq,
        })
    }
}
