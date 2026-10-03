//! Hyperparameters for Mixtral MoE GGUF (`general.architecture = mixtral`).

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

#[derive(Debug, Clone)]
pub struct MixtralConfig {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub max_seq: usize,
}

fn metadata_i64(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<i64> {
    md.get(key).and_then(|v| match v {
        GgufValue::U8(x) => Some(*x as i64),
        GgufValue::I8(x) => Some(*x as i64),
        GgufValue::U16(x) => Some(*x as i64),
        GgufValue::I16(x) => Some(*x as i64),
        GgufValue::U32(x) => Some(*x as i64),
        GgufValue::I32(x) => Some(*x as i64),
        GgufValue::U64(x) => i64::try_from(*x).ok(),
        GgufValue::I64(x) => Some(*x),
        _ => None,
    })
}

fn metadata_f32(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    md.get(key).and_then(|v| match v {
        GgufValue::F32(x) => Some(*x),
        GgufValue::F64(x) => Some(*x as f32),
        _ => None,
    })
}

fn metadata_usize(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<usize> {
    metadata_i64(md, key).and_then(|v| usize::try_from(v).ok())
}

fn first_usize(md: &std::collections::HashMap<String, GgufValue>, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|k| metadata_usize(md, k))
}

fn first_f32(md: &std::collections::HashMap<String, GgufValue>, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|k| metadata_f32(md, k))
}

fn infer_block_count(archive: &GgufArchive) -> Option<usize> {
    archive
        .tensors
        .iter()
        .filter_map(|t| {
            let mut parts = t.name.split('.');
            match (parts.next(), parts.next()) {
                (Some("blk"), Some(idx)) => idx.parse::<usize>().ok(),
                _ => None,
            }
        })
        .max()
        .map(|m| m + 1)
}

fn tensor_dim(archive: &GgufArchive, name: &str, dim: usize) -> Option<usize> {
    archive
        .tensor_by_name(name)
        .and_then(|t| t.dimensions.get(dim).copied())
        .and_then(|v| usize::try_from(v).ok())
}

impl MixtralConfig {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let md = &archive.metadata;
        let arch = archive.normalized_architecture().unwrap_or_default();
        if arch != "mixtral" {
            return Err(BitNetError::Inference(format!(
                "unsupported Mixtral architecture key `{arch}` (expected mixtral)"
            )));
        }

        let n_embd = first_usize(md, &["mixtral.embedding_length", "llama.embedding_length"])
            .or_else(|| tensor_dim(archive, "token_embd.weight", 0))
            .ok_or_else(|| BitNetError::Inference("missing mixtral.embedding_length".into()))?;
        let n_vocab = first_usize(md, &["mixtral.vocab_size", "llama.vocab_size"])
            .or_else(|| tensor_dim(archive, "token_embd.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing mixtral.vocab_size".into()))?;
        let n_layer = first_usize(md, &["mixtral.block_count", "llama.block_count"])
            .or_else(|| infer_block_count(archive))
            .ok_or_else(|| BitNetError::Inference("missing mixtral.block_count".into()))?;
        let n_head = first_usize(
            md,
            &["mixtral.attention.head_count", "llama.attention.head_count"],
        )
        .ok_or_else(|| BitNetError::Inference("missing mixtral.attention.head_count".into()))?;

        let q_out = tensor_dim(archive, "blk.0.attn_q.weight", 1)
            .ok_or_else(|| BitNetError::Inference("missing blk.0.attn_q.weight".into()))?;
        if n_head == 0 || q_out % n_head != 0 {
            return Err(BitNetError::Inference(
                "invalid mixtral attention.head_count / q projection shape".into(),
            ));
        }
        let head_dim = q_out / n_head;

        let n_kv = first_usize(
            md,
            &[
                "mixtral.attention.head_count_kv",
                "llama.attention.head_count_kv",
            ],
        )
        .or_else(|| tensor_dim(archive, "blk.0.attn_k.weight", 1).map(|v| v / head_dim))
        .unwrap_or(n_head);
        if n_kv == 0 || n_head % n_kv != 0 {
            return Err(BitNetError::Inference(
                "invalid mixtral attention.head_count_kv".into(),
            ));
        }

        let n_ff = first_usize(
            md,
            &["mixtral.feed_forward_length", "llama.feed_forward_length"],
        )
        .or_else(|| tensor_dim(archive, "blk.0.ffn_up_exps.weight", 1))
        .ok_or_else(|| BitNetError::Inference("missing mixtral.feed_forward_length".into()))?;

        let n_expert = first_usize(md, &["mixtral.expert_count", "llama.expert_count"])
            .or_else(|| tensor_dim(archive, "blk.0.ffn_gate_inp.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing mixtral.expert_count".into()))?;
        let n_expert_used = first_usize(
            md,
            &["mixtral.expert_used_count", "llama.expert_used_count"],
        )
        .unwrap_or(2)
        .clamp(1, n_expert.max(1));

        let rope_theta = first_f32(md, &["mixtral.rope.freq_base", "llama.rope.freq_base"])
            .unwrap_or(1_000_000.0);
        let norm_eps = first_f32(
            md,
            &[
                "mixtral.attention.layer_norm_rms_epsilon",
                "llama.attention.layer_norm_rms_epsilon",
            ],
        )
        .unwrap_or(1e-5);
        let max_seq = crate::context_capacity::capacity_from_env(
            first_usize(md, &["mixtral.context_length", "llama.context_length"]).unwrap_or(2048),
            8192,
            Some(8192),
        )?;

        Ok(Self {
            n_embd,
            n_vocab,
            n_layer,
            n_head,
            n_kv,
            head_dim,
            n_ff,
            n_expert,
            n_expert_used,
            rope_theta,
            norm_eps,
            max_seq,
        })
    }
}
