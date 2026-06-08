//! Hyperparameters for dense `general.architecture = qwen3` GGUF files.

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
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

impl Qwen3Config {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let md = &archive.metadata;
        let arch = archive.normalized_architecture().unwrap_or_default();
        if arch != "qwen3" {
            return Err(BitNetError::Inference(format!(
                "unsupported Qwen3 architecture key `{arch}`"
            )));
        }

        let n_embd = metadata_usize(md, "qwen3.embedding_length")
            .or_else(|| tensor_dim(archive, "token_embd.weight", 0))
            .ok_or_else(|| BitNetError::Inference("missing qwen3.embedding_length".into()))?;
        let n_vocab = metadata_usize(md, "qwen3.vocab_size")
            .or_else(|| tensor_dim(archive, "token_embd.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing qwen3.vocab_size".into()))?;
        let n_layer = metadata_usize(md, "qwen3.block_count")
            .or_else(|| infer_block_count(archive))
            .ok_or_else(|| BitNetError::Inference("missing qwen3.block_count".into()))?;
        let n_head = metadata_usize(md, "qwen3.attention.head_count")
            .ok_or_else(|| BitNetError::Inference("missing qwen3.attention.head_count".into()))?;

        let q_out = tensor_dim(archive, "blk.0.attn_q.weight", 1)
            .ok_or_else(|| BitNetError::Inference("missing blk.0.attn_q.weight".into()))?;
        if n_head == 0 || q_out % n_head != 0 {
            return Err(BitNetError::Inference(
                "invalid qwen3 attention.head_count / q projection shape".into(),
            ));
        }
        let head_dim = q_out / n_head;

        let n_kv = metadata_usize(md, "qwen3.attention.head_count_kv")
            .or_else(|| tensor_dim(archive, "blk.0.attn_k.weight", 1).map(|v| v / head_dim))
            .unwrap_or(n_head);
        if n_kv == 0 || n_head % n_kv != 0 {
            return Err(BitNetError::Inference(
                "invalid qwen3 attention.head_count_kv".into(),
            ));
        }

        let n_ff = metadata_usize(md, "qwen3.feed_forward_length")
            .or_else(|| tensor_dim(archive, "blk.0.ffn_up.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing qwen3.feed_forward_length".into()))?;

        let rope_theta = metadata_f32(md, "qwen3.rope.freq_base").unwrap_or(1_000_000.0);
        let norm_eps = metadata_f32(md, "qwen3.attention.layer_norm_rms_epsilon")
            .or_else(|| metadata_f32(md, "qwen3.attention.layer_norm_epsilon"))
            .unwrap_or(1e-6);
        let max_seq = metadata_usize(md, "qwen3.context_length")
            .unwrap_or(2048)
            .min(8192);

        Ok(Self {
            n_embd,
            n_vocab,
            n_layer,
            n_head,
            n_kv,
            head_dim,
            n_ff,
            rope_theta,
            norm_eps,
            max_seq,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    fn write_u32(w: &mut File, x: u32) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn write_u64(w: &mut File, x: u64) -> std::io::Result<()> {
        w.write_all(&x.to_le_bytes())
    }

    fn write_str(w: &mut File, s: &str) -> std::io::Result<()> {
        write_u64(w, s.len() as u64)?;
        w.write_all(s.as_bytes())
    }

    fn write_kv_str(w: &mut File, key: &str, val: &str) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 8)?;
        write_str(w, val)
    }

    fn write_kv_u32(w: &mut File, key: &str, val: u32) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 4)?;
        write_u32(w, val)
    }

    fn write_kv_f32(w: &mut File, key: &str, val: f32) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 6)?;
        w.write_all(&val.to_le_bytes())
    }

    fn write_tensor(w: &mut File, name: &str, dims: &[u64]) -> std::io::Result<()> {
        write_str(w, name)?;
        write_u32(w, dims.len() as u32)?;
        for &d in dims {
            write_u64(w, d)?;
        }
        write_u32(w, 0)?; // F32; payload is not touched by these config tests.
        write_u64(w, 0)
    }

    fn write_minimal_qwen3(path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        let kvs = 11u64;
        let tensors = 4u64;
        f.write_all(b"GGUF")?;
        write_u32(&mut f, 3)?;
        write_u64(&mut f, tensors)?;
        write_u64(&mut f, kvs)?;
        write_kv_str(&mut f, "general.architecture", "qwen3")?;
        write_kv_u32(&mut f, "qwen3.embedding_length", 16)?;
        write_kv_u32(&mut f, "qwen3.vocab_size", 32)?;
        write_kv_u32(&mut f, "qwen3.block_count", 1)?;
        write_kv_u32(&mut f, "qwen3.attention.head_count", 2)?;
        write_kv_u32(&mut f, "qwen3.attention.head_count_kv", 1)?;
        write_kv_u32(&mut f, "qwen3.feed_forward_length", 24)?;
        write_kv_u32(&mut f, "qwen3.context_length", 128)?;
        write_kv_f32(&mut f, "qwen3.rope.freq_base", 1_000_000.0)?;
        write_kv_f32(&mut f, "qwen3.attention.layer_norm_rms_epsilon", 1e-6)?;
        write_kv_u32(&mut f, "general.alignment", 32)?;
        write_tensor(&mut f, "token_embd.weight", &[16, 32])?;
        write_tensor(&mut f, "blk.0.attn_q.weight", &[16, 32])?;
        write_tensor(&mut f, "blk.0.attn_k.weight", &[16, 16])?;
        write_tensor(&mut f, "blk.0.ffn_up.weight", &[16, 24])?;
        let pos = f.metadata()?.len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }

    #[test]
    fn qwen3_config_reads_qwen_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen3.gguf");
        write_minimal_qwen3(&path).unwrap();
        let archive = GgufArchive::mmap_path(&path).unwrap();
        let cfg = Qwen3Config::from_gguf(&archive).unwrap();
        assert_eq!(cfg.n_embd, 16);
        assert_eq!(cfg.n_vocab, 32);
        assert_eq!(cfg.n_layer, 1);
        assert_eq!(cfg.n_head, 2);
        assert_eq!(cfg.n_kv, 1);
        assert_eq!(cfg.head_dim, 16);
        assert_eq!(cfg.n_ff, 24);
        assert_eq!(cfg.max_seq, 128);
    }
}
