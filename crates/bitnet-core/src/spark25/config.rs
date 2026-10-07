//! Hyperparameters for dense `general.architecture = spark2_5` GGUF files.

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

#[derive(Debug, Clone)]
pub struct Spark25Config {
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_theta_full: f32,
    pub rope_theta_swa: f32,
    /// RoPE dims for full-attention layers (`spark2_5.rope.dimension_count`).
    pub rope_dim_full: usize,
    /// RoPE dims for SWA layers (`spark2_5.rope.dimension_count_swa`).
    pub rope_dim_swa: usize,
    pub norm_eps: f32,
    pub sliding_window: usize,
    /// Per-layer SWA flag (`true` = sliding window, `false` = full attention).
    pub is_swa: Vec<bool>,
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

fn metadata_bool_array(
    md: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Option<Vec<bool>> {
    md.get(key).and_then(|v| match v {
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    GgufValue::Bool(b) => out.push(*b),
                    GgufValue::U8(x) => out.push(*x != 0),
                    GgufValue::I8(x) => out.push(*x != 0),
                    GgufValue::U32(x) => out.push(*x != 0),
                    GgufValue::I32(x) => out.push(*x != 0),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    })
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

fn default_iswa_pattern(n_layer: usize) -> Vec<bool> {
    // HF Spark-X2.5: 3 sliding + 1 full, repeating.
    (0..n_layer).map(|il| (il % 4) != 3).collect()
}

impl Spark25Config {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let md = &archive.metadata;
        let arch = archive.normalized_architecture().unwrap_or_default();
        if !matches!(
            arch.as_str(),
            "spark2_5" | "spark2-5" | "spark25" | "spark_2_5"
        ) {
            return Err(BitNetError::Inference(format!(
                "unsupported Spark architecture key `{arch}`"
            )));
        }

        let n_embd = metadata_usize(md, "spark2_5.embedding_length")
            .or_else(|| tensor_dim(archive, "token_embd.weight", 0))
            .ok_or_else(|| BitNetError::Inference("missing spark2_5.embedding_length".into()))?;
        let n_vocab = metadata_usize(md, "spark2_5.vocab_size")
            .or_else(|| tensor_dim(archive, "token_embd.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing spark2_5.vocab_size".into()))?;
        let n_layer = metadata_usize(md, "spark2_5.block_count")
            .or_else(|| infer_block_count(archive))
            .ok_or_else(|| BitNetError::Inference("missing spark2_5.block_count".into()))?;
        let n_head = metadata_usize(md, "spark2_5.attention.head_count").ok_or_else(|| {
            BitNetError::Inference("missing spark2_5.attention.head_count".into())
        })?;
        let head_dim = metadata_usize(md, "spark2_5.attention.key_length").ok_or_else(|| {
            BitNetError::Inference("missing spark2_5.attention.key_length".into())
        })?;
        if n_head == 0 || head_dim == 0 {
            return Err(BitNetError::Inference(
                "invalid spark2_5 attention head geometry".into(),
            ));
        }
        let n_kv = metadata_usize(md, "spark2_5.attention.head_count_kv").unwrap_or(n_head);
        if n_kv == 0 || n_head % n_kv != 0 {
            return Err(BitNetError::Inference(
                "invalid spark2_5 attention.head_count_kv".into(),
            ));
        }
        let n_ff = metadata_usize(md, "spark2_5.feed_forward_length")
            .or_else(|| tensor_dim(archive, "blk.0.ffn_up.weight", 1))
            .ok_or_else(|| BitNetError::Inference("missing spark2_5.feed_forward_length".into()))?;

        let qkv_out = tensor_dim(archive, "blk.0.attn_qkv.weight", 1).ok_or_else(|| {
            BitNetError::Inference("missing blk.0.attn_qkv.weight (fused QKV required)".into())
        })?;
        let expected_qkv = n_head * head_dim + 2 * n_kv * head_dim;
        if qkv_out != expected_qkv {
            return Err(BitNetError::Inference(format!(
                "spark2_5 attn_qkv width {qkv_out} != n_q+n_k+n_v {expected_qkv}"
            )));
        }
        let gate_out = tensor_dim(archive, "blk.0.attn_gate.weight", 1).ok_or_else(|| {
            BitNetError::Inference("missing blk.0.attn_gate.weight".into())
        })?;
        if gate_out != n_head {
            return Err(BitNetError::Inference(format!(
                "spark2_5 attn_gate width {gate_out} != n_head {n_head}"
            )));
        }

        let rope_theta_full = metadata_f32(md, "spark2_5.rope.freq_base").unwrap_or(5_000_000.0);
        let rope_theta_swa =
            metadata_f32(md, "spark2_5.rope.freq_base_swa").unwrap_or(10_000.0);
        let rope_dim_full = metadata_usize(md, "spark2_5.rope.dimension_count").unwrap_or(head_dim);
        let rope_dim_swa =
            metadata_usize(md, "spark2_5.rope.dimension_count_swa").unwrap_or(head_dim);
        if rope_dim_full == 0
            || rope_dim_swa == 0
            || rope_dim_full > head_dim
            || rope_dim_swa > head_dim
            || rope_dim_full % 2 != 0
            || rope_dim_swa % 2 != 0
        {
            return Err(BitNetError::Inference(
                "invalid spark2_5 rope.dimension_count(_swa)".into(),
            ));
        }

        let norm_eps = metadata_f32(md, "spark2_5.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);
        let sliding_window =
            metadata_usize(md, "spark2_5.attention.sliding_window").unwrap_or(512);
        if sliding_window == 0 {
            return Err(BitNetError::Inference(
                "spark2_5.attention.sliding_window must be > 0".into(),
            ));
        }

        let mut is_swa = metadata_bool_array(md, "spark2_5.attention.sliding_window_pattern")
            .unwrap_or_else(|| default_iswa_pattern(n_layer));
        if is_swa.len() != n_layer {
            if is_swa.len() > n_layer {
                is_swa.truncate(n_layer);
            } else {
                // Pad with the HF 3+1 pattern if the array is short.
                let pad = default_iswa_pattern(n_layer);
                is_swa.extend_from_slice(&pad[is_swa.len()..]);
            }
        }

        let max_seq = crate::context_capacity::capacity_from_env(
            metadata_usize(md, "spark2_5.context_length").unwrap_or(2048),
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
            rope_theta_full,
            rope_theta_swa,
            rope_dim_full,
            rope_dim_swa,
            norm_eps,
            sliding_window,
            is_swa,
            max_seq,
        })
    }

    pub fn layer_is_swa(&self, il: usize) -> bool {
        self.is_swa.get(il).copied().unwrap_or(true)
    }

    pub fn rope_theta_for_layer(&self, il: usize) -> f32 {
        if self.layer_is_swa(il) {
            self.rope_theta_swa
        } else {
            self.rope_theta_full
        }
    }

    pub fn rope_dim_for_layer(&self, il: usize) -> usize {
        if self.layer_is_swa(il) {
            self.rope_dim_swa
        } else {
            self.rope_dim_full
        }
    }

    pub fn kv_key_start(&self, il: usize, pos: usize) -> usize {
        if self.layer_is_swa(il) {
            (pos + 1).saturating_sub(self.sliding_window)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn write_kv_bool_array(w: &mut File, key: &str, vals: &[bool]) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 9)?; // array
        write_u32(w, 7)?; // bool element type
        write_u64(w, vals.len() as u64)?;
        for b in vals {
            w.write_all(&[u8::from(*b)])?;
        }
        Ok(())
    }
    fn write_tensor(w: &mut File, name: &str, dims: &[u64]) -> std::io::Result<()> {
        write_str(w, name)?;
        write_u32(w, dims.len() as u32)?;
        for &d in dims {
            write_u64(w, d)?;
        }
        write_u32(w, 0)?;
        write_u64(w, 0)
    }

    pub(crate) fn write_minimal_spark(path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        let kvs = 16u64;
        let tensors = 5u64;
        f.write_all(b"GGUF")?;
        write_u32(&mut f, 3)?;
        write_u64(&mut f, tensors)?;
        write_u64(&mut f, kvs)?;
        write_kv_str(&mut f, "general.architecture", "spark2_5")?;
        write_kv_u32(&mut f, "spark2_5.embedding_length", 16)?;
        write_kv_u32(&mut f, "spark2_5.vocab_size", 32)?;
        write_kv_u32(&mut f, "spark2_5.block_count", 4)?;
        write_kv_u32(&mut f, "spark2_5.attention.head_count", 2)?;
        write_kv_u32(&mut f, "spark2_5.attention.head_count_kv", 1)?;
        write_kv_u32(&mut f, "spark2_5.attention.key_length", 8)?;
        write_kv_u32(&mut f, "spark2_5.attention.value_length", 8)?;
        write_kv_u32(&mut f, "spark2_5.feed_forward_length", 24)?;
        write_kv_u32(&mut f, "spark2_5.context_length", 128)?;
        write_kv_u32(&mut f, "spark2_5.attention.sliding_window", 16)?;
        write_kv_u32(&mut f, "spark2_5.rope.dimension_count", 4)?;
        write_kv_u32(&mut f, "spark2_5.rope.dimension_count_swa", 8)?;
        write_kv_f32(&mut f, "spark2_5.rope.freq_base", 5_000_000.0)?;
        write_kv_f32(&mut f, "spark2_5.rope.freq_base_swa", 10_000.0)?;
        write_kv_bool_array(
            &mut f,
            "spark2_5.attention.sliding_window_pattern",
            &[true, true, true, false],
        )?;
        // Q=2*8=16, K=1*8=8, V=1*8=8 → 32
        write_tensor(&mut f, "token_embd.weight", &[16, 32])?;
        write_tensor(&mut f, "blk.0.attn_qkv.weight", &[16, 32])?;
        write_tensor(&mut f, "blk.0.attn_gate.weight", &[16, 2])?;
        write_tensor(&mut f, "blk.0.ffn_up.weight", &[16, 24])?;
        write_tensor(&mut f, "blk.3.attn_norm.weight", &[16])?;
        let pos = f.metadata()?.len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }

}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use std::path::Path;

    pub fn write_minimal_spark(path: &Path) -> std::io::Result<()> {
        super::tests::write_minimal_spark(path)
    }
}

#[cfg(test)]
mod config_tests {
    use super::test_fixtures::write_minimal_spark;
    use super::Spark25Config;
    use crate::gguf::GgufArchive;

    #[test]
    fn spark25_config_reads_iswa_and_dual_rope() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spark.gguf");
        write_minimal_spark(&path).unwrap();
        let archive = GgufArchive::mmap_path(&path).unwrap();
        let cfg = Spark25Config::from_gguf(&archive).unwrap();
        assert_eq!(cfg.n_embd, 16);
        assert_eq!(cfg.n_head, 2);
        assert_eq!(cfg.n_kv, 1);
        assert_eq!(cfg.head_dim, 8);
        assert_eq!(cfg.sliding_window, 16);
        assert_eq!(cfg.is_swa, vec![true, true, true, false]);
        assert!(cfg.layer_is_swa(0));
        assert!(!cfg.layer_is_swa(3));
        assert_eq!(cfg.rope_dim_for_layer(0), 8);
        assert_eq!(cfg.rope_dim_for_layer(3), 4);
        assert_eq!(cfg.kv_key_start(0, 20), 5);
        assert_eq!(cfg.kv_key_start(3, 20), 0);
    }
}
