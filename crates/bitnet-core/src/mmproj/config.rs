//! Vision mmproj hyperparameters loaded from a CLIP / LLaVA GGUF sidecar.

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

/// CLIP defaults used when `clip.vision.image_mean` / `image_std` are absent.
pub const CLIP_IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Activation used in the ViT FFN (not the LLaVA MLP projector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitFfnOp {
    /// `x * sigmoid(1.702 * x)` — default when `clip.use_gelu` is false.
    GeluQuick,
    /// Standard GGML GELU (tanh approximation).
    Gelu,
}

/// Hyperparameters for a LLaVA-class CLIP ViT + MLP projector.
#[derive(Debug, Clone)]
pub struct MmprojConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub projection_dim: usize,
    pub layer_norm_eps: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub ffn_op: VitFfnOp,
    pub has_llava_projector: bool,
    pub projector_type: Option<String>,
}

impl MmprojConfig {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let md = &archive.metadata;
        let image_size = meta_usize(md, "clip.vision.image_size")
            .ok_or_else(|| BitNetError::InvalidGguf("missing clip.vision.image_size".into()))?;
        let patch_size = meta_usize(md, "clip.vision.patch_size")
            .ok_or_else(|| BitNetError::InvalidGguf("missing clip.vision.patch_size".into()))?;
        let n_embd = meta_usize(md, "clip.vision.embedding_length")
            .ok_or_else(|| BitNetError::InvalidGguf("missing clip.vision.embedding_length".into()))?;
        let n_layer = meta_usize(md, "clip.vision.block_count")
            .ok_or_else(|| BitNetError::InvalidGguf("missing clip.vision.block_count".into()))?;
        let n_head = meta_usize(md, "clip.vision.attention.head_count")
            .ok_or_else(|| BitNetError::InvalidGguf("missing clip.vision.attention.head_count".into()))?;
        let n_ff = meta_usize(md, "clip.vision.feed_forward_length").unwrap_or(4 * n_embd);
        let projection_dim = meta_usize(md, "clip.vision.projection_dim").unwrap_or(n_embd);
        let layer_norm_eps =
            meta_f32(md, "clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-5);
        let image_mean = meta_f32_3(md, "clip.vision.image_mean").unwrap_or(CLIP_IMAGE_MEAN);
        let image_std = meta_f32_3(md, "clip.vision.image_std").unwrap_or(CLIP_IMAGE_STD);

        let use_gelu = meta_bool(md, "clip.use_gelu").unwrap_or(false);
        let ffn_op = if use_gelu {
            VitFfnOp::Gelu
        } else {
            VitFfnOp::GeluQuick
        };

        let has_llava_projector = meta_bool(md, "clip.has_llava_projector").unwrap_or(false);
        let projector_type = archive
            .metadata_str("clip.projector_type")
            .or_else(|| archive.metadata_str("projector_type"))
            .map(|s| s.to_string());

        if image_size == 0 || patch_size == 0 || n_embd == 0 || n_layer == 0 || n_head == 0 {
            return Err(BitNetError::InvalidGguf(
                "mmproj vision hparams contain a zero dimension".into(),
            ));
        }
        if image_size % patch_size != 0 {
            return Err(BitNetError::InvalidGguf(format!(
                "image_size {image_size} not divisible by patch_size {patch_size}"
            )));
        }
        if n_embd % n_head != 0 {
            return Err(BitNetError::InvalidGguf(format!(
                "embedding_length {n_embd} not divisible by head_count {n_head}"
            )));
        }

        Ok(Self {
            image_size,
            patch_size,
            n_embd,
            n_layer,
            n_head,
            n_ff,
            projection_dim,
            layer_norm_eps,
            image_mean,
            image_std,
            ffn_op,
            has_llava_projector,
            projector_type,
        })
    }

    /// Number of spatial patches for a square `image_size` grid.
    pub fn n_patches(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Exclusive upper bound for ViT layers to run for LLaVA feature extract.
    ///
    /// Matches llama.cpp: `max_feature_layer = n_layer - 1`, loop `il < max_feature_layer`
    /// → layers `0..(n_layer - 2)` inclusive (second-to-last).
    pub fn max_feature_layer(&self) -> usize {
        self.n_layer.saturating_sub(1)
    }
}

fn meta_usize(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<usize> {
    md.get(key).and_then(|v| match v {
        GgufValue::U8(x) => Some(*x as usize),
        GgufValue::U16(x) => Some(*x as usize),
        GgufValue::U32(x) => Some(*x as usize),
        GgufValue::U64(x) => Some(*x as usize),
        GgufValue::I8(x) if *x >= 0 => Some(*x as usize),
        GgufValue::I16(x) if *x >= 0 => Some(*x as usize),
        GgufValue::I32(x) if *x >= 0 => Some(*x as usize),
        GgufValue::I64(x) if *x >= 0 => Some(*x as usize),
        _ => None,
    })
}

fn meta_f32(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    md.get(key).and_then(|v| match v {
        GgufValue::F32(x) => Some(*x),
        GgufValue::F64(x) => Some(*x as f32),
        _ => None,
    })
}

fn meta_bool(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<bool> {
    md.get(key).and_then(|v| match v {
        GgufValue::Bool(b) => Some(*b),
        _ => None,
    })
}

fn meta_f32_3(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<[f32; 3]> {
    let GgufValue::Array(items) = md.get(key)? else {
        return None;
    };
    if items.len() < 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (i, item) in items.iter().take(3).enumerate() {
        out[i] = match item {
            GgufValue::F32(x) => *x,
            GgufValue::F64(x) => *x as f32,
            _ => return None,
        };
    }
    Some(out)
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
    fn write_kv_u32(w: &mut File, key: &str, val: u32) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 4)?; // U32
        write_u32(w, val)
    }
    fn write_kv_bool(w: &mut File, key: &str, val: bool) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 7)?; // Bool
        w.write_all(&[u8::from(val)])
    }
    fn write_kv_str(w: &mut File, key: &str, val: &str) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 8)?;
        write_str(w, val)
    }
    fn write_kv_f32_arr(w: &mut File, key: &str, vals: &[f32]) -> std::io::Result<()> {
        write_str(w, key)?;
        write_u32(w, 9)?; // Array
        write_u32(w, 6)?; // F32 elements
        write_u64(w, vals.len() as u64)?;
        for v in vals {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_vision_meta_gguf(path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"GGUF")?;
        write_u32(&mut f, 3)?;
        write_u64(&mut f, 0)?; // tensors
        write_u64(&mut f, 11)?; // kv
        write_kv_str(&mut f, "general.architecture", "clip")?;
        write_kv_u32(&mut f, "clip.vision.image_size", 336)?;
        write_kv_u32(&mut f, "clip.vision.patch_size", 14)?;
        write_kv_u32(&mut f, "clip.vision.embedding_length", 1024)?;
        write_kv_u32(&mut f, "clip.vision.block_count", 23)?;
        write_kv_u32(&mut f, "clip.vision.attention.head_count", 16)?;
        write_kv_u32(&mut f, "clip.vision.feed_forward_length", 4096)?;
        write_kv_bool(&mut f, "clip.use_gelu", false)?;
        write_kv_bool(&mut f, "clip.has_llava_projector", true)?;
        write_kv_f32_arr(&mut f, "clip.vision.image_mean", &CLIP_IMAGE_MEAN)?;
        write_kv_f32_arr(&mut f, "clip.vision.image_std", &CLIP_IMAGE_STD)?;
        // Align tensor-data section start (even with zero tensors).
        let pos = f.metadata()?.len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }

    #[test]
    fn config_from_gguf_reads_vision_hparams() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vision-meta.gguf");
        write_vision_meta_gguf(&path).unwrap();
        let archive = GgufArchive::mmap_path(&path).unwrap();
        let cfg = MmprojConfig::from_gguf(&archive).unwrap();
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.n_embd, 1024);
        assert_eq!(cfg.n_layer, 23);
        assert_eq!(cfg.n_head, 16);
        assert_eq!(cfg.n_ff, 4096);
        assert_eq!(cfg.n_patches(), 576);
        assert_eq!(cfg.max_feature_layer(), 22);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.ffn_op, VitFfnOp::GeluQuick);
        assert!(cfg.has_llava_projector);
        assert!((cfg.image_mean[0] - CLIP_IMAGE_MEAN[0]).abs() < 1e-5);
    }
}
