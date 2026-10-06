//! Metadata surface for a mmap'd mmproj GGUF (no vision encode).

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

#[derive(Debug, Clone)]
pub struct MmprojInfo {
    pub architecture: Option<String>,
    pub projector_type: Option<String>,
    pub clip_keys: Vec<String>,
    pub vision_tensor_names: Vec<String>,
    pub tensor_count: usize,
}

impl MmprojInfo {
    pub fn from_gguf(archive: &GgufArchive) -> Self {
        let architecture = archive.normalized_architecture();
        let projector_type = archive
            .metadata_str("clip.projector_type")
            .or_else(|| archive.metadata_str("projector_type"))
            .map(|s| s.to_string())
            .or_else(|| {
                archive.metadata.get("clip.projector_type").and_then(|v| {
                    if let GgufValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
            });

        let mut clip_keys: Vec<String> = archive
            .metadata
            .keys()
            .filter(|k| k.starts_with("clip.") || k.starts_with("projector"))
            .cloned()
            .collect();
        clip_keys.sort();

        let mut vision_tensor_names: Vec<String> = archive
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .filter(|n| {
                let l = n.to_ascii_lowercase();
                l.contains("mm.")
                    || l.contains("clip")
                    || l.contains("vision")
                    || l.contains("v.")
                    || l.starts_with("blk.")
                    || l.contains("proj")
            })
            .collect();
        vision_tensor_names.sort();
        // Cap listing for large projectors.
        if vision_tensor_names.len() > 64 {
            vision_tensor_names.truncate(64);
        }

        Self {
            architecture,
            projector_type,
            clip_keys,
            vision_tensor_names,
            tensor_count: archive.tensors.len(),
        }
    }

    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let archive = GgufArchive::mmap_path(path).map_err(|e| {
            BitNetError::InvalidGguf(format!("mmproj mmap {}: {e}", path.display()))
        })?;
        Ok(Self::from_gguf(&archive))
    }

    pub fn summary_line(&self) -> String {
        format!(
            "mmproj arch={} projector_type={} tensors={} clip_keys={}",
            self.architecture.as_deref().unwrap_or("-"),
            self.projector_type.as_deref().unwrap_or("-"),
            self.tensor_count,
            self.clip_keys.len()
        )
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
    fn write_tensor(w: &mut File, name: &str, dims: &[u64]) -> std::io::Result<()> {
        write_str(w, name)?;
        write_u32(w, dims.len() as u32)?;
        for &d in dims {
            write_u64(w, d)?;
        }
        write_u32(w, 0)?;
        write_u64(w, 0)
    }

    fn write_minimal_mmproj(path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"GGUF")?;
        write_u32(&mut f, 3)?;
        write_u64(&mut f, 2)?; // tensors
        write_u64(&mut f, 3)?; // kv
        write_kv_str(&mut f, "general.architecture", "clip")?;
        write_kv_str(&mut f, "clip.projector_type", "mlp")?;
        write_kv_str(&mut f, "clip.vision.model_type", "vit")?;
        write_tensor(&mut f, "v.blk.0.attn_q.weight", &[16, 16])?;
        write_tensor(&mut f, "mm.0.weight", &[16, 32])?;
        let pos = f.metadata()?.len() as usize;
        let pad = (32 - (pos % 32)) % 32;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }

    #[test]
    fn mmproj_info_reads_clip_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmproj.gguf");
        write_minimal_mmproj(&path).unwrap();
        let info = MmprojInfo::from_path(&path).unwrap();
        assert_eq!(info.architecture.as_deref(), Some("clip"));
        assert_eq!(info.projector_type.as_deref(), Some("mlp"));
        assert!(info.clip_keys.iter().any(|k| k == "clip.projector_type"));
        assert_eq!(info.tensor_count, 2);
        assert!(info.summary_line().contains("projector_type=mlp"));
    }
}
