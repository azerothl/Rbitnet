//! GGUF support: header-only helper (legacy) and full archive parse.

mod bitnet_meta;
mod parse;

pub use bitnet_meta::LlamaHyperParams;
pub use parse::{GgufArchive, GgufTensorInfo, GgufValue};

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{BitNetError, Result};

const GGUF_MAGIC: u32 = 0x46554747;

/// Lightweight view of a GGUF file (header fields only — prefer [`GgufArchive`]).
#[derive(Debug, Clone)]
pub struct GgufFileInfo {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

/// Memory-map and read only the GGUF header (fast check).
pub fn mmap_gguf_header(path: &Path) -> Result<(Mmap, GgufFileInfo)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 24 {
        return Err(BitNetError::InvalidGguf("file too small".into()));
    }
    let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
    if magic != GGUF_MAGIC {
        return Err(BitNetError::InvalidGguf(format!(
            "bad magic {magic:#x}, expected GGUF"
        )));
    }
    let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap());
    Ok((
        mmap,
        GgufFileInfo {
            version,
            tensor_count,
            kv_count,
        },
    ))
}
