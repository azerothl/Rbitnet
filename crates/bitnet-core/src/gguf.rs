//! Minimal GGUF reader: magic, version, tensor names (metadata enumeration).
//!
//! Full weight loading for BitNet-packed tensors will extend this module.
//! Format reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{BitNetError, Result};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

/// Lightweight view of a GGUF file after validating header.
#[derive(Debug, Clone)]
pub struct GgufFileInfo {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

/// Memory-map a file and parse the GGUF header (versions 2–3).
pub fn mmap_gguf_header(path: &Path) -> Result<(Mmap, GgufFileInfo)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 24 {
        return Err(BitNetError::InvalidGguf("file too small".into()));
    }
    let magic = read_u32_le(&mmap, 0);
    if magic != GGUF_MAGIC {
        return Err(BitNetError::InvalidGguf(format!(
            "bad magic {magic:#x}, expected GGUF"
        )));
    }
    let version = read_u32_le(&mmap, 4);
    let tensor_count = read_u64_le(&mmap, 8);
    let kv_count = read_u64_le(&mmap, 16);

    Ok((
        mmap,
        GgufFileInfo {
            version,
            tensor_count,
            kv_count,
        },
    ))
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Read a GGUF string at `offset` into `mmap` (length-prefixed u64 LE + UTF-8).
///
/// Used when extending this module to walk metadata KV entries.
#[allow(dead_code)]
pub fn read_gguf_string(mmap: &[u8], offset: usize) -> Result<(String, usize)> {
    if offset + 8 > mmap.len() {
        return Err(BitNetError::InvalidGguf("string length OOB".into()));
    }
    let len = read_u64_le(mmap, offset) as usize;
    let start = offset + 8;
    let end = start.checked_add(len).ok_or_else(|| {
        BitNetError::InvalidGguf("string length overflow".into())
    })?;
    if end > mmap.len() {
        return Err(BitNetError::InvalidGguf("string data OOB".into()));
    }
    let s = std::str::from_utf8(&mmap[start..end])
        .map_err(|e| BitNetError::InvalidGguf(format!("utf8: {e}")))?;
    Ok((s.to_string(), end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_gguf() {
        let buf = [0u8; 32];
        let magic = read_u32_le(&buf, 0);
        assert_ne!(magic, GGUF_MAGIC);
    }
}
