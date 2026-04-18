//! Full GGUF parse: metadata KV, tensor infos, tensor data blob (memory-mapped).
//!
//! Tensor `ggml_type` is kept as raw `u32` so unknown / future quantizations (e.g. BitNet) still load.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{BitNetError, Result};

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    /// Raw `ggml_type` as stored in the file.
    pub ggml_type: u32,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufArchive {
    mmap: Arc<Mmap>,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset in file where tensor payload starts (after header + KV + tensor infos + padding).
    tensor_data_offset: usize,
}

impl GgufArchive {
    /// Look up a tensor by exact name.
    pub fn tensor_by_name(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw payload bytes for a tensor (size matches [`crate::ggml::ggml_nbytes`]).
    pub fn tensor_payload(&self, t: &GgufTensorInfo) -> Result<&[u8]> {
        let n = crate::ggml::ggml_nbytes(&t.dimensions, t.ggml_type)?;
        let start = t.offset as usize;
        let end = start
            .checked_add(n)
            .ok_or_else(|| BitNetError::InvalidGguf("tensor offset overflow".into()))?;
        let blob = self.tensor_data();
        if end > blob.len() {
            return Err(BitNetError::InvalidGguf(format!(
                "tensor payload [{start}, {end}) out of range (blob len {})",
                blob.len()
            )));
        }
        Ok(&blob[start..end])
    }

    /// Memory-map and parse a `.gguf` file.
    pub fn mmap_path(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(Arc::new(mmap))
    }

    fn parse(mmap: Arc<Mmap>) -> Result<Self> {
        let buf = mmap.as_ref();
        if buf.len() < 24 {
            return Err(BitNetError::InvalidGguf("file too small".into()));
        }
        let magic = read_u32_le(buf, 0);
        if magic != GGUF_MAGIC {
            return Err(BitNetError::InvalidGguf(format!(
                "bad magic {magic:#x}, expected GGUF"
            )));
        }
        let version = read_u32_le(buf, 4);
        let tensor_count = read_u64_le(buf, 8);
        let kv_count = read_u64_le(buf, 16);

        // Sanity limits to guard against malformed/crafted GGUF files.
        const MAX_KV_COUNT: u64 = 65_536;
        const MAX_TENSOR_COUNT: u64 = 1_000_000;
        if kv_count > MAX_KV_COUNT {
            return Err(BitNetError::InvalidGguf(format!(
                "kv_count {kv_count} exceeds sanity limit {MAX_KV_COUNT}"
            )));
        }
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(BitNetError::InvalidGguf(format!(
                "tensor_count {tensor_count} exceeds sanity limit {MAX_TENSOR_COUNT}"
            )));
        }

        let mut off = 24usize;
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let (key, o2) = read_gguf_string(buf, off)?;
            off = o2;
            if off + 4 > buf.len() {
                return Err(BitNetError::InvalidGguf("truncated KV value_type".into()));
            }
            let value_type = read_u32_le(buf, off);
            off += 4;
            let (val, o3) = read_metadata_value(buf, off, value_type)?;
            off = o3;
            metadata.insert(key, val);
        }

        // Maximum sane alignment: must be a positive power-of-two no larger than 4096.
        const MAX_ALIGNMENT: u64 = 4096;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                GgufValue::U32(u) => Some(*u as u64),
                GgufValue::U64(u) => Some(*u),
                // Accept I32 only when positive and within range.
                GgufValue::I32(i) if *i > 0 => Some(*i as u64),
                _ => None,
            })
            .unwrap_or(32);
        if alignment == 0 || alignment > MAX_ALIGNMENT {
            return Err(BitNetError::InvalidGguf(format!(
                "general.alignment {alignment} is out of range (1..={MAX_ALIGNMENT})"
            )));
        }

        off = align_usize(off, alignment as usize);

        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let (name, o2) = read_gguf_string(buf, off)?;
            off = o2;
            if off + 4 > buf.len() {
                return Err(BitNetError::InvalidGguf("truncated n_dimensions".into()));
            }
            let n_dims = read_u32_le(buf, off) as usize;
            off += 4;
            if off + n_dims * 8 > buf.len() {
                return Err(BitNetError::InvalidGguf("truncated dimensions".into()));
            }
            let mut dimensions = Vec::with_capacity(n_dims);
            for i in 0..n_dims {
                dimensions.push(read_u64_le(buf, off + i * 8));
            }
            off += n_dims * 8;
            if off + 12 > buf.len() {
                return Err(BitNetError::InvalidGguf("truncated tensor type/offset".into()));
            }
            let ggml_type = read_u32_le(buf, off);
            off += 4;
            let tensor_offset = read_u64_le(buf, off);
            off += 8;
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset: tensor_offset,
            });
        }

        off = align_usize(off, alignment as usize);
        if off > buf.len() {
            return Err(BitNetError::InvalidGguf("tensor_data offset OOB".into()));
        }

        Ok(GgufArchive {
            mmap,
            version,
            metadata,
            tensors,
            tensor_data_offset: off,
        })
    }

    pub fn tensor_data(&self) -> &[u8] {
        let b = self.mmap.as_ref();
        &b[self.tensor_data_offset..]
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Summarize for logs / `GET /v1/models` diagnostics.
    pub fn summary_line(&self) -> String {
        let arch = self
            .architecture()
            .unwrap_or("unknown");
        format!(
            "GGUF v{} arch={} tensors={} tensor_data@{}",
            self.version,
            arch,
            self.tensors.len(),
            self.tensor_data_offset
        )
    }

    /// `general.architecture` metadata (e.g. `llama`, `bitnet`).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| match v {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Stable id for OpenAI-style `model` fields (e.g. `rbitnet-llama`).
    pub fn suggested_openai_model_id(&self) -> String {
        self.architecture()
            .map(|a| format!("rbitnet-{}", a.replace('.', "-")))
            .unwrap_or_else(|| "rbitnet-gguf".into())
    }
}

fn align_usize(offset: usize, align: usize) -> usize {
    if align == 0 {
        return offset;
    }
    let m = offset % align;
    if m == 0 {
        offset
    } else {
        offset + (align - m)
    }
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn read_i64_le(buf: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_f64_le(buf: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn read_gguf_string(buf: &[u8], offset: usize) -> Result<(String, usize)> {
    if offset + 8 > buf.len() {
        return Err(BitNetError::InvalidGguf("string length OOB".into()));
    }
    let len = read_u64_le(buf, offset) as usize;
    let start = offset + 8;
    let end = start.checked_add(len).ok_or_else(|| {
        BitNetError::InvalidGguf("string length overflow".into())
    })?;
    if end > buf.len() {
        return Err(BitNetError::InvalidGguf("string data OOB".into()));
    }
    let s = std::str::from_utf8(&buf[start..end])
        .map_err(|e| BitNetError::InvalidGguf(format!("utf8: {e}")))?;
    Ok((s.to_string(), end))
}

fn read_metadata_value(buf: &[u8], mut off: usize, typ: u32) -> Result<(GgufValue, usize)> {
    match typ {
        0 => {
            if off + 1 > buf.len() {
                return Err(BitNetError::InvalidGguf("u8 OOB".into()));
            }
            let v = buf[off];
            Ok((GgufValue::U8(v), off + 1))
        }
        1 => {
            if off + 1 > buf.len() {
                return Err(BitNetError::InvalidGguf("i8 OOB".into()));
            }
            let v = buf[off] as i8;
            Ok((GgufValue::I8(v), off + 1))
        }
        2 => {
            if off + 2 > buf.len() {
                return Err(BitNetError::InvalidGguf("u16 OOB".into()));
            }
            let v = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
            Ok((GgufValue::U16(v), off + 2))
        }
        3 => {
            if off + 2 > buf.len() {
                return Err(BitNetError::InvalidGguf("i16 OOB".into()));
            }
            let v = i16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
            Ok((GgufValue::I16(v), off + 2))
        }
        4 => {
            if off + 4 > buf.len() {
                return Err(BitNetError::InvalidGguf("u32 OOB".into()));
            }
            let v = read_u32_le(buf, off);
            Ok((GgufValue::U32(v), off + 4))
        }
        5 => {
            if off + 4 > buf.len() {
                return Err(BitNetError::InvalidGguf("i32 OOB".into()));
            }
            let v = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            Ok((GgufValue::I32(v), off + 4))
        }
        6 => {
            if off + 4 > buf.len() {
                return Err(BitNetError::InvalidGguf("f32 OOB".into()));
            }
            let v = read_f32_le(buf, off);
            Ok((GgufValue::F32(v), off + 4))
        }
        7 => {
            if off + 1 > buf.len() {
                return Err(BitNetError::InvalidGguf("bool OOB".into()));
            }
            let b = buf[off];
            if b > 1 {
                return Err(BitNetError::InvalidGguf("invalid bool byte".into()));
            }
            Ok((GgufValue::Bool(b != 0), off + 1))
        }
        8 => {
            let (s, end) = read_gguf_string(buf, off)?;
            Ok((GgufValue::String(s), end))
        }
        9 => {
            if off + 12 > buf.len() {
                return Err(BitNetError::InvalidGguf("array header OOB".into()));
            }
            let elem_type = read_u32_le(buf, off);
            let len = read_u64_le(buf, off + 4);
            off += 12;
            // Guard against malformed files claiming huge arrays.
            const MAX_ARRAY_LEN: u64 = 10_000_000;
            if len > MAX_ARRAY_LEN {
                return Err(BitNetError::InvalidGguf(format!(
                    "metadata array length {len} exceeds limit {MAX_ARRAY_LEN}"
                )));
            }
            let len = len as usize;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                let (v, o2) = read_metadata_value(buf, off, elem_type)?;
                off = o2;
                out.push(v);
            }
            Ok((GgufValue::Array(out), off))
        }
        10 => {
            if off + 8 > buf.len() {
                return Err(BitNetError::InvalidGguf("u64 OOB".into()));
            }
            let v = read_u64_le(buf, off);
            Ok((GgufValue::U64(v), off + 8))
        }
        11 => {
            if off + 8 > buf.len() {
                return Err(BitNetError::InvalidGguf("i64 OOB".into()));
            }
            let v = read_i64_le(buf, off);
            Ok((GgufValue::I64(v), off + 8))
        }
        12 => {
            if off + 8 > buf.len() {
                return Err(BitNetError::InvalidGguf("f64 OOB".into()));
            }
            let v = read_f64_le(buf, off);
            Ok((GgufValue::F64(v), off + 8))
        }
        _ => Err(BitNetError::InvalidGguf(format!(
            "unknown metadata value type {typ}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_works() {
        assert_eq!(align_usize(24, 32), 32);
        assert_eq!(align_usize(32, 32), 32);
    }
}
