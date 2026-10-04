//! Sealed F32 checkpoint envelope, checked before native restoration.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8; 8] = b"RBSTATE1";
const MAX_HEADER: usize = 65536;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Compatibility {
    pub model_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    pub native_library_sha256: [u8; 32],
    /// Includes architecture, actual shape, RoPE, precision and attention/GEMM variants.
    pub execution_config_sha256: [u8; 32],
    pub layout: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    version: u32,
    key: Compatibility,
    tokens: Vec<u32>,
    elements: u64,
}
pub(crate) struct HostCheckpoint {
    pub tokens: Vec<u32>,
    pub values: Vec<f32>,
}
pub(crate) struct CheckpointInfo {
    pub tokens: Vec<u32>,
    pub payload_bytes: usize,
}
/// Bound the pending temporary file as well as the final sealed object.
pub(crate) fn sealed_size(
    key: &Compatibility,
    tokens: &[u32],
    payload_bytes: usize,
) -> std::io::Result<usize> {
    if payload_bytes == 0 || payload_bytes % 4 != 0 || tokens.is_empty() || tokens.len() > 8192 {
        return Err(invalid("invalid sealed geometry"));
    }
    let header = serde_json::to_vec(&Header {
        version: 1,
        key: key.clone(),
        tokens: tokens.to_vec(),
        elements: (payload_bytes / 4) as u64,
    })?;
    if header.len() > MAX_HEADER {
        return Err(invalid("header exceeds bound"));
    }
    payload_bytes
        .checked_add(header.len())
        .and_then(|size| size.checked_add(52))
        .ok_or_else(|| invalid("sealed size overflow"))
}
/// Untrusted index metadata only: read_checkpoint verifies the entire seal again.
pub(crate) fn inspect_checkpoint(
    path: &Path,
    expected: &Compatibility,
    maximum_bytes: usize,
) -> std::io::Result<CheckpointInfo> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(invalid("checkpoint must be regular"));
    }
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut fixed = [0u8; 20];
    reader.read_exact(&mut fixed)?;
    if &fixed[..8] != MAGIC {
        return Err(invalid("unknown checkpoint magic"));
    }
    let header_bytes = u32::from_le_bytes(fixed[8..12].try_into().unwrap()) as usize;
    let payload_bytes = usize::try_from(u64::from_le_bytes(fixed[12..20].try_into().unwrap()))
        .map_err(|_| invalid("payload overflow"))?;
    if header_bytes == 0
        || header_bytes > MAX_HEADER
        || payload_bytes == 0
        || payload_bytes % 4 != 0
        || payload_bytes > maximum_bytes
        || metadata.len() != 52 + header_bytes as u64 + payload_bytes as u64
    {
        return Err(invalid("invalid checkpoint size"));
    }
    let mut header = vec![0u8; header_bytes];
    reader.read_exact(&mut header)?;
    let decoded: Header = serde_json::from_slice(&header)?;
    if decoded.version != 1
        || &decoded.key != expected
        || decoded.tokens.is_empty()
        || decoded.tokens.len() > 8192
        || decoded.elements != payload_bytes as u64 / 4
    {
        return Err(invalid("incompatible checkpoint identity"));
    }
    Ok(CheckpointInfo {
        tokens: decoded.tokens,
        payload_bytes,
    })
}
fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn wire_bytes(values: &[f32]) -> std::io::Result<&[u8]> {
    if !cfg!(target_endian = "little") {
        return Err(invalid("this native F32 layout requires little endian"));
    }
    let bytes = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("payload overflow"))?;
    // u8 can represent all initialized f32 bits. The borrowed byte slice cannot
    // outlive the source allocation or mutate it during hashing/writing.
    Ok(unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), bytes) })
}
pub(crate) fn write_checkpoint(
    directory: &Path,
    key: &Compatibility,
    tokens: &[u32],
    values: &[f32],
    maximum_bytes: usize,
) -> std::io::Result<PathBuf> {
    let payload = wire_bytes(values)?;
    if tokens.is_empty()
        || tokens.len() > 8192
        || payload.is_empty()
        || payload.len() > maximum_bytes
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid or over-budget checkpoint"));
    }
    let header = serde_json::to_vec(&Header {
        version: 1,
        key: key.clone(),
        tokens: tokens.to_vec(),
        elements: values.len() as u64,
    })?;
    if header.len() > MAX_HEADER {
        return Err(invalid("header exceeds bound"));
    }
    let mut hash = Sha256::new();
    hash.update(&header);
    hash.update(payload);
    let checksum = hash.finalize();
    let final_path = directory.join(format!("{}.state", hex(&checksum)));
    std::fs::create_dir_all(directory)?;
    let mut created = None;
    for _ in 0..16 {
        let temporary = directory.join(format!(
            ".{}-{}-{}.tmp",
            hex(&checksum),
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => {
                created = Some((temporary, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let (temporary, file) = created.ok_or_else(|| invalid("temporary names exhausted"))?;
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let cleanup = Cleanup(temporary.clone());
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC)?;
    writer.write_all(&(header.len() as u32).to_le_bytes())?;
    writer.write_all(&(payload.len() as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    writer.write_all(&checksum)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    match std::fs::rename(&temporary, &final_path) {
        Ok(()) => (),
        // Windows rename refuses an existing target. Only an identical sealed
        // object is reusable; a corrupt or incompatible target remains an error.
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists || final_path.exists() => {
            let existing = read_checkpoint(&final_path, key, maximum_bytes, payload.len())?;
            if existing.tokens != tokens || wire_bytes(&existing.values)? != payload {
                return Err(invalid("existing object differs"));
            }
        }
        Err(error) => return Err(error),
    }
    drop(cleanup);
    Ok(final_path)
}
pub(crate) fn read_checkpoint(
    path: &Path,
    expected: &Compatibility,
    maximum_bytes: usize,
    expected_payload_bytes: usize,
) -> std::io::Result<HostCheckpoint> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(invalid("checkpoint must be a regular file"));
    }
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut fixed = [0u8; 20];
    reader.read_exact(&mut fixed)?;
    if &fixed[..8] != MAGIC {
        return Err(invalid("unknown checkpoint magic"));
    }
    let header_bytes = u32::from_le_bytes(fixed[8..12].try_into().unwrap()) as usize;
    let payload_bytes = usize::try_from(u64::from_le_bytes(fixed[12..20].try_into().unwrap()))
        .map_err(|_| invalid("payload overflow"))?;
    if header_bytes == 0
        || header_bytes > MAX_HEADER
        || payload_bytes == 0
        || payload_bytes % 4 != 0
        || payload_bytes > maximum_bytes
        || payload_bytes != expected_payload_bytes
        || metadata.len() != 20 + header_bytes as u64 + payload_bytes as u64 + 32
    {
        return Err(invalid("invalid checkpoint size"));
    }
    let mut header = vec![0u8; header_bytes];
    reader.read_exact(&mut header)?;
    let decoded: Header = serde_json::from_slice(&header)?;
    if decoded.version != 1
        || &decoded.key != expected
        || decoded.tokens.is_empty()
        || decoded.tokens.len() > 8192
        || decoded.elements != payload_bytes as u64 / 4
    {
        return Err(invalid("incompatible checkpoint identity or layout"));
    }
    let mut values = Vec::<f32>::new();
    values
        .try_reserve_exact(payload_bytes / 4)
        .map_err(|_| invalid("host memory refused"))?;
    values.resize(payload_bytes / 4, 0.0);
    if !cfg!(target_endian = "little") {
        return Err(invalid("this native F32 layout requires little endian"));
    }
    // The vector is initialized; arbitrary file bytes are valid f32 bits. Its
    // mutable byte view ends before any floating-point value is examined.
    let payload =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), payload_bytes) };
    reader.read_exact(payload)?;
    let mut checksum = [0u8; 32];
    reader.read_exact(&mut checksum)?;
    let mut hash = Sha256::new();
    hash.update(&header);
    hash.update(&*payload);
    if hash.finalize().as_slice() != checksum
        || path.file_stem().and_then(|s| s.to_str()) != Some(hex(&checksum).as_str())
    {
        return Err(invalid("checkpoint checksum or content address differs"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite checkpoint"));
    }
    Ok(HostCheckpoint {
        tokens: decoded.tokens,
        values,
    })
}
