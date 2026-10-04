//! Typed caller ownership around the optional Native F32 checkpoint ABI.
use crate::context_tiers::{Policy, Store};
use crate::portable_envelope::Compatibility;
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! counters {
    ($($name:ident),*) => {$(static $name:AtomicU64=AtomicU64::new(0);)*};
}
counters!(
    RAM_HITS,
    DISK_HITS,
    MISSES,
    CAPTURES,
    CAPTURE_REFUSALS,
    WRITES,
    WRITE_FAILURES,
    READ_FAILURES,
    EVICTIONS,
    READ_NS,
    WRITE_NS,
    RESTORED_TOKENS,
    IMPORT_FAILURES
);
fn update(before: &crate::context_tiers::Stats, after: &crate::context_tiers::Stats) {
    macro_rules! add {
        ($counter:ident,$field:ident) => {
            $counter.fetch_add(
                after.$field.saturating_sub(before.$field),
                Ordering::Relaxed,
            );
        };
    }
    add!(RAM_HITS, ram_hits);
    add!(DISK_HITS, disk_hits);
    add!(MISSES, misses);
    add!(CAPTURES, captures);
    add!(CAPTURE_REFUSALS, capture_refusals);
    add!(WRITES, writes);
    add!(WRITE_FAILURES, write_failures);
    add!(READ_FAILURES, read_failures);
    add!(EVICTIONS, evictions);
    add!(READ_NS, read_ns);
    add!(WRITE_NS, write_ns);
}
pub(crate) fn prometheus_text() -> String {
    use std::fmt::Write;
    let mut result = String::new();
    for (name, counter) in [
        ("ram_hits", &RAM_HITS),
        ("disk_hits", &DISK_HITS),
        ("misses", &MISSES),
        ("captures", &CAPTURES),
        ("capture_refusals", &CAPTURE_REFUSALS),
        ("writes", &WRITES),
        ("write_failures", &WRITE_FAILURES),
        ("read_failures", &READ_FAILURES),
        ("evictions", &EVICTIONS),
        ("read_ns", &READ_NS),
        ("write_ns", &WRITE_NS),
        ("restored_tokens", &RESTORED_TOKENS),
        ("import_failures", &IMPORT_FAILURES),
    ] {
        let _ = writeln!(
            result,
            "rbitnet_core_context_{name}_total {}",
            counter.load(Ordering::Relaxed)
        );
    }
    result
}

type Bytes = unsafe extern "C" fn(*mut c_void, u32) -> usize;
type Export = unsafe extern "C" fn(*mut c_void, u32, *mut f32, usize) -> i32;
type Import = unsafe extern "C" fn(*mut c_void, u32, *const f32, usize) -> i32;
pub(crate) struct Handle {
    bytes: Bytes,
    export: Export,
    import: Import,
    pub store: Store,
}
pub(crate) fn enabled() -> bool {
    matches!(
        std::env::var("RBITNET_CONTEXT_TIERS").as_deref(),
        Ok("1" | "true")
    )
}

fn execution_options(
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<(&'static str, Option<String>)> {
    [
        "RBITNET_CUDA_RESIDENT_GRAPH",
        "RBITNET_CUDA_QWEN_FULL_GRAPH",
        "RBITNET_CUDA_QWEN_PREFILL",
        "RBITNET_CUDA_PREFILL",
        "RBITNET_CUDA_PREFILL_TOKENS",
        "RBITNET_PREFILL_CHUNK_TOKENS",
        "RBITNET_QWEN_ORDERED_BLOCK_TEST",
        "RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS",
        "RBITNET_PREFIX_KV",
        "RBITNET_PREFIX_KV_MIN_TOKENS",
        "RBITNET_CUDA_KV_PAGE_LIMIT",
        "RBITNET_CUDA_PREFILL_TF32X3",
        "RBITNET_CUDA_KV_FORMAT",
        "RBITNET_CUDA_SPLIT_KV",
    ]
    .into_iter()
    .map(|name| (name, lookup(name)))
    .collect()
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn prefill_partitions_ordering_prefix_boundaries_and_page_geometry_change_identity() {
        let baseline = serde_json::to_vec(&execution_options(|_| None)).unwrap();
        for key in [
            "RBITNET_CUDA_PREFILL_TOKENS",
            "RBITNET_PREFILL_CHUNK_TOKENS",
            "RBITNET_QWEN_ORDERED_BLOCK_TEST",
            "RBITNET_QWEN_PREFIX_CHECKPOINT_TOKENS",
            "RBITNET_PREFIX_KV",
            "RBITNET_PREFIX_KV_MIN_TOKENS",
            "RBITNET_CUDA_KV_PAGE_LIMIT",
        ] {
            let first = serde_json::to_vec(&execution_options(|name| {
                (name == key).then(|| "1".to_string())
            }))
            .unwrap();
            let second = serde_json::to_vec(&execution_options(|name| {
                (name == key).then(|| "2".to_string())
            }))
            .unwrap();
            assert_ne!(baseline, first, "missing state compatibility option {key}");
            assert_ne!(first, second, "option value omitted from identity: {key}");
        }
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn digest_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        };
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().into())
}
impl Handle {
    pub fn new(
        family: &str,
        archive: &crate::gguf::GgufArchive,
        tokenizer: &Path,
        configuration: &str,
    ) -> io::Result<Option<Self>> {
        let Some(policy) = Policy::from_env()? else {
            return Ok(None);
        };
        let lib = crate::ggml::load_cuda_quant_library()
            .ok_or_else(|| invalid("context transport requires Native CUDA"))?;
        let names = match family {
            "llama" => [
                b"rbitnet_cuda_llama_portable_bytes\0".as_slice(),
                b"rbitnet_cuda_llama_portable_export\0",
                b"rbitnet_cuda_llama_portable_import\0",
            ],
            "qwen" => [
                b"rbitnet_cuda_qwen_portable_bytes\0".as_slice(),
                b"rbitnet_cuda_qwen_portable_export\0",
                b"rbitnet_cuda_qwen_portable_import\0",
            ],
            _ => return Err(invalid("unsupported context transport family")),
        };
        let (bytes, export, import) = unsafe {
            (
                *lib.get::<Bytes>(names[0])
                    .map_err(|_| invalid("context size ABI unavailable"))?,
                *lib.get::<Export>(names[1])
                    .map_err(|_| invalid("context export ABI unavailable"))?,
                *lib.get::<Import>(names[2])
                    .map_err(|_| invalid("context import ABI unavailable"))?,
            )
        };
        type DeviceKey = unsafe extern "C" fn(*mut u8, usize) -> i32;
        let device_key = unsafe {
            *lib.get::<DeviceKey>(b"rbitnet_cuda_portable_device_key\0")
                .map_err(|_| invalid("Native device identity unavailable"))?
        };
        let mut device = [0u8; 32];
        if unsafe { device_key(device.as_mut_ptr(), device.len()) } != 0 {
            return Err(invalid("Native device identity query failed"));
        }
        let options = execution_options(|name| std::env::var(name).ok());
        static EXECUTABLE: std::sync::OnceLock<Option<[u8; 32]>> = std::sync::OnceLock::new();
        let executable = EXECUTABLE
            .get_or_init(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|path| digest_file(&path).ok())
            })
            .ok_or_else(|| invalid("current executable identity unavailable"))?;
        let mut configuration_hash = Sha256::new();
        configuration_hash.update(executable);
        configuration_hash.update(configuration.as_bytes());
        configuration_hash.update(device);
        configuration_hash.update(serde_json::to_vec(&options)?);
        configuration_hash.update(
            format!(
                "ordered_f32_lanes={:?}",
                crate::ggml::f32_accumulator_lanes()
            )
            .as_bytes(),
        );
        let key = Compatibility {
            model_sha256: archive.content_sha256(),
            tokenizer_sha256: digest_file(tokenizer)?,
            native_library_sha256: crate::ggml::cuda_quant_library_identity()
                .ok_or_else(|| invalid("actual loaded Native DLL identity unavailable"))?,
            execution_config_sha256: configuration_hash.finalize().into(),
            layout: format!("{family}-native-f32-v1"),
        };
        match Store::open(key, policy) {
            Ok(store) => Ok(Some(Self {
                bytes,
                export,
                import,
                store,
            })),
            Err(error) => {
                tracing::warn!(%error,"context store unavailable; normal prompt replay remains enabled");
                Ok(None)
            }
        }
    }
    /// The caller exclusively owns this live Native context for the full operation.
    pub unsafe fn capture(&mut self, context: *mut c_void, tokens: &[u32]) -> io::Result<bool> {
        let length = u32::try_from(tokens.len()).map_err(|_| invalid("context length overflow"))?;
        let bytes = (self.bytes)(context, length);
        let export = self.export;
        let before = self.store.stats.clone();
        let result = self.store.capture(tokens, bytes, |values| {
            if export(context, length, values.as_mut_ptr(), bytes) != 0 {
                return Err(invalid("Native context export refused"));
            }
            crate::perf::record_gpu_transfer(0, bytes as u64, 0);
            Ok(())
        });
        update(&before, &self.store.stats);
        tracing::info!(ram_bytes=self.store.ram_used(),disk_bytes=self.store.disk_used(),stats=?self.store.stats,"context tier capture");
        result
    }
    /// Restore a full exact prefix; Native length is zeroed by a failed transfer.
    pub unsafe fn restore(&mut self, context: *mut c_void, tokens: &[u32]) -> io::Result<usize> {
        let size = self.bytes;
        let before = self.store.stats.clone();
        let saved = self.store.lookup(tokens, |length| {
            u32::try_from(length)
                .ok()
                .map_or(0, |length| size(context, length))
        });
        update(&before, &self.store.stats);
        let Some(lease) = saved? else {
            return Ok(0);
        };
        let length = u32::try_from(lease.checkpoint.tokens.len())
            .map_err(|_| invalid("context length overflow"))?;
        let bytes = lease
            .checkpoint
            .values
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("context size overflow"))?;
        if (self.import)(context, length, lease.checkpoint.values.as_ptr(), bytes) != 0 {
            IMPORT_FAILURES.fetch_add(1, Ordering::Relaxed);
            return Err(invalid("Native context import refused; replay required"));
        }
        crate::perf::record_gpu_transfer(bytes as u64, 0, 0);
        crate::perf::record_prefix_hit(bytes);
        RESTORED_TOKENS.fetch_add(length as u64, Ordering::Relaxed);
        tracing::info!(length,ram_bytes=self.store.ram_used(),disk_bytes=self.store.disk_used(),stats=?self.store.stats,"context tier restored");
        Ok(length as usize)
    }
}
