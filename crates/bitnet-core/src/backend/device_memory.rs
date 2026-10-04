//! Optional allocator shared with all native CUDA contexts. Old DLLs keep their
//! previous allocator unless a managed limit is explicitly requested.
use crate::error::{BitNetError, Result};
use std::ffi::c_void;
use std::sync::OnceLock;

pub(super) const WEIGHTS: u32 = 0;
pub(super) const EXPERTS: u32 = 4;
pub(super) const SCRATCH: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CudaMemoryStats {
    pub version: u64,
    pub limit: u64,
    pub live: u64,
    pub peak: u64,
    pub allocations: u64,
    pub refusals: u64,
    pub categories: [u64; 7],
}
type Alloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32;
type Free = unsafe extern "C" fn(*mut c_void) -> i32;
type Stats = unsafe extern "C" fn(*mut CudaMemoryStats) -> i32;
type Configure = unsafe extern "C" fn(u64, u64) -> i32;
#[derive(Clone, Copy)]
pub(super) struct DeviceMemory {
    pub alloc: Alloc,
    pub free: Free,
    stats: Stats,
}
static ACTIVE: OnceLock<DeviceMemory> = OnceLock::new();

pub fn cuda_managed_memory_stats() -> Option<CudaMemoryStats> {
    ACTIVE.get()?.snapshot()
}
pub(super) fn requested_budget() -> Result<Option<(u64, u64)>> {
    let megabytes = |key: &str, default: Option<u64>| -> Result<Option<u64>> {
        let value = match std::env::var(key) {
            Ok(s) => Some(s.parse::<u64>().map_err(|_| {
                BitNetError::Inference(format!("{key}: expected nonnegative MiB integer"))
            })?),
            Err(std::env::VarError::NotPresent) => default,
            Err(_) => return Err(BitNetError::Inference(format!("{key}: invalid value"))),
        };
        value
            .map(|n| {
                n.checked_mul(1024 * 1024)
                    .ok_or_else(|| BitNetError::Inference(format!("{key}: bytes overflow")))
            })
            .transpose()
    };
    let limit = megabytes("RBITNET_CUDA_DEVICE_BUDGET_MB", None)?.unwrap_or(0);
    if limit == 0 {
        return Ok(None);
    }
    Ok(Some((
        limit,
        megabytes("RBITNET_CUDA_DEVICE_MARGIN_MB", Some(256))?.unwrap(),
    )))
}
impl DeviceMemory {
    pub fn load() -> Result<Option<Self>> {
        let budget = requested_budget()?;
        let api = crate::ggml::load_cuda_quant_library().and_then(|lib| unsafe {
            Some((
                Self {
                    alloc: *lib.get::<Alloc>(b"rbitnet_cuda_memory_alloc\0").ok()?,
                    free: *lib.get::<Free>(b"rbitnet_cuda_memory_free\0").ok()?,
                    stats: *lib.get::<Stats>(b"rbitnet_cuda_memory_stats\0").ok()?,
                },
                *lib.get::<Configure>(b"rbitnet_cuda_memory_configure\0")
                    .ok()?,
            ))
        });
        let Some((api, configure)) = api else {
            if budget.is_some() {
                return Err(BitNetError::Inference(
                    "managed CUDA memory budget requires the native memory ABI; rebuild the quant library".into(),
                ));
            }
            return Ok(None);
        };
        if api.snapshot().is_none() {
            return Err(BitNetError::Inference(
                "native CUDA memory ABI mismatch".into(),
            ));
        }
        if let Some((limit, margin)) = budget {
            let status = unsafe { configure(limit, margin) };
            if status != 0 {
                return Err(BitNetError::Inference(format!(
                    "CUDA memory cap unavailable (status {status}): live allocations, free-device margin or device mismatch"
                )));
            }
        }
        // The loader retains the native library for the process lifetime.
        let _ = ACTIVE.set(api);
        Ok(Some(api))
    }
    pub fn snapshot(self) -> Option<CudaMemoryStats> {
        let mut out = CudaMemoryStats::default();
        (unsafe { (self.stats)(&mut out) } == 0 && out.version == 1).then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_shared_allocator_bounds_views_native_rollback_fallback_and_concurrent_admission() {
        if std::env::var("RBITNET_CUDA_MEMORY_TEST").as_deref() != Ok("1") {
            return;
        }
        let rt = crate::backend::CudaRuntime::try_load().expect("CUDA and new memory ABI required");
        let api = rt.memory.unwrap();
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        let configure = unsafe {
            *lib.get::<Configure>(b"rbitnet_cuda_memory_configure\0")
                .unwrap()
        };
        type Create = unsafe extern "C" fn(usize, usize, usize, usize, usize) -> *mut c_void;
        type Destroy = unsafe extern "C" fn(*mut c_void);
        let create = unsafe {
            *lib.get::<Create>(b"rbitnet_cuda_attention_create\0")
                .unwrap()
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_attention_destroy\0")
                .unwrap()
        };
        let snapshot = || api.snapshot().unwrap();
        assert_eq!(snapshot().live, 0, "run this test in its own process");
        assert_eq!(unsafe { configure(64 * 1024, 256 * 1024 * 1024) }, 0);
        let payload = || {
            vec![1.0f32; 4096]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let matrix =
            crate::backend::CudaDeviceQuantMatrix::from_payload(Some(&rt), 0, payload(), 32, 128)
                .unwrap();
        assert!(matrix.is_device_resident());
        let clone = matrix.clone();
        drop(matrix);
        assert_eq!(snapshot().live, 16384);
        assert_eq!(snapshot().categories[WEIGHTS as usize], 16384);
        let context = unsafe { create(16, 1, 64, 64, 2) };
        assert!(!context.is_null());
        assert_eq!(snapshot().categories[1], 8192);
        assert_eq!(snapshot().categories[2], 1032);
        let allocate = |bytes, category| {
            let mut p = std::ptr::null_mut();
            let status = unsafe { (api.alloc)(&mut p, bytes, category) };
            (status, p)
        };
        let (rc, prefix) = allocate(8192, 3);
        assert_eq!(rc, 0);
        let (rc, experts) = allocate(24576, EXPERTS);
        assert_eq!(rc, 0);
        let before = snapshot();
        let (rc, empty) = allocate(0, 6);
        assert_eq!(rc, 0);
        assert!(empty.is_null());
        assert_eq!(snapshot().live, before.live);
        let (rc, failed) = allocate(16384, WEIGHTS);
        assert_ne!(rc, 0);
        assert!(failed.is_null());
        assert_eq!(snapshot().live, before.live);
        assert_eq!(snapshot().refusals, before.refusals + 1);
        // Best-effort residency retains the real CPU payload on cap refusal.
        let cpu =
            crate::backend::CudaDeviceQuantMatrix::from_payload(Some(&rt), 0, payload(), 32, 128)
                .unwrap();
        assert!(!cpu.is_device_resident());
        assert_eq!(cpu.matvec(&[1.0; 128]).unwrap(), vec![128.0; 32]);
        assert_eq!(snapshot().live, before.live);
        // First KV allocation fits; the second fails. Destruction rolls back
        // the first allocation without freeing another context's memory.
        assert!(unsafe { create(16, 1, 64, 64, 2) }.is_null());
        assert_eq!(snapshot().live, before.live);
        assert!(snapshot().peak <= snapshot().limit);
        unsafe { destroy(context) };
        assert_eq!(unsafe { (api.free)(prefix) }, 0);
        assert_ne!(
            unsafe { (api.free)(prefix) },
            0,
            "double free rejected before CUDA"
        );
        assert_eq!(unsafe { (api.free)(experts) }, 0);
        drop(cpu);
        drop(clone);
        drop(rt);
        assert_eq!(snapshot().live, 0);
        assert_eq!(snapshot().categories, [0; 7]);

        let gate = std::sync::Arc::new(std::sync::Barrier::new(9));
        let workers = (0..8)
            .map(|_| {
                let gate = std::sync::Arc::clone(&gate);
                std::thread::spawn(move || {
                    let mut p = std::ptr::null_mut();
                    let rc = unsafe { (api.alloc)(&mut p, 16384, 6) };
                    gate.wait();
                    gate.wait();
                    if rc == 0 {
                        assert_eq!(unsafe { (api.free)(p) }, 0);
                        true
                    } else {
                        assert!(p.is_null());
                        false
                    }
                })
            })
            .collect::<Vec<_>>();
        gate.wait();
        assert_eq!(snapshot().live, 65536);
        assert_eq!(snapshot().categories[6], 65536);
        gate.wait();
        assert_eq!(
            workers
                .into_iter()
                .map(|w| usize::from(w.join().unwrap()))
                .sum::<usize>(),
            4
        );
        assert_eq!(snapshot().live, 0);
        assert_eq!(snapshot().categories, [0; 7]);
        assert_eq!(unsafe { configure(32768, 0) }, 0);
        assert_eq!(snapshot().peak, 0, "a new cap starts a new peak epoch");
        assert_eq!(unsafe { configure(0, 0) }, 0);
        eprintln!("CUDA shared ledger: exact categories, shared views, CPU fallback, partial-create rollback, and 8-thread cap admission passed");
    }
}
