//! Compute backend abstractions for portable execution.

use crate::error::Result;
use libloading::Library;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
pub(crate) mod async_upload;
mod device_memory;
pub(crate) mod expert_arena;
pub use device_memory::{cuda_managed_memory_stats, CudaMemoryStats};

/// Backend identifiers used by runtime selection and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Hybrid,
    Rocm,
    Vulkan,
    Metal,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Hybrid => "hybrid",
            BackendKind::Rocm => "rocm",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Metal => "metal",
        }
    }

    pub fn from_env() -> Self {
        let raw = std::env::var("RBITNET_BACKEND").unwrap_or_else(|_| "auto".into());
        Self::select_from_value(&raw, "", Self::detect_best)
    }

    /// Automatic selection must account for what the model executor implements.
    /// Explicit accelerator requests remain explicit and are validated at load.
    pub(crate) fn from_env_for_architecture(architecture: &str) -> Self {
        let raw = std::env::var("RBITNET_BACKEND").unwrap_or_else(|_| "auto".into());
        Self::select_from_value(&raw, architecture, Self::detect_best)
    }

    fn select_from_value(raw: &str, architecture: &str, detect: impl FnOnce() -> Self) -> Self {
        let explicit = match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "hybrid" | "cpu-gpu" | "gpu-cpu" => Some(Self::Hybrid),
            "rocm" => Some(Self::Rocm),
            "vulkan" | "intel" | "level-zero" | "oneapi" => Some(Self::Vulkan),
            "metal" => Some(Self::Metal),
            _ => None,
        };
        match explicit {
            Some(kind) => kind,
            None if matches!(architecture, "mixtral" | "qwen3") => Self::Cpu,
            None => detect(),
        }
    }

    /// Prefer implemented CUDA/ROCm compute backends; else CPU. Loader-only
    /// Vulkan/Metal prototypes must not win automatic inference selection.
    pub fn detect_best() -> Self {
        if CudaRuntime::try_load().is_some() {
            return BackendKind::Cuda;
        }
        if RocmBackend::runtime_available() {
            return BackendKind::Rocm;
        }
        BackendKind::Cpu
    }
}

/// Minimal backend contract for memory and kernel dispatch.
pub trait ComputeBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn is_native_accelerated(&self) -> bool {
        false
    }
    fn alloc(&self, len: usize) -> Result<Vec<f32>>;
    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>>;
    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>>;
    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>>;
}

#[derive(Debug, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; len])
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        Ok(src.to_vec())
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        Ok(src.to_vec())
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; out_rows];
        for row in 0..out_rows {
            let mut acc = 0.0f32;
            let base = row * in_cols;
            for col in 0..in_cols {
                acc += w[base + col] * x[col];
            }
            out[row] = acc;
        }
        Ok(out)
    }
}

/// CUDA MVP backend: API-compatible with CPU path, using native cuBLAS GEMV when available.
///
/// Generic host `matvec` paths reuse pooled device buffers inside [`CudaRuntime`] (see
/// [`CudaRuntime::gemv_host_f32`]) instead of allocating per call.
#[derive(Debug)]
pub struct CudaBackend {
    cpu: CpuBackend,
    runtime: Option<Arc<CudaRuntime>>,
}

#[allow(non_camel_case_types)]
type cudaError_t = i32;
#[allow(non_camel_case_types)]
type cudaMemcpyKind = i32;
#[allow(non_camel_case_types)]
type cublasStatus_t = i32;
#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUDA_SUCCESS: cudaError_t = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: cudaMemcpyKind = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: cudaMemcpyKind = 2;
const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
const CUBLAS_OP_T: i32 = 1;

/// CUDA bootstrap state (CUDA runtime + optional cuBLAS) used by [`CudaBackend`] and native Qwen paths.
///
/// Loaded dynamically from the system CUDA stack; callers should treat failures as unavailable GPU.
///
/// `matvec_cuda` / [`CudaRuntime::gemv_device_weight_f32`] reuse a small triple of device buffers
/// (`d_w`, `d_x`, `d_y`) to avoid per-call `cudaMalloc` / `cudaFree` when shapes fit within the
/// pooled capacities (they grow as needed and are released on [`Drop`]).
///
/// Residency note (#22): pooled `d_w` in `matvec_cuda` is scratch for host-uploaded weights each
/// call. True device-resident weights go through [`CudaDeviceMatrix`] /
/// [`Self::gemv_device_weight_f32`], which increments
/// [`CudaRuntimeMetrics::device_resident_gemv_calls`].
pub struct CudaRuntime {
    _lib: Library,
    _cublas_lib: Option<Library>,
    cuda_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
    cuda_free: unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    cuda_memcpy:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t,
    cuda_device_synchronize: unsafe extern "C" fn() -> cudaError_t,
    cuda_stream_synchronize: unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    memory: Option<device_memory::DeviceMemory>,
    cublas_create_v2: Option<unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t>,
    cublas_destroy_v2: Option<unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t>,
    cublas_sgemv_v2: Option<
        unsafe extern "C" fn(
            cublasHandle_t,
            i32,
            i32,
            i32,
            *const f32,
            *const f32,
            i32,
            *const f32,
            i32,
            *const f32,
            *mut f32,
            i32,
        ) -> cublasStatus_t,
    >,
    cublas_handle: Mutex<Option<usize>>,
    pooled_gemv: Mutex<PooledGemvBufs>,
    upload_bytes: AtomicU64,
    download_bytes: AtomicU64,
    gemv_calls: AtomicU64,
    /// Successful GEMVs where weight matrix was already on device (not re-uploaded).
    device_resident_gemv_calls: AtomicU64,
    /// Successful quantized GEMVs where **W** stayed on device ([`CudaDeviceQuantMatrix`]).
    device_resident_quant_gemv_calls: AtomicU64,
}

/// Reusable device allocations for the generic `f32` GEMV helper (`matvec_cuda`).
struct PooledGemvBufs {
    d_w: *mut c_void,
    d_x: *mut c_void,
    d_y: *mut c_void,
    cap_w: usize,
    cap_x: usize,
    cap_y: usize,
}

impl Default for PooledGemvBufs {
    fn default() -> Self {
        Self {
            d_w: std::ptr::null_mut(),
            d_x: std::ptr::null_mut(),
            d_y: std::ptr::null_mut(),
            cap_w: 0,
            cap_x: 0,
            cap_y: 0,
        }
    }
}

// Device pointers are owned by this struct and only accessed while holding `pooled_gemv` lock
// on `CudaRuntime` (CUDA API is not thread-safe across arbitrary concurrent callers anyway).
unsafe impl Send for PooledGemvBufs {}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaRuntime(loaded)")
    }
}

impl CudaRuntime {
    /// Try loading CUDA runtime and cuBLAS from the usual system library names for this platform.
    pub fn try_load() -> Option<std::sync::Arc<Self>> {
        Self::load().map(std::sync::Arc::new)
    }

    pub fn roundtrip_host_f32(&self, src: &[f32]) -> Option<Vec<f32>> {
        self.roundtrip_f32(src)
    }

    pub fn gemv_host_f32(
        &self,
        w: &[f32],
        x: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Option<Vec<f32>> {
        self.matvec_cuda(w, x, out_rows, in_cols)
    }

    pub fn has_cublas_gemv(&self) -> bool {
        self.cublas_create_v2.is_some()
            && self.cublas_destroy_v2.is_some()
            && self.cublas_sgemv_v2.is_some()
    }

    pub fn metrics_snapshot(&self) -> CudaRuntimeMetrics {
        CudaRuntimeMetrics {
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            download_bytes: self.download_bytes.load(Ordering::Relaxed),
            gemv_calls: self.gemv_calls.load(Ordering::Relaxed),
            device_resident_gemv_calls: self.device_resident_gemv_calls.load(Ordering::Relaxed),
            device_resident_quant_gemv_calls: self
                .device_resident_quant_gemv_calls
                .load(Ordering::Relaxed),
        }
    }

    pub fn managed_memory_stats(&self) -> Option<CudaMemoryStats> {
        self.memory?.snapshot()
    }

    pub fn upload_f32(self: &Arc<Self>, src: &[f32]) -> Option<CudaDeviceBuffer> {
        let nbytes = src.len().checked_mul(std::mem::size_of::<f32>())?;
        self.upload_raw(src.as_ptr().cast::<c_void>(), nbytes)
    }

    /// Upload an arbitrary host byte blob (e.g. GGML quantized payload) once.
    pub fn upload_raw(
        self: &Arc<Self>,
        src: *const c_void,
        nbytes: usize,
    ) -> Option<CudaDeviceBuffer> {
        self.upload_raw_category(src, nbytes, device_memory::WEIGHTS)
    }

    fn upload_raw_category(
        self: &Arc<Self>,
        src: *const c_void,
        nbytes: usize,
        category: u32,
    ) -> Option<CudaDeviceBuffer> {
        if nbytes == 0 {
            return None;
        }
        let ptr = self.alloc_device_category(nbytes, category)?;
        if !self.copy_host_to_device(ptr, src, nbytes) {
            self.free_device(ptr);
            return None;
        }
        Some(CudaDeviceBuffer {
            rt: Arc::clone(self),
            ptr: ptr as usize,
            nbytes,
            owner: None,
        })
    }

    pub fn upload_u8(self: &Arc<Self>, src: &[u8]) -> Option<CudaDeviceBuffer> {
        self.upload_raw(src.as_ptr().cast::<c_void>(), src.len())
    }

    pub(crate) fn record_device_resident_quant_gemv(&self) {
        self.device_resident_quant_gemv_calls
            .fetch_add(1, Ordering::Relaxed);
        self.gemv_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// GEMV with weight matrix already resident on device (`d_w`).
    ///
    /// On success increments both `gemv_calls` (via cuBLAS) and
    /// [`CudaRuntimeMetrics::device_resident_gemv_calls`] so operators can tell resident-weight
    /// traffic apart from host-upload `matvec_cuda` (see `docs/GPU_NATIVE_ROADMAP.md`).
    pub fn gemv_device_weight_f32(
        &self,
        d_w: *mut c_void,
        x: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != in_cols {
            return None;
        }
        let x_bytes = x.len().checked_mul(std::mem::size_of::<f32>())?;
        let y_bytes = out_rows.checked_mul(std::mem::size_of::<f32>())?;
        let mut guard = self.pooled_gemv.lock().ok()?;
        let pool = &mut *guard;
        self.pooled_ensure_xy(pool, x_bytes, y_bytes)?;
        let d_x = pool.d_x;
        let d_y = pool.d_y;
        let mut out = vec![0.0f32; out_rows];
        let ok = self.copy_host_to_device(d_x, x.as_ptr().cast::<c_void>(), x_bytes)
            && self.cublas_sgemv_device(d_w, d_x, d_y, out_rows, in_cols)
            && self.copy_device_to_host(out.as_mut_ptr().cast::<c_void>(), d_y, y_bytes);
        let _ = unsafe { (self.cuda_device_synchronize)() };
        if ok {
            self.device_resident_gemv_calls
                .fetch_add(1, Ordering::Relaxed);
            Some(out)
        } else {
            None
        }
    }

    fn load() -> Option<Self> {
        // Finish the CUDA runtime's lazy driver initialization before another model loader
        // opens the same DLL. Keep per-model buffers and metrics independent; only cold
        // initialization is serialized, never kernel execution or graph replay.
        static INITIALIZATION: Mutex<()> = Mutex::new(());
        let _initialization = INITIALIZATION.lock().ok()?;
        let mut candidates: Vec<String> = Vec::new();
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            for rel in [
                "bin/x64/cudart64_13.dll",
                "bin/cudart64_13.dll",
                "bin/x64/cudart64_12.dll",
                "bin/cudart64_12.dll",
                "bin/x64/cudart64_11.dll",
                "bin/cudart64_11.dll",
                "lib64/libcudart.so",
                "lib64/libcudart.so.13",
                "lib64/libcudart.so.12",
            ] {
                candidates.push(format!(
                    "{}{}{}",
                    cuda_path.trim_end_matches(['/', '\\']),
                    std::path::MAIN_SEPARATOR,
                    rel.replace('/', std::path::MAIN_SEPARATOR_STR)
                ));
            }
        }
        for path in [
            "cudart64_13.dll",
            "cudart64_12.dll",
            "cudart64_11.dll",
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.dylib",
        ] {
            candidates.push(path.to_string());
        }
        for path in &candidates {
            let Ok(lib) = (unsafe { Library::new(path.as_str()) }) else {
                continue;
            };
            let cuda_malloc = unsafe {
                let sym: libloading::Symbol<
                    unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
                > = lib.get(b"cudaMalloc").ok()?;
                *sym
            };
            let cuda_free = unsafe {
                let sym: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> cudaError_t> =
                    lib.get(b"cudaFree").ok()?;
                *sym
            };
            if unsafe { cuda_free(null_mut()) } != CUDA_SUCCESS {
                continue;
            }
            let cuda_memcpy = unsafe {
                let sym: libloading::Symbol<
                    unsafe extern "C" fn(
                        *mut c_void,
                        *const c_void,
                        usize,
                        cudaMemcpyKind,
                    ) -> cudaError_t,
                > = lib.get(b"cudaMemcpy").ok()?;
                *sym
            };
            let cuda_device_synchronize = unsafe {
                let sym: libloading::Symbol<unsafe extern "C" fn() -> cudaError_t> =
                    lib.get(b"cudaDeviceSynchronize").ok()?;
                *sym
            };
            let cuda_stream_synchronize = unsafe {
                *lib.get::<unsafe extern "C" fn(*mut c_void) -> cudaError_t>(
                    b"cudaStreamSynchronize",
                )
                .ok()?
            };
            let mut cublas_lib: Option<Library> = None;
            let mut cublas_create_v2 = None;
            let mut cublas_destroy_v2 = None;
            let mut cublas_sgemv_v2 = None;
            let mut cublas_candidates: Vec<String> = Vec::new();
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                for rel in [
                    "bin/x64/cublas64_13.dll",
                    "bin/cublas64_13.dll",
                    "bin/x64/cublas64_12.dll",
                    "bin/cublas64_12.dll",
                    "bin/x64/cublas64_11.dll",
                    "bin/cublas64_11.dll",
                    "lib64/libcublas.so",
                    "lib64/libcublas.so.13",
                    "lib64/libcublas.so.12",
                ] {
                    cublas_candidates.push(format!(
                        "{}{}{}",
                        cuda_path.trim_end_matches(['/', '\\']),
                        std::path::MAIN_SEPARATOR,
                        rel.replace('/', std::path::MAIN_SEPARATOR_STR)
                    ));
                }
            }
            for cb in [
                "cublas64_13.dll",
                "cublas64_12.dll",
                "cublas64_11.dll",
                "libcublas.so",
                "libcublas.so.13",
                "libcublas.so.12",
                "libcublas.dylib",
            ] {
                cublas_candidates.push(cb.to_string());
            }
            for cb_path in &cublas_candidates {
                let Ok(cb_lib) = (unsafe { Library::new(cb_path.as_str()) }) else {
                    continue;
                };
                unsafe {
                    let s_create: libloading::Symbol<
                        unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t,
                    > = match cb_lib.get(b"cublasCreate_v2") {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let s_destroy: libloading::Symbol<
                        unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t,
                    > = match cb_lib.get(b"cublasDestroy_v2") {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let s_gemv: libloading::Symbol<
                        unsafe extern "C" fn(
                            cublasHandle_t,
                            i32,
                            i32,
                            i32,
                            *const f32,
                            *const f32,
                            i32,
                            *const f32,
                            i32,
                            *const f32,
                            *mut f32,
                            i32,
                        ) -> cublasStatus_t,
                    > = match cb_lib.get(b"cublasSgemv_v2") {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    cublas_create_v2 = Some(*s_create);
                    cublas_destroy_v2 = Some(*s_destroy);
                    cublas_sgemv_v2 = Some(*s_gemv);
                    cublas_lib = Some(cb_lib);
                    break;
                }
            }
            let memory = match device_memory::DeviceMemory::load() {
                Ok(memory) => memory,
                Err(error) => {
                    tracing::warn!(%error, "CUDA managed allocator unavailable");
                    return None;
                }
            };
            return Some(Self {
                _lib: lib,
                _cublas_lib: cublas_lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cuda_device_synchronize,
                cuda_stream_synchronize,
                memory,
                cublas_create_v2,
                cublas_destroy_v2,
                cublas_sgemv_v2,
                cublas_handle: Mutex::new(None),
                pooled_gemv: Mutex::new(PooledGemvBufs::default()),
                upload_bytes: AtomicU64::new(0),
                download_bytes: AtomicU64::new(0),
                gemv_calls: AtomicU64::new(0),
                device_resident_gemv_calls: AtomicU64::new(0),
                device_resident_quant_gemv_calls: AtomicU64::new(0),
            });
        }
        None
    }

    fn alloc_device(&self, nbytes: usize) -> Option<*mut c_void> {
        self.alloc_device_category(nbytes, device_memory::SCRATCH)
    }

    fn alloc_device_category(&self, nbytes: usize, category: u32) -> Option<*mut c_void> {
        let mut dev_ptr: *mut c_void = null_mut();
        let status = unsafe {
            match self.memory {
                Some(api) => (api.alloc)(&mut dev_ptr, nbytes, category),
                None => (self.cuda_malloc)(&mut dev_ptr, nbytes),
            }
        };
        if status != CUDA_SUCCESS {
            tracing::warn!(status, nbytes, "CUDA allocation failed");
        }
        let ok = status == CUDA_SUCCESS;
        ok.then_some(dev_ptr)
    }

    fn free_device(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            let status = unsafe {
                match self.memory {
                    Some(api) => (api.free)(ptr),
                    None => (self.cuda_free)(ptr),
                }
            };
            if status != CUDA_SUCCESS {
                tracing::warn!(status, "CUDA allocation release failed; charge retained");
            }
        }
    }

    fn copy_host_to_device(&self, dst: *mut c_void, src: *const c_void, nbytes: usize) -> bool {
        let mut status =
            unsafe { (self.cuda_memcpy)(dst, src, nbytes, CUDA_MEMCPY_HOST_TO_DEVICE) };
        // Pageable H2D cudaMemcpy may return after staging, before DMA ends.
        // Our consumers use nonblocking private streams and cannot implicitly
        // depend on the default stream. An owned resident/refilled weight is
        // published only after that copy has actually reached the device.
        if status == CUDA_SUCCESS {
            status = unsafe { (self.cuda_stream_synchronize)(null_mut()) };
        }
        if status != CUDA_SUCCESS {
            tracing::warn!(status, nbytes, "CUDA host upload failed");
        }
        let ok = status == CUDA_SUCCESS;
        if ok {
            self.upload_bytes
                .fetch_add(nbytes as u64, Ordering::Relaxed);
            crate::perf::record_gpu_transfer(nbytes as u64, 0, 0);
        }
        ok
    }

    fn copy_device_to_host(&self, dst: *mut c_void, src: *const c_void, nbytes: usize) -> bool {
        let ok = unsafe { (self.cuda_memcpy)(dst, src, nbytes, CUDA_MEMCPY_DEVICE_TO_HOST) }
            == CUDA_SUCCESS;
        if ok {
            self.download_bytes
                .fetch_add(nbytes as u64, Ordering::Relaxed);
            crate::perf::record_gpu_transfer(0, nbytes as u64, 0);
        }
        ok
    }

    fn pooled_ensure_triple(
        &self,
        pool: &mut PooledGemvBufs,
        w_bytes: usize,
        x_bytes: usize,
        y_bytes: usize,
    ) -> Option<()> {
        if pool.cap_w < w_bytes {
            self.free_device(pool.d_w);
            pool.d_w = null_mut();
            pool.cap_w = 0;
            pool.d_w = self.alloc_device(w_bytes)?;
            pool.cap_w = w_bytes;
        }
        if pool.cap_x < x_bytes {
            self.free_device(pool.d_x);
            pool.d_x = null_mut();
            pool.cap_x = 0;
            pool.d_x = self.alloc_device(x_bytes)?;
            pool.cap_x = x_bytes;
        }
        if pool.cap_y < y_bytes {
            self.free_device(pool.d_y);
            pool.d_y = null_mut();
            pool.cap_y = 0;
            pool.d_y = self.alloc_device(y_bytes)?;
            pool.cap_y = y_bytes;
        }
        Some(())
    }

    /// Resize only `d_x` / `d_y` for [`Self::gemv_device_weight_f32`] (weight already on device).
    fn pooled_ensure_xy(
        &self,
        pool: &mut PooledGemvBufs,
        x_bytes: usize,
        y_bytes: usize,
    ) -> Option<()> {
        if pool.cap_x < x_bytes {
            self.free_device(pool.d_x);
            pool.d_x = null_mut();
            pool.cap_x = 0;
            pool.d_x = self.alloc_device(x_bytes)?;
            pool.cap_x = x_bytes;
        }
        if pool.cap_y < y_bytes {
            self.free_device(pool.d_y);
            pool.d_y = null_mut();
            pool.cap_y = 0;
            pool.d_y = self.alloc_device(y_bytes)?;
            pool.cap_y = y_bytes;
        }
        Some(())
    }

    fn cublas_handle(&self) -> Option<cublasHandle_t> {
        let create = self.cublas_create_v2?;
        let mut guard = self.cublas_handle.lock().ok()?;
        if let Some(raw) = *guard {
            return Some(raw as cublasHandle_t);
        }
        let mut handle: cublasHandle_t = null_mut();
        let ok = unsafe { create(&mut handle as *mut cublasHandle_t) } == CUBLAS_STATUS_SUCCESS;
        if ok {
            *guard = Some(handle as usize);
            Some(handle)
        } else {
            None
        }
    }

    fn cublas_sgemv_device(
        &self,
        d_w: *const c_void,
        d_x: *const c_void,
        d_y: *mut c_void,
        out_rows: usize,
        in_cols: usize,
    ) -> bool {
        let Some(sgemv) = self.cublas_sgemv_v2 else {
            return false;
        };
        let Some(handle) = self.cublas_handle() else {
            return false;
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            sgemv(
                handle,
                CUBLAS_OP_T,
                in_cols as i32,
                out_rows as i32,
                &alpha as *const f32,
                d_w.cast::<f32>(),
                in_cols as i32,
                d_x.cast::<f32>(),
                1,
                &beta as *const f32,
                d_y.cast::<f32>(),
                1,
            )
        };
        let ok = status == CUBLAS_STATUS_SUCCESS;
        if ok {
            self.gemv_calls.fetch_add(1, Ordering::Relaxed);
            crate::perf::record_gpu_transfer(0, 0, 1);
        }
        ok
    }

    fn roundtrip_f32(&self, src: &[f32]) -> Option<Vec<f32>> {
        let nbytes = src.len().checked_mul(std::mem::size_of::<f32>())?;
        let dev_ptr = self.alloc_device(nbytes)?;
        let mut out = vec![0.0f32; src.len()];
        unsafe {
            let ok_h2d = (self.cuda_memcpy)(
                dev_ptr,
                src.as_ptr().cast::<c_void>(),
                nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ) == CUDA_SUCCESS;
            let ok_d2h = (self.cuda_memcpy)(
                out.as_mut_ptr().cast::<c_void>(),
                dev_ptr,
                nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ) == CUDA_SUCCESS;
            let _ = (self.cuda_device_synchronize)();
            self.free_device(dev_ptr);
            if !ok_h2d || !ok_d2h {
                return None;
            }
        }
        Some(out)
    }

    fn matvec_cuda(
        &self,
        w: &[f32],
        x: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Option<Vec<f32>> {
        let w_bytes = w.len().checked_mul(std::mem::size_of::<f32>())?;
        let x_bytes = x.len().checked_mul(std::mem::size_of::<f32>())?;
        let y_bytes = out_rows.checked_mul(std::mem::size_of::<f32>())?;
        let mut guard = self.pooled_gemv.lock().ok()?;
        let pool = &mut *guard;
        self.pooled_ensure_triple(pool, w_bytes, x_bytes, y_bytes)?;
        let d_w = pool.d_w;
        let d_x = pool.d_x;
        let d_y = pool.d_y;
        let mut out = vec![0.0f32; out_rows];
        let ok = self.copy_host_to_device(d_w, w.as_ptr().cast::<c_void>(), w_bytes)
            && self.copy_host_to_device(d_x, x.as_ptr().cast::<c_void>(), x_bytes)
            && self.cublas_sgemv_device(d_w, d_x, d_y, out_rows, in_cols)
            && self.copy_device_to_host(out.as_mut_ptr().cast::<c_void>(), d_y, y_bytes);
        let _ = unsafe { (self.cuda_device_synchronize)() };
        if !ok {
            return None;
        }
        Some(out)
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pooled_gemv.lock() {
            self.free_device(pool.d_w);
            self.free_device(pool.d_x);
            self.free_device(pool.d_y);
            pool.d_w = null_mut();
            pool.d_x = null_mut();
            pool.d_y = null_mut();
            pool.cap_w = 0;
            pool.cap_x = 0;
            pool.cap_y = 0;
        }
        if let (Some(destroy), Ok(mut guard)) = (self.cublas_destroy_v2, self.cublas_handle.lock())
        {
            if let Some(raw) = guard.take() {
                let _ = unsafe { destroy(raw as cublasHandle_t) };
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaRuntimeMetrics {
    pub upload_bytes: u64,
    pub download_bytes: u64,
    /// All successful cuBLAS SGEMV calls (host-upload and device-resident weight paths).
    pub gemv_calls: u64,
    /// Successful GEMVs where **W** was already on device ([`CudaRuntime::gemv_device_weight_f32`]).
    /// Spike metric for #22 residency checklist; CI does not require CUDA to exercise this.
    pub device_resident_gemv_calls: u64,
    /// Successful quantized GEMVs with device-resident **W** ([`CudaDeviceQuantMatrix`]).
    /// Requires optional `librbitnet_cuda_quant` device symbols; CI stays CUDA-free.
    pub device_resident_quant_gemv_calls: u64,
}

#[derive(Debug)]
pub struct CudaDeviceBuffer {
    rt: Arc<CudaRuntime>,
    ptr: usize,
    nbytes: usize,
    owner: Option<Arc<CudaDeviceBuffer>>,
}

unsafe impl Send for CudaDeviceBuffer {}
unsafe impl Sync for CudaDeviceBuffer {}

impl CudaDeviceBuffer {
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn as_device_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }

    pub fn download_f32(&self) -> Option<Vec<f32>> {
        if self.nbytes % std::mem::size_of::<f32>() != 0 {
            return None;
        }
        let n = self.nbytes / std::mem::size_of::<f32>();
        let mut out = vec![0.0f32; n];
        let ok = self.rt.copy_device_to_host(
            out.as_mut_ptr().cast::<c_void>(),
            self.ptr as *const c_void,
            self.nbytes,
        );
        ok.then_some(out)
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.owner.is_none() {
            self.rt.free_device(self.ptr as *mut c_void);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaDeviceMatrix {
    buffer: Arc<CudaDeviceBuffer>,
    out_rows: usize,
    in_cols: usize,
}

impl CudaDeviceMatrix {
    pub fn upload(
        rt: Arc<CudaRuntime>,
        w: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Option<Self> {
        if w.len() != out_rows.checked_mul(in_cols)? {
            return None;
        }
        let buffer = Arc::new(rt.upload_f32(w)?);
        Some(Self {
            buffer,
            out_rows,
            in_cols,
        })
    }

    pub fn matvec(&self, x: &[f32]) -> Option<Vec<f32>> {
        self.buffer.rt.gemv_device_weight_f32(
            self.buffer.ptr as *mut c_void,
            x,
            self.out_rows,
            self.in_cols,
        )
    }

    pub fn bytes(&self) -> usize {
        self.buffer.nbytes()
    }
}

/// Device-resident **quantized** weight matrix for the #22 CUDA vertical.
///
/// Host payload is always retained for CPU golden / fallback. When CUDA loads, the same
/// bytes are uploaded once; matvec prefers optional `*_matvec_device` symbols from
/// `librbitnet_cuda_quant`, else falls back to host [`crate::ggml::matvec_payload_quant`].
enum QuantHostBacking {
    Owned(Vec<u8>),
    Mapped {
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: crate::gguf::GgufTensorInfo,
        start: usize,
        length: usize,
    },
}
impl std::fmt::Debug for QuantHostBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(bytes) => f
                .debug_struct("OwnedQuantBytes")
                .field("length", &bytes.len())
                .finish(),
            Self::Mapped {
                tensor,
                start,
                length,
                ..
            } => f
                .debug_struct("MappedQuantBytes")
                .field("tensor", &tensor.name)
                .field("start", start)
                .field("length", length)
                .finish(),
        }
    }
}
impl std::ops::Deref for QuantHostBacking {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped {
                archive,
                tensor,
                start,
                length,
            } => {
                // Both tensor descriptor and checked span are immutable owners.
                &archive
                    .tensor_payload(tensor)
                    .expect("validated immutable quant tensor")[*start..*start + *length]
            }
        }
    }
}
impl QuantHostBacking {
    fn mapped(
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: &crate::gguf::GgufTensorInfo,
        start: usize,
        length: usize,
    ) -> crate::error::Result<Self> {
        let end = start.checked_add(length).ok_or_else(|| {
            crate::error::BitNetError::Inference("mapped quant range overflow".into())
        })?;
        if length == 0 || end > archive.tensor_payload(tensor)?.len() {
            return Err(crate::error::BitNetError::Inference(
                "mapped quant range out of bounds".into(),
            ));
        }
        Ok(Self::Mapped {
            archive,
            tensor: tensor.clone(),
            start,
            length,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CudaDeviceQuantMatrix {
    host: Arc<QuantHostBacking>,
    device: Option<Arc<CudaDeviceBuffer>>,
    rt: Option<Arc<CudaRuntime>>,
    ggml_type: u32,
    out_rows: usize,
    in_cols: usize,
    row_bytes: usize,
}

impl CudaDeviceQuantMatrix {
    /// Build a resident quant matrix. Device upload is best-effort (None without CUDA).
    pub fn from_payload(
        rt: Option<&Arc<CudaRuntime>>,
        ggml_type: u32,
        payload: Vec<u8>,
        out_rows: usize,
        in_cols: usize,
    ) -> crate::error::Result<Self> {
        Self::from_payload_category(
            rt,
            ggml_type,
            payload,
            out_rows,
            in_cols,
            device_memory::WEIGHTS,
        )
    }

    fn from_payload_category(
        rt: Option<&Arc<CudaRuntime>>,
        ggml_type: u32,
        payload: Vec<u8>,
        out_rows: usize,
        in_cols: usize,
        category: u32,
    ) -> crate::error::Result<Self> {
        Self::from_backing_category(
            rt,
            ggml_type,
            QuantHostBacking::Owned(payload),
            out_rows,
            in_cols,
            category,
        )
    }
    fn from_backing_category(
        rt: Option<&Arc<CudaRuntime>>,
        ggml_type: u32,
        payload: QuantHostBacking,
        out_rows: usize,
        in_cols: usize,
        category: u32,
    ) -> crate::error::Result<Self> {
        let row_bytes = crate::ggml::ggml_row_size(ggml_type, in_cols as u64)?;
        let need = row_bytes.checked_mul(out_rows).ok_or_else(|| {
            crate::error::BitNetError::Inference("quant payload size overflow".into())
        })?;
        if payload.len() < need {
            return Err(crate::error::BitNetError::Inference(
                "quant payload truncated for device residency".into(),
            ));
        }
        let host = Arc::new(payload);
        let (device, rt_keep) = if let Some(rt) = rt {
            match rt.upload_raw_category(host.as_ptr().cast::<c_void>(), need, category) {
                Some(buf) => (Some(Arc::new(buf)), Some(Arc::clone(rt))),
                None => (None, Some(Arc::clone(rt))),
            }
        } else {
            (None, None)
        };
        Ok(Self {
            host,
            device,
            rt: rt_keep,
            ggml_type,
            out_rows,
            in_cols,
            row_bytes,
        })
    }

    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }

    pub fn out_rows(&self) -> usize {
        self.out_rows
    }

    pub fn in_cols(&self) -> usize {
        self.in_cols
    }

    pub fn host_payload(&self) -> &[u8] {
        &self.host
    }

    pub fn is_device_resident(&self) -> bool {
        self.device.is_some()
    }

    /// Snapshot of the shared device runtime counters backing this matrix.
    ///
    /// All matrices built from one model load share a single `CudaRuntime`,
    /// so any resident matrix reports the same process for that load.
    /// Returns `None` for CPU-only matrices without a device runtime.
    pub fn runtime_metrics(&self) -> Option<CudaRuntimeMetrics> {
        self.rt.as_ref().map(|rt| rt.metrics_snapshot())
    }

    pub(crate) fn device_address(&self) -> Option<usize> {
        self.device.as_ref().map(|buffer| buffer.ptr)
    }

    /// Inspect the allocation itself; host backing alone cannot prove a refill.
    /// This synchronous diagnostic is compiled only into tests.
    #[cfg(test)]
    pub(crate) fn download_device_payload_for_test(&self) -> Option<Vec<u8>> {
        let device = self.device.as_ref()?;
        let mut bytes = vec![0; self.bytes()];
        device
            .rt
            .copy_device_to_host(
                bytes.as_mut_ptr().cast(),
                device.as_device_ptr(),
                bytes.len(),
            )
            .then_some(bytes)
    }

    pub fn bytes(&self) -> usize {
        self.row_bytes.saturating_mul(self.out_rows)
    }

    /// Reuse an exclusively owned, compatible expert slot after its last step.
    #[cfg(test)]
    pub(crate) fn refill(&mut self, payload: Vec<u8>) -> crate::error::Result<bool> {
        if payload.len() != self.bytes() {
            return Err(crate::error::BitNetError::Inference(
                "expert refill shape mismatch".into(),
            ));
        }
        let (Some(device), Some(rt)) = (&mut self.device, &self.rt) else {
            return Ok(false);
        };
        let Some(buffer) = Arc::get_mut(device) else {
            return Ok(false);
        };
        if !rt.copy_host_to_device(
            buffer.as_device_ptr(),
            payload.as_ptr().cast(),
            payload.len(),
        ) {
            return Err(crate::error::BitNetError::Inference(
                "expert refill CUDA upload failed".into(),
            ));
        }
        self.host = Arc::new(QuantHostBacking::Owned(payload));
        Ok(true)
    }
    /// Immutable GGUF span: keep the original mmap for CPU fallback without a
    /// permanent Vec mirror. The caller chooses weights or experts allocation.
    pub(crate) fn from_archive_range(
        rt: Option<&Arc<CudaRuntime>>,
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: &crate::gguf::GgufTensorInfo,
        start: usize,
        out_rows: usize,
        in_cols: usize,
        expert: bool,
    ) -> crate::error::Result<Self> {
        let row_bytes = crate::ggml::ggml_row_size(tensor.ggml_type, in_cols as u64)?;
        if tensor.dimensions.first().copied() != Some(in_cols as u64)
            || row_bytes == 0
            || start % row_bytes != 0
        {
            return Err(crate::error::BitNetError::Inference(
                "mapped quant row geometry mismatch".into(),
            ));
        }
        let length = row_bytes.checked_mul(out_rows).ok_or_else(|| {
            crate::error::BitNetError::Inference("mapped quant size overflow".into())
        })?;
        let backing = QuantHostBacking::mapped(archive, tensor, start, length)?;
        Self::from_backing_category(
            rt,
            tensor.ggml_type,
            backing,
            out_rows,
            in_cols,
            if expert {
                device_memory::EXPERTS
            } else {
                device_memory::WEIGHTS
            },
        )
    }
    pub(crate) fn refill_archive_range(
        &mut self,
        archive: Arc<crate::gguf::GgufArchive>,
        tensor: &crate::gguf::GgufTensorInfo,
        start: usize,
    ) -> crate::error::Result<bool> {
        if tensor.ggml_type != self.ggml_type
            || tensor.dimensions.first().copied() != Some(self.in_cols as u64)
            || start % self.row_bytes != 0
        {
            return Err(crate::error::BitNetError::Inference(
                "mapped expert refill type mismatch".into(),
            ));
        }
        let backing = QuantHostBacking::mapped(archive, tensor, start, self.bytes())?;
        let (Some(device), Some(rt)) = (&mut self.device, &self.rt) else {
            return Ok(false);
        };
        let Some(buffer) = Arc::get_mut(device) else {
            return Ok(false);
        };
        if !rt.copy_host_to_device(
            buffer.as_device_ptr(),
            backing.as_ptr().cast(),
            backing.len(),
        ) {
            return Err(crate::error::BitNetError::Inference(
                "mapped expert refill CUDA upload failed".into(),
            ));
        }
        self.host = Arc::new(backing);
        Ok(true)
    }

    /// Prefer device-resident CUDA quant kernel; otherwise CPU payload matvec (golden parity).
    pub fn matvec(&self, x: &[f32]) -> crate::error::Result<Vec<f32>> {
        self.matvec_rows(x, 0, self.out_rows)
    }

    /// A row-aligned view, used for selected expert slabs without uploading them again.
    pub fn matvec_rows(
        &self,
        x: &[f32],
        first_row: usize,
        rows: usize,
    ) -> crate::error::Result<Vec<f32>> {
        if x.len() != self.in_cols
            || first_row
                .checked_add(rows)
                .filter(|&n| n <= self.out_rows)
                .is_none()
        {
            return Err(crate::error::BitNetError::Inference(
                "quant matrix view out of bounds".into(),
            ));
        }
        let offset = first_row * self.row_bytes;
        if let (Some(dev), Some(rt)) = (self.device.as_ref(), self.rt.as_ref()) {
            if let Some(result) = crate::ggml::matvec_device_quant_optional(
                self.ggml_type,
                unsafe { dev.as_device_ptr().cast::<u8>().add(offset).cast() },
                self.row_bytes,
                x,
                rows,
            ) {
                let y = result?;
                rt.record_device_resident_quant_gemv();
                crate::perf::record_gpu_transfer((x.len() * 4) as u64, (y.len() * 4) as u64, 1);
                return Ok(y);
            }
        }
        crate::ggml::matvec_payload_quant(
            self.ggml_type,
            &self.host[offset..offset + rows * self.row_bytes],
            x,
            self.in_cols,
            rows,
        )
    }

    /// Independent input vector for each equally sized slab (e.g. MLA heads).
    pub fn matvec_batch(&self, x: &[f32], rows_per_batch: usize) -> crate::error::Result<Vec<f32>> {
        if rows_per_batch == 0 || self.out_rows % rows_per_batch != 0 {
            return Err(crate::error::BitNetError::Inference(
                "invalid quant batch dimensions".into(),
            ));
        }
        let batches = self.out_rows / rows_per_batch;
        if x.len() != self.in_cols.saturating_mul(batches) {
            return Err(crate::error::BitNetError::Inference(
                "invalid quant batch input".into(),
            ));
        }
        if let (Some(dev), Some(rt)) = (self.device.as_ref(), self.rt.as_ref()) {
            if let Some(result) = crate::ggml::matvec_device_quant_batch_optional(
                self.ggml_type,
                dev.as_device_ptr(),
                self.row_bytes,
                x,
                self.in_cols,
                rows_per_batch,
                batches,
            ) {
                let output = result?;
                rt.record_device_resident_quant_gemv();
                crate::perf::record_gpu_transfer(
                    (x.len() * 4) as u64,
                    (output.len() * 4) as u64,
                    1,
                );
                return Ok(output);
            }
        }
        let mut result = Vec::with_capacity(self.out_rows);
        for batch in 0..batches {
            result.extend(self.matvec_rows(
                &x[batch * self.in_cols..(batch + 1) * self.in_cols],
                batch * rows_per_batch,
                rows_per_batch,
            )?);
        }
        Ok(result)
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime: CudaRuntime::try_load(),
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }
    fn is_native_accelerated(&self) -> bool {
        self.runtime.is_some()
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        self.cpu.alloc(len)
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        if let Some(rt) = &self.runtime {
            if let Some(out) = rt.matvec_cuda(w, x, out_rows, in_cols) {
                return Ok(out);
            }
        }
        self.cpu.matvec(w, x, out_rows, in_cols)
    }
}

/// Hybrid CPU/GPU backend: CPU for orchestration and fallback, CUDA for model-specific offload.
#[derive(Debug)]
pub struct HybridBackend {
    cpu: CpuBackend,
    runtime: Option<Arc<CudaRuntime>>,
}

impl Default for HybridBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime: CudaRuntime::try_load(),
        }
    }
}

impl ComputeBackend for HybridBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Hybrid
    }

    fn is_native_accelerated(&self) -> bool {
        self.runtime.is_some()
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        self.cpu.alloc(len)
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        if let Some(rt) = &self.runtime {
            if let Some(out) = rt.gemv_host_f32(w, x, out_rows, in_cols) {
                return Ok(out);
            }
        }
        self.cpu.matvec(w, x, out_rows, in_cols)
    }
}

/// ROCm backend: hipBLAS SGEMV when AMD HIP + hipBLAS load; otherwise CPU fallback.
#[derive(Debug)]
pub struct RocmBackend {
    cpu: CpuBackend,
    runtime_loaded: bool,
    runtime: Option<Arc<RocmRuntime>>,
}

/// ROCm / hipBLAS runtime (dynamically loaded). Mirrors the CUDA f32 GEMV path.
pub struct RocmRuntime {
    _hip: Library,
    _hipblas: Library,
    hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    hip_free: unsafe extern "C" fn(*mut c_void) -> i32,
    hip_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    hip_sync: unsafe extern "C" fn() -> i32,
    hipblas_create: unsafe extern "C" fn(*mut *mut c_void) -> i32,
    hipblas_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    hipblas_sgemv: unsafe extern "C" fn(
        *mut c_void,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
    ) -> i32,
    handle: Mutex<Option<usize>>,
}

unsafe impl Send for RocmRuntime {}
unsafe impl Sync for RocmRuntime {}

impl std::fmt::Debug for RocmRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RocmRuntime(loaded)")
    }
}

impl RocmRuntime {
    const HIP_SUCCESS: i32 = 0;
    const HIPBLAS_SUCCESS: i32 = 0;
    const HIP_MEMCPY_H2D: i32 = 1;
    const HIP_MEMCPY_D2H: i32 = 2;
    const HIPBLAS_OP_T: i32 = 1;

    pub fn try_load() -> Option<Arc<Self>> {
        Self::load().map(Arc::new)
    }

    fn load() -> Option<Self> {
        let hip_candidates = ["amdhip64.dll", "libamdhip64.so", "libhip_hcc.so"];
        let blas_candidates = ["hipblas.dll", "libhipblas.so"];
        let mut hip_lib = None;
        for path in hip_candidates {
            if let Ok(lib) = unsafe { Library::new(path) } {
                hip_lib = Some(lib);
                break;
            }
        }
        let hip = hip_lib?;
        let hip_malloc = unsafe {
            let s: libloading::Symbol<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32> =
                hip.get(b"hipMalloc").ok()?;
            *s
        };
        let hip_free = unsafe {
            let s: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> i32> =
                hip.get(b"hipFree").ok()?;
            *s
        };
        let hip_memcpy = unsafe {
            let s: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
            > = hip.get(b"hipMemcpy").ok()?;
            *s
        };
        let hip_sync = unsafe {
            let s: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                hip.get(b"hipDeviceSynchronize").ok()?;
            *s
        };
        let mut blas_lib = None;
        for path in blas_candidates {
            if let Ok(lib) = unsafe { Library::new(path) } {
                blas_lib = Some(lib);
                break;
            }
        }
        let hipblas = blas_lib?;
        let hipblas_create = unsafe {
            let s: libloading::Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32> =
                hipblas.get(b"hipblasCreate").ok()?;
            *s
        };
        let hipblas_destroy = unsafe {
            let s: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> i32> =
                hipblas.get(b"hipblasDestroy").ok()?;
            *s
        };
        let hipblas_sgemv = unsafe {
            let s: libloading::Symbol<
                unsafe extern "C" fn(
                    *mut c_void,
                    i32,
                    i32,
                    i32,
                    *const f32,
                    *const f32,
                    i32,
                    *const f32,
                    i32,
                    *const f32,
                    *mut f32,
                    i32,
                ) -> i32,
            > = hipblas.get(b"hipblasSgemv").ok()?;
            *s
        };
        Some(Self {
            _hip: hip,
            _hipblas: hipblas,
            hip_malloc,
            hip_free,
            hip_memcpy,
            hip_sync,
            hipblas_create,
            hipblas_destroy,
            hipblas_sgemv,
            handle: Mutex::new(None),
        })
    }

    fn handle(&self) -> Option<*mut c_void> {
        let mut guard = self.handle.lock().ok()?;
        if let Some(raw) = *guard {
            return Some(raw as *mut c_void);
        }
        let mut h: *mut c_void = null_mut();
        if unsafe { (self.hipblas_create)(&mut h) } != Self::HIPBLAS_SUCCESS {
            return None;
        }
        *guard = Some(h as usize);
        Some(h)
    }

    pub fn gemv_host_f32(
        &self,
        w: &[f32],
        x: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != in_cols || w.len() != out_rows.checked_mul(in_cols)? {
            return None;
        }
        let handle = self.handle()?;
        let w_bytes = w.len() * std::mem::size_of::<f32>();
        let x_bytes = x.len() * std::mem::size_of::<f32>();
        let y_bytes = out_rows * std::mem::size_of::<f32>();
        let mut d_w: *mut c_void = null_mut();
        let mut d_x: *mut c_void = null_mut();
        let mut d_y: *mut c_void = null_mut();
        let mut out = vec![0.0f32; out_rows];
        let ok = unsafe {
            (self.hip_malloc)(&mut d_w, w_bytes) == Self::HIP_SUCCESS
                && (self.hip_malloc)(&mut d_x, x_bytes) == Self::HIP_SUCCESS
                && (self.hip_malloc)(&mut d_y, y_bytes) == Self::HIP_SUCCESS
                && (self.hip_memcpy)(d_w, w.as_ptr().cast(), w_bytes, Self::HIP_MEMCPY_H2D)
                    == Self::HIP_SUCCESS
                && (self.hip_memcpy)(d_x, x.as_ptr().cast(), x_bytes, Self::HIP_MEMCPY_H2D)
                    == Self::HIP_SUCCESS
        };
        if !ok {
            unsafe {
                if !d_w.is_null() {
                    let _ = (self.hip_free)(d_w);
                }
                if !d_x.is_null() {
                    let _ = (self.hip_free)(d_x);
                }
                if !d_y.is_null() {
                    let _ = (self.hip_free)(d_y);
                }
            }
            return None;
        }
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let status = unsafe {
            (self.hipblas_sgemv)(
                handle,
                Self::HIPBLAS_OP_T,
                in_cols as i32,
                out_rows as i32,
                &alpha,
                d_w.cast::<f32>(),
                in_cols as i32,
                d_x.cast::<f32>(),
                1,
                &beta,
                d_y.cast::<f32>(),
                1,
            )
        };
        let copy_ok = status == Self::HIPBLAS_SUCCESS
            && unsafe {
                (self.hip_memcpy)(out.as_mut_ptr().cast(), d_y, y_bytes, Self::HIP_MEMCPY_D2H)
                    == Self::HIP_SUCCESS
            };
        let _ = unsafe { (self.hip_sync)() };
        unsafe {
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_y);
        }
        copy_ok.then_some(out)
    }
}

impl Drop for RocmRuntime {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.handle.lock() {
            if let Some(raw) = guard.take() {
                let _ = unsafe { (self.hipblas_destroy)(raw as *mut c_void) };
            }
        }
    }
}

impl RocmBackend {
    pub(crate) fn runtime_available() -> bool {
        RocmRuntime::try_load().is_some()
    }
}

impl Default for RocmBackend {
    fn default() -> Self {
        let runtime = RocmRuntime::try_load();
        Self {
            cpu: CpuBackend,
            runtime_loaded: runtime.is_some(),
            runtime,
        }
    }
}

impl ComputeBackend for RocmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rocm
    }
    fn is_native_accelerated(&self) -> bool {
        self.runtime_loaded
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        self.cpu.alloc(len)
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        if let Some(rt) = &self.runtime {
            if let Some(out) = rt.gemv_host_f32(w, x, out_rows, in_cols) {
                return Ok(out);
            }
        }
        self.cpu.matvec(w, x, out_rows, in_cols)
    }
}

/// Vulkan bootstrap backend: functional parity stub.
#[derive(Debug)]
pub struct VulkanBackend {
    cpu: CpuBackend,
    runtime_loaded: bool,
}

impl VulkanBackend {
    /// A library probe is diagnostic; this prototype still computes on CPU.
    pub fn runtime_detected(&self) -> bool {
        self.runtime_loaded
    }
    /// Probe system Vulkan loader (Intel discrete/iGPU often via this path — #22).
    pub(crate) fn runtime_available() -> bool {
        ["vulkan-1.dll", "libvulkan.so", "libvulkan.dylib"]
            .iter()
            .any(|p| unsafe { Library::new(p).is_ok() })
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime_loaded: Self::runtime_available(),
        }
    }
}

impl ComputeBackend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }
    fn is_native_accelerated(&self) -> bool {
        false
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        self.cpu.alloc(len)
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        self.cpu.matvec(w, x, out_rows, in_cols)
    }
}

/// Metal bootstrap backend: planned second stage, parity stub for now.
#[derive(Debug)]
pub struct MetalBackend {
    cpu: CpuBackend,
    runtime_loaded: bool,
}

impl MetalBackend {
    /// A library probe is diagnostic; this prototype still computes on CPU.
    pub fn runtime_detected(&self) -> bool {
        self.runtime_loaded
    }
    pub(crate) fn runtime_available() -> bool {
        ["Metal.framework/Metal", "libMetal.dylib"]
            .iter()
            .any(|p| unsafe { Library::new(p).is_ok() })
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime_loaded: Self::runtime_available(),
        }
    }
}

impl ComputeBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }
    fn is_native_accelerated(&self) -> bool {
        false
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>> {
        self.cpu.alloc(len)
    }

    fn copy_from_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
        self.cpu.matvec(w, x, out_rows, in_cols)
    }
}

pub fn make_backend(kind: BackendKind) -> Box<dyn ComputeBackend> {
    match kind {
        BackendKind::Cpu => Box::<CpuBackend>::default(),
        BackendKind::Cuda => Box::<CudaBackend>::default(),
        BackendKind::Hybrid => Box::<HybridBackend>::default(),
        BackendKind::Rocm => Box::<RocmBackend>::default(),
        BackendKind::Vulkan => Box::<VulkanBackend>::default(),
        BackendKind::Metal => Box::<MetalBackend>::default(),
    }
}

#[cfg(test)]
mod transfer_tests {
    #[test]
    fn auto_selection_respects_mixtral_cpu_executor_without_gpu_probe() {
        use super::BackendKind;
        for raw in ["auto", "detect", "gpu", "", "unknown", " AUTO "] {
            assert_eq!(
                BackendKind::select_from_value(raw, "mixtral", || panic!(
                    "CPU-only architecture must not probe GPU"
                )),
                BackendKind::Cpu
            );
            assert_eq!(
                BackendKind::select_from_value(raw, "qwen3", || panic!(
                    "CPU-only architecture must not probe GPU"
                )),
                BackendKind::Cpu
            );
            assert_eq!(
                BackendKind::select_from_value(raw, "qwen35", || BackendKind::Cuda),
                BackendKind::Cuda
            );
        }
        for (raw, expected) in [
            ("cpu", BackendKind::Cpu),
            ("cuda", BackendKind::Cuda),
            ("hybrid", BackendKind::Hybrid),
            ("rocm", BackendKind::Rocm),
            ("intel", BackendKind::Vulkan),
            ("metal", BackendKind::Metal),
        ] {
            assert_eq!(
                BackendKind::select_from_value(raw, "mixtral", || panic!(
                    "explicit backend must remain explicit"
                )),
                expected
            );
        }
    }

    #[test]
    fn loader_only_prototypes_never_claim_native_acceleration() {
        use super::{ComputeBackend, CpuBackend, MetalBackend, VulkanBackend};
        let v = VulkanBackend {
            cpu: CpuBackend,
            runtime_loaded: true,
        };
        let m = MetalBackend {
            cpu: CpuBackend,
            runtime_loaded: true,
        };
        assert!(v.runtime_detected() && m.runtime_detected());
        assert!(!v.is_native_accelerated() && !m.is_native_accelerated());
    }
    #[test]
    fn optional_pageable_upload_and_refill_finish_before_private_stream_consumption() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let rt = super::CudaRuntime::try_load().expect("CUDA required");
        let query = unsafe {
            *rt._lib
                .get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>(b"cudaStreamQuery")
                .unwrap()
        };
        // Large pageable copies expose the staging-vs-DMA lifetime distinction.
        let mut matrix = super::CudaDeviceQuantMatrix::from_payload(
            Some(&rt),
            0,
            vec![0; 64 * 1024 * 1024],
            4096,
            4096,
        )
        .unwrap();
        assert!(matrix.is_device_resident());
        assert_eq!(unsafe { query(std::ptr::null_mut()) }, 0);
        for value in [0x3fu8, 0x40, 0x41, 0] {
            assert!(matrix.refill(vec![value; matrix.bytes()]).unwrap());
            assert_eq!(
                unsafe { query(std::ptr::null_mut()) },
                0,
                "weights were published before the DMA finished"
            );
        }
        let held = matrix.clone();
        assert!(
            !matrix.refill(vec![0; matrix.bytes()]).unwrap(),
            "shared device allocations cannot be overwritten"
        );
        drop(held);
    }
    #[test]
    fn optional_native_completion_guard_finishes_pinned_download_on_early_return() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        type Check = unsafe extern "C" fn(u32, *mut f32) -> i32;
        let lib = crate::ggml::load_cuda_quant_library().expect("native CUDA DLL required");
        let check = unsafe {
            *lib.get::<Check>(b"rbitnet_cuda_native_completion_check\0")
                .unwrap()
        };
        for early in [0, 1, 2, 1, 0] {
            let mut value = -1.;
            assert_eq!(unsafe { check(early, &mut value) }, 0);
            assert_eq!(value, 123.25);
        }
    }
}

#[cfg(test)]
#[path = "backend/mapped_quant_tests.rs"]
mod mapped_quant_tests;
