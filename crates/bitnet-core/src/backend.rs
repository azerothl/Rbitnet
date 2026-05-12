//! Compute backend abstractions for portable execution.

use crate::error::Result;
use libloading::Library;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
        let raw = std::env::var("RBITNET_BACKEND").unwrap_or_else(|_| "cpu".into());
        match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => BackendKind::Cpu,
            "cuda" => BackendKind::Cuda,
            "hybrid" | "cpu-gpu" | "gpu-cpu" => BackendKind::Hybrid,
            "rocm" => BackendKind::Rocm,
            "vulkan" => BackendKind::Vulkan,
            "metal" => BackendKind::Metal,
            _ => BackendKind::Cpu,
        }
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
/// Host copy operations intentionally remain host-side until the backend exposes real reusable
/// device buffers; copying host -> device -> host here only adds latency without preserving GPU
/// state across calls.
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
pub struct CudaRuntime {
    _lib: Library,
    _cublas_lib: Option<Library>,
    cuda_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
    cuda_free: unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    cuda_memcpy:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t,
    cuda_device_synchronize: unsafe extern "C" fn() -> cudaError_t,
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
    upload_bytes: AtomicU64,
    download_bytes: AtomicU64,
    gemv_calls: AtomicU64,
}

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
        }
    }

    pub fn upload_f32(self: &Arc<Self>, src: &[f32]) -> Option<CudaDeviceBuffer> {
        let nbytes = src.len().checked_mul(std::mem::size_of::<f32>())?;
        let ptr = self.alloc_device(nbytes)?;
        if !self.copy_host_to_device(ptr, src.as_ptr().cast::<c_void>(), nbytes) {
            self.free_device(ptr);
            return None;
        }
        Some(CudaDeviceBuffer {
            rt: Arc::clone(self),
            ptr: ptr as usize,
            len: src.len(),
        })
    }

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
        let d_x = self.alloc_device(x_bytes)?;
        let d_y = self.alloc_device(y_bytes)?;
        let mut out = vec![0.0f32; out_rows];
        let ok = self.copy_host_to_device(d_x, x.as_ptr().cast::<c_void>(), x_bytes)
            && self.cublas_sgemv_device(d_w, d_x, d_y, out_rows, in_cols)
            && self.copy_device_to_host(out.as_mut_ptr().cast::<c_void>(), d_y, y_bytes);
        let _ = unsafe { (self.cuda_device_synchronize)() };
        self.free_device(d_x);
        self.free_device(d_y);
        if ok {
            Some(out)
        } else {
            None
        }
    }

    fn load() -> Option<Self> {
        let candidates = [
            "cudart64_12.dll",
            "cudart64_11.dll",
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.dylib",
        ];
        for path in candidates {
            let Ok(lib) = (unsafe { Library::new(path) }) else {
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
            let mut cublas_lib: Option<Library> = None;
            let mut cublas_create_v2 = None;
            let mut cublas_destroy_v2 = None;
            let mut cublas_sgemv_v2 = None;
            for cb_path in [
                "cublas64_12.dll",
                "cublas64_11.dll",
                "libcublas.so",
                "libcublas.so.12",
                "libcublas.dylib",
            ] {
                let Ok(cb_lib) = (unsafe { Library::new(cb_path) }) else {
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
            return Some(Self {
                _lib: lib,
                _cublas_lib: cublas_lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cuda_device_synchronize,
                cublas_create_v2,
                cublas_destroy_v2,
                cublas_sgemv_v2,
                cublas_handle: Mutex::new(None),
                upload_bytes: AtomicU64::new(0),
                download_bytes: AtomicU64::new(0),
                gemv_calls: AtomicU64::new(0),
            });
        }
        None
    }

    fn alloc_device(&self, nbytes: usize) -> Option<*mut c_void> {
        let mut dev_ptr: *mut c_void = null_mut();
        let ok = unsafe { (self.cuda_malloc)(&mut dev_ptr, nbytes) } == CUDA_SUCCESS;
        ok.then_some(dev_ptr)
    }

    fn free_device(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            let _ = unsafe { (self.cuda_free)(ptr) };
        }
    }

    fn copy_host_to_device(&self, dst: *mut c_void, src: *const c_void, nbytes: usize) -> bool {
        let ok = unsafe { (self.cuda_memcpy)(dst, src, nbytes, CUDA_MEMCPY_HOST_TO_DEVICE) }
            == CUDA_SUCCESS;
        if ok {
            self.upload_bytes
                .fetch_add(nbytes as u64, Ordering::Relaxed);
        }
        ok
    }

    fn copy_device_to_host(&self, dst: *mut c_void, src: *const c_void, nbytes: usize) -> bool {
        let ok = unsafe { (self.cuda_memcpy)(dst, src, nbytes, CUDA_MEMCPY_DEVICE_TO_HOST) }
            == CUDA_SUCCESS;
        if ok {
            self.download_bytes
                .fetch_add(nbytes as u64, Ordering::Relaxed);
        }
        ok
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
        }
        ok
    }

    fn roundtrip_f32(&self, src: &[f32]) -> Option<Vec<f32>> {
        let nbytes = src.len().checked_mul(std::mem::size_of::<f32>())?;
        let mut dev_ptr: *mut c_void = null_mut();
        let mut out = vec![0.0f32; src.len()];
        unsafe {
            if (self.cuda_malloc)(&mut dev_ptr, nbytes) != CUDA_SUCCESS {
                return None;
            }
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
            let _ = (self.cuda_free)(dev_ptr);
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
        let d_w = self.alloc_device(w_bytes)?;
        let d_x = self.alloc_device(x_bytes)?;
        let d_y = self.alloc_device(y_bytes)?;
        let mut out = vec![0.0f32; out_rows];
        let ok = self.copy_host_to_device(d_w, w.as_ptr().cast::<c_void>(), w_bytes)
            && self.copy_host_to_device(d_x, x.as_ptr().cast::<c_void>(), x_bytes)
            && self.cublas_sgemv_device(d_w, d_x, d_y, out_rows, in_cols)
            && self.copy_device_to_host(out.as_mut_ptr().cast::<c_void>(), d_y, y_bytes);
        let _ = unsafe { (self.cuda_device_synchronize)() };
        self.free_device(d_w);
        self.free_device(d_x);
        self.free_device(d_y);
        if !ok {
            return None;
        }
        Some(out)
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
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
    pub gemv_calls: u64,
}

#[derive(Debug)]
pub struct CudaDeviceBuffer {
    rt: Arc<CudaRuntime>,
    ptr: usize,
    len: usize,
}

unsafe impl Send for CudaDeviceBuffer {}
unsafe impl Sync for CudaDeviceBuffer {}

impl CudaDeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn download_f32(&self) -> Option<Vec<f32>> {
        let nbytes = self.len.checked_mul(std::mem::size_of::<f32>())?;
        let mut out = vec![0.0f32; self.len];
        let ok = self.rt.copy_device_to_host(
            out.as_mut_ptr().cast::<c_void>(),
            self.ptr as *const c_void,
            nbytes,
        );
        ok.then_some(out)
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        self.rt.free_device(self.ptr as *mut c_void);
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
        self.buffer.len() * std::mem::size_of::<f32>()
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

/// ROCm bootstrap backend: functional parity stub.
#[derive(Debug)]
pub struct RocmBackend {
    cpu: CpuBackend,
    runtime_loaded: bool,
}

impl RocmBackend {
    fn runtime_available() -> bool {
        [
            "amdhip64.dll",
            "hipblas.dll",
            "libamdhip64.so",
            "libhipblas.so",
        ]
        .iter()
        .any(|p| unsafe { Library::new(p).is_ok() })
    }
}

impl Default for RocmBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime_loaded: Self::runtime_available(),
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
    fn runtime_available() -> bool {
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
    fn runtime_available() -> bool {
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
