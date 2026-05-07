//! Compute backend abstractions for portable execution.

use crate::error::Result;
use libloading::Library;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

/// Backend identifiers used by runtime selection and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
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
        self.kind() == BackendKind::Cpu
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

/// CUDA MVP backend: API-compatible with CPU path, currently forwarding to CPU kernels.
#[derive(Debug)]
pub struct CudaBackend {
    cpu: CpuBackend,
    runtime: Option<Arc<CudaRuntime>>,
}

#[allow(non_camel_case_types)]
type cudaError_t = i32;
#[allow(non_camel_case_types)]
type cudaMemcpyKind = i32;

const CUDA_SUCCESS: cudaError_t = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: cudaMemcpyKind = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: cudaMemcpyKind = 2;

struct CudaRuntime {
    _lib: Library,
    cuda_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
    cuda_free: unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    cuda_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t,
    cuda_device_synchronize: unsafe extern "C" fn() -> cudaError_t,
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaRuntime(loaded)")
    }
}

impl CudaRuntime {
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
                let sym: libloading::Symbol<unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t> =
                    lib.get(b"cudaMalloc").ok()?;
                *sym
            };
            let cuda_free = unsafe {
                let sym: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> cudaError_t> =
                    lib.get(b"cudaFree").ok()?;
                *sym
            };
            let cuda_memcpy = unsafe {
                let sym: libloading::Symbol<
                    unsafe extern "C" fn(*mut c_void, *const c_void, usize, cudaMemcpyKind) -> cudaError_t,
                > = lib.get(b"cudaMemcpy").ok()?;
                *sym
            };
            let cuda_device_synchronize = unsafe {
                let sym: libloading::Symbol<unsafe extern "C" fn() -> cudaError_t> =
                    lib.get(b"cudaDeviceSynchronize").ok()?;
                *sym
            };
            return Some(Self {
                _lib: lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cuda_device_synchronize,
            });
        }
        None
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
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            cpu: CpuBackend,
            runtime: CudaRuntime::load().map(Arc::new),
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
        if let Some(rt) = &self.runtime {
            if let Some(out) = rt.roundtrip_f32(src) {
                return Ok(out);
            }
        }
        self.cpu.copy_from_host(src)
    }

    fn copy_to_host(&self, src: &[f32]) -> Result<Vec<f32>> {
        if let Some(rt) = &self.runtime {
            if let Some(out) = rt.roundtrip_f32(src) {
                return Ok(out);
            }
        }
        self.cpu.copy_to_host(src)
    }

    fn matvec(&self, w: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Result<Vec<f32>> {
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
        BackendKind::Rocm => Box::<RocmBackend>::default(),
        BackendKind::Vulkan => Box::<VulkanBackend>::default(),
        BackendKind::Metal => Box::<MetalBackend>::default(),
    }
}
