//! Compute backend abstractions for portable execution.

use crate::error::Result;

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
#[derive(Debug, Default)]
pub struct CudaBackend {
    cpu: CpuBackend,
}

impl ComputeBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
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

/// ROCm bootstrap backend: functional parity stub.
#[derive(Debug, Default)]
pub struct RocmBackend {
    cpu: CpuBackend,
}

impl ComputeBackend for RocmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rocm
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
#[derive(Debug, Default)]
pub struct VulkanBackend {
    cpu: CpuBackend,
}

impl ComputeBackend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
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
#[derive(Debug, Default)]
pub struct MetalBackend {
    cpu: CpuBackend,
}

impl ComputeBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
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
