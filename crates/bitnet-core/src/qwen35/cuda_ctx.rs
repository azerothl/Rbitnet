//! Optional CUDA facades (`cuBLAS` GEMV after host-side dequant).

use std::sync::Arc;

use crate::backend::CudaRuntime;

#[derive(Clone)]
pub struct QwenCudaContext {
    pub rt: Arc<CudaRuntime>,
}

impl QwenCudaContext {
    pub fn try_load() -> Option<Self> {
        CudaRuntime::try_load().map(|rt| Self { rt })
    }

    pub fn logits_gemv_maybe(
        &self,
        w_chunk: &[f32],
        x: &[f32],
        out_rows_chunk: usize,
        in_cols: usize,
        cpu_fallback: impl FnOnce() -> Vec<f32>,
    ) -> Vec<f32> {
        self.rt
            .gemv_host_f32(w_chunk, x, out_rows_chunk, in_cols)
            .unwrap_or_else(cpu_fallback)
    }
}
