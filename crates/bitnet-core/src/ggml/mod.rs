//! GGML type sizes and dequantization (reference: llama.cpp `ggml`).

mod dequant;
mod quant_dot;
mod quant_simd;
pub(crate) mod simd;
mod types;

pub use dequant::tensor_to_f32;
pub(crate) use quant_dot::load_cuda_quant_library;
pub(crate) use quant_dot::matvec_device_quant_batch_optional;
pub use quant_dot::{
    cuda_quant_library_available, decode_row_to_f32, dot_row, embedding_row_mmap,
    ggml_type_supported_mmap_matvec, ggml_type_supports_cuda_quant, matvec_device_quant_optional,
    matvec_embd_out_mmap, matvec_ff_mmap, matvec_payload_quant, QuantKernelBackend,
    QuantMatvecKernel,
};
pub(crate) use quant_simd::f32_accumulator_lanes;
pub use types::{ggml_nbytes, ggml_row_size};
