//! GGML type sizes and dequantization (reference: llama.cpp `ggml`).

mod dequant;
mod quant_dot;
mod types;

pub use dequant::tensor_to_f32;
pub use quant_dot::{
    embedding_row_mmap, ggml_type_supported_mmap_matvec, matvec_embd_out_mmap, matvec_ff_mmap,
    matvec_payload_quant, QuantKernelBackend, QuantMatvecKernel,
};
pub use types::{ggml_nbytes, ggml_row_size};
