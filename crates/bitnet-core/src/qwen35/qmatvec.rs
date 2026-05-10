//! Quantised matmul for GGUF-packed weights (row-major stripes along `ne[0]`).

use crate::error::{BitNetError, Result};
use crate::ggml::{ggml_row_size, tensor_to_f32};
use crate::gguf::{GgufArchive, GgufTensorInfo};

/// Embedding row `token_embd[token_id]` for matrix `[ne0=n_embd, ne1=vocab]` (Llama GGUF convention).
pub fn token_embedding_row(
    archive: &GgufArchive,
    t: &GgufTensorInfo,
    token_id: usize,
    n_embd: usize,
    n_vocab: usize,
) -> Result<Vec<f32>> {
    if t.dimensions.len() != 2 {
        return Err(BitNetError::Inference(
            "token_embd: expected dims [embd,vocab]".into(),
        ));
    }
    let dim0 = usize::try_from(t.dimensions[0])
        .map_err(|_| BitNetError::Inference("ne0 overflow".into()))?;
    let dim1 = usize::try_from(t.dimensions[1])
        .map_err(|_| BitNetError::Inference("ne1 overflow".into()))?;
    if dim0 != n_embd || dim1 != n_vocab || token_id >= n_vocab {
        return Err(BitNetError::Inference(
            "token_embd shape mismatch vs config".into(),
        ));
    }
    let row_bs = ggml_row_size(t.ggml_type, t.dimensions[0])?;
    let row_start = token_id
        .checked_mul(row_bs)
        .ok_or_else(|| BitNetError::Inference("embed offset overflow".into()))?;
    let payload = archive.tensor_payload(t)?;
    let end = row_start
        .checked_add(row_bs)
        .filter(|e| *e <= payload.len())
        .ok_or_else(|| BitNetError::Inference("token_embd row OOB".into()))?;
    let slice = &payload[row_start..end];
    tensor_to_f32(slice, t.ggml_type, &[t.dimensions[0]])
}

/// Compute `y = W @ x` for `W.shape() == [ne0, ne1]` (contiguous GGUF stripes along `ne0`).
pub fn quant_matmul_vec(
    payload: &[u8],
    ggml_ty: u32,
    ne0: usize,
    ne1: usize,
    x: &[f32],
) -> Result<Vec<f32>> {
    if x.len() != ne0 {
        return Err(BitNetError::Inference(format!(
            "quant_matmul_vec size mismatch x.len={} ne0={ne0}",
            x.len()
        )));
    }
    let stride = ggml_row_size(ggml_ty, ne0 as u64)?;
    let need = stride
        .checked_mul(ne1)
        .filter(|n| *n <= payload.len())
        .ok_or_else(|| BitNetError::Inference("quant matmul payload too small".into()))?;
    if need != payload.len() && ne1 != 0 {
        // allow supersets (view prefix)
        if payload.len() < need {
            return Err(BitNetError::Inference(
                "quant matmul truncated payload".into(),
            ));
        }
    }
    let mut y = Vec::with_capacity(ne1);
    for j in 0..ne1 {
        let row_start = j * stride;
        let slice = &payload[row_start..row_start + stride];
        let row = tensor_to_f32(slice, ggml_ty, &[ne0 as u64])?;
        let dot: f32 = row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
        y.push(dot);
    }
    Ok(y)
}

/// Matmul for slab `payload[slice_start..]` covering `[ne0, ne1]`.
pub fn quant_matmul_vec_offset(
    payload: &[u8],
    ggml_ty: u32,
    ne0: usize,
    ne1: usize,
    offset: usize,
    x: &[f32],
) -> Result<Vec<f32>> {
    let stride = ggml_row_size(ggml_ty, ne0 as u64)?;
    let total = stride
        .checked_mul(ne1)
        .ok_or_else(|| BitNetError::Inference("stride overflow".into()))?;
    if offset
        .checked_add(total)
        .filter(|e| *e <= payload.len())
        .is_none()
    {
        return Err(BitNetError::Inference("quant slab OOB".into()));
    }
    quant_matmul_vec(&payload[offset..offset + total], ggml_ty, ne0, ne1, x)
}
