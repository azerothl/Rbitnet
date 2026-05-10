//! Row-wise dot products and matmul helpers over mmap GGUF payloads without full-tensor dequant.

use half::{bf16, f16};

use crate::error::{BitNetError, Result};
use crate::ggml::types;
use crate::ggml::dequant::{
    q4_0_block_dequant, q4_k_superblock_dequant, q6_k_superblock_dequant, q8_0_block_dequant,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};

const QK_K: usize = 256;
const QK4_0: usize = 32;

#[inline]
fn fp16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Types that have mmap row GEMV / row decode in this module (match [`dot_row`]).
pub fn ggml_type_supported_mmap_matvec(ty: u32) -> bool {
    matches!(ty, 0 | 1 | 2 | 8 | 12 | 14 | 30)
}

/// Dot product of one logical row (along `ne[0]`) with `x`.
#[inline]
pub fn dot_row(ty: u32, row_payload: &[u8], x: &[f32]) -> Result<f32> {
    if row_payload.len() != types::ggml_row_size(ty, x.len() as u64)? {
        return Err(BitNetError::InvalidGguf(
            "quant_dot: row payload length mismatch vs x / ne[0]".into(),
        ));
    }
    match ty {
        0 => dot_row_f32(row_payload, x),
        1 => dot_row_f16(row_payload, x),
        30 => dot_row_bf16(row_payload, x),
        2 => dot_row_q4_0(row_payload, x),
        8 => dot_row_q8_0(row_payload, x),
        12 => dot_row_q4_k(row_payload, x),
        14 => dot_row_q6_k(row_payload, x),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

fn dot_row_f32(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 4 {
        return Err(BitNetError::InvalidGguf("f32 row size".into()));
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let b = u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
        s += f32::from_bits(b) * x[i];
    }
    Ok(s)
}

fn dot_row_f16(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 2 {
        return Err(BitNetError::InvalidGguf("f16 row size".into()));
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let h = u16::from_le_bytes(row[i * 2..i * 2 + 2].try_into().unwrap());
        s += fp16_to_f32(h) * x[i];
    }
    Ok(s)
}

fn dot_row_bf16(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 2 {
        return Err(BitNetError::InvalidGguf("bf16 row size".into()));
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let h = u16::from_le_bytes(row[i * 2..i * 2 + 2].try_into().unwrap());
        s += bf16_to_f32(h) * x[i];
    }
    Ok(s)
}

fn dot_row_q4_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK4_0 != 0 {
        return Err(BitNetError::InvalidGguf("q4_0 ne0 % 32".into()));
    }
    let nb = x.len() / QK4_0;
    if row.len() != nb * 18 {
        return Err(BitNetError::InvalidGguf("q4_0 row bytes".into()));
    }
    let mut buf = [0.0f32; 32];
    let mut acc = 0.0f32;
    for b in 0..nb {
        q4_0_block_dequant(&row[b * 18..b * 18 + 18], &mut buf)?;
        let xb = &x[b * QK4_0..(b + 1) * QK4_0];
        for i in 0..QK4_0 {
            acc += buf[i] * xb[i];
        }
    }
    Ok(acc)
}

fn dot_row_q8_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % 32 != 0 {
        return Err(BitNetError::InvalidGguf("q8_0 ne0 % 32".into()));
    }
    let nb = x.len() / 32;
    if row.len() != nb * 34 {
        return Err(BitNetError::InvalidGguf("q8_0 row bytes".into()));
    }
    let mut buf = [0.0f32; 32];
    let mut acc = 0.0f32;
    for b in 0..nb {
        q8_0_block_dequant(&row[b * 34..b * 34 + 34], &mut buf)?;
        let xb = &x[b * 32..(b + 1) * 32];
        for i in 0..32 {
            acc += buf[i] * xb[i];
        }
    }
    Ok(acc)
}

fn dot_row_q6_k(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("q6_K ne0 % 256".into()));
    }
    let nb = x.len() / QK_K;
    if row.len() != nb * 210 {
        return Err(BitNetError::InvalidGguf("q6_K row bytes".into()));
    }
    let mut buf = [0.0f32; QK_K];
    let mut acc = 0.0f32;
    for b in 0..nb {
        q6_k_superblock_dequant(&row[b * 210..b * 210 + 210], &mut buf)?;
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        for i in 0..QK_K {
            acc += buf[i] * xb[i];
        }
    }
    Ok(acc)
}

fn dot_row_q4_k(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("q4_K ne0 % 256".into()));
    }
    let nb = x.len() / QK_K;
    if row.len() != nb * 144 {
        return Err(BitNetError::InvalidGguf("q4_K row bytes".into()));
    }
    let mut buf = [0.0f32; QK_K];
    let mut acc = 0.0f32;
    for b in 0..nb {
        q4_k_superblock_dequant(&row[b * 144..b * 144 + 144], &mut buf)?;
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        for i in 0..QK_K {
            acc += buf[i] * xb[i];
        }
    }
    Ok(acc)
}

/// Decode one matrix row (second index `row`) to `out` (`len == ne0`).
#[inline]
pub fn decode_row_to_f32(
    ty: u32,
    row_payload: &[u8],
    out: &mut [f32],
) -> Result<()> {
    if row_payload.len() != types::ggml_row_size(ty, out.len() as u64)? {
        return Err(BitNetError::InvalidGguf("decode_row: row size".into()));
    }
    match ty {
        0 => {
            if row_payload.len() != out.len() * 4 {
                return Err(BitNetError::InvalidGguf("f32 decode".into()));
            }
            for i in 0..out.len() {
                let b = u32::from_le_bytes(row_payload[i * 4..i * 4 + 4].try_into().unwrap());
                out[i] = f32::from_bits(b);
            }
        }
        1 => {
            for i in 0..out.len() {
                let h = u16::from_le_bytes(row_payload[i * 2..i * 2 + 2].try_into().unwrap());
                out[i] = fp16_to_f32(h);
            }
        }
        30 => {
            for i in 0..out.len() {
                let h = u16::from_le_bytes(row_payload[i * 2..i * 2 + 2].try_into().unwrap());
                out[i] = bf16_to_f32(h);
            }
        }
        2 => {
            if out.len() % QK4_0 != 0 {
                return Err(BitNetError::InvalidGguf("q4_0 decode ne0".into()));
            }
            let nb = out.len() / QK4_0;
            let mut buf = [0.0f32; 32];
            for b in 0..nb {
                q4_0_block_dequant(&row_payload[b * 18..b * 18 + 18], &mut buf)?;
                out[b * QK4_0..(b + 1) * QK4_0].copy_from_slice(&buf);
            }
        }
        8 => {
            if out.len() % 32 != 0 {
                return Err(BitNetError::InvalidGguf("q8_0 decode ne0".into()));
            }
            let nb = out.len() / 32;
            let mut buf = [0.0f32; 32];
            for b in 0..nb {
                q8_0_block_dequant(&row_payload[b * 34..b * 34 + 34], &mut buf)?;
                out[b * 32..(b + 1) * 32].copy_from_slice(&buf);
            }
        }
        12 => {
            if out.len() % QK_K != 0 {
                return Err(BitNetError::InvalidGguf("q4_K decode ne0".into()));
            }
            let nb = out.len() / QK_K;
            let mut buf = [0.0f32; QK_K];
            for b in 0..nb {
                q4_k_superblock_dequant(&row_payload[b * 144..b * 144 + 144], &mut buf)?;
                out[b * QK_K..(b + 1) * QK_K].copy_from_slice(&buf);
            }
        }
        14 => {
            if out.len() % QK_K != 0 {
                return Err(BitNetError::InvalidGguf("q6_K decode ne0".into()));
            }
            let nb = out.len() / QK_K;
            let mut buf = [0.0f32; QK_K];
            for b in 0..nb {
                q6_k_superblock_dequant(&row_payload[b * 210..b * 210 + 210], &mut buf)?;
                out[b * QK_K..(b + 1) * QK_K].copy_from_slice(&buf);
            }
        }
        _ => return Err(BitNetError::UnsupportedGgmlType(ty)),
    }
    Ok(())
}

/// `y[o] = dot(W[o,:], x)` with W shaped `[ne0, ne1]` in GGUF row layout.
pub fn matvec_embd_out_mmap(
    archive: &GgufArchive,
    t: &GgufTensorInfo,
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    if t.dimensions.len() < 2 {
        return Err(BitNetError::InvalidGguf("matvec: expected 2D tensor".into()));
    }
    if t.dimensions[0] as usize != ne0 || t.dimensions[1] as usize != ne1 {
        return Err(BitNetError::Inference("quant matvec: tensor dims mismatch".into()));
    }
    if x.len() != ne0 {
        return Err(BitNetError::Inference("matvec: x len".into()));
    }
    let ty = t.ggml_type;
    let payload = archive.tensor_payload(t)?;
    let row_bytes = types::ggml_row_size(ty, ne0 as u64)?;
    let mut y = vec![0.0f32; ne1];
    for o in 0..ne1 {
        let row_start = o * row_bytes;
        let row = &payload[row_start..row_start + row_bytes];
        y[o] = dot_row(ty, row, x)?;
    }
    Ok(y)
}

/// `ffn_down`: shape `[n_ff, n_embd]` — `y[o] = dot(W[o,:], x)` for `o` in `0..n_embd`, `x.len()==n_ff`.
pub fn matvec_ff_mmap(
    archive: &GgufArchive,
    t: &GgufTensorInfo,
    x: &[f32],
    n_ff: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    matvec_embd_out_mmap(archive, t, x, n_ff, n_embd)
}

/// Token embedding row `tok` into `out` (`len == n_embd`). Tensor `[n_embd, n_vocab]`.
pub fn embedding_row_mmap(
    archive: &GgufArchive,
    t: &GgufTensorInfo,
    tok: usize,
    n_embd: usize,
    n_vocab: usize,
    out: &mut [f32],
) -> Result<()> {
    if t.dimensions.len() < 2 {
        return Err(BitNetError::InvalidGguf("embedding: expected 2D tensor".into()));
    }
    if t.dimensions[0] as usize != n_embd || t.dimensions[1] as usize != n_vocab {
        return Err(BitNetError::Inference("embedding: tensor dims mismatch".into()));
    }
    if tok >= n_vocab {
        return Err(BitNetError::Inference("embedding: token id OOB".into()));
    }
    if out.len() != n_embd {
        return Err(BitNetError::Inference("embedding: out len".into()));
    }
    let ty = t.ggml_type;
    let payload = archive.tensor_payload(t)?;
    let row_bytes = types::ggml_row_size(ty, n_embd as u64)?;
    let row_start = tok * row_bytes;
    let row = &payload[row_start..row_start + row_bytes];
    decode_row_to_f32(ty, row, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::tensor_to_f32;

    #[test]
    fn q6_k_row_dot_matches_full_dequant() {
        let mut payload = vec![0u8; 210];
        for i in 0..210 {
            payload[i] = (i as u8).wrapping_mul(11).wrapping_add(7);
        }
        let dims = vec![256u64, 1u64];
        let dense = tensor_to_f32(&payload, 14, &dims).unwrap();
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let ref_dot: f32 = dense.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let q = dot_row(14, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 2e-2, "q={q} ref={ref_dot}");
    }

    #[test]
    fn q4_k_row_dot_matches_full_dequant() {
        let mut payload = vec![0u8; 144];
        for i in 0..144 {
            payload[i] = (i as u8).wrapping_mul(13).wrapping_add(3);
        }
        let dims = vec![256u64, 1u64];
        let dense = tensor_to_f32(&payload, 12, &dims).unwrap();
        assert_eq!(dense.len(), 256);
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.25).collect();
        let mut ref_dot = 0.0f32;
        for i in 0..256 {
            ref_dot += dense[i] * x[i];
        }
        let q = dot_row(12, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 1e-3, "q={q} ref={ref_dot}");
    }

    #[test]
    fn q4_0_row_matches_dense_dot() {
        let mut payload = vec![0u8; 18];
        for i in 0..18 {
            payload[i] = (i as u8).wrapping_add(11);
        }
        let dims = vec![32u64, 1u64];
        let dense = tensor_to_f32(&payload, 2, &dims).unwrap();
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let ref_dot: f32 = dense.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let q = dot_row(2, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 1e-3);
    }
}
