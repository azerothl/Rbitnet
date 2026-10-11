//! `ggml_nbytes` / row sizing (see llama.cpp `ggml_type_traits`).

use crate::error::{BitNetError, Result};

/// `(blck_size_elements, block_size_bytes)` for each `ggml_type` enum value.
pub(crate) fn type_layout(ty: u32) -> Result<(usize, usize)> {
    let v = match ty {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        4 | 5 | 31..=33 | 37 | 38 => {
            return Err(BitNetError::InvalidGguf(format!(
                "deprecated or removed ggml_type {ty}"
            )));
        }
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 40),    // Q8_1
        10 => (256, 84),  // Q2_K: 2*f16 + 16 + 64
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K: f32 + 256 + 32 int16
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S: half + 32 + 16
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS: half + u16 + 4 + 128
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M: 32+16+8
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0: half + 4 + 48 (from ggml-common assert)
        35 => (256, 66),  // TQ2_0: half + 64
        // BitNet I2_S: 128 ternary weights in 32 code bytes. A per-tensor f32
        // scale sits in a 32-byte trailer after the codes (`quantize_i2_s`).
        36 => (128, 32),
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        _ => {
            return Err(BitNetError::UnsupportedGgmlType(ty));
        }
    };
    Ok(v)
}

/// Bytes for one row along `ne[0]` (contiguous quantized blocks).
pub fn ggml_row_size(ty: u32, ne0: u64) -> Result<usize> {
    let (bs, bb) = type_layout(ty)?;
    let bs = bs as u64;
    if ne0 % bs != 0 {
        return Err(BitNetError::InvalidGguf(format!(
            "tensor ne[0]={ne0} not divisible by block size {bs} for ggml_type {ty}"
        )));
    }
    let nblocks = ne0 / bs;
    Ok((nblocks as usize).saturating_mul(bb))
}

/// Total tensor payload size in bytes.
pub fn ggml_nbytes(dims: &[u64], ty: u32) -> Result<usize> {
    if dims.is_empty() {
        return Ok(0);
    }
    let mut n = ggml_row_size(ty, dims[0])?;
    for &d in &dims[1..] {
        n = n
            .checked_mul(d as usize)
            .ok_or_else(|| BitNetError::InvalidGguf("tensor nbytes overflow".into()))?;
    }
    if ty == 36 {
        n = n
            .checked_add(32)
            .ok_or_else(|| BitNetError::InvalidGguf("tensor nbytes overflow".into()))?;
    }
    Ok(n)
}

/// Human-readable GGML type label for `inspect_gguf` / load errors.
pub fn ggml_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        36 => "I2_S",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        _ => "unknown",
    }
}

/// True when row GEMV or dense `tensor_to_f32` can decode `ty`.
pub fn ggml_type_dequant_supported(ty: u32) -> bool {
    crate::ggml::ggml_type_supported_mmap_matvec(ty) || matches!(ty, 3 | 7 | 9 | 10 | 11)
}
