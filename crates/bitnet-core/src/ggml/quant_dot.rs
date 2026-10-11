//! Row-wise dot products and matmul helpers over mmap GGUF payloads without full-tensor dequant.

use half::{bf16, f16};
use libloading::Library;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::thread;
use std::time::Instant;

use crate::error::{BitNetError, Result};
use crate::ggml::dequant::{
    q4_0_block_dequant, q4_k_superblock_dequant, q6_k_superblock_dequant, q8_0_block_dequant,
};
use crate::ggml::types;
use crate::gguf::{GgufArchive, GgufTensorInfo};

const QK_K: usize = 256;
const QK4_0: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKernelBackend {
    CpuScalar,
    CpuParallel,
    CudaQuantStub,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantMatvecKernel {
    backend: QuantKernelBackend,
    parallel_min_rows: usize,
}

impl Default for QuantMatvecKernel {
    fn default() -> Self {
        Self::from_env()
    }
}

impl QuantMatvecKernel {
    /// Explicit SIMD CPU dispatch, independent of the process CUDA preference.
    pub(crate) fn cpu_parallel() -> Self {
        Self {
            backend: QuantKernelBackend::CpuParallel,
            parallel_min_rows: 128,
        }
    }

    pub fn from_env() -> Self {
        let backend = match std::env::var("RBITNET_QUANT_KERNEL")
            .unwrap_or_else(|_| "auto".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "scalar" | "cpu-scalar" => QuantKernelBackend::CpuScalar,
            "cuda" | "gpu" => QuantKernelBackend::CudaQuantStub,
            _ => QuantKernelBackend::CpuParallel,
        };
        let parallel_min_rows = std::env::var("RBITNET_QUANT_PAR_MIN_ROWS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(128);
        Self {
            backend,
            parallel_min_rows,
        }
    }

    pub fn matvec_mmap(
        &self,
        archive: &GgufArchive,
        t: &GgufTensorInfo,
        x: &[f32],
        ne0: usize,
        ne1: usize,
    ) -> Result<Vec<f32>> {
        validate_matvec_shape(t, x, ne0, ne1)?;
        let started = Instant::now();
        let ty = t.ggml_type;
        let payload = archive.tensor_payload(t)?;
        let row_bytes = types::ggml_row_size(ty, ne0 as u64)?;
        let y = match self.backend {
            QuantKernelBackend::CpuScalar => matvec_rows_scalar(ty, payload, row_bytes, x, ne1),
            QuantKernelBackend::CpuParallel
                if ne1 >= effective_parallel_min_rows(self.parallel_min_rows, ne0, ne1) =>
            {
                matvec_rows_parallel(ty, payload, row_bytes, x, ne1)
            }
            QuantKernelBackend::CudaQuantStub => {
                if let Some(result) =
                    matvec_rows_cuda_quant_optional(ty, payload, row_bytes, x, ne1)
                {
                    result
                } else {
                    matvec_rows_parallel(ty, payload, row_bytes, x, ne1)
                }
            }
            QuantKernelBackend::CpuParallel => matvec_rows_scalar(ty, payload, row_bytes, x, ne1),
        }?;
        crate::perf::record_quant_matvec(
            ty,
            ne1,
            ne0,
            started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        );
        Ok(y)
    }

    pub fn matvec_payload(
        &self,
        ggml_type: u32,
        payload: &[u8],
        x: &[f32],
        ne0: usize,
        ne1: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != ne0 {
            return Err(BitNetError::Inference("matvec payload: x len".into()));
        }
        let started = Instant::now();
        let row_bytes = types::ggml_row_size(ggml_type, ne0 as u64)?;
        let need = row_bytes
            .checked_mul(ne1)
            .ok_or_else(|| BitNetError::Inference("matvec payload size overflow".into()))?;
        if payload.len() < need {
            return Err(BitNetError::Inference("matvec payload truncated".into()));
        }
        let y = match self.backend {
            QuantKernelBackend::CpuScalar => {
                matvec_rows_scalar(ggml_type, payload, row_bytes, x, ne1)
            }
            QuantKernelBackend::CpuParallel
                if ne1 >= effective_parallel_min_rows(self.parallel_min_rows, ne0, ne1) =>
            {
                matvec_rows_parallel(ggml_type, payload, row_bytes, x, ne1)
            }
            QuantKernelBackend::CudaQuantStub => {
                if let Some(result) =
                    matvec_rows_cuda_quant_optional(ggml_type, payload, row_bytes, x, ne1)
                {
                    result
                } else {
                    matvec_rows_parallel(ggml_type, payload, row_bytes, x, ne1)
                }
            }
            QuantKernelBackend::CpuParallel => {
                matvec_rows_scalar(ggml_type, payload, row_bytes, x, ne1)
            }
        }?;
        crate::perf::record_quant_matvec(
            ggml_type,
            ne1,
            ne0,
            started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        );
        Ok(y)
    }
}

pub fn matvec_payload_quant(
    ggml_type: u32,
    payload: &[u8],
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    QuantMatvecKernel::default().matvec_payload(ggml_type, payload, x, ne0, ne1)
}

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
    matches!(
        ty,
        0 | 1 | 2 | 6 | 8 | 10 | 12 | 13 | 14 | 16 | 17 | 18 | 19 | 21 | 22 | 23 | 29 | 30 | 34
            | 35 | 36 | 39
    )
}

/// Dot product of one logical row (along `ne[0]`) with `x`.
#[inline]
pub fn dot_row(ty: u32, row_payload: &[u8], x: &[f32]) -> Result<f32> {
    if row_payload.len() != types::ggml_row_size(ty, x.len() as u64)? {
        return Err(BitNetError::InvalidGguf(
            "quant_dot: row payload length mismatch vs x / ne[0]".into(),
        ));
    }
    if let Some(value) = super::quant_simd::dot(ty, row_payload, x) {
        return Ok(value);
    }
    match ty {
        0 => dot_row_f32(row_payload, x),
        1 => dot_row_f16(row_payload, x),
        30 => dot_row_bf16(row_payload, x),
        2 => dot_row_q4_0(row_payload, x),
        6 | 10 | 13 | 16 | 17 | 18 | 19 | 21 | 22 | 23 | 29 | 39 => {
            dot_row_extra_quant(ty, row_payload, x)
        }
        8 => dot_row_q8_0(row_payload, x),
        12 => dot_row_q4_k(row_payload, x),
        14 => dot_row_q6_k(row_payload, x),
        34 => dot_row_tq1_0(row_payload, x),
        35 => dot_row_tq2_0(row_payload, x),
        36 => Ok(dot_row_i2_s(row_payload, x)),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

fn dot_row_extra_quant(ty: u32, row: &[u8], x: &[f32]) -> Result<f32> {
    let (elements, bytes) = types::type_layout(ty)?;
    let mut buf = [0.0f32; 256];
    let mut sum = 0.0;
    for (block, xb) in row.chunks_exact(bytes).zip(x.chunks_exact(elements)) {
        crate::ggml::dequant::decode_extra_block(ty, block, &mut buf[..elements])?;
        sum += crate::ggml::simd::dot(&buf[..elements], xb);
    }
    Ok(sum)
}

fn dot_row_f32(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 4 {
        return Err(BitNetError::InvalidGguf("f32 row size".into()));
    }
    let mut sum = 0.0;
    let mut buf = [0.0f32; 32];
    for (weights, xb) in row.chunks(128).zip(x.chunks(32)) {
        for (slot, bytes) in buf.iter_mut().zip(weights.chunks_exact(4)) {
            *slot = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        sum += crate::ggml::simd::dot(&buf[..xb.len()], xb);
    }
    Ok(sum)
}

fn dot_row_f16(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 2 {
        return Err(BitNetError::InvalidGguf("f16 row size".into()));
    }
    #[cfg(target_arch = "x86_64")]
    if f16c_enabled() {
        return Ok(unsafe { dot_row_f16_avx(row, x) });
    }
    Ok(dot_row_f16_scalar(row, x))
}

fn dot_row_f16_scalar(row: &[u8], x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let h = u16::from_le_bytes(row[i * 2..i * 2 + 2].try_into().unwrap());
        s = fp16_to_f32(h).mul_add(x[i], s);
    }
    s
}

#[cfg(target_arch = "x86_64")]
fn f16c_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_row_f16_avx(row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let n = x.len();
    let p = row.as_ptr();
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_cvtph_ps(_mm_loadu_si128(p.add(i * 2).cast()));
        let b = _mm256_cvtph_ps(_mm_loadu_si128(p.add((i + 8) * 2).cast()));
        s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x.as_ptr().add(i)), s0);
        s1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(x.as_ptr().add(i + 8)), s1);
        i += 16;
    }
    let mut acc = hsum256(_mm256_add_ps(s0, s1));
    while i < n {
        let h = u16::from_le_bytes(row[i * 2..i * 2 + 2].try_into().unwrap());
        acc = fp16_to_f32(h).mul_add(x[i], acc);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

fn dot_row_bf16(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 2 {
        return Err(BitNetError::InvalidGguf("bf16 row size".into()));
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let h = u16::from_le_bytes(row[i * 2..i * 2 + 2].try_into().unwrap());
        s = bf16_to_f32(h).mul_add(x[i], s);
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
        acc += crate::ggml::simd::dot(&buf, xb);
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
        acc += crate::ggml::simd::dot(&buf, xb);
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
        acc += crate::ggml::simd::dot(&buf, xb);
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
        acc += crate::ggml::simd::dot(&buf, xb);
    }
    Ok(acc)
}

/// TQ1_0 row dot using a stack scratch block (avoids per-call heap `Vec`).
fn dot_row_tq1_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq1_0 ne0 % 256".into()));
    }
    let nb = x.len() / QK_K;
    const BLOCK: usize = 2 + 48 + 4;
    if row.len() != nb * BLOCK {
        return Err(BitNetError::InvalidGguf("tq1_0 row bytes".into()));
    }
    let mut acc = 0.0f32;
    let mut buf = [0.0f32; QK_K];
    for b in 0..nb {
        let o = b * BLOCK;
        decode_tq1_0_to_f32(&row[o..o + BLOCK], &mut buf)?;
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        acc += crate::ggml::simd::dot(&buf, xb);
    }
    Ok(acc)
}

/// TQ2_0 row dot. `block_tq2_0` stores 64 code bytes then the fp16 scale.
/// Codes are −1, 0, +1, or +2 (`(q − 1)`), matching [`decode_tq2_0_to_f32`].
fn dot_row_i2_s(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let n = x.len();
    for block in 0..(n / 128) {
        let base = block * 32;
        let x0 = block * 128;
        for j in 0..128 {
            let group = j / 32;
            let gp = j % 32;
            let code = row[base + gp] >> (6 - 2 * group);
            acc = crate::ggml::dequant::i2s_trit(code).mul_add(x[x0 + j], acc);
        }
    }
    acc
}

fn i2s_scale(payload: &[u8], row_bytes: usize, ne1: usize) -> Result<f32> {
    let off = row_bytes
        .checked_mul(ne1)
        .ok_or_else(|| BitNetError::InvalidGguf("I2_S scale offset overflow".into()))?;
    let end = off
        .checked_add(4)
        .ok_or_else(|| BitNetError::InvalidGguf("I2_S scale offset overflow".into()))?;
    if payload.len() < end {
        return Err(BitNetError::InvalidGguf(
            "I2_S payload missing the f32 scale trailer".into(),
        ));
    }
    Ok(f32::from_le_bytes(
        payload[off..off + 4].try_into().unwrap(),
    ))
}

fn apply_i2s_scale(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    ne1: usize,
    y: &mut [f32],
) -> Result<()> {
    if ty != 36 {
        return Ok(());
    }
    let scale = i2s_scale(payload, row_bytes, ne1)?;
    for value in y {
        *value *= scale;
    }
    Ok(())
}

fn dot_row_tq2_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq2_0 ne0 % 256".into()));
    }
    let nb = x.len() / QK_K;
    const BLOCK: usize = 2 + 64;
    if row.len() != nb * BLOCK {
        return Err(BitNetError::InvalidGguf("tq2_0 row bytes".into()));
    }
    #[cfg(target_arch = "x86_64")]
    if tq2_avx512_enabled() {
        return Ok(unsafe { dot_row_tq2_0_avx512(row, x, nb) });
    }
    #[cfg(target_arch = "x86_64")]
    if tq2_avx2_enabled() {
        return Ok(unsafe { dot_row_tq2_0_avx2(row, x, nb) });
    }
    Ok(dot_row_tq2_0_scalar(row, x, nb))
}

fn dot_row_tq2_0_scalar(row: &[u8], x: &[f32], nb: usize) -> f32 {
    const QS_LEN: usize = 64;
    const BLOCK: usize = 2 + QS_LEN;
    let mut acc = 0.0f32;
    for b in 0..nb {
        let o = b * BLOCK;
        let qs = &row[o..o + QS_LEN];
        let d = fp16_to_f32(u16::from_le_bytes(
            row[o + QS_LEN..o + QS_LEN + 2].try_into().unwrap(),
        ));
        let xb = &x[b * QK_K..(b + 1) * QK_K];
        let mut block = 0.0f32;
        let mut yp = 0usize;
        for j in (0..QS_LEN).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (qs[j + m] >> (l * 2)) & 3;
                    block += ((q as i8) - 1) as f32 * xb[yp];
                    yp += 1;
                }
            }
        }
        acc += block * d;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
fn tq2_avx512_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn dot_row_tq2_0_avx512(row: &[u8], x: &[f32], nb: usize) -> f32 {
    use std::arch::x86_64::*;
    const QS_LEN: usize = 64;
    const BLOCK: usize = 2 + QS_LEN;
    let mut acc = _mm512_setzero_ps();
    let ones = _mm256_set1_epi8(1);
    let three = _mm256_set1_epi8(3);
    for b in 0..nb {
        let o = b * BLOCK;
        let qs = row.as_ptr().add(o);
        let d = fp16_to_f32(u16::from_le_bytes(
            row[o + QS_LEN..o + BLOCK].try_into().unwrap(),
        ));
        let scale = _mm512_set1_ps(d);
        let xb = x.as_ptr().add(b * QK_K);
        let mut b0 = _mm512_setzero_ps();
        let mut b1 = _mm512_setzero_ps();
        let mut yp = 0usize;
        for j in (0..QS_LEN).step_by(32) {
            let packed = _mm256_loadu_si256(qs.add(j).cast());
            let planes = [
                packed,
                _mm256_srli_epi16::<2>(packed),
                _mm256_srli_epi16::<4>(packed),
                _mm256_srli_epi16::<6>(packed),
            ];
            for shifted in planes {
                let codes = _mm256_sub_epi8(_mm256_and_si256(shifted, three), ones);
                let lo = _mm256_castsi256_si128(codes);
                let hi = _mm256_extracti128_si256(codes, 1);
                b0 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),
                    _mm512_loadu_ps(xb.add(yp)),
                    b0,
                );
                b1 = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),
                    _mm512_loadu_ps(xb.add(yp + 16)),
                    b1,
                );
                yp += 32;
            }
        }
        acc = _mm512_fmadd_ps(_mm512_add_ps(b0, b1), scale, acc);
    }
    let low = _mm512_castps512_ps256(acc);
    let high = _mm512_extractf32x8_ps(acc, 1);
    hsum256(_mm256_add_ps(low, high))
}

#[cfg(target_arch = "x86_64")]
fn tq2_avx2_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_tq2_0_avx2(row: &[u8], x: &[f32], nb: usize) -> f32 {
    use std::arch::x86_64::*;
    const QS_LEN: usize = 64;
    const BLOCK: usize = 2 + QS_LEN;
    let mut acc = _mm256_setzero_ps();
    let ones = _mm256_set1_epi8(1);
    let three = _mm256_set1_epi8(3);
    for b in 0..nb {
        let o = b * BLOCK;
        let qs = row.as_ptr().add(o);
        let d = fp16_to_f32(u16::from_le_bytes(
            row[o + QS_LEN..o + BLOCK].try_into().unwrap(),
        ));
        let scale = _mm256_set1_ps(d);
        let xb = x.as_ptr().add(b * QK_K);
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut yp = 0usize;
        for j in (0..QS_LEN).step_by(32) {
            let packed = _mm256_loadu_si256(qs.add(j).cast());
            // Shifts of 0, 2, 4 and 6 stay inside each byte: a 16-bit shift does not
            // mix the high byte into the low byte, so a single mask recovers the plane.
            let planes = [
                packed,
                _mm256_srli_epi16::<2>(packed),
                _mm256_srli_epi16::<4>(packed),
                _mm256_srli_epi16::<6>(packed),
            ];
            for shifted in planes {
                let codes = _mm256_sub_epi8(_mm256_and_si256(shifted, three), ones);
                tq2_fmadd_i8_32_split(codes, xb.add(yp), &mut b0, &mut b1, &mut b2, &mut b3);
                yp += 32;
            }
        }
        let block = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        acc = _mm256_fmadd_ps(block, scale, acc);
    }
    hsum256(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tq2_fmadd_i8_32_split(
    codes: std::arch::x86_64::__m256i,
    x: *const f32,
    a0: &mut std::arch::x86_64::__m256,
    a1: &mut std::arch::x86_64::__m256,
    a2: &mut std::arch::x86_64::__m256,
    a3: &mut std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let lo = _mm256_extracti128_si256(codes, 0);
    let hi = _mm256_extracti128_si256(codes, 1);
    *a0 = tq2_fmadd_i8_8(lo, x, *a0);
    *a1 = tq2_fmadd_i8_8(_mm_bsrli_si128(lo, 8), x.add(8), *a1);
    *a2 = tq2_fmadd_i8_8(hi, x.add(16), *a2);
    *a3 = tq2_fmadd_i8_8(_mm_bsrli_si128(hi, 8), x.add(24), *a3);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tq2_fmadd_i8_8(
    eight: std::arch::x86_64::__m128i,
    x: *const f32,
    acc: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eight));
    _mm256_fmadd_ps(w, _mm256_loadu_ps(x), acc)
}

fn decode_tq1_0_to_f32(row: &[u8], out: &mut [f32]) -> Result<()> {
    if out.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq1_0 decode ne0".into()));
    }
    const QS_LEN: usize = 48;
    const QH_LEN: usize = 4;
    const BLOCK: usize = 2 + QS_LEN + QH_LEN;
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let nb = out.len() / QK_K;
    if row.len() != nb * BLOCK {
        return Err(BitNetError::InvalidGguf("tq1_0 row bytes".into()));
    }
    for b in 0..nb {
        let o = b * BLOCK;
        let d = fp16_to_f32(u16::from_le_bytes(row[o..o + 2].try_into().unwrap()));
        let qs = &row[o + 2..o + 2 + QS_LEN];
        let qh = &row[o + 2 + QS_LEN..o + 2 + QS_LEN + QH_LEN];
        let mut yp = b * QK_K;
        for j in (0..(QS_LEN - QS_LEN % 32)).step_by(32) {
            for p in POW3.iter().take(5) {
                for m in 0..32 {
                    let q = qs[j + m].wrapping_mul(*p);
                    let xi = (((q as u16) * 3) >> 8) as i16;
                    out[yp] = (xi - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
        for j in (QS_LEN - QS_LEN % 32..QS_LEN).step_by(16) {
            for p in POW3.iter().take(5) {
                for m in 0..16 {
                    let q = qs[j + m].wrapping_mul(*p);
                    let xi = (((q as u16) * 3) >> 8) as i16;
                    out[yp] = (xi - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
        for p in POW3.iter().take(4) {
            for qh_byte in qh {
                let q = qh_byte.wrapping_mul(*p);
                let xi = (((q as u16) * 3) >> 8) as i16;
                out[yp] = (xi - 1) as f32 * d;
                yp += 1;
            }
        }
        debug_assert_eq!(yp, (b + 1) * QK_K);
    }
    Ok(())
}

fn decode_tq2_0_to_f32(row: &[u8], out: &mut [f32]) -> Result<()> {
    if out.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq2_0 decode ne0".into()));
    }
    const QS_LEN: usize = 64;
    const BLOCK: usize = 2 + QS_LEN;
    let nb = out.len() / QK_K;
    if row.len() != nb * BLOCK {
        return Err(BitNetError::InvalidGguf("tq2_0 row bytes".into()));
    }
    for b in 0..nb {
        let o = b * BLOCK;
        let qs = &row[o..o + QS_LEN];
        let d = fp16_to_f32(u16::from_le_bytes(
            row[o + QS_LEN..o + QS_LEN + 2].try_into().unwrap(),
        ));
        let mut yp = b * QK_K;
        for j in (0..QS_LEN).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (qs[j + m] >> (l * 2)) & 3;
                    out[yp] = (q as i8 - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
        debug_assert_eq!(yp, (b + 1) * QK_K);
    }
    Ok(())
}

/// Decode one matrix row (second index `row`) to `out` (`len == ne0`).
#[inline]
pub fn decode_row_to_f32(ty: u32, row_payload: &[u8], out: &mut [f32]) -> Result<()> {
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
        6 | 10 | 13 | 16 | 17 | 18 | 19 | 21 | 22 | 23 | 29 | 39 => {
            let (elements, bytes) = types::type_layout(ty)?;
            for (block, dst) in row_payload
                .chunks_exact(bytes)
                .zip(out.chunks_exact_mut(elements))
            {
                crate::ggml::dequant::decode_extra_block(ty, block, dst)?;
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
        34 => decode_tq1_0_to_f32(row_payload, out)?,
        35 => decode_tq2_0_to_f32(row_payload, out)?,
        36 => crate::ggml::dequant::decode_i2_s_row(row_payload, out),
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
    QuantMatvecKernel::default().matvec_mmap(archive, t, x, ne0, ne1)
}

/// Several activation rows against the same weights. `xs` and `y` are token-major.
/// Each output element uses the same [`dot_row`] as a single matvec.
pub fn matvec_batch_mmap(
    archive: &GgufArchive,
    t: &GgufTensorInfo,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    if n_tokens == 0 || xs.len() != n_tokens * ne0 {
        return Err(BitNetError::Inference(
            "matvec batch: activation shape".into(),
        ));
    }
    if t.dimensions.len() < 2
        || t.dimensions[0] as usize != ne0
        || t.dimensions[1] as usize != ne1
    {
        return Err(BitNetError::Inference(
            "quant matvec batch: tensor dims mismatch".into(),
        ));
    }
    let payload = archive.tensor_payload(t)?;
    let row_bytes = types::ggml_row_size(t.ggml_type, ne0 as u64)?;
    matvec_rows_batch(t.ggml_type, payload, row_bytes, xs, n_tokens, ne0, ne1)
}

/// One weight row, dequantized once, then one dot per activation row.
/// Q4_K and Q6_K keep the per-superblock sum used by [`dot_row`].
fn dots_for_weight_row(
    ty: u32,
    w: &[u8],
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    dense: &mut [f32],
    dots: &mut [f32],
) -> Result<()> {
    // Type 36 keeps its scale in a tensor trailer applied after the dots.
    if ty != 36 && decode_row_to_f32(ty, w, dense).is_ok() {
        let block = matches!(ty, 12 | 14).then_some(QK_K);
        for t in 0..n_tokens {
            let x = &xs[t * ne0..(t + 1) * ne0];
            dots[t] = if let Some(bk) = block {
                let mut acc = 0.0f32;
                for (d, xb) in dense.chunks_exact(bk).zip(x.chunks_exact(bk)) {
                    acc += super::simd::dot(d, xb);
                }
                acc
            } else {
                super::simd::dot(dense, x)
            };
        }
        return Ok(());
    }
    for t in 0..n_tokens {
        dots[t] = dot_row(ty, w, &xs[t * ne0..(t + 1) * ne0])?;
    }
    Ok(())
}

fn i2s_q8k_layout(row_bytes: usize, ne0: usize) -> bool {
    ne0 > 0 && ne0.is_multiple_of(256) && row_bytes == ne0 / 4
}

/// `code 0 → -1`, `2 → +1`, `1` and `3 → 0`. Stored as `trit + 1` so `maddubs` applies.
fn i2s_trit_i32(code: i32) -> i32 {
    match code & 3 {
        0 => -1,
        2 => 1,
        _ => 0,
    }
}

fn dot_i2s_q8k_scalar(row: &[u8], act: &Q8KAct) -> f32 {
    let nb = act.d.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let codes = &row[i * 64..(i + 1) * 64];
        let q8 = &act.qs[i * 256..(i + 1) * 256];
        let mut sumi = 0i32;
        for sub in 0..2 {
            let bytes = &codes[sub * 32..sub * 32 + 32];
            for group in 0..4 {
                let shift = 6 - 2 * group;
                for gp in 0..32 {
                    let code = i32::from(bytes[gp] >> shift);
                    let activation = i32::from(q8[sub * 128 + group * 32 + gp]);
                    sumi += i2s_trit_i32(code) * activation;
                }
            }
        }
        sumf += (sumi as f32) * act.d[i];
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i2s_q8k_avx2(row: *const u8, act: &Q8KAct) -> f32 {
    use std::arch::x86_64::*;
    let nb = act.d.len();
    let mask = _mm256_set1_epi8(3);
    let three = _mm256_set1_epi8(3);
    let one_b = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let mut acc = _mm256_setzero_si256();
        let base = row.add(i * 64);
        let q8 = act.qs.as_ptr().add(i * 256);
        for sub in 0..2usize {
            let packed = _mm256_loadu_si256(base.add(sub * 32).cast());
            let planes = [
                _mm256_and_si256(_mm256_srli_epi16::<6>(packed), mask),
                _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask),
                _mm256_and_si256(_mm256_srli_epi16::<2>(packed), mask),
                _mm256_and_si256(packed, mask),
            ];
            for (group, plane) in planes.into_iter().enumerate() {
                let eq3 = _mm256_cmpeq_epi8(plane, three);
                let mapped = _mm256_blendv_epi8(plane, one_b, eq3);
                let activation = _mm256_loadu_si256(q8.add(sub * 128 + group * 32).cast());
                acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(mapped, activation));
            }
        }
        let mut prod_lanes = [0i32; 8];
        _mm256_storeu_si256(
            prod_lanes.as_mut_ptr().cast(),
            _mm256_madd_epi16(acc, ones16),
        );
        let mut bsum_lanes = [0i16; 16];
        _mm256_storeu_si256(
            bsum_lanes.as_mut_ptr().cast(),
            _mm256_loadu_si256(act.bsums.as_ptr().add(i * 16).cast()),
        );
        let prod: i32 = prod_lanes.iter().sum();
        let bsum: i32 = bsum_lanes.iter().map(|v| i32::from(*v)).sum();
        sumf += (prod - bsum) as f32 * *act.d.as_ptr().add(i);
    }
    sumf
}

fn dot_i2s_q8k(row: &[u8], act: &Q8KAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { dot_i2s_q8k_avx2(row.as_ptr(), act) };
    }
    dot_i2s_q8k_scalar(row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i2s_f32_avx2(row: *const u8, x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(3);
    let zero = _mm256_setzero_si256();
    let two = _mm256_set1_epi32(2);
    let mut acc = _mm256_setzero_ps();
    let mut base = 0usize;
    while base + 128 <= n {
        let packed = _mm256_loadu_si256(row.add(base / 4).cast());
        let planes = [
            _mm256_and_si256(_mm256_srli_epi16::<6>(packed), mask),
            _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask),
            _mm256_and_si256(_mm256_srli_epi16::<2>(packed), mask),
            _mm256_and_si256(packed, mask),
        ];
        for (group, plane) in planes.into_iter().enumerate() {
            let low = _mm256_castsi256_si128(plane);
            let high = _mm256_extracti128_si256(plane, 1);
            let chunks = [
                _mm256_cvtepu8_epi32(low),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(low)),
                _mm256_cvtepu8_epi32(high),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(high)),
            ];
            for (chunk, codes) in chunks.into_iter().enumerate() {
                let xv = _mm256_loadu_ps(x.add(base + group * 32 + chunk * 8));
                let is0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, zero));
                let is2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(codes, two));
                let pos = _mm256_and_ps(xv, is2);
                let neg = _mm256_and_ps(xv, is0);
                acc = _mm256_add_ps(acc, _mm256_sub_ps(pos, neg));
            }
        }
        base += 128;
    }
    hsum256(acc)
}

fn dot_i2s_f32(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && x.len().is_multiple_of(128) {
        return unsafe { dot_i2s_f32_avx2(row.as_ptr(), x.as_ptr(), x.len()) };
    }
    dot_row_i2_s(row, x)
}

fn fill_i2s_f32(payload: &[u8], row_bytes: usize, first: usize, y: &mut [f32], x: &[f32]) {
    for (local, slot) in y.iter_mut().enumerate() {
        let start = (first + local) * row_bytes;
        *slot = dot_i2s_f32(&payload[start..start + row_bytes], x);
    }
}

fn i2s_f32_preferred() -> bool {
    static PREF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PREF.get_or_init(|| {
        !matches!(
            std::env::var("RBITNET_I2S_Q8K").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
    })
}

fn matvec_i2s_f32(payload: &[u8], row_bytes: usize, x: &[f32], ne1: usize) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; ne1];
    if ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1);
    if threads <= 1 {
        fill_i2s_f32(payload, row_bytes, 0, &mut y, x);
    } else {
        let chunk_rows = ne1.div_ceil(threads);
        pool.install(|| {
            y.par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk, rows)| {
                    fill_i2s_f32(payload, row_bytes, chunk * chunk_rows, rows, x);
                });
        });
    }
    apply_i2s_scale(36, payload, row_bytes, ne1, &mut y)?;
    Ok(y)
}

fn matvec_i2s_f32_batch(
    payload: &[u8],
    row_bytes: usize,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; n_tokens.saturating_mul(ne1)];
    if n_tokens == 0 || ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 {
        let mut column = vec![0.0f32; ne1];
        for token in 0..n_tokens {
            fill_i2s_f32(
                payload,
                row_bytes,
                0,
                &mut column,
                &xs[token * ne0..(token + 1) * ne0],
            );
            for (row, value) in column.iter().copied().enumerate() {
                y[token * ne1 + row] = value;
            }
        }
    } else {
        let chunk_rows = ne1.div_ceil(threads);
        let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
        let parts: Vec<(usize, Vec<f32>)> = pool.install(|| {
            chunk_starts
                .par_iter()
                .copied()
                .map(|chunk_start| {
                    let chunk_end = (chunk_start + chunk_rows).min(ne1);
                    let rows = chunk_end - chunk_start;
                    let mut block = vec![0.0f32; rows * n_tokens];
                    let mut column = vec![0.0f32; rows];
                    for token in 0..n_tokens {
                        fill_i2s_f32(
                            payload,
                            row_bytes,
                            chunk_start,
                            &mut column,
                            &xs[token * ne0..(token + 1) * ne0],
                        );
                        for (local, value) in column.iter().copied().enumerate() {
                            block[local * n_tokens + token] = value;
                        }
                    }
                    (chunk_start, block)
                })
                .collect()
        });
        for (chunk_start, block) in parts {
            let rows = block.len() / n_tokens.max(1);
            for local in 0..rows {
                let row = chunk_start + local;
                for token in 0..n_tokens {
                    y[token * ne1 + row] = block[local * n_tokens + token];
                }
            }
        }
    }
    for token in 0..n_tokens {
        apply_i2s_scale(
            36,
            payload,
            row_bytes,
            ne1,
            &mut y[token * ne1..(token + 1) * ne1],
        )?;
    }
    Ok(y)
}

fn fill_i2s_q8k(payload: &[u8], row_bytes: usize, first: usize, y: &mut [f32], act: &Q8KAct) {
    for (local, slot) in y.iter_mut().enumerate() {
        let start = (first + local) * row_bytes;
        *slot = dot_i2s_q8k(&payload[start..start + row_bytes], act);
    }
}

fn matvec_i2s_q8k(payload: &[u8], row_bytes: usize, x: &[f32], ne1: usize) -> Result<Vec<f32>> {
    let act = quantize_q8_k(x);
    let mut y = vec![0.0f32; ne1];
    if ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1);
    if threads <= 1 {
        fill_i2s_q8k(payload, row_bytes, 0, &mut y, &act);
    } else {
        let chunk_rows = ne1.div_ceil(threads);
        pool.install(|| {
            y.par_chunks_mut(chunk_rows).enumerate().for_each(|(chunk, rows)| {
                fill_i2s_q8k(payload, row_bytes, chunk * chunk_rows, rows, &act);
            });
        });
    }
    apply_i2s_scale(36, payload, row_bytes, ne1, &mut y)?;
    Ok(y)
}

fn matvec_i2s_q8k_batch(
    payload: &[u8],
    row_bytes: usize,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    let acts: Vec<Q8KAct> = (0..n_tokens)
        .map(|token| quantize_q8_k(&xs[token * ne0..(token + 1) * ne0]))
        .collect();
    let mut y = vec![0.0f32; n_tokens.saturating_mul(ne1)];
    if n_tokens == 0 || ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 {
        let mut column = vec![0.0f32; ne1];
        for token in 0..n_tokens {
            fill_i2s_q8k(payload, row_bytes, 0, &mut column, &acts[token]);
            for (row, value) in column.iter().copied().enumerate() {
                y[token * ne1 + row] = value;
            }
        }
    } else {
        let chunk_rows = ne1.div_ceil(threads);
        let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
        let parts: Vec<(usize, Vec<f32>)> = pool.install(|| {
            chunk_starts
                .par_iter()
                .copied()
                .map(|chunk_start| {
                    let chunk_end = (chunk_start + chunk_rows).min(ne1);
                    let rows = chunk_end - chunk_start;
                    let mut block = vec![0.0f32; rows * n_tokens];
                    let mut column = vec![0.0f32; rows];
                    for token in 0..n_tokens {
                        fill_i2s_q8k(
                            payload,
                            row_bytes,
                            chunk_start,
                            &mut column,
                            &acts[token],
                        );
                        for (local, value) in column.iter().copied().enumerate() {
                            block[local * n_tokens + token] = value;
                        }
                    }
                    (chunk_start, block)
                })
                .collect()
        });
        for (chunk_start, block) in parts {
            let rows = block.len() / n_tokens.max(1);
            for local in 0..rows {
                let row = chunk_start + local;
                for token in 0..n_tokens {
                    y[token * ne1 + row] = block[local * n_tokens + token];
                }
            }
        }
    }
    for token in 0..n_tokens {
        apply_i2s_scale(
            36,
            payload,
            row_bytes,
            ne1,
            &mut y[token * ne1..(token + 1) * ne1],
        )?;
    }
    Ok(y)
}

fn matvec_rows_batch(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    if ty == 35 && tq2_q8k_enabled() && tq2_q8k_layout(row_bytes, ne0) {
        return matvec_tq2_q8k_batch(payload, row_bytes, xs, n_tokens, ne0, ne1);
    }
    if ty == 36 && i2s_q8k_layout(row_bytes, ne0) {
        return if i2s_f32_preferred() {
            matvec_i2s_f32_batch(payload, row_bytes, xs, n_tokens, ne0, ne1)
        } else {
            matvec_i2s_q8k_batch(payload, row_bytes, xs, n_tokens, ne0, ne1)
        };
    }
    let mut y = vec![0.0f32; n_tokens.saturating_mul(ne1)];
    if n_tokens == 0 || ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 || ne1 < 2 {
        let mut dense = vec![0.0f32; ne0];
        let mut dots = vec![0.0f32; n_tokens];
        for row in 0..ne1 {
            let w = &payload[row * row_bytes..row * row_bytes + row_bytes];
            dots_for_weight_row(ty, w, xs, n_tokens, ne0, &mut dense, &mut dots)?;
            for t in 0..n_tokens {
                y[t * ne1 + row] = dots[t];
            }
        }
    } else {
        let chunk_rows = ne1.div_ceil(threads);
        let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
        let parts: Vec<Result<(usize, Vec<f32>)>> = pool.install(|| {
            chunk_starts
                .par_iter()
                .copied()
                .map(|chunk_start| {
                    let chunk_end = (chunk_start + chunk_rows).min(ne1);
                    let rows = chunk_end - chunk_start;
                    let mut block = vec![0.0f32; rows * n_tokens];
                    let mut dense = vec![0.0f32; ne0];
                    for (local, row) in (chunk_start..chunk_end).enumerate() {
                        let w = &payload[row * row_bytes..row * row_bytes + row_bytes];
                        let dots = &mut block[local * n_tokens..local * n_tokens + n_tokens];
                        dots_for_weight_row(ty, w, xs, n_tokens, ne0, &mut dense, dots)?;
                    }
                    Ok((chunk_start, block))
                })
                .collect()
        });
        for part in parts {
            let (chunk_start, block) = part?;
            let rows = block.len() / n_tokens.max(1);
            for local in 0..rows {
                let row = chunk_start + local;
                for t in 0..n_tokens {
                    y[t * ne1 + row] = block[local * n_tokens + t];
                }
            }
        }
    }
    apply_i2s_scale(ty, payload, row_bytes, ne1, &mut y)?;
    Ok(y)
}

fn validate_matvec_shape(t: &GgufTensorInfo, x: &[f32], ne0: usize, ne1: usize) -> Result<()> {
    if t.dimensions.len() < 2 {
        return Err(BitNetError::InvalidGguf(
            "matvec: expected 2D tensor".into(),
        ));
    }
    if t.dimensions[0] as usize != ne0 || t.dimensions[1] as usize != ne1 {
        return Err(BitNetError::Inference(
            "quant matvec: tensor dims mismatch".into(),
        ));
    }
    if x.len() != ne0 {
        return Err(BitNetError::Inference("matvec: x len".into()));
    }
    Ok(())
}

fn quant_matvec_thread_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or_else(|| {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("rbitnet-quant-{i}"))
            .build()
            .expect("quant matvec thread pool")
    })
}

/// llama.cpp `nearest_int`: add 1.5·2^23 and recover the integer from the mantissa.
fn ggml_nearest_int(fval: f32) -> i32 {
    let bits = (fval + 12582912.0f32).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

struct Q8KAct {
    /// One scale per block of 256, `1 / (-127 / signed_peak)`.
    d: Vec<f32>,
    qs: Vec<i8>,
    bsums: Vec<i16>,
}

/// Activation quantizer matching `quantize_row_q8_K_ref`.
///
/// The peak keeps its sign (`iscale = -127 / max`, not `/ amax`). Equal absolute
/// values keep the earlier element, because the scan uses a strict greater-than.
fn quantize_q8_k(x: &[f32]) -> Q8KAct {
    let nb = x.len() / 256;
    let mut d = vec![0.0f32; nb];
    let mut qs = vec![0i8; nb * 256];
    let mut bsums = vec![0i16; nb * 16];
    for i in 0..nb {
        let xb = &x[i * 256..(i + 1) * 256];
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in xb {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = value;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127.0f32 / max;
        let q = &mut qs[i * 256..(i + 1) * 256];
        for (slot, &value) in q.iter_mut().zip(xb) {
            let rounded = ggml_nearest_int(iscale * value).min(127);
            *slot = rounded as i8;
        }
        for group in 0..16 {
            let mut sum = 0i32;
            for lane in 0..16 {
                sum += q[group * 16 + lane] as i32;
            }
            bsums[i * 16 + group] = sum as i16;
        }
        d[i] = 1.0 / iscale;
    }
    Q8KAct { d, qs, bsums }
}

fn tq2_q8k_layout(row_bytes: usize, ne0: usize) -> bool {
    ne0 > 0 && ne0.is_multiple_of(256) && row_bytes == (ne0 / 256) * 66
}

fn tq2_q8k_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("RBITNET_TQ2_F32").ok().as_deref(),
            Some("1" | "true" | "yes")
        )
    })
}

/// Integer dot from `ggml_vec_dot_tq2_0_q8_K_generic`: `(code - 1) * q8`, then both scales.
fn dot_tq2_q8k_scalar(row: &[u8], act: &Q8KAct) -> f32 {
    let nb = act.d.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q2 = &row[i * 66..i * 66 + 64];
        let q8 = &act.qs[i * 256..(i + 1) * 256];
        let mut sumi = 0i32;
        let mut j = 0usize;
        while j < 64 {
            for plane in 0..4 {
                for k in 0..32 {
                    let raw = i32::from((q2[j + k] >> (plane * 2)) & 3);
                    let activation = q8[j * 4 + plane * 32 + k] as i32;
                    sumi += activation * (raw - 1);
                }
            }
            j += 32;
        }
        let weight_scale = fp16_to_f32(u16::from_le_bytes(
            row[i * 66 + 64..i * 66 + 66].try_into().unwrap(),
        ));
        sumf += (sumi as f32) * act.d[i] * weight_scale;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_tq2_q8k_avx2(row: *const u8, act: &Q8KAct) -> f32 {
    use std::arch::x86_64::*;
    let nb = act.d.len();
    let mask = _mm256_set1_epi8(3);
    let one = _mm256_set1_epi16(1);
    let mut sumf = _mm256_setzero_ps();
    for i in 0..nb {
        let mut sumi0 = _mm256_setzero_si256();
        let mut sumi1 = _mm256_setzero_si256();
        let q2 = row.add(i * 66);
        let q8 = act.qs.as_ptr().add(i * 256);
        let mut j = 0usize;
        while j < 64 {
            let packed = _mm256_loadu_si256(q2.add(j).cast());
            let qx0 = _mm256_and_si256(packed, mask);
            let qx1 = _mm256_and_si256(_mm256_srli_epi16::<2>(packed), mask);
            let qx2 = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask);
            let qx3 = _mm256_and_si256(_mm256_srli_epi16::<6>(packed), mask);
            let base = q8.add(j * 4);
            let p0 = _mm256_maddubs_epi16(qx0, _mm256_loadu_si256(base.cast()));
            let p1 = _mm256_maddubs_epi16(qx1, _mm256_loadu_si256(base.add(32).cast()));
            let p2 = _mm256_maddubs_epi16(qx2, _mm256_loadu_si256(base.add(64).cast()));
            let p3 = _mm256_maddubs_epi16(qx3, _mm256_loadu_si256(base.add(96).cast()));
            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(p0, p1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(p2, p3));
            j += 32;
        }
        let ysum = _mm256_loadu_si256(act.bsums.as_ptr().add(i * 16).cast());
        let weight_scale = fp16_to_f32(u16::from_le_bytes([*q2.add(64), *q2.add(65)]));
        let scale = _mm256_set1_ps(*act.d.as_ptr().add(i) * weight_scale);
        let mut acc = _mm256_add_epi16(sumi0, sumi1);
        acc = _mm256_sub_epi16(acc, ysum);
        acc = _mm256_madd_epi16(acc, one);
        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc), scale), sumf);
    }
    hsum256(sumf)
}

#[cfg(test)]
fn dot_tq2_q8k(row: &[u8], act: &Q8KAct) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { dot_tq2_q8k_avx2(row.as_ptr(), act) };
    }
    dot_tq2_q8k_scalar(row, act)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_tq2_q8k_avx2(
    payload: &[u8],
    row_bytes: usize,
    first: usize,
    y: &mut [f32],
    act: &Q8KAct,
) {
    for (local, slot) in y.iter_mut().enumerate() {
        let start = (first + local) * row_bytes;
        *slot = dot_tq2_q8k_avx2(payload.as_ptr().add(start), act);
    }
}

fn fill_tq2_q8k(payload: &[u8], row_bytes: usize, first: usize, y: &mut [f32], act: &Q8KAct) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { fill_tq2_q8k_avx2(payload, row_bytes, first, y, act) }
        return;
    }
    for (local, slot) in y.iter_mut().enumerate() {
        let start = (first + local) * row_bytes;
        *slot = dot_tq2_q8k_scalar(&payload[start..start + row_bytes], act);
    }
}

fn matvec_tq2_q8k(payload: &[u8], row_bytes: usize, x: &[f32], ne1: usize) -> Result<Vec<f32>> {
    let act = quantize_q8_k(x);
    let mut y = vec![0.0f32; ne1];
    if ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1);
    if threads <= 1 {
        fill_tq2_q8k(payload, row_bytes, 0, &mut y, &act);
        return Ok(y);
    }
    let chunk_rows = ne1.div_ceil(threads);
    pool.install(|| {
        y.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk, rows)| {
                fill_tq2_q8k(payload, row_bytes, chunk * chunk_rows, rows, &act);
            });
    });
    Ok(y)
}

fn matvec_tq2_q8k_batch(
    payload: &[u8],
    row_bytes: usize,
    xs: &[f32],
    n_tokens: usize,
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>> {
    let acts: Vec<Q8KAct> = (0..n_tokens)
        .map(|token| quantize_q8_k(&xs[token * ne0..(token + 1) * ne0]))
        .collect();
    let mut y = vec![0.0f32; n_tokens.saturating_mul(ne1)];
    if n_tokens == 0 || ne1 == 0 {
        return Ok(y);
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 {
        let mut column = vec![0.0f32; ne1];
        for token in 0..n_tokens {
            fill_tq2_q8k(payload, row_bytes, 0, &mut column, &acts[token]);
            for (row, value) in column.iter().copied().enumerate() {
                y[token * ne1 + row] = value;
            }
        }
        return Ok(y);
    }
    let chunk_rows = ne1.div_ceil(threads);
    let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
    let parts: Vec<(usize, Vec<f32>)> = pool.install(|| {
        chunk_starts
            .par_iter()
            .copied()
            .map(|chunk_start| {
                let chunk_end = (chunk_start + chunk_rows).min(ne1);
                let rows = chunk_end - chunk_start;
                let mut block = vec![0.0f32; rows * n_tokens];
                let mut column = vec![0.0f32; rows];
                for token in 0..n_tokens {
                    fill_tq2_q8k(payload, row_bytes, chunk_start, &mut column, &acts[token]);
                    for (local, value) in column.iter().copied().enumerate() {
                        block[local * n_tokens + token] = value;
                    }
                }
                (chunk_start, block)
            })
            .collect()
    });
    for (chunk_start, block) in parts {
        let rows = block.len() / n_tokens.max(1);
        for local in 0..rows {
            let row = chunk_start + local;
            for token in 0..n_tokens {
                y[token * ne1 + row] = block[local * n_tokens + token];
            }
        }
    }
    Ok(y)
}

fn effective_parallel_min_rows(cfg_min: usize, ne0: usize, ne1: usize) -> usize {
    // Large inner dimension: parallelize smaller output rows too.
    let boosted = if ne0 >= 2048 {
        cfg_min / 2
    } else if ne0 >= 1024 {
        (cfg_min * 3) / 4
    } else {
        cfg_min
    };
    boosted.max(32).min(ne1.max(1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fill_f16_avx2(payload: &[u8], row_bytes: usize, first: usize, y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let xp = x.as_ptr();
    let mut row = 0usize;
    while row + 2 <= y.len() {
        let p0 = payload.as_ptr().add((first + row) * row_bytes);
        let p1 = payload.as_ptr().add((first + row + 1) * row_bytes);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            let x0 = _mm256_loadu_ps(xp.add(i));
            let x1 = _mm256_loadu_ps(xp.add(i + 8));
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i * 2).cast())), x0, a0);
            a1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p0.add((i + 8) * 2).cast())),
                x1,
                a1,
            );
            b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i * 2).cast())), x0, b0);
            b1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p1.add((i + 8) * 2).cast())),
                x1,
                b1,
            );
            i += 16;
        }
        let mut acc0 = hsum256(_mm256_add_ps(a0, a1));
        let mut acc1 = hsum256(_mm256_add_ps(b0, b1));
        while i < n {
            let h0 = u16::from_le_bytes([*p0.add(i * 2), *p0.add(i * 2 + 1)]);
            let h1 = u16::from_le_bytes([*p1.add(i * 2), *p1.add(i * 2 + 1)]);
            acc0 = fp16_to_f32(h0).mul_add(*xp.add(i), acc0);
            acc1 = fp16_to_f32(h1).mul_add(*xp.add(i), acc1);
            i += 1;
        }
        y[row] = acc0;
        y[row + 1] = acc1;
        row += 2;
    }
    if row < y.len() {
        let p = payload.as_ptr().add((first + row) * row_bytes);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p.add(i * 2).cast())), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p.add((i + 8) * 2).cast())),
                _mm256_loadu_ps(xp.add(i + 8)),
                a1,
            );
            i += 16;
        }
        let mut acc = hsum256(_mm256_add_ps(a0, a1));
        while i < n {
            let h = u16::from_le_bytes([*p.add(i * 2), *p.add(i * 2 + 1)]);
            acc = fp16_to_f32(h).mul_add(*xp.add(i), acc);
            i += 1;
        }
        y[row] = acc;
    }
}

/// Q8_0 GEMV: 2-byte fp16 scale then 32 int8 values. Two rows share each activation load.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fill_q8_0_avx2(payload: &[u8], row_bytes: usize, first: usize, y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let blocks = n / 32;
    let xp = x.as_ptr();
    let mut row = 0usize;
    while row + 2 <= y.len() {
        let p0 = payload.as_ptr().add((first + row) * row_bytes);
        let p1 = payload.as_ptr().add((first + row + 1) * row_bytes);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        for b in 0..blocks {
            let s0 = _mm256_set1_ps(fp16_to_f32(u16::from_le_bytes([
                *p0.add(b * 34),
                *p0.add(b * 34 + 1),
            ])));
            let s1 = _mm256_set1_ps(fp16_to_f32(u16::from_le_bytes([
                *p1.add(b * 34),
                *p1.add(b * 34 + 1),
            ])));
            let q0 = p0.add(b * 34 + 2);
            let q1 = p1.add(b * 34 + 2);
            let xb = xp.add(b * 32);
            let x0 = _mm256_loadu_ps(xb);
            let x1 = _mm256_loadu_ps(xb.add(8));
            let x2 = _mm256_loadu_ps(xb.add(16));
            let x3 = _mm256_loadu_ps(xb.add(24));
            a0 = _mm256_fmadd_ps(q8_weight8(q0, 0, s0), x0, a0);
            b0 = _mm256_fmadd_ps(q8_weight8(q1, 0, s1), x0, b0);
            a1 = _mm256_fmadd_ps(q8_weight8(q0, 8, s0), x1, a1);
            b1 = _mm256_fmadd_ps(q8_weight8(q1, 8, s1), x1, b1);
            a0 = _mm256_fmadd_ps(q8_weight8(q0, 16, s0), x2, a0);
            b0 = _mm256_fmadd_ps(q8_weight8(q1, 16, s1), x2, b0);
            a1 = _mm256_fmadd_ps(q8_weight8(q0, 24, s0), x3, a1);
            b1 = _mm256_fmadd_ps(q8_weight8(q1, 24, s1), x3, b1);
        }
        y[row] = hsum256(_mm256_add_ps(a0, a1));
        y[row + 1] = hsum256(_mm256_add_ps(b0, b1));
        row += 2;
    }
    if row < y.len() {
        let p = payload.as_ptr().add((first + row) * row_bytes);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for b in 0..blocks {
            let scale = _mm256_set1_ps(fp16_to_f32(u16::from_le_bytes([
                *p.add(b * 34),
                *p.add(b * 34 + 1),
            ])));
            let q = p.add(b * 34 + 2);
            let xb = xp.add(b * 32);
            let mut k = 0usize;
            while k < 32 {
                let v = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(q.add(k).cast()))),
                    scale,
                );
                let xv = _mm256_loadu_ps(xb.add(k));
                if k & 8 == 0 {
                    a0 = _mm256_fmadd_ps(v, xv, a0);
                } else {
                    a1 = _mm256_fmadd_ps(v, xv, a1);
                }
                k += 8;
            }
        }
        y[row] = hsum256(_mm256_add_ps(a0, a1));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q8_weight8(q: *const u8, off: usize, scale: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(q.add(off).cast()))),
        scale,
    )
}

/// Pack an f16 weight matrix (`[ne0, ne1]`, 2 bytes per value) into Q8_0 rows.
pub(crate) fn quantize_f16_rows_to_q8_0(payload: &[u8], ne0: usize, ne1: usize) -> Result<Vec<u8>> {
    if ne0 == 0 || ne0 % 32 != 0 || payload.len() < ne1.saturating_mul(ne0.saturating_mul(2)) {
        return Err(BitNetError::Inference(
            "f16 to q8_0: shape".into(),
        ));
    }
    let q8_row = (ne0 / 32) * 34;
    let mut out = vec![0u8; q8_row * ne1];
    let src_row = ne0 * 2;
    out.par_chunks_mut(q8_row)
        .enumerate()
        .for_each(|(row, dest_row)| {
            let src = &payload[row * src_row..(row + 1) * src_row];
            for b in 0..(ne0 / 32) {
                let mut vals = [0.0f32; 32];
                let mut amax = 0.0f32;
                for j in 0..32 {
                    let o = (b * 32 + j) * 2;
                    vals[j] = fp16_to_f32(u16::from_le_bytes([src[o], src[o + 1]]));
                    amax = amax.max(vals[j].abs());
                }
                let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                let dest = b * 34;
                dest_row[dest..dest + 2]
                    .copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
                for j in 0..32 {
                    dest_row[dest + 2 + j] =
                        (vals[j] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        });
    Ok(out)
}

pub(crate) fn matvec_q8_0_rows(
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    if x.len() % 32 != 0 || row_bytes != (x.len() / 32) * 34 || payload.len() < ne1 * row_bytes {
        return Err(BitNetError::Inference("q8_0 matvec: shape".into()));
    }
    let mut y = vec![0.0f32; ne1];
    if ne1 == 0 {
        return Ok(y);
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        let pool = quant_matvec_thread_pool();
        let threads = pool.current_num_threads().min(ne1).max(1);
        if threads <= 1 {
            unsafe { fill_q8_0_avx2(payload, row_bytes, 0, &mut y, x) }
            return Ok(y);
        }
        let chunk_rows = ne1.div_ceil(threads);
        pool.install(|| {
            y.par_chunks_mut(chunk_rows).enumerate().for_each(|(chunk, rows)| {
                unsafe { fill_q8_0_avx2(payload, row_bytes, chunk * chunk_rows, rows, x) }
            });
        });
        return Ok(y);
    }
    for (row, slot) in y.iter_mut().enumerate() {
        let start = row * row_bytes;
        let mut acc = 0.0f32;
        let blocks = x.len() / 32;
        for b in 0..blocks {
            let o = start + b * 34;
            let d = fp16_to_f32(u16::from_le_bytes([payload[o], payload[o + 1]]));
            for j in 0..32 {
                acc += (payload[o + 2 + j] as i8 as f32) * d * x[b * 32 + j];
            }
        }
        *slot = acc;
    }
    Ok(y)
}

fn f16_matvec_layout(row_bytes: usize, ne0: usize) -> bool {
    ne0 > 0 && row_bytes == ne0 * 2
}

fn matvec_f16_fast(payload: &[u8], row_bytes: usize, x: &[f32], ne1: usize) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; ne1];
    if ne1 == 0 {
        return Ok(y);
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        let pool = quant_matvec_thread_pool();
        let threads = pool.current_num_threads().min(ne1).max(1);
        if threads <= 1 {
            unsafe { fill_f16_avx2(payload, row_bytes, 0, &mut y, x) }
            return Ok(y);
        }
        let chunk_rows = ne1.div_ceil(threads);
        pool.install(|| {
            y.par_chunks_mut(chunk_rows).enumerate().for_each(|(chunk, rows)| {
                unsafe { fill_f16_avx2(payload, row_bytes, chunk * chunk_rows, rows, x) }
            });
        });
        return Ok(y);
    }
    for (row, slot) in y.iter_mut().enumerate() {
        let start = row * row_bytes;
        *slot = dot_row_f16_scalar(&payload[start..start + row_bytes], x);
    }
    Ok(y)
}

fn matvec_rows_scalar(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    if ty == 1 && f16_matvec_layout(row_bytes, x.len()) {
        return matvec_f16_fast(payload, row_bytes, x, ne1);
    }
    if ty == 35 && tq2_q8k_enabled() && tq2_q8k_layout(row_bytes, x.len()) {
        return matvec_tq2_q8k(payload, row_bytes, x, ne1);
    }
    if ty == 36 && i2s_q8k_layout(row_bytes, x.len()) {
        return if i2s_f32_preferred() {
            matvec_i2s_f32(payload, row_bytes, x, ne1)
        } else {
            matvec_i2s_q8k(payload, row_bytes, x, ne1)
        };
    }
    let mut y = vec![0.0f32; ne1];
    for o in 0..ne1 {
        let row_start = o * row_bytes;
        let row = &payload[row_start..row_start + row_bytes];
        y[o] = dot_row(ty, row, x)?;
    }
    apply_i2s_scale(ty, payload, row_bytes, ne1, &mut y)?;
    Ok(y)
}

fn matvec_rows_parallel_original(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    if ty == 1 && f16_matvec_layout(row_bytes, x.len()) {
        return matvec_f16_fast(payload, row_bytes, x, ne1);
    }
    if ty == 35 && tq2_q8k_enabled() && tq2_q8k_layout(row_bytes, x.len()) {
        return matvec_tq2_q8k(payload, row_bytes, x, ne1);
    }
    if ty == 36 && i2s_q8k_layout(row_bytes, x.len()) {
        return if i2s_f32_preferred() {
            matvec_i2s_f32(payload, row_bytes, x, ne1)
        } else {
            matvec_i2s_q8k(payload, row_bytes, x, ne1)
        };
    }
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 || ne1 < 2 {
        return matvec_rows_scalar(ty, payload, row_bytes, x, ne1);
    }
    let chunk_rows = ne1.div_ceil(threads);
    let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
    let chunk_results: Vec<std::result::Result<(usize, Vec<f32>), BitNetError>> =
        pool.install(|| {
            chunk_starts
                .par_iter()
                .copied()
                .map(|chunk_start| {
                    let chunk_end = (chunk_start + chunk_rows).min(ne1);
                    let mut y = vec![0.0f32; chunk_end - chunk_start];
                    for (local, o) in (chunk_start..chunk_end).enumerate() {
                        let row_start = o * row_bytes;
                        let row = &payload[row_start..row_start + row_bytes];
                        y[local] = dot_row(ty, row, x)?;
                    }
                    Ok((chunk_start, y))
                })
                .collect()
        });
    let mut rows = Vec::with_capacity(chunk_results.len());
    for r in chunk_results {
        rows.push(r?);
    }
    rows.sort_by_key(|(start, _)| *start);
    let mut y = vec![0.0f32; ne1];
    for (start, chunk) in rows {
        y[start..start + chunk.len()].copy_from_slice(&chunk);
    }
    apply_i2s_scale(ty, payload, row_bytes, ne1, &mut y)?;
    Ok(y)
}

type CudaQuantMatvecFn =
    unsafe extern "C" fn(*const c_void, usize, *const f32, usize, usize, *mut f32) -> i32;

/// GGML types with optional native CUDA quant kernels (`librbitnet_cuda_quant`).
pub fn ggml_type_supports_cuda_quant(ty: u32) -> bool {
    matches!(ty, 0 | 2 | 6 | 8 | 12 | 13 | 14 | 35 | 39)
}

fn cuda_quant_symbol(ty: u32, device_resident: bool) -> Option<&'static [u8]> {
    match (ty, device_resident) {
        (2, false) => Some(b"rbitnet_cuda_q4_0_matvec\0"),
        (8, false) => Some(b"rbitnet_cuda_q8_0_matvec\0"),
        (12, false) => Some(b"rbitnet_cuda_q4_k_matvec\0"),
        (14, false) => Some(b"rbitnet_cuda_q6_k_matvec\0"),
        (2, true) => Some(b"rbitnet_cuda_q4_0_matvec_device\0"),
        (8, true) => Some(b"rbitnet_cuda_q8_0_matvec_device\0"),
        (12, true) => Some(b"rbitnet_cuda_q4_k_matvec_device\0"),
        (14, true) => Some(b"rbitnet_cuda_q6_k_matvec_device\0"),
        (0, false) => Some(b"rbitnet_cuda_f32_matvec\0"),
        (0, true) => Some(b"rbitnet_cuda_f32_matvec_device\0"),
        (6, false) => Some(b"rbitnet_cuda_q5_0_matvec\0"),
        (6, true) => Some(b"rbitnet_cuda_q5_0_matvec_device\0"),
        (13, false) => Some(b"rbitnet_cuda_q5_k_matvec\0"),
        (13, true) => Some(b"rbitnet_cuda_q5_k_matvec_device\0"),
        (39, false) => Some(b"rbitnet_cuda_mxfp4_matvec\0"),
        (39, true) => Some(b"rbitnet_cuda_mxfp4_matvec_device\0"),
        (35, false) => Some(b"rbitnet_cuda_tq2_0_matvec\0"),
        (35, true) => Some(b"rbitnet_cuda_tq2_0_matvec_device\0"),
        _ => None,
    }
}

fn matvec_rows_cuda_quant_optional(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Option<Result<Vec<f32>>> {
    let symbol = cuda_quant_symbol(ty, false)?;
    let lib = load_cuda_quant_library()?;
    let kernel = unsafe { lib.get::<CudaQuantMatvecFn>(symbol).ok()? };
    let mut y = vec![0.0f32; ne1];
    let status = unsafe {
        kernel(
            payload.as_ptr().cast::<c_void>(),
            row_bytes,
            x.as_ptr(),
            x.len(),
            ne1,
            y.as_mut_ptr(),
        )
    };
    if status == 0 {
        Some(Ok(y))
    } else {
        Some(Err(BitNetError::Inference(format!(
            "CUDA quant matvec kernel failed for ggml_type={ty} status={status}"
        ))))
    }
}

/// Optional device-resident quantized matvec (`d_w` already on GPU).
///
/// Looks up `rbitnet_cuda_*_matvec_device` in `librbitnet_cuda_quant`. Returns `None` when the
/// library or symbol is absent so callers can fall back to CPU without failing CI.
pub fn matvec_device_quant_optional(
    ty: u32,
    d_w: *mut c_void,
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Option<Result<Vec<f32>>> {
    if d_w.is_null() || !ggml_type_supports_cuda_quant(ty) {
        return None;
    }
    let symbol = cuda_quant_symbol(ty, true)?;
    let lib = load_cuda_quant_library()?;
    let kernel = unsafe { lib.get::<CudaQuantMatvecFn>(symbol).ok()? };
    let mut y = vec![0.0f32; ne1];
    let status = unsafe {
        kernel(
            d_w as *const c_void,
            row_bytes,
            x.as_ptr(),
            x.len(),
            ne1,
            y.as_mut_ptr(),
        )
    };
    if status == 0 {
        Some(Ok(y))
    } else {
        Some(Err(BitNetError::Inference(format!(
            "CUDA device-resident quant matvec failed for ggml_type={ty} status={status}"
        ))))
    }
}

struct LoadedCudaLibrary {
    library: Library,
    identity: Option<[u8; 32]>,
}
static CUDA_QUANT_LIBRARY: OnceLock<Option<LoadedCudaLibrary>> = OnceLock::new();

pub(crate) fn cuda_quant_library_identity() -> Option<[u8; 32]> {
    let _ = load_cuda_quant_library();
    CUDA_QUANT_LIBRARY.get()?.as_ref()?.identity
}
pub(crate) fn load_cuda_quant_library() -> Option<&'static Library> {
    CUDA_QUANT_LIBRARY
        .get_or_init(|| {
            let mut candidates: Vec<String> = Vec::new();
            if let Ok(explicit) = std::env::var("RBITNET_CUDA_QUANT_LIB") {
                let trimmed = explicit.trim();
                if !trimmed.is_empty() {
                    candidates.push(trimmed.to_string());
                }
            }
            for path in [
                "rbitnet_cuda_quant64.dll",
                "rbitnet_cuda_quant.dll",
                "librbitnet_cuda_quant.so",
                "librbitnet_cuda_quant.dylib",
                "native/cuda_quant/build/rbitnet_cuda_quant64.dll",
                "native/cuda_quant/build/rbitnet_cuda_quant.dll",
                "native/cuda_quant/build/librbitnet_cuda_quant.so",
                "native/cuda_quant/build/librbitnet_cuda_quant.dylib",
            ] {
                candidates.push(path.to_string());
            }
            for path in candidates {
                use sha2::Digest;
                let identity = std::fs::read(&path)
                    .ok()
                    .map(|bytes| sha2::Sha256::digest(&bytes).into());
                if let Ok(lib) = unsafe { Library::new(path.as_str()) } {
                    tracing::info!(path = %path, "loaded librbitnet_cuda_quant");
                    return Some(LoadedCudaLibrary {
                        library: lib,
                        identity,
                    });
                }
            }
            None
        })
        .as_ref()
        .map(|loaded| &loaded.library)
}

pub(crate) fn matvec_device_quant_batch_optional(
    ty: u32,
    w: *mut c_void,
    row_bytes: usize,
    x: &[f32],
    cols: usize,
    rows: usize,
    batches: usize,
) -> Option<Result<Vec<f32>>> {
    type Batch = unsafe extern "C" fn(
        u32,
        *const c_void,
        usize,
        *const f32,
        usize,
        usize,
        usize,
        *mut f32,
    ) -> i32;
    let lib = load_cuda_quant_library()?;
    let kernel = unsafe {
        lib.get::<Batch>(b"rbitnet_cuda_quant_matvec_batch_device\0")
            .ok()?
    };
    let mut y = vec![0.0; rows.checked_mul(batches)?];
    let status = unsafe {
        kernel(
            ty,
            w,
            row_bytes,
            x.as_ptr(),
            cols,
            rows,
            batches,
            y.as_mut_ptr(),
        )
    };
    Some(if status == 0 {
        Ok(y)
    } else {
        Err(BitNetError::Inference(format!(
            "CUDA batch matvec failed: {status}"
        )))
    })
}

/// True when optional `librbitnet_cuda_quant` was found (device or host symbols may still vary).
pub fn cuda_quant_library_available() -> bool {
    load_cuda_quant_library().is_some()
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
        return Err(BitNetError::InvalidGguf(
            "embedding: expected 2D tensor".into(),
        ));
    }
    if t.dimensions[0] as usize != n_embd || t.dimensions[1] as usize != n_vocab {
        return Err(BitNetError::Inference(
            "embedding: tensor dims mismatch".into(),
        ));
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

    #[test]
    fn tq1_0_row_dot_matches_full_dequant() {
        let mut payload = vec![0u8; 54];
        payload[0..2].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        for i in 2..payload.len() {
            payload[i] = (i as u8).wrapping_mul(17).wrapping_add(5);
        }
        let dims = vec![256u64, 1u64];
        let dense = tensor_to_f32(&payload, 34, &dims).unwrap();
        let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.015 - 1.7).collect();
        let ref_dot: f32 = dense.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let q = dot_row(34, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 1e-5, "q={q} ref={ref_dot}");
    }

    #[test]
    fn tq2_q8k_avx_matches_generic_integer_dot() {
        let ne0 = 512;
        let nb = ne0 / 256;
        let mut row = vec![0u8; nb * 66];
        for (i, byte) in row.iter_mut().enumerate() {
            *byte = ((i * 19 + 7) % 251) as u8;
        }
        for block in 0..nb {
            let scale = f16::from_f32(0.02 * (block as f32 + 1.0)).to_bits().to_le_bytes();
            row[block * 66 + 64..block * 66 + 66].copy_from_slice(&scale);
        }
        let x: Vec<f32> = (0..ne0)
            .map(|i| ((i * 13) % 71) as f32 / 30.0 - 1.1)
            .collect();
        let act = quantize_q8_k(&x);
        let scalar = dot_tq2_q8k_scalar(&row, &act);
        let simd = dot_tq2_q8k(&row, &act);
        assert_eq!(
            simd.to_bits(),
            scalar.to_bits(),
            "simd={simd} scalar={scalar}"
        );
        let tied = quantize_q8_k(&[1.0, -1.0].repeat(128));
        assert!(tied.d[0] < 0.0, "the first peak keeps its sign, d={}", tied.d[0]);
        let ne0 = 2560;
        let ne1 = 7;
        let row_bytes = (ne0 / 256) * 66;
        let mut payload = vec![0u8; ne1 * row_bytes];
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte = ((i * 17 + 3) % 251) as u8;
        }
        for row in 0..ne1 {
            for block in 0..(ne0 / 256) {
                let at = row * row_bytes + block * 66 + 64;
                payload[at..at + 2]
                    .copy_from_slice(&f16::from_f32(0.03 * (row as f32 + 1.0)).to_bits().to_le_bytes());
            }
        }
        let wide: Vec<f32> = (0..ne0).map(|i| (i as f32) * 0.001 - 0.4).collect();
        let got = matvec_tq2_q8k(&payload, row_bytes, &wide, ne1).unwrap();
        let act = quantize_q8_k(&wide);
        for row in 0..ne1 {
            let slice = &payload[row * row_bytes..(row + 1) * row_bytes];
            let simd = dot_tq2_q8k(slice, &act);
            assert_eq!(got[row].to_bits(), simd.to_bits(), "row {row}");
            let scalar = dot_tq2_q8k_scalar(slice, &act);
            let scale = scalar.abs().max(1.0);
            assert!(
                (got[row] - scalar).abs() / scale < 1e-5,
                "row {row} got={} scalar={scalar}",
                got[row]
            );
        }
    }

    #[test]
    fn tq2_0_row_dot_matches_full_dequant() {
        let mut payload = vec![0u8; 66];
        for i in 0..64 {
            payload[i] = (i as u8).wrapping_mul(19).wrapping_add(7);
        }
        payload[64..66].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
        let dims = vec![256u64, 1u64];
        let dense = tensor_to_f32(&payload, 35, &dims).unwrap();
        let q0 = ((payload[0] & 3) as i8 - 1) as f32 * 0.125;
        assert!((dense[0] - q0).abs() < 1e-6, "scale must follow the codes");
        let x: Vec<f32> = (0..256).map(|i| i as f32 * -0.02 + 2.0).collect();
        let ref_dot: f32 = dense.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let q = dot_row(35, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 1e-4, "q={q} ref={ref_dot}");
    }

    #[test]
    fn tq2_batch_matvec_matches_one_token_rows() {
        let ne0 = 256;
        let ne1 = 4;
        let n_tokens = 3;
        let row_bytes = 66;
        let mut payload = vec![0u8; ne1 * row_bytes];
        for row in 0..ne1 {
            let base = row * row_bytes;
            for i in 0..64 {
                payload[base + i] = (row as u8).wrapping_mul(13).wrapping_add(i as u8);
            }
            payload[base + 64..base + 66]
                .copy_from_slice(&f16::from_f32(0.05 * (row as f32 + 1.0)).to_bits().to_le_bytes());
        }
        let xs: Vec<f32> = (0..n_tokens * ne0)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let batch = matvec_rows_batch(35, &payload, row_bytes, &xs, n_tokens, ne0, ne1).unwrap();
        for t in 0..n_tokens {
            let serial = matvec_rows_scalar(35, &payload, row_bytes, &xs[t * ne0..(t + 1) * ne0], ne1)
                .unwrap();
            for row in 0..ne1 {
                let got = batch[t * ne1 + row];
                let scale = serial[row].abs().max(1.0);
                assert!(
                    (got - serial[row]).abs() / scale < 1e-4,
                    "t={t} row={row} got={got} serial={}",
                    serial[row]
                );
            }
        }
    }

    #[test]
    fn q4_k_batch_matvec_matches_one_token_rows() {
        let ne0 = 256;
        let ne1 = 3;
        let n_tokens = 4;
        let row_bytes = 144;
        let mut payload = vec![0u8; ne1 * row_bytes];
        for row in 0..ne1 {
            let base = row * row_bytes;
            for i in 0..row_bytes {
                payload[base + i] = (row as u8).wrapping_mul(9).wrapping_add(i as u8);
            }
            payload[base..base + 2]
                .copy_from_slice(&f16::from_f32(0.02 * (row as f32 + 1.0)).to_bits().to_le_bytes());
            payload[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
        }
        let xs: Vec<f32> = (0..n_tokens * ne0)
            .map(|i| (i as f32) * 0.02 - 0.4)
            .collect();
        let batch = matvec_rows_batch(12, &payload, row_bytes, &xs, n_tokens, ne0, ne1).unwrap();
        for t in 0..n_tokens {
            let serial =
                matvec_rows_scalar(12, &payload, row_bytes, &xs[t * ne0..(t + 1) * ne0], ne1)
                    .unwrap();
            for row in 0..ne1 {
                let got = batch[t * ne1 + row];
                let scale = serial[row].abs().max(1.0);
                assert!(
                    (got - serial[row]).abs() / scale < 1e-5,
                    "t={t} row={row} got={got} serial={}",
                    serial[row]
                );
            }
        }
    }

    #[test]
    fn f16_grouped_matvec_matches_row_dot() {
        let n = 256;
        let rows = 5;
        let mut payload = vec![0u8; rows * n * 2];
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.7).collect();
        for row in 0..rows {
            for i in 0..n {
                let bits = f16::from_f32((i as f32) * 0.03 - 1.2 + row as f32).to_bits();
                payload[(row * n + i) * 2..(row * n + i) * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let got = matvec_f16_fast(&payload, n * 2, &x, rows).unwrap();
        for row in 0..rows {
            let slice = &payload[row * n * 2..(row + 1) * n * 2];
            let scalar = dot_row_f16_scalar(slice, &x);
            let simd = dot_row(1, slice, &x).unwrap();
            assert_eq!(got[row].to_bits(), simd.to_bits(), "row {row}");
            let scale = scalar.abs().max(1.0);
            assert!((got[row] - scalar).abs() / scale < 1e-5);
        }
    }

    #[test]
    fn q8_output_head_matches_scalar_and_reports_speed() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            return;
        }
        let ne0 = 2560usize;
        let ne1 = 32768usize;
        let mut f16_payload = vec![0u8; ne0 * ne1 * 2];
        for row in 0..ne1 {
            for i in 0..ne0 {
                let bits = f16::from_f32(((row * 17 + i * 3) % 97) as f32 * 0.01 - 0.4).to_bits();
                let o = (row * ne0 + i) * 2;
                f16_payload[o..o + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let q8_row = (ne0 / 32) * 34;
        let mut q8 = vec![0u8; q8_row * ne1];
        for row in 0..ne1 {
            for b in 0..(ne0 / 32) {
                let mut vals = [0.0f32; 32];
                let mut amax = 0.0f32;
                for j in 0..32 {
                    let o = (row * ne0 + b * 32 + j) * 2;
                    vals[j] = fp16_to_f32(u16::from_le_bytes([
                        f16_payload[o],
                        f16_payload[o + 1],
                    ]));
                    amax = amax.max(vals[j].abs());
                }
                let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                let dest = row * q8_row + b * 34;
                q8[dest..dest + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
                for j in 0..32 {
                    q8[dest + 2 + j] =
                        (vals[j] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        }
        let x_owned: Vec<f32> = (0..ne0).map(|i| (i as f32) * 0.002 - 0.3).collect();
        let x: &[f32] = &x_owned;
        let mut sample = [0.0f32; 4];
        unsafe { fill_q8_0_avx2(&q8, q8_row, 0, &mut sample, x) };
        for row in 0..4 {
            let mut acc = 0.0f32;
            for b in 0..(ne0 / 32) {
                let dest = row * q8_row + b * 34;
                let d = fp16_to_f32(u16::from_le_bytes([q8[dest], q8[dest + 1]]));
                for j in 0..32 {
                    acc += (q8[dest + 2 + j] as i8 as f32) * d * x[b * 32 + j];
                }
            }
            let scale = acc.abs().max(1.0);
            assert!((sample[row] - acc).abs() / scale < 1e-4);
        }
        let pool = quant_matvec_thread_pool();
        let threads = pool.current_num_threads().max(1);
        let rounds = 3u32;
        let mut f16_us = 0u128;
        let mut q8_us = 0u128;
        let mut sink = 0u32;
        for _ in 0..rounds {
            let mut y = vec![0.0f32; ne1];
            let base = ne1 / threads;
            let rem = ne1 % threads;
            let t0 = Instant::now();
            pool.install(|| {
                std::thread::scope(|scope| {
                    let mut rest = y.as_mut_slice();
                    let mut offset = 0usize;
                    for thread in 0..threads {
                        let count = base + usize::from(thread < rem);
                        let (chunk, tail) = rest.split_at_mut(count);
                        rest = tail;
                        let first = offset;
                        offset += count;
                        let payload = f16_payload.as_slice();
                        scope.spawn(move || unsafe {
                            fill_f16_avx2(payload, ne0 * 2, first, chunk, x);
                        });
                    }
                });
            });
            f16_us += t0.elapsed().as_micros();
            sink ^= y[0].to_bits();
            let t1 = Instant::now();
            pool.install(|| {
                std::thread::scope(|scope| {
                    let mut rest = y.as_mut_slice();
                    let mut offset = 0usize;
                    for thread in 0..threads {
                        let count = base + usize::from(thread < rem);
                        let (chunk, tail) = rest.split_at_mut(count);
                        rest = tail;
                        let first = offset;
                        offset += count;
                        let payload = q8.as_slice();
                        scope.spawn(move || unsafe {
                            fill_q8_0_avx2(payload, q8_row, first, chunk, x);
                        });
                    }
                });
            });
            q8_us += t1.elapsed().as_micros();
            sink ^= y[0].to_bits();
        }
        eprintln!(
            "q8_head ne1={ne1} rounds={rounds} f16_us={f16_us} q8_us={q8_us} sink={sink}"
        );
    }

    #[test]
    fn f16_row_dot_matches_scalar_conversion() {
        let n = 256;
        let mut payload = vec![0u8; n * 2];
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.7).collect();
        for i in 0..n {
            let bits = f16::from_f32((i as f32) * 0.03 - 1.2).to_bits().to_le_bytes();
            payload[i * 2..i * 2 + 2].copy_from_slice(&bits);
        }
        let scalar = dot_row_f16_scalar(&payload, &x);
        let got = dot_row(1, &payload, &x).unwrap();
        let scale = scalar.abs().max(1.0);
        assert!(
            (got - scalar).abs() / scale < 1e-5,
            "got={got} scalar={scalar}"
        );
    }

    #[test]
    fn i2s_row_dot_matches_bitnet_cpp_quantize_layout() {
        let n = 256;
        let weights: Vec<i8> = (0..n)
            .map(|i| match i % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let scale = 2.1631613f32;
        let mut payload = vec![0u8; n / 4 + 32];
        for block in 0..(n / 128) {
            for j in 0..128 {
                let q = match weights[block * 128 + j] {
                    0 => 1u8,
                    1 => 2,
                    _ => 0,
                };
                let group = j / 32;
                let gp = j % 32;
                payload[block * 32 + gp] |= q << (6 - 2 * group);
            }
        }
        payload[n / 4..n / 4 + 4].copy_from_slice(&scale.to_le_bytes());
        let dense = tensor_to_f32(&payload, 36, &[n as u64]).unwrap();
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let row_bytes = crate::ggml::ggml_row_size(36, n as u64).unwrap();
        let y = matvec_rows_scalar(36, &payload, row_bytes, &x, 1).unwrap();
        let reference: f32 = dense.iter().zip(x.iter()).map(|(w, a)| w * a).sum();
        assert!((y[0] - reference).abs() < 1e-4, "y={} ref={reference}", y[0]);
        assert!((dense[1] - scale).abs() < 1e-5, "first +1 weight {}", dense[1]);
        assert!((dense[2] + scale).abs() < 1e-5, "first -1 weight {}", dense[2]);
    }

    #[test]
    fn i2s_q8k_matches_scalar_integer_and_tracks_f32() {
        let n = 256;
        let mut codes = vec![0u8; n / 4];
        let weights: Vec<i8> = (0..n)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => 1,
                _ => 0,
            })
            .collect();
        for block in 0..(n / 128) {
            for j in 0..128 {
                let q = match weights[block * 128 + j] {
                    0 => 1u8,
                    1 => 2,
                    _ => 0,
                };
                let group = j / 32;
                let gp = j % 32;
                codes[block * 32 + gp] |= q << (6 - 2 * group);
            }
        }
        let x: Vec<f32> = (0..n).map(|i| ((i * 17) % 11) as f32 * 0.07 - 0.3).collect();
        let exact = dot_row_i2_s(&codes, &x);
        let avx = dot_i2s_f32(&codes, &x);
        let denom = exact.abs().max(1.0);
        assert!(
            (avx - exact).abs() / denom < 1e-4,
            "avx={avx} exact={exact}"
        );
        let act = quantize_q8_k(&x);
        let scalar = dot_i2s_q8k_scalar(&codes, &act);
        let fast = dot_i2s_q8k(&codes, &act);
        assert!((fast - scalar).abs() < 1e-3, "fast={fast} scalar={scalar}");
        let mut payload = codes.clone();
        let scale = 1.25f32;
        payload.extend_from_slice(&scale.to_le_bytes());
        payload.resize(payload.len() + 28, 0);
        let y = matvec_i2s_q8k(&payload, n / 4, &x, 1).unwrap();
        let dense = tensor_to_f32(&payload, 36, &[n as u64]).unwrap();
        let reference: f32 = dense.iter().zip(x.iter()).map(|(w, a)| w * a).sum();
        let denom = reference.abs().max(1.0);
        assert!(
            (y[0] - reference).abs() / denom < 0.02,
            "y={} ref={reference}",
            y[0]
        );
    }

    #[test]
    fn microsoft_i2s_gguf_uses_per_tensor_f32_scale() {
        let path = std::path::Path::new(
            r"D:\rbitnet-bench\models\bitnet-b158-2b\ggml-model-i2_s.gguf",
        );
        if !path.is_file() {
            return;
        }
        let archive = GgufArchive::mmap_path(path).unwrap();
        assert_eq!(
            archive.architecture().map(|s| s.to_ascii_lowercase()),
            Some("bitnet-b1.58".into())
        );
        let tensor = archive
            .tensor_by_name("blk.0.ffn_down.weight")
            .expect("ffn_down");
        assert_eq!(tensor.ggml_type, 36);
        let payload = archive.tensor_payload(tensor).unwrap();
        let nels: u64 = tensor.dimensions.iter().product();
        let scale = f32::from_le_bytes(payload[(nels / 4) as usize..][..4].try_into().unwrap());
        assert!(
            (scale - 2.1631613).abs() < 1e-4,
            "blk.0.ffn_down scale {scale}"
        );
        assert_eq!(
            payload.len(),
            crate::ggml::ggml_nbytes(&tensor.dimensions, 36).unwrap()
        );
    }
}

fn matvec_rows_parallel(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    static DIRECT: OnceLock<bool> = OnceLock::new();
    if *DIRECT.get_or_init(|| std::env::var("RBITNET_CPU_DIRECT_ROWS").as_deref() == Ok("1")) {
        matvec_rows_direct(ty, payload, row_bytes, x, ne1)
    } else {
        matvec_rows_parallel_original(ty, payload, row_bytes, x, ne1)
    }
}
/// Write disjoint output bands directly. Each row retains its original dot
/// kernel and accumulation order; no intermediate result vectors are merged.
static DIRECT_ROW_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[doc(hidden)]
pub fn direct_row_calls() -> u64 {
    DIRECT_ROW_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

fn matvec_rows_direct(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    DIRECT_ROW_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pool = quant_matvec_thread_pool();
    let threads = pool.current_num_threads().min(ne1.max(1));
    if threads <= 1 || ne1 < 2 {
        return matvec_rows_scalar(ty, payload, row_bytes, x, ne1);
    }
    let chunk_rows = ne1.div_ceil(threads);
    let mut output = vec![0.0f32; ne1];
    pool.install(|| {
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk, rows)| -> Result<()> {
                let first = chunk * chunk_rows;
                for (local, target) in rows.iter_mut().enumerate() {
                    let start = (first + local) * row_bytes;
                    *target = dot_row(ty, &payload[start..start + row_bytes], x)?;
                }
                Ok(())
            })
    })?;
    apply_i2s_scale(ty, payload, row_bytes, ne1, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod direct_row_tests {
    use super::*;

    #[test]
    fn cpu_direct_rows_match_original_bits_across_quant_formats_and_tail_bands() {
        let x = (0..512)
            .map(|i| ((i * 13 % 71) as f32 - 35.) / 256.)
            .collect::<Vec<_>>();
        let mut cases = 0;
        for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
            let (elements, block_bytes) = types::type_layout(ty).unwrap();
            let row_bytes = (x.len() / elements) * block_bytes;
            for rows in [0, 1, 7, 17, 33, 65] {
                let mut payload = vec![0u8; row_bytes * rows];
                for (i, byte) in payload.iter_mut().enumerate() {
                    *byte = ((i * 37 + 11) % 256) as u8;
                }
                if ty == 0 {
                    for (i, bytes) in payload.chunks_exact_mut(4).enumerate() {
                        bytes
                            .copy_from_slice(&(((i * 11 % 113) as f32 - 56.) / 256.).to_le_bytes());
                    }
                } else {
                    for block in payload.chunks_exact_mut(block_bytes) {
                        let scale = half::f16::from_f32(0.0625).to_le_bytes();
                        match ty {
                            2 | 6 | 8 => block[..2].copy_from_slice(&scale),
                            12 | 13 => {
                                block[..2].copy_from_slice(&scale);
                                block[2..4]
                                    .copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                            }
                            14 => block[208..210].copy_from_slice(&scale),
                            39 => block[0] = 127,
                            _ => unreachable!(),
                        }
                    }
                }
                let original =
                    matvec_rows_parallel_original(ty, &payload, row_bytes, &x, rows).unwrap();
                let actual = matvec_rows_direct(ty, &payload, row_bytes, &x, rows).unwrap();
                assert!(
                    original.iter().all(|x| x.is_finite()),
                    "finite fixture {ty}/{rows}"
                );
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    original.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "type={ty}, rows={rows}"
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 48);
        println!("CPU_DIRECT_ROWS_BITS_DONE formats=8 tail_shapes=6 cases=48");
    }

    #[test]
    fn optional_actual_cpu_direct_rows_match_original_gguf_matrices() {
        if std::env::var("RBITNET_CPU_DIRECT_ROWS_REAL_TEST").as_deref() != Ok("1") {
            return;
        }
        let archive = GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap();
        let mut cases = 0;
        for tensor in archive
            .tensors
            .iter()
            .filter(|t| t.dimensions.len() >= 2 && ggml_type_supported_mmap_matvec(t.ggml_type))
            .take(12)
        {
            let columns = tensor.dimensions[0] as usize;
            let available = tensor.dimensions[1] as usize;
            let row_bytes = crate::ggml::ggml_row_size(tensor.ggml_type, columns as u64).unwrap();
            let payload = archive.tensor_payload(tensor).unwrap();
            let x = (0..columns)
                .map(|i| ((i * 13 % 71) as f32 - 35.) / 256.)
                .collect::<Vec<_>>();
            for rows in [1, 7, 17, 33, 65].map(|rows| rows.min(available)) {
                let original = matvec_rows_parallel_original(
                    tensor.ggml_type,
                    &payload[..rows * row_bytes],
                    row_bytes,
                    &x,
                    rows,
                )
                .unwrap();
                let actual = matvec_rows_direct(
                    tensor.ggml_type,
                    &payload[..rows * row_bytes],
                    row_bytes,
                    &x,
                    rows,
                )
                .unwrap();
                assert!(
                    original.iter().all(|x| x.is_finite()),
                    "finite actual {}",
                    tensor.name
                );
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    original.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "{} rows={rows}",
                    tensor.name
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 60);
        println!("CPU_DIRECT_ROWS_ACTUAL_DONE matrices=12 cases=60 exact_bits=true");
    }
}
