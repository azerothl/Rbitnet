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
                if ne1 >= effective_parallel_min_rows(self.parallel_min_rows, ne0, ne1) => {
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
                if ne1 >= effective_parallel_min_rows(self.parallel_min_rows, ne0, ne1) => {
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
    matches!(ty, 0 | 1 | 2 | 8 | 12 | 14 | 30 | 34 | 35)
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
        34 => dot_row_tq1_0(row_payload, x),
        35 => dot_row_tq2_0(row_payload, x),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

fn dot_row_f32(row: &[u8], x: &[f32]) -> Result<f32> {
    if row.len() != x.len() * 4 {
        return Err(BitNetError::InvalidGguf("f32 row size".into()));
    }
    let mut s = 0.0f32;
    let mut i = 0;
    while i + 4 <= x.len() {
        let w0 = f32::from_bits(u32::from_le_bytes(
            row[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
        let w1 = f32::from_bits(u32::from_le_bytes(
            row[(i + 1) * 4..(i + 1) * 4 + 4].try_into().unwrap(),
        ));
        let w2 = f32::from_bits(u32::from_le_bytes(
            row[(i + 2) * 4..(i + 2) * 4 + 4].try_into().unwrap(),
        ));
        let w3 = f32::from_bits(u32::from_le_bytes(
            row[(i + 3) * 4..(i + 3) * 4 + 4].try_into().unwrap(),
        ));
        s = w0.mul_add(x[i], s);
        s = w1.mul_add(x[i + 1], s);
        s = w2.mul_add(x[i + 2], s);
        s = w3.mul_add(x[i + 3], s);
        i += 4;
    }
    while i < x.len() {
        let b = u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
        s = f32::from_bits(b).mul_add(x[i], s);
        i += 1;
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
        s = fp16_to_f32(h).mul_add(x[i], s);
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
        for i in 0..QK4_0 {
            acc = buf[i].mul_add(xb[i], acc);
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
            acc = buf[i].mul_add(xb[i], acc);
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
            acc = buf[i].mul_add(xb[i], acc);
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
            acc = buf[i].mul_add(xb[i], acc);
        }
    }
    Ok(acc)
}

fn dot_row_tq1_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq1_0 ne0 % 256".into()));
    }
    let mut buf = vec![0.0f32; x.len()];
    decode_tq1_0_to_f32(row, &mut buf)?;
    Ok(buf.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum())
}

fn dot_row_tq2_0(row: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq2_0 ne0 % 256".into()));
    }
    let mut buf = vec![0.0f32; x.len()];
    decode_tq2_0_to_f32(row, &mut buf)?;
    Ok(buf.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum())
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
        let d = fp16_to_f32(u16::from_le_bytes(row[o..o + 2].try_into().unwrap()));
        let qs = &row[o + 2..o + 2 + QS_LEN];
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
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("rbitnet-quant-{i}"))
            .build()
            .expect("quant matvec thread pool")
    })
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

fn matvec_rows_scalar(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    let mut y = vec![0.0f32; ne1];
    for o in 0..ne1 {
        let row_start = o * row_bytes;
        let row = &payload[row_start..row_start + row_bytes];
        y[o] = dot_row(ty, row, x)?;
    }
    Ok(y)
}

fn matvec_rows_parallel(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Result<Vec<f32>> {
    let threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
        .min(ne1.max(1));
    let pool = quant_matvec_thread_pool();
    if threads <= 1 || ne1 < 2 {
        return matvec_rows_scalar(ty, payload, row_bytes, x, ne1);
    }
    let chunk_rows = (ne1 + threads - 1) / threads;
    let chunk_starts: Vec<usize> = (0..ne1).step_by(chunk_rows).collect();
    let chunk_results: Vec<std::result::Result<(usize, Vec<f32>), BitNetError>> = pool.install(|| {
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
    Ok(y)
}

type CudaQuantMatvecFn =
    unsafe extern "C" fn(*const c_void, usize, *const f32, usize, usize, *mut f32) -> i32;

fn matvec_rows_cuda_quant_optional(
    ty: u32,
    payload: &[u8],
    row_bytes: usize,
    x: &[f32],
    ne1: usize,
) -> Option<Result<Vec<f32>>> {
    let symbol = match ty {
        2 => b"rbitnet_cuda_q4_0_matvec\0".as_slice(),
        8 => b"rbitnet_cuda_q8_0_matvec\0".as_slice(),
        12 => b"rbitnet_cuda_q4_k_matvec\0".as_slice(),
        14 => b"rbitnet_cuda_q6_k_matvec\0".as_slice(),
        _ => return None,
    };
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

fn load_cuda_quant_library() -> Option<Library> {
    for path in [
        "rbitnet_cuda_quant64.dll",
        "rbitnet_cuda_quant.dll",
        "librbitnet_cuda_quant.so",
        "librbitnet_cuda_quant.dylib",
    ] {
        if let Ok(lib) = unsafe { Library::new(path) } {
            return Some(lib);
        }
    }
    None
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
    fn tq2_0_row_dot_matches_full_dequant() {
        let mut payload = vec![0u8; 66];
        payload[0..2].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
        for i in 2..payload.len() {
            payload[i] = (i as u8).wrapping_mul(19).wrapping_add(7);
        }
        let dims = vec![256u64, 1u64];
        let dense = tensor_to_f32(&payload, 35, &dims).unwrap();
        let x: Vec<f32> = (0..256).map(|i| i as f32 * -0.02 + 2.0).collect();
        let ref_dot: f32 = dense.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let q = dot_row(35, &payload, &x).unwrap();
        assert!((q - ref_dot).abs() < 1e-5, "q={q} ref={ref_dot}");
    }
}
