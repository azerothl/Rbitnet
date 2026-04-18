//! Dequantize GGML tensor payloads to `Vec<f32>` (layout matches `ggml-quants.c` / `ggml.c`).

use half::{bf16, f16};

use crate::error::{BitNetError, Result};
use crate::ggml::types;

const QK_K: usize = 256;
const QK4_0: usize = 32;
const QK8_1: usize = 32;

fn fp16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// `nelements` = product of tensor dimensions.
pub fn tensor_to_f32(data: &[u8], ty: u32, dims: &[u64]) -> Result<Vec<f32>> {
    let nelements: usize = dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
        .ok_or_else(|| BitNetError::InvalidGguf("tensor element count overflow".into()))?;
    let nbytes = types::ggml_nbytes(dims, ty)?;
    if data.len() < nbytes {
        return Err(BitNetError::InvalidGguf(format!(
            "tensor data len {} < expected nbytes {}",
            data.len(),
            nbytes
        )));
    }
    let data = &data[..nbytes];
    match ty {
        0 => dequant_f32(data, nelements),
        1 => dequant_f16(data, nelements),
        2 => dequant_q4_0(data, nelements),
        3 => dequant_q4_1(data, nelements),
        6 => dequant_q5_0(data, nelements),
        7 => dequant_q5_1(data, nelements),
        8 => dequant_q8_0(data, nelements),
        9 => dequant_q8_1(data, nelements),
        10 => Err(BitNetError::UnsupportedGgmlType(10)),
        11 => Err(BitNetError::UnsupportedGgmlType(11)),
        12 => dequant_q4_k(data, nelements),
        13 => Err(BitNetError::UnsupportedGgmlType(13)),
        14 => dequant_q6_k(data, nelements),
        15 => Err(BitNetError::UnsupportedGgmlType(15)),
        30 => dequant_bf16(data, nelements),
        34 => dequant_tq1_0(data, nelements),
        35 => dequant_tq2_0(data, nelements),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

fn dequant_f32(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if data.len() < n * 4 {
        return Err(BitNetError::InvalidGguf("f32 tensor truncated".into()));
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let b = u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
        v.push(f32::from_bits(b));
    }
    Ok(v)
}

fn dequant_f16(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if data.len() < n * 2 {
        return Err(BitNetError::InvalidGguf("f16 tensor truncated".into()));
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let h = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
        v.push(fp16_to_f32(h));
    }
    Ok(v)
}

fn dequant_bf16(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if data.len() < n * 2 {
        return Err(BitNetError::InvalidGguf("bf16 tensor truncated".into()));
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let h = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
        v.push(bf16_to_f32(h));
    }
    Ok(v)
}

fn dequant_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK4_0 != 0 {
        return Err(BitNetError::InvalidGguf("q4_0 nelements % 32 != 0".into()));
    }
    let nb = n / QK4_0;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 18;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        for j in 0..QK4_0 / 2 {
            let q = data[o + 2 + j];
            let x0 = ((q & 0x0f) as f32) - 8.0;
            let x1 = ((q >> 4) as f32) - 8.0;
            y[i * QK4_0 + j] = x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
    Ok(y)
}

fn dequant_q4_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK4_0 != 0 {
        return Err(BitNetError::InvalidGguf("q4_1 nelements % 32 != 0".into()));
    }
    let nb = n / QK4_0;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 20;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        let m = fp16_to_f32(u16::from_le_bytes(data[o + 2..o + 4].try_into().unwrap()));
        for j in 0..QK4_0 / 2 {
            let q = data[o + 4 + j];
            let x0 = ((q & 0x0f) as f32) - 8.0;
            let x1 = ((q >> 4) as f32) - 8.0;
            y[i * QK4_0 + j] = x0 * d + m;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d + m;
        }
    }
    Ok(y)
}

fn dequant_q5_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % 32 != 0 {
        return Err(BitNetError::InvalidGguf("q5_0 nelements % 32 != 0".into()));
    }
    let nb = n / 32;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 22;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        let qh = u32::from_le_bytes(data[o + 2..o + 6].try_into().unwrap());
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q = data[o + 6 + j];
            let x0 = (((q & 0x0F) as u32 | xh0) as i32) - 16;
            let x1 = (((q >> 4) as u32 | xh1) as i32) - 16;
            y[i * 32 + j] = x0 as f32 * d;
            y[i * 32 + j + 16] = x1 as f32 * d;
        }
    }
    Ok(y)
}

fn dequant_q5_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % 32 != 0 {
        return Err(BitNetError::InvalidGguf("q5_1 nelements % 32 != 0".into()));
    }
    let nb = n / 32;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 24;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        let m = fp16_to_f32(u16::from_le_bytes(data[o + 2..o + 4].try_into().unwrap()));
        let qh = u32::from_le_bytes(data[o + 4..o + 8].try_into().unwrap());
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q = data[o + 8 + j];
            let x0 = (((q & 0x0F) as u32 | xh0) as i32) - 16;
            let x1 = (((q >> 4) as u32 | xh1) as i32) - 16;
            y[i * 32 + j] = x0 as f32 * d + m;
            y[i * 32 + j + 16] = x1 as f32 * d + m;
        }
    }
    Ok(y)
}

fn dequant_q8_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % 32 != 0 {
        return Err(BitNetError::InvalidGguf("q8_0 nelements % 32 != 0".into()));
    }
    let nb = n / 32;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 34;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        for j in 0..32 {
            let q = data[o + 2 + j] as i8 as f32;
            y[i * 32 + j] = q * d;
        }
    }
    Ok(y)
}

fn dequant_q8_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK8_1 != 0 {
        return Err(BitNetError::InvalidGguf("q8_1 nelements % 32 != 0".into()));
    }
    let nb = n / QK8_1;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 40;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        for j in 0..QK8_1 {
            let q = data[o + 4 + j] as i8 as f32;
            y[i * QK8_1 + j] = q * d;
        }
    }
    Ok(y)
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(q.len() >= 8);
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("q4_K nelements % 256 != 0".into()));
    }
    let nb = n / QK_K;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 144;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        let min = fp16_to_f32(u16::from_le_bytes(data[o + 2..o + 4].try_into().unwrap()));
        let scales = &data[o + 4..o + 16];
        let qs = &data[o + 16..o + 16 + 128];
        let mut is = 0usize;
        let mut qo = 0usize;
        let mut y_off = i * QK_K;
        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let d2 = d * sc2 as f32;
            let m2 = min * m2 as f32;
            for l in 0..32 {
                y[y_off + l] = d1 * ((qs[qo + l] & 0x0F) as f32) - m1;
            }
            for l in 0..32 {
                y[y_off + 32 + l] = d2 * ((qs[qo + l] >> 4) as f32) - m2;
            }
            qo += 32;
            y_off += 64;
            is += 2;
        }
    }
    Ok(y)
}

fn dequant_q6_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("q6_K nelements % 256 != 0".into()));
    }
    let nb = n / QK_K;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * 210;
        let d = fp16_to_f32(u16::from_le_bytes(data[o + 208..o + 210].try_into().unwrap()));
        let ql = &data[o..o + 128];
        let qh = &data[o + 128..o + 192];
        let sc = &data[o + 192..o + 208];
        let yb = &mut y[i * QK_K..(i + 1) * QK_K];
        let mut ql_o = 0usize;
        let mut qh_o = 0usize;
        let mut sc_o = 0usize;
        let mut yp = 0usize;
        for _ in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_o + l] & 0xF) as i32 | ((((qh[qh_o + l] >> 0) & 3) as i32) << 4)) - 32;
                let q2 = ((ql[ql_o + l + 32] & 0xF) as i32
                    | ((((qh[qh_o + l] >> 2) & 3) as i32) << 4))
                    - 32;
                let q3 = ((ql[ql_o + l] >> 4) as i32 | ((((qh[qh_o + l] >> 4) & 3) as i32) << 4)) - 32;
                let q4 = ((ql[ql_o + l + 32] >> 4) as i32
                    | ((((qh[qh_o + l] >> 6) & 3) as i32) << 4))
                    - 32;
                yb[yp + l] = d * sc[sc_o + is + 0] as f32 * q1 as f32;
                yb[yp + l + 32] = d * sc[sc_o + is + 2] as f32 * q2 as f32;
                yb[yp + l + 64] = d * sc[sc_o + is + 4] as f32 * q3 as f32;
                yb[yp + l + 96] = d * sc[sc_o + is + 6] as f32 * q4 as f32;
            }
            yp += 128;
            ql_o += 64;
            qh_o += 32;
            sc_o += 8;
        }
    }
    Ok(y)
}

fn dequant_tq1_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq1_0 nelements % 256 != 0".into()));
    }
    const QS_LEN: usize = 48;
    const QH_LEN: usize = 4;
    let nb = n / QK_K;
    let block = 2 + QS_LEN + QH_LEN;
    let mut y = vec![0.0f32; n];
    let pow3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    for i in 0..nb {
        let o = i * block;
        let d = fp16_to_f32(u16::from_le_bytes(data[o + 0..o + 2].try_into().unwrap()));
        let qs = &data[o + 2..o + 2 + QS_LEN];
        let qh = &data[o + 2 + QS_LEN..o + 2 + QS_LEN + QH_LEN];
        let mut yp = i * QK_K;
        // `ggml-quants.c` — first j runs while `j < sizeof(qs) - sizeof(qs)%32` (here: j < 32).
        for j in (0..(QS_LEN - QS_LEN % 32)).step_by(32) {
            for n in 0..5 {
                for m in 0..32 {
                    let q = qs[j + m].wrapping_mul(pow3[n]);
                    let xi = (((q as u16) * 3) >> 8) as i16;
                    y[yp] = (xi - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
        for j in (QS_LEN - QS_LEN % 32..QS_LEN).step_by(16) {
            for n in 0..5 {
                for m in 0..16 {
                    let q = qs[j + m].wrapping_mul(pow3[n]);
                    let xi = (((q as u16) * 3) >> 8) as i16;
                    y[yp] = (xi - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
        for n in 0..4 {
            for j in 0..QH_LEN {
                let q = qh[j].wrapping_mul(pow3[n]);
                let xi = (((q as u16) * 3) >> 8) as i16;
                y[yp] = (xi - 1) as f32 * d;
                yp += 1;
            }
        }
        debug_assert_eq!(yp, (i + 1) * QK_K);
    }
    Ok(y)
}

fn dequant_tq2_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        return Err(BitNetError::InvalidGguf("tq2_0 nelements % 256 != 0".into()));
    }
    const QS_LEN: usize = 64;
    let nb = n / QK_K;
    let block = 2 + QS_LEN;
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        let o = i * block;
        let d = fp16_to_f32(u16::from_le_bytes(data[o..o + 2].try_into().unwrap()));
        let qs = &data[o + 2..o + 2 + QS_LEN];
        let mut yp = i * QK_K;
        for j in (0..QS_LEN).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (qs[j + m] >> (l * 2)) & 3;
                    y[yp] = (q as i8 - 1) as f32 * d;
                    yp += 1;
                }
            }
        }
    }
    Ok(y)
}
