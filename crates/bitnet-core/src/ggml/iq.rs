//! I-quant dequant (llama.cpp `ggml-quants.c` layouts for IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S).

use half::f16;

use crate::error::{BitNetError, Result};
use crate::ggml::iq_tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
};

const QK_K: usize = 256;

/// GGML type ids used by GSQ-RCO mixed-precision GGUFs.
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;

pub fn is_supported_iquant(ty: u32) -> bool {
    matches!(ty, IQ2_XS | IQ3_XXS | IQ3_S | IQ2_S)
}

fn fp16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

fn grid8_iq2xs(idx: usize) -> [u8; 8] {
    IQ2XS_GRID[idx].to_le_bytes()
}

fn grid8_iq2s(idx: usize) -> [u8; 8] {
    IQ2S_GRID[idx].to_le_bytes()
}

fn grid4_iq3xxs(idx: usize) -> [u8; 4] {
    IQ3XXS_GRID[idx].to_le_bytes()
}

fn grid4_iq3s(idx: usize) -> [u8; 4] {
    IQ3S_GRID[idx].to_le_bytes()
}

fn signed_grid(val: u8, signs: u8, bit: usize) -> f32 {
    let s = if signs & KMASK_IQ2XS[bit] != 0 {
        -1.0
    } else {
        1.0
    };
    val as f32 * s
}

/// Decode one 256-element I-quant block into `out`.
pub fn decode_iq_block(ty: u32, data: &[u8], out: &mut [f32]) -> Result<()> {
    if out.len() != QK_K {
        return Err(BitNetError::InvalidGguf("i-quant block out len".into()));
    }
    match ty {
        IQ2_XS => decode_iq2_xs(data, out),
        IQ2_S => decode_iq2_s(data, out),
        IQ3_XXS => decode_iq3_xxs(data, out),
        IQ3_S => decode_iq3_s(data, out),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

pub fn dequant_iquant(ty: u32, data: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        return Err(BitNetError::InvalidGguf(
            "i-quant nelements must be divisible by 256".into(),
        ));
    }
    let (_, block) = crate::ggml::types::type_layout(ty)?;
    let nb = n / QK_K;
    let need = nb
        .checked_mul(block)
        .ok_or_else(|| BitNetError::InvalidGguf("i-quant payload overflow".into()))?;
    if data.len() < need {
        return Err(BitNetError::InvalidGguf(format!(
            "i-quant data len {} < {}",
            data.len(),
            need
        )));
    }
    let mut y = vec![0.0f32; n];
    for i in 0..nb {
        decode_iq_block(
            ty,
            &data[i * block..(i + 1) * block],
            &mut y[i * QK_K..(i + 1) * QK_K],
        )?;
    }
    Ok(y)
}

fn decode_iq2_xs(data: &[u8], y: &mut [f32]) -> Result<()> {
    if data.len() != 74 {
        return Err(BitNetError::InvalidGguf("iq2_xs block size".into()));
    }
    let d = fp16_to_f32(u16::from_le_bytes(data[0..2].try_into().unwrap()));
    let mut yp = 0;
    for ib32 in 0..8 {
        let scale = data[66 + ib32];
        let db0 = d * (0.5 + (scale & 0xf) as f32) * 0.25;
        let db1 = d * (0.5 + (scale >> 4) as f32) * 0.25;
        for l in 0..4 {
            let q = u16::from_le_bytes(
                data[2 + (4 * ib32 + l) * 2..2 + (4 * ib32 + l) * 2 + 2]
                    .try_into()
                    .unwrap(),
            );
            let grid = grid8_iq2xs((q & 511) as usize);
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            let db = if l / 2 == 0 { db0 } else { db1 };
            for j in 0..8 {
                y[yp] = db * signed_grid(grid[j], signs, j);
                yp += 1;
            }
        }
    }
    debug_assert_eq!(yp, QK_K);
    Ok(())
}

fn decode_iq2_s(data: &[u8], y: &mut [f32]) -> Result<()> {
    if data.len() != 82 {
        return Err(BitNetError::InvalidGguf("iq2_s block size".into()));
    }
    let d = fp16_to_f32(u16::from_le_bytes(data[0..2].try_into().unwrap()));
    let qs = &data[2..34];
    let signs = &data[34..66];
    let qh = &data[66..74];
    let scales = &data[74..82];
    let mut yp = 0;
    for ib32 in 0..8 {
        let db0 = d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25;
        let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
        for l in 0..4 {
            let dl = if l / 2 == 0 { db0 } else { db1 };
            let shift = 8 - 2 * l;
            let idx = qs[ib32 * 4 + l] as usize
                | (((qh[ib32] as usize) << shift) & 0x300);
            let grid = grid8_iq2s(idx);
            let sg = signs[ib32 * 4 + l];
            for j in 0..8 {
                y[yp] = dl * signed_grid(grid[j], sg, j);
                yp += 1;
            }
        }
    }
    debug_assert_eq!(yp, QK_K);
    Ok(())
}

fn decode_iq3_xxs(data: &[u8], y: &mut [f32]) -> Result<()> {
    if data.len() != 98 {
        return Err(BitNetError::InvalidGguf("iq3_xxs block size".into()));
    }
    let d = fp16_to_f32(u16::from_le_bytes(data[0..2].try_into().unwrap()));
    let qs = &data[2..66];
    let scales_and_signs = &data[66..98];
    let mut yp = 0;
    for ib32 in 0..8 {
        let aux32 = u32::from_le_bytes(
            scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap(),
        );
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let grid1 = grid4_iq3xxs(qs[ib32 * 8 + 2 * l] as usize);
            let grid2 = grid4_iq3xxs(qs[ib32 * 8 + 2 * l + 1] as usize);
            for j in 0..4 {
                y[yp + j] = db * signed_grid(grid1[j], signs, j);
                y[yp + j + 4] = db * signed_grid(grid2[j], signs, j + 4);
            }
            yp += 8;
        }
    }
    debug_assert_eq!(yp, QK_K);
    Ok(())
}

fn decode_iq3_s(data: &[u8], y: &mut [f32]) -> Result<()> {
    if data.len() != 110 {
        return Err(BitNetError::InvalidGguf("iq3_s block size".into()));
    }
    let d = fp16_to_f32(u16::from_le_bytes(data[0..2].try_into().unwrap()));
    let qs_all = &data[2..66];
    let qh_all = &data[66..74];
    let signs_all = &data[74..106];
    let scales = &data[106..110];
    let mut yp = 0;
    let mut qs_off = 0;
    let mut signs_off = 0;
    let mut qh_off = 0;
    for ib32 in (0..8).step_by(2) {
        let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0xf) as f32);
        let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
        let qh0 = qh_all[qh_off];
        for l in 0..4 {
            let grid1 = grid4_iq3s(
                qs_all[qs_off + 2 * l] as usize | (((qh0 as usize) << (8 - 2 * l)) & 256),
            );
            let grid2 = grid4_iq3s(
                qs_all[qs_off + 2 * l + 1] as usize | (((qh0 as usize) << (7 - 2 * l)) & 256),
            );
            let sg = signs_all[signs_off + l];
            for j in 0..4 {
                y[yp + j] = db1 * signed_grid(grid1[j], sg, j);
                y[yp + j + 4] = db1 * signed_grid(grid2[j], sg, j + 4);
            }
            yp += 8;
        }
        qs_off += 8;
        signs_off += 4;
        let qh1 = qh_all[qh_off + 1];
        for l in 0..4 {
            let grid1 = grid4_iq3s(
                qs_all[qs_off + 2 * l] as usize | (((qh1 as usize) << (8 - 2 * l)) & 256),
            );
            let grid2 = grid4_iq3s(
                qs_all[qs_off + 2 * l + 1] as usize | (((qh1 as usize) << (7 - 2 * l)) & 256),
            );
            let sg = signs_all[signs_off + l];
            for j in 0..4 {
                y[yp + j] = db2 * signed_grid(grid1[j], sg, j);
                y[yp + j + 4] = db2 * signed_grid(grid2[j], sg, j + 4);
            }
            yp += 8;
        }
        qh_off += 2;
        qs_off += 8;
        signs_off += 4;
    }
    debug_assert_eq!(yp, QK_K);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp16_one() -> [u8; 2] {
        f16::from_f32(1.0).to_bits().to_le_bytes()
    }

    #[test]
    fn iquant_types_are_mmap_matvec() {
        for ty in [IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S] {
            assert!(crate::ggml::ggml_type_supported_mmap_matvec(ty), "ty={ty}");
            assert!(is_supported_iquant(ty));
        }
        assert!(!is_supported_iquant(16)); // IQ2_XXS still unwired
    }

    #[test]
    fn iq2_xs_zero_qs_matches_grid0() {
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&fp16_one());
        let y = dequant_iquant(IQ2_XS, &block, 256).unwrap();
        // d=1, scale=0 → db = 0.125; qs=0 → grid 0, signs 0 (all +)
        let g = IQ2XS_GRID[0].to_le_bytes();
        for ib in 0..32 {
            for j in 0..8 {
                let expect = 0.125 * g[j] as f32;
                assert!(
                    (y[ib * 8 + j] - expect).abs() < 1e-6,
                    "idx {} {} vs {}",
                    ib * 8 + j,
                    y[ib * 8 + j],
                    expect
                );
            }
        }
        let x = [1.0f32; 256];
        let dot = crate::ggml::quant_dot::dot_row(IQ2_XS, &block, &x).unwrap();
        let sum: f32 = y.iter().sum();
        assert!((dot - sum).abs() < 1e-4);
    }

    #[test]
    fn iq3_s_nonzero_scale_is_finite() {
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&fp16_one());
        block[106] = 0x31; // two 4-bit scales
        let y = dequant_iquant(IQ3_S, &block, 256).unwrap();
        assert_eq!(y.len(), 256);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn unknown_iq_still_unsupported() {
        let err = crate::ggml::tensor_to_f32(&[0u8; 66], 16, &[256, 1]).unwrap_err();
        match err {
            BitNetError::UnsupportedGgmlType(16) => {}
            other => panic!("expected UnsupportedGgmlType(16), got {other:?}"),
        }
    }
}
