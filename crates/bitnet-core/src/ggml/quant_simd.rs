//! Decode packed weights directly into SIMD registers; activations retain F32 precision.
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
macro_rules! dispatch {
    ($kernel:ident, $ty:expr, $row:expr, $x:expr) => {
        match $ty {
            0 => $kernel::<0>($row, $x),
            2 => $kernel::<2>($row, $x),
            6 => $kernel::<6>($row, $x),
            8 => $kernel::<8>($row, $x),
            12 => $kernel::<12>($row, $x),
            13 => $kernel::<13>($row, $x),
            14 => $kernel::<14>($row, $x),
            39 => $kernel::<39>($row, $x),
            _ => unreachable!(),
        }
    };
}

static ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static WIDE: OnceLock<bool> = OnceLock::new();

pub(crate) fn f32_accumulator_lanes() -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if *ENABLED.get_or_init(|| {
        std::env::var("RBITNET_CPU_SIMD_QUANT").as_deref() != Ok("0")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }) {
        if *WIDE.get_or_init(|| {
            std::env::var("RBITNET_CPU_AVX512").as_deref() != Ok("0")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
        }) {
            return Some(16);
        }
        return Some(8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = &ENABLED;
    None
}

pub(super) fn dot(ty: u32, row: &[u8], x: &[f32]) -> Option<f32> {
    let lanes = f32_accumulator_lanes()?;
    #[cfg(target_arch = "x86_64")]
    {
        let (elements, bytes) = super::types::type_layout(ty).ok()?;
        if !matches!(ty, 0 | 2 | 6 | 8 | 12 | 13 | 14 | 39)
            || x.len() % elements != 0
            || row.len() != (x.len() / elements).checked_mul(bytes)?
        {
            return None;
        }
        // The shape and CPU features have been checked before unaligned loads.
        if lanes == 16 {
            Some(unsafe { dispatch!(dot_avx512, ty, row, x) })
        } else {
            Some(unsafe { dispatch!(dot_avx2, ty, row, x) })
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (lanes, ty, row, x);
        None
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_avx2<const TY: u32>(row: &[u8], x: &[f32]) -> f32 {
    let ty = TY;
    use std::arch::x86_64::*;
    let mut a = _mm256_setzero_ps();
    let mut b = _mm256_setzero_ps();
    let mask = _mm_set1_epi8(15);
    let half = |p: usize| {
        let bits = std::ptr::read_unaligned(row.as_ptr().add(p).cast::<u16>());
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
    };
    let magnitudes = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    let accumulate = |weights: __m256, offset: usize, accumulator: &mut __m256| {
        *accumulator = _mm256_fmadd_ps(
            weights,
            _mm256_loadu_ps(x.as_ptr().add(offset)),
            *accumulator,
        );
    };
    match ty {
        0 => {
            let mut i = 0;
            while i + 8 <= x.len() {
                accumulate(_mm256_loadu_ps(row.as_ptr().add(i * 4).cast()), i, &mut a);
                i += 8;
            }
            let mut lanes = [0.0; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), a);
            let mut sum: f32 = lanes.into_iter().sum();
            for (i, &xi) in x.iter().enumerate().skip(i) {
                sum += f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()) * xi;
            }
            return sum;
        }
        2 | 6 | 8 | 39 => {
            let bytes = match ty {
                2 => 18,
                6 => 22,
                8 => 34,
                _ => 17,
            };
            for block in 0..x.len() / 32 {
                let base = block * bytes;
                let d = if ty == 39 {
                    let e = row[base];
                    f32::from_bits(if e < 2 {
                        0x00200000u32 << e
                    } else {
                        (u32::from(e) - 1) << 23
                    })
                } else {
                    half(base)
                };
                let scale = _mm256_set1_ps(d);
                if ty == 8 {
                    for j in (0..32).step_by(8) {
                        let q = _mm_loadl_epi64(row.as_ptr().add(base + 2 + j).cast());
                        let weights =
                            _mm256_mul_ps(scale, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)));
                        accumulate(
                            weights,
                            block * 32 + j,
                            if j & 8 == 0 { &mut a } else { &mut b },
                        );
                    }
                    continue;
                }
                let offset = match ty {
                    2 => 2,
                    6 => 6,
                    _ => 1,
                };
                let high = if ty == 6 {
                    u32::from_le_bytes(row[base + 2..base + 6].try_into().unwrap())
                } else {
                    0
                };
                for j in [0, 8] {
                    let packed = _mm_loadl_epi64(row.as_ptr().add(base + offset + j).cast());
                    for upper in 0..2 {
                        let q = _mm_and_si128(
                            if upper == 0 {
                                packed
                            } else {
                                _mm_srli_epi16::<4>(packed)
                            },
                            mask,
                        );
                        let integers = if ty == 39 {
                            _mm256_cvtepi8_epi32(_mm_shuffle_epi8(magnitudes, q))
                        } else {
                            let mut q = _mm256_cvtepu8_epi32(q);
                            if ty == 6 {
                                let first = (j + upper * 16) as i32;
                                let shifts = _mm256_setr_epi32(
                                    first,
                                    first + 1,
                                    first + 2,
                                    first + 3,
                                    first + 4,
                                    first + 5,
                                    first + 6,
                                    first + 7,
                                );
                                let bits = _mm256_and_si256(
                                    _mm256_srlv_epi32(_mm256_set1_epi32(high as i32), shifts),
                                    _mm256_set1_epi32(1),
                                );
                                q = _mm256_or_si256(q, _mm256_slli_epi32::<4>(bits));
                            }
                            _mm256_sub_epi32(q, _mm256_set1_epi32(if ty == 6 { 16 } else { 8 }))
                        };
                        let weights = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(integers));
                        accumulate(
                            weights,
                            block * 32 + j + upper * 16,
                            if j == 0 { &mut a } else { &mut b },
                        );
                    }
                }
            }
        }
        12 | 13 => {
            let bytes = if ty == 12 { 144 } else { 176 };
            let qoff = if ty == 12 { 16 } else { 48 };
            for block in 0..x.len() / 256 {
                let base = block * bytes;
                let d = half(base);
                let dmin = half(base + 2);
                for pair in 0..4 {
                    for j in (0..32).step_by(8) {
                        let packed =
                            _mm_loadl_epi64(row.as_ptr().add(base + qoff + pair * 32 + j).cast());
                        for upper in 0..2 {
                            let group = pair * 2 + upper;
                            let scales = &row[base + 4..base + 16];
                            let (sc, min) = if group < 4 {
                                (scales[group] & 63, scales[group + 4] & 63)
                            } else {
                                (
                                    (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                                    (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                                )
                            };
                            let nibble = _mm_and_si128(
                                if upper == 0 {
                                    packed
                                } else {
                                    _mm_srli_epi16::<4>(packed)
                                },
                                mask,
                            );
                            let mut q = _mm256_cvtepu8_epi32(nibble);
                            if ty == 13 {
                                let high = _mm_loadl_epi64(row.as_ptr().add(base + 16 + j).cast());
                                let bits = _mm256_and_si256(
                                    _mm256_srlv_epi32(
                                        _mm256_cvtepu8_epi32(high),
                                        _mm256_set1_epi32(group as i32),
                                    ),
                                    _mm256_set1_epi32(1),
                                );
                                q = _mm256_or_si256(q, _mm256_slli_epi32::<4>(bits));
                            }
                            let weights = _mm256_sub_ps(
                                _mm256_mul_ps(
                                    _mm256_set1_ps(d * f32::from(sc)),
                                    _mm256_cvtepi32_ps(q),
                                ),
                                _mm256_set1_ps(dmin * f32::from(min)),
                            );
                            accumulate(
                                weights,
                                block * 256 + pair * 64 + upper * 32 + j,
                                if j & 8 == 0 { &mut a } else { &mut b },
                            );
                        }
                    }
                }
            }
        }
        14 => {
            for block in 0..x.len() / 256 {
                let base = block * 210;
                let d = half(base + 208);
                for pass in 0..2 {
                    for quarter in 0..4 {
                        for j in (0..32).step_by(8) {
                            let low = _mm_loadl_epi64(
                                row.as_ptr()
                                    .add(base + pass * 64 + (quarter % 2) * 32 + j)
                                    .cast(),
                            );
                            let low = _mm_and_si128(
                                if quarter < 2 {
                                    low
                                } else {
                                    _mm_srli_epi16::<4>(low)
                                },
                                mask,
                            );
                            let high = _mm_loadl_epi64(
                                row.as_ptr().add(base + 128 + pass * 32 + j).cast(),
                            );
                            let high = match quarter {
                                0 => high,
                                1 => _mm_srli_epi16::<2>(high),
                                2 => _mm_srli_epi16::<4>(high),
                                _ => _mm_srli_epi16::<6>(high),
                            };
                            let high = _mm_and_si128(high, _mm_set1_epi8(3));
                            let q = _mm256_sub_epi32(
                                _mm256_or_si256(
                                    _mm256_cvtepu8_epi32(low),
                                    _mm256_slli_epi32::<4>(_mm256_cvtepu8_epi32(high)),
                                ),
                                _mm256_set1_epi32(32),
                            );
                            let sc = row[base + 192 + pass * 8 + quarter * 2 + j / 16] as i8;
                            let weights = _mm256_mul_ps(
                                _mm256_set1_ps(d * f32::from(sc)),
                                _mm256_cvtepi32_ps(q),
                            );
                            accumulate(
                                weights,
                                block * 256 + pass * 128 + quarter * 32 + j,
                                if j & 8 == 0 { &mut a } else { &mut b },
                            );
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    let mut lanes = [0.0; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(a, b));
    lanes.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx2,fma,f16c")]
unsafe fn dot_avx512<const TY: u32>(row: &[u8], x: &[f32]) -> f32 {
    let ty = TY;
    use std::arch::x86_64::*;
    let mut a = _mm512_setzero_ps();
    let mut b = _mm512_setzero_ps();
    let mask = _mm_set1_epi8(15);
    let half = |p: usize| {
        let bits = std::ptr::read_unaligned(row.as_ptr().add(p).cast::<u16>());
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
    };
    let magnitudes = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    let accumulate = |weights: __m512, offset: usize, accumulator: &mut __m512| {
        *accumulator = _mm512_fmadd_ps(
            weights,
            _mm512_loadu_ps(x.as_ptr().add(offset)),
            *accumulator,
        );
    };
    match ty {
        0 => {
            let mut i = 0;
            while i + 16 <= x.len() {
                accumulate(_mm512_loadu_ps(row.as_ptr().add(i * 4).cast()), i, &mut a);
                i += 16;
            }
            let mut lanes = [0.0; 16];
            _mm512_storeu_ps(lanes.as_mut_ptr(), a);
            let mut sum: f32 = lanes.into_iter().sum();
            for (i, &xi) in x.iter().enumerate().skip(i) {
                sum += f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()) * xi;
            }
            return sum;
        }
        2 | 6 | 8 | 39 => {
            let bytes = match ty {
                2 => 18,
                6 => 22,
                8 => 34,
                _ => 17,
            };
            for block in 0..x.len() / 32 {
                let base = block * bytes;
                let d = if ty == 39 {
                    let e = row[base];
                    f32::from_bits(if e < 2 {
                        0x00200000u32 << e
                    } else {
                        (u32::from(e) - 1) << 23
                    })
                } else {
                    half(base)
                };
                let scale = _mm512_set1_ps(d);
                if ty == 8 {
                    for j in (0..32).step_by(16) {
                        let q = _mm_loadu_si128(row.as_ptr().add(base + 2 + j).cast());
                        let weights =
                            _mm512_mul_ps(scale, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)));
                        accumulate(
                            weights,
                            block * 32 + j,
                            if j & 16 == 0 { &mut a } else { &mut b },
                        );
                    }
                    continue;
                }
                let offset = match ty {
                    2 => 2,
                    6 => 6,
                    _ => 1,
                };
                let high = if ty == 6 {
                    u32::from_le_bytes(row[base + 2..base + 6].try_into().unwrap())
                } else {
                    0
                };
                for j in [0] {
                    let packed = _mm_loadu_si128(row.as_ptr().add(base + offset + j).cast());
                    for upper in 0..2 {
                        let q = _mm_and_si128(
                            if upper == 0 {
                                packed
                            } else {
                                _mm_srli_epi16::<4>(packed)
                            },
                            mask,
                        );
                        let integers = if ty == 39 {
                            _mm512_cvtepi8_epi32(_mm_shuffle_epi8(magnitudes, q))
                        } else {
                            let mut q = _mm512_cvtepu8_epi32(q);
                            if ty == 6 {
                                let first = (j + upper * 16) as i32;
                                let shifts = _mm512_setr_epi32(
                                    first,
                                    first + 1,
                                    first + 2,
                                    first + 3,
                                    first + 4,
                                    first + 5,
                                    first + 6,
                                    first + 7,
                                    first + 8,
                                    first + 9,
                                    first + 10,
                                    first + 11,
                                    first + 12,
                                    first + 13,
                                    first + 14,
                                    first + 15,
                                );
                                let bits = _mm512_and_si512(
                                    _mm512_srlv_epi32(_mm512_set1_epi32(high as i32), shifts),
                                    _mm512_set1_epi32(1),
                                );
                                q = _mm512_or_si512(q, _mm512_slli_epi32::<4>(bits));
                            }
                            _mm512_sub_epi32(q, _mm512_set1_epi32(if ty == 6 { 16 } else { 8 }))
                        };
                        let weights = _mm512_mul_ps(scale, _mm512_cvtepi32_ps(integers));
                        accumulate(
                            weights,
                            block * 32 + j + upper * 16,
                            if j == 0 { &mut a } else { &mut b },
                        );
                    }
                }
            }
        }
        12 | 13 => {
            let bytes = if ty == 12 { 144 } else { 176 };
            let qoff = if ty == 12 { 16 } else { 48 };
            for block in 0..x.len() / 256 {
                let base = block * bytes;
                let d = half(base);
                let dmin = half(base + 2);
                for pair in 0..4 {
                    for j in (0..32).step_by(16) {
                        let packed =
                            _mm_loadu_si128(row.as_ptr().add(base + qoff + pair * 32 + j).cast());
                        for upper in 0..2 {
                            let group = pair * 2 + upper;
                            let scales = &row[base + 4..base + 16];
                            let (sc, min) = if group < 4 {
                                (scales[group] & 63, scales[group + 4] & 63)
                            } else {
                                (
                                    (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                                    (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                                )
                            };
                            let nibble = _mm_and_si128(
                                if upper == 0 {
                                    packed
                                } else {
                                    _mm_srli_epi16::<4>(packed)
                                },
                                mask,
                            );
                            let mut q = _mm512_cvtepu8_epi32(nibble);
                            if ty == 13 {
                                let high = _mm_loadu_si128(row.as_ptr().add(base + 16 + j).cast());
                                let bits = _mm512_and_si512(
                                    _mm512_srlv_epi32(
                                        _mm512_cvtepu8_epi32(high),
                                        _mm512_set1_epi32(group as i32),
                                    ),
                                    _mm512_set1_epi32(1),
                                );
                                q = _mm512_or_si512(q, _mm512_slli_epi32::<4>(bits));
                            }
                            let weights = _mm512_sub_ps(
                                _mm512_mul_ps(
                                    _mm512_set1_ps(d * f32::from(sc)),
                                    _mm512_cvtepi32_ps(q),
                                ),
                                _mm512_set1_ps(dmin * f32::from(min)),
                            );
                            accumulate(
                                weights,
                                block * 256 + pair * 64 + upper * 32 + j,
                                if j & 16 == 0 { &mut a } else { &mut b },
                            );
                        }
                    }
                }
            }
        }
        14 => {
            for block in 0..x.len() / 256 {
                let base = block * 210;
                let d = half(base + 208);
                for pass in 0..2 {
                    for quarter in 0..4 {
                        for j in (0..32).step_by(16) {
                            let low = _mm_loadu_si128(
                                row.as_ptr()
                                    .add(base + pass * 64 + (quarter % 2) * 32 + j)
                                    .cast(),
                            );
                            let low = _mm_and_si128(
                                if quarter < 2 {
                                    low
                                } else {
                                    _mm_srli_epi16::<4>(low)
                                },
                                mask,
                            );
                            let high = _mm_loadu_si128(
                                row.as_ptr().add(base + 128 + pass * 32 + j).cast(),
                            );
                            let high = match quarter {
                                0 => high,
                                1 => _mm_srli_epi16::<2>(high),
                                2 => _mm_srli_epi16::<4>(high),
                                _ => _mm_srli_epi16::<6>(high),
                            };
                            let high = _mm_and_si128(high, _mm_set1_epi8(3));
                            let q = _mm512_sub_epi32(
                                _mm512_or_si512(
                                    _mm512_cvtepu8_epi32(low),
                                    _mm512_slli_epi32::<4>(_mm512_cvtepu8_epi32(high)),
                                ),
                                _mm512_set1_epi32(32),
                            );
                            let sc = row[base + 192 + pass * 8 + quarter * 2 + j / 16] as i8;
                            let weights = _mm512_mul_ps(
                                _mm512_set1_ps(d * f32::from(sc)),
                                _mm512_cvtepi32_ps(q),
                            );
                            accumulate(
                                weights,
                                block * 256 + pass * 128 + quarter * 32 + j,
                                if j & 16 == 0 { &mut a } else { &mut b },
                            );
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    let mut lanes = [0.0; 16];
    _mm512_storeu_ps(lanes.as_mut_ptr(), _mm512_add_ps(a, b));
    lanes.into_iter().sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_register_dots_match_independently_dequantized_weights() {
        for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
            let (block, bytes) = super::super::types::type_layout(ty).unwrap();
            let n = if ty == 0 { 1024 } else { block * 4 };
            let mut payload: Vec<u8> = (0..n / block * bytes)
                .map(|i| (i * 37 + 19) as u8)
                .collect();
            if ty == 0 {
                for (i, slot) in payload.chunks_exact_mut(4).enumerate() {
                    slot.copy_from_slice(&(i as f32 * 0.37).sin().to_le_bytes());
                }
            } else {
                for b in payload.chunks_exact_mut(bytes) {
                    if ty == 39 {
                        b[0] = 120;
                    } else {
                        let offset = if ty == 14 { 208 } else { 0 };
                        b[offset..offset + 2]
                            .copy_from_slice(&half::f16::from_f32(0.003).to_bits().to_le_bytes());
                        if ty == 12 || ty == 13 {
                            b[2..4].copy_from_slice(
                                &half::f16::from_f32(0.002).to_bits().to_le_bytes(),
                            );
                        }
                    }
                }
            }
            let weights = super::super::tensor_to_f32(&payload, ty, &[n as u64, 1]).unwrap();
            for phase in [0.0, 0.19, 1.0] {
                let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23 + phase).cos()).collect();
                let expected: f64 = weights
                    .iter()
                    .zip(&x)
                    .map(|(&w, &x)| w as f64 * x as f64)
                    .sum();
                if let Some(actual) = super::dot(ty, &payload, &x) {
                    assert!(
                        (actual as f64 - expected).abs() < 3e-4 * expected.abs().max(1.0),
                        "type {ty}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}
