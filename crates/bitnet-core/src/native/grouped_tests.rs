//! Pending CUDA oracle for expert-grouped block FFNs. No throughput claims.
use super::*;

type Check = unsafe extern "C" fn(
    *const Config,
    *const Matrix,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const u32,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    *mut f32,
    *mut f32,
    *mut f32,
) -> i32;

fn payload(ty: u32, columns: usize, rows: usize, seed: usize) -> Vec<u8> {
    let rb = crate::ggml::ggml_row_size(ty, columns as u64).unwrap();
    let mut data: Vec<u8> = (0..rows * rb)
        .map(|i| (i * 37 + seed * 53 + 19) as u8)
        .collect();
    if ty == 0 {
        for (i, bytes) in data.chunks_exact_mut(4).enumerate() {
            bytes.copy_from_slice(&((i as f32 * 0.031 + seed as f32).cos() * 0.035).to_le_bytes());
        }
    } else {
        let bytes = match ty {
            2 => 18,
            6 => 22,
            8 => 34,
            12 => 144,
            13 => 176,
            14 => 210,
            39 => 17,
            _ => unreachable!(),
        };
        for (i, b) in data.chunks_exact_mut(bytes).enumerate() {
            if ty == 39 {
                b[0] = [113, 117, 119, 121][i % 4];
            } else {
                let offset = if ty == 14 { 208 } else { 0 };
                b[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(0.00013).to_bits().to_le_bytes());
                if ty == 12 || ty == 13 {
                    b[2..4].copy_from_slice(&half::f16::from_f32(0.00007).to_bits().to_le_bytes());
                }
            }
        }
    }
    data
}

#[test]
fn optional_grouped_moe_exact_serial_bits_all_formats_router_slots_tails_graphs_and_f64() {
    if std::env::var("RBITNET_CUDA_GROUPED_MOE_TEST").as_deref() != Ok("1") {
        return;
    }
    let rt = crate::backend::CudaRuntime::try_load().expect("CUDA required");
    let lib = crate::ggml::load_cuda_quant_library().expect("CUDA DLL required");
    let run = unsafe {
        *lib.get::<Check>(b"rbitnet_cuda_moe_group_check\0")
            .expect("group oracle ABI")
    };
    const N: usize = 256;
    const EXPERTS: usize = 7;
    const USED: usize = 3;
    let before = rt.managed_memory_stats().unwrap();
    let formats: Vec<[u32; 3]> = [0, 2, 6, 8, 12, 13, 14, 39]
        .map(|ty| [ty; 3])
        .into_iter()
        .chain([[39, 8, 14]])
        .collect();
    for types in formats {
        let banks: Vec<Vec<u8>> = (0..3)
            .map(|p| payload(types[p], N, N * EXPERTS, p))
            .collect();
        let matrices: Vec<Matrix> = (0..3)
            .map(|p| Matrix {
                weights: banks[p].as_ptr().cast(),
                row_bytes: crate::ggml::ggml_row_size(types[p], N as u64).unwrap(),
                ty: types[p],
                cols: N as u32,
                rows: (N * EXPERTS) as u32,
            })
            .collect();
        let dense: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                crate::ggml::tensor_to_f32(
                    &banks[p],
                    types[p],
                    &[N as u64, N as u64, EXPERTS as u64],
                )
                .unwrap()
            })
            .collect();
        let biases: Vec<Vec<f32>> = (0..3)
            .map(|p| {
                (0..N * EXPERTS)
                    .map(|i| (i as f32 * 0.17 + p as f32).sin() * 0.08)
                    .collect()
            })
            .collect();
        for oai in [0u32, 1] {
            let cfg = Config {
                embd: N as u32,
                ffn: N as u32,
                experts: EXPERTS as u32,
                used: USED as u32,
                oai,
            };
            for count in [1usize, 3, 7, 17, 32] {
                let input: Vec<f32> = (0..count * N)
                    .map(|i| (i as f32 * 0.23).sin() * 2.)
                    .collect();
                // Permuted selected slots, an unused bank, and incomplete chunks.
                let ids: Vec<u32> = (0..count)
                    .flat_map(|t| [[3, 1, 4], [4, 0, 2], [1, 4, 3], [5, 2, 0]][t % 4])
                    .collect();
                let probabilities: Vec<f32> = (0..count)
                    .flat_map(|t| {
                        if t % 2 == 0 {
                            [0.5, 0.3, 0.2]
                        } else {
                            [0.2, 0.5, 0.3]
                        }
                    })
                    .collect();
                for has_bias in [false, true] {
                    let bias: Vec<*const f32> = biases
                        .iter()
                        .map(|b| {
                            if has_bias {
                                b.as_ptr()
                            } else {
                                std::ptr::null()
                            }
                        })
                        .collect();
                    let oracle = if count == 3 {
                        Some(
                            (0..count)
                                .flat_map(|token| {
                                    let x: Vec<f64> = input[token * N..(token + 1) * N]
                                        .iter()
                                        .map(|&v| v as f64)
                                        .collect();
                                    let mut out = vec![0f64; N];
                                    let dot = |p: usize, e: usize, row: usize, x: &[f64]| {
                                        dense[p][(e * N + row) * N..(e * N + row + 1) * N]
                                            .iter()
                                            .zip(x)
                                            .map(|(&w, &v)| w as f64 * v)
                                            .sum::<f64>()
                                            + if has_bias {
                                                biases[p][e * N + row] as f64
                                            } else {
                                                0.
                                            }
                                    };
                                    for slot in 0..USED {
                                        let e = ids[token * USED + slot] as usize;
                                        let hidden: Vec<f64> = (0..N)
                                            .map(|row| {
                                                let mut g = dot(0, e, row, &x);
                                                let mut u = dot(1, e, row, &x);
                                                if oai == 1 {
                                                    g = g.min(7.);
                                                    u = u.clamp(-7., 7.) + 1.;
                                                }
                                                g / (1.
                                                    + (-(if oai == 1 {
                                                        1.702f32 as f64
                                                    } else {
                                                        1.
                                                    }) * g)
                                                        .exp())
                                                    * u
                                            })
                                            .collect();
                                        for (row, v) in out.iter_mut().enumerate() {
                                            *v += probabilities[token * USED + slot] as f64
                                                * dot(2, e, row, &hidden);
                                        }
                                    }
                                    out
                                })
                                .collect::<Vec<f64>>(),
                        )
                    } else {
                        None
                    };
                    for graphs in [0u32, 1] {
                        for mode in [0u32, 1, 2] {
                            let mut actual = vec![f32::NAN; count * N];
                            let mut reference = actual.clone();
                            let mut ms = 0.;
                            assert_eq!(
                                unsafe {
                                    run(
                                        &cfg,
                                        matrices.as_ptr(),
                                        bias[0],
                                        bias[1],
                                        bias[2],
                                        input.as_ptr(),
                                        ids.as_ptr(),
                                        probabilities.as_ptr(),
                                        count as u32,
                                        mode,
                                        graphs,
                                        2,
                                        actual.as_mut_ptr(),
                                        reference.as_mut_ptr(),
                                        &mut ms,
                                    )
                                },
                                0
                            );
                            assert!(ms.is_finite() && ms >= 0.);
                            for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
                                assert!(a.is_finite() && b.is_finite());
                                assert_eq!(a.to_bits(),b.to_bits(),"{types:?} oai={oai} count={count} bias={has_bias} mode={mode} graphs={graphs} output={i}");
                                if let Some(oracle) = &oracle {
                                    assert!(
                                        (a as f64 - oracle[i]).abs()
                                            < 5e-4 * (1. + oracle[i].abs()),
                                        "F64 {types:?} output={i}: {a}/{}",
                                        oracle[i]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            rt.managed_memory_stats().unwrap().categories,
            before.categories,
            "diagnostic must release banks/workspace after each group"
        );
    }
}
