//! Actual-device exactness of optional FFN fusion, all supported GGUF layouts.
use super::*;
type Configure = unsafe extern "C" fn(*mut c_void, u32) -> i32;
struct Handle {
    ptr: *mut c_void,
    destroy: Destroy,
}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.ptr) }
    }
}
fn payload(ty: u32, n: usize, rows: usize, seed: usize) -> Vec<u8> {
    let rb = crate::ggml::ggml_row_size(ty, n as u64).unwrap();
    let mut b: Vec<u8> = (0..rb * rows)
        .map(|i| (i * 37 + seed * 53 + 19) as u8)
        .collect();
    if ty == 0 {
        for (i, s) in b.chunks_exact_mut(4).enumerate() {
            s.copy_from_slice(&((i as f32 * 0.031 + seed as f32).cos() * 0.1).to_le_bytes());
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
        for s in b.chunks_exact_mut(bytes) {
            if ty == 39 {
                s[0] = 119;
            } else {
                let at = if ty == 14 { 208 } else { 0 };
                s[at..at + 2].copy_from_slice(&half::f16::from_f32(0.0007).to_bits().to_le_bytes());
                if ty == 12 || ty == 13 {
                    s[2..4].copy_from_slice(&half::f16::from_f32(0.0003).to_bits().to_le_bytes());
                }
            }
        }
    }
    b
}
#[test]
fn fused_fixed_dynamic_all_formats_mixed_bias_graph_refill_matches_original_bits_and_f64() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let rt = crate::backend::CudaRuntime::try_load().expect("real CUDA required");
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let create = unsafe { *lib.get::<Create>(b"rbitnet_cuda_moe_create\0").unwrap() };
    let dynamic_create = unsafe {
        *lib.get::<Create>(b"rbitnet_cuda_moe_dynamic_create\0")
            .unwrap()
    };
    let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").unwrap() };
    let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_moe_step\0").unwrap() };
    let dynamic_step = unsafe {
        *lib.get::<DynamicStep>(b"rbitnet_cuda_moe_dynamic_step\0")
            .unwrap()
    };
    let configure = unsafe {
        *lib.get::<Configure>(b"rbitnet_cuda_moe_configure_fused\0")
            .unwrap()
    };
    const N: usize = 256;
    const EXP: usize = 5;
    const USED: usize = 3;
    let mut formats: Vec<[u32; 3]> = [0, 2, 6, 8, 12, 13, 14, 39]
        .into_iter()
        .map(|ty| [ty; 3])
        .collect();
    formats.extend([[39, 14, 39], [8, 39, 14]]);
    for types in formats {
        let mut owned = Vec::new();
        let mut dense = Vec::new();
        let mut rb = Vec::new();
        let mut biases = Vec::new();
        for (p, &ty) in types.iter().enumerate() {
            let bytes = payload(ty, N, N * EXP, p);
            rb.push(crate::ggml::ggml_row_size(ty, N as u64).unwrap());
            dense.push(
                crate::ggml::tensor_to_f32(&bytes, ty, &[N as u64, N as u64, EXP as u64]).unwrap(),
            );
            owned.push(
                CudaDeviceQuantMatrix::from_payload(Some(&rt), ty, bytes, N * EXP, N).unwrap(),
            );
            biases.push(
                (0..N * EXP)
                    .map(|i| (i as f32 * 0.17 + p as f32).sin() * 0.2)
                    .collect::<Vec<_>>(),
            );
        }
        let matrices: Vec<_> = owned
            .iter()
            .enumerate()
            .map(|(p, m)| Matrix {
                weights: m.device_address().unwrap() as *const c_void,
                row_bytes: rb[p],
                ty: types[p],
                cols: N as u32,
                rows: (N * EXP) as u32,
            })
            .collect();
        for (oai, dynamic, biased) in [
            (0, false, false),
            (1, false, true),
            (0, true, true),
            (1, true, false),
        ] {
            let cfg = Config {
                embd: N as u32,
                ffn: N as u32,
                experts: EXP as u32,
                used: USED as u32,
                oai,
            };
            let bias = |p: usize| {
                if biased {
                    biases[p].as_ptr()
                } else {
                    std::ptr::null()
                }
            };
            let make = || Handle {
                ptr: unsafe {
                    (if dynamic { dynamic_create } else { create })(
                        &cfg,
                        &matrices[0],
                        &matrices[1],
                        &matrices[2],
                        bias(0),
                        bias(1),
                        bias(2),
                    )
                },
                destroy,
            };
            let baseline = make();
            let fused = make();
            assert!(!baseline.ptr.is_null() && !fused.ptr.is_null());
            assert_ne!(unsafe { configure(fused.ptr, 2) }, 0);
            assert_eq!(unsafe { configure(fused.ptr, 1) }, 0);
            let probabilities = [0.5, 0.3, 0.2];
            let mut slots: Vec<CudaDeviceQuantMatrix> = Vec::new();
            for (iteration, selected) in [[3u32, 1, 4], [4, 0, 2], [1, 4, 3], [3, 1, 4]]
                .into_iter()
                .enumerate()
            {
                if iteration == 2 {
                    slots.clear();
                }
                if dynamic {
                    for (p, bank) in owned.iter().enumerate() {
                        for (s, &e) in selected.iter().enumerate() {
                            let slab = rb[p] * N;
                            let data = bank.host_payload()
                                [e as usize * slab..(e as usize + 1) * slab]
                                .to_vec();
                            if iteration == 1 || iteration == 3 {
                                assert!(slots[p * USED + s].refill(data).unwrap());
                            } else {
                                slots.push(
                                    CudaDeviceQuantMatrix::from_payload(
                                        Some(&rt),
                                        types[p],
                                        data,
                                        N,
                                        N,
                                    )
                                    .unwrap(),
                                );
                            }
                        }
                    }
                }
                let x: Vec<f32> = (0..N)
                    .map(|i| (i as f32 * 0.23 + iteration as f32 * 0.1).sin() * 4.0)
                    .collect();
                let pointers: Vec<_> = slots
                    .iter()
                    .map(|m| m.device_address().unwrap() as *const c_void)
                    .collect();
                let run = |context: *mut c_void| {
                    let mut output = vec![0.0; N];
                    let status = unsafe {
                        if dynamic {
                            dynamic_step(
                                context,
                                x.as_ptr(),
                                selected.as_ptr(),
                                probabilities.as_ptr(),
                                output.as_mut_ptr(),
                                pointers.as_ptr(),
                            )
                        } else {
                            step(
                                context,
                                x.as_ptr(),
                                selected.as_ptr(),
                                probabilities.as_ptr(),
                                output.as_mut_ptr(),
                            )
                        }
                    };
                    assert_eq!(status, 0);
                    output
                };
                let expected = run(baseline.ptr);
                let actual = run(fused.ptr);
                assert_eq!(actual,expected,"types={types:?} oai={oai} dynamic={dynamic} bias={biased} iteration={iteration}");
                assert_ne!(
                    unsafe { configure(fused.ptr, 0) },
                    0,
                    "captured graph configuration must be sealed"
                );
                let dot = |p: usize, e: usize, row: usize, input: &[f64]| {
                    dense[p][(e * N + row) * N..(e * N + row + 1) * N]
                        .iter()
                        .zip(input)
                        .map(|(&w, &v)| w as f64 * v)
                        .sum::<f64>()
                        + if biased {
                            biases[p][e * N + row] as f64
                        } else {
                            0.0
                        }
                };
                let input: Vec<_> = x.iter().map(|&v| v as f64).collect();
                let mut oracle = vec![0.0f64; N];
                for (slot, &e) in selected.iter().enumerate() {
                    let hidden: Vec<_> = (0..N)
                        .map(|r| {
                            let mut g = dot(0, e as usize, r, &input);
                            let mut u = dot(1, e as usize, r, &input);
                            if oai == 1 {
                                g = g.min(7.0);
                                u = u.clamp(-7.0, 7.0) + 1.0;
                            }
                            g / (1.0 + (-(if oai == 1 { 1.702f32 as f64 } else { 1.0 }) * g).exp())
                                * u
                        })
                        .collect();
                    for (r, value) in oracle.iter_mut().enumerate() {
                        *value += probabilities[slot] as f64 * dot(2, e as usize, r, &hidden);
                    }
                }
                for (i, (&got, &reference)) in actual.iter().zip(&oracle).enumerate() {
                    assert!(
                        (got as f64 - reference).abs() < 5e-4 * (1.0 + reference.abs()),
                        "F64 types={types:?} row={i}"
                    );
                }
            }
        }
    }
}
