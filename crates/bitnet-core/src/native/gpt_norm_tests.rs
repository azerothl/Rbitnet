// Native memory-ledger ABI category MemoryScratch, also used by alloc_scratch.
const SCRATCH_CATEGORY: usize = 5;
type NormCheck = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    u32,
    u32,
    f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
) -> i32;
#[test]
fn optional_gpt_staged_norm_preserves_original_bits_independent_f64_and_true_scratch() {
    if std::env::var("RBITNET_GPT_ORDERED_FASTPATH_TEST").as_deref() != Ok("1") {
        return;
    }
    let rt = crate::backend::CudaRuntime::try_load().unwrap();
    let library = crate::ggml::load_cuda_quant_library().unwrap();
    let norm = unsafe {
        *library
            .get::<NormCheck>(b"rbitnet_cuda_gpt_ordered_norm_check\0")
            .unwrap()
    };
    let before = rt.managed_memory_stats().unwrap().categories[SCRATCH_CATEGORY];
    let mut norms = 0;
    for n in [1usize, 31, 257, 4096, 8192, 8193] {
        for count in [1usize, 2, 32] {
            for residual in [false, true] {
                for pattern in 0..3 {
                    let x: Vec<_> = (0..n * count)
                        .map(|i| match pattern {
                            0 => ((i * 13 % 71) as f32 - 35.) / 37.,
                            1 => {
                                if i % 2 == 0 {
                                    0.0
                                } else {
                                    -0.0
                                }
                            }
                            _ => {
                                if i % 5 == 0 {
                                    f32::MAX / 8.
                                } else {
                                    f32::from_bits((i % 15 + 1) as u32)
                                }
                            }
                        })
                        .collect();
                    let add: Vec<_> = (0..n * count)
                        .map(|i| ((i * 7 % 43) as f32 - 21.) / 63.)
                        .collect();
                    let weights: Vec<_> = (0..n).map(|i| 1. + (i % 13) as f32 / 128.).collect();
                    let mut reference = vec![0.; x.len()];
                    let mut actual = reference.clone();
                    let mut reference_x = reference.clone();
                    let mut actual_x = reference.clone();
                    assert_eq!(
                        unsafe {
                            norm(
                                x.as_ptr(),
                                if residual {
                                    add.as_ptr()
                                } else {
                                    std::ptr::null()
                                },
                                weights.as_ptr(),
                                n as u32,
                                count as u32,
                                1e-6,
                                reference.as_mut_ptr(),
                                actual.as_mut_ptr(),
                                reference_x.as_mut_ptr(),
                                actual_x.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    assert!(reference.iter().chain(&actual).all(|x| x.is_finite()));
                    for i in 0..x.len() {
                        assert_eq!(
                            reference[i].to_bits(),
                            actual[i].to_bits(),
                            "norm n={n} count={count} residual={residual} pattern={pattern} i={i}"
                        );
                        assert_eq!(reference_x[i].to_bits(), actual_x[i].to_bits());
                        let expected = if residual { x[i] + add[i] } else { x[i] };
                        assert_eq!(actual_x[i].to_bits(), expected.to_bits());
                    }
                    if pattern < 2 {
                        // F64 RMS independently checks the intended mathematics;
                        // strict old/new bits are the separate arithmetic-order gate.
                        for token in 0..count {
                            let values = &reference_x[token * n..(token + 1) * n];
                            let rms =
                                values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64;
                            let inv = 1. / (rms + 1e-6f32 as f64).sqrt();
                            for i in 0..n {
                                let expected = values[i] as f64 * inv * weights[i] as f64;
                                assert!(
                                    (actual[token * n + i] as f64 - expected).abs()
                                        <= 2e-5 * (1. + expected.abs())
                                );
                            }
                        }
                    }
                    assert_eq!(
                        rt.managed_memory_stats().unwrap().categories[SCRATCH_CATEGORY],
                        before
                    );
                    norms += 1;
                }
            }
        }
    }
    let x = [1f32];
    let mut output = [0f32];
    assert_eq!(
        unsafe {
            norm(
                x.as_ptr(),
                std::ptr::null(),
                x.as_ptr(),
                0,
                1,
                1e-6,
                output.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        -1
    );
    println!("GPT_NORM_ORACLE_DONE norms={norms}");
    assert_eq!(norms, 108);
}
