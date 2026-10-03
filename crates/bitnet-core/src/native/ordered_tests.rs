//! Exact batched dot ordering, checked against the original GEMV and F64.
use std::ffi::c_void;
type Check = unsafe extern "C" fn(
    u32,
    *const c_void,
    usize,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut f32,
    *mut f32,
    *mut f32,
) -> i32;

#[test]
fn opt_in_ordered_gemm_matches_original_bits_all_formats_tails_and_f64() {
    if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let run = unsafe {
        *lib.get::<Check>(b"rbitnet_cuda_ordered_gemm_check\0")
            .expect("ordered diagnostic ABI required")
    };
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        let columns: Vec<usize> = if ty == 0 {
            vec![17, 255, 256, 513]
        } else if [12, 13, 14].contains(&ty) {
            vec![256, 512]
        } else {
            vec![32, 96, 256, 288, 512]
        };
        for cols in columns {
            for (rows, tokens) in [(3, 1), (9, 3), (13, 7), (17, 17)] {
                let rb = crate::ggml::ggml_row_size(ty, cols as u64).unwrap();
                let mut p: Vec<u8> = (0..rows * rb).map(|i| (i * 37 + 19) as u8).collect();
                if ty == 0 {
                    for (i, b) in p.chunks_exact_mut(4).enumerate() {
                        b.copy_from_slice(&((i as f32 * 0.031).cos() * 0.015).to_le_bytes());
                    }
                } else {
                    let size = match ty {
                        2 => 18,
                        6 => 22,
                        8 => 34,
                        12 => 144,
                        13 => 176,
                        14 => 210,
                        39 => 17,
                        _ => unreachable!(),
                    };
                    for (i, b) in p.chunks_exact_mut(size).enumerate() {
                        if ty == 39 {
                            b[0] = [111, 119, 127, 135][i % 4];
                        } else {
                            let offset = if ty == 14 { 208 } else { 0 };
                            b[offset..offset + 2].copy_from_slice(
                                &half::f16::from_f32(if ty == 14 { 0.00003 } else { 0.00007 })
                                    .to_bits()
                                    .to_le_bytes(),
                            );
                            if ty == 12 || ty == 13 {
                                b[2..4].copy_from_slice(
                                    &half::f16::from_f32(0.00003).to_bits().to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                let dense =
                    crate::ggml::tensor_to_f32(&p, ty, &[cols as u64, rows as u64]).unwrap();
                let x: Vec<f32> = (0..cols * tokens)
                    .map(|i| (i as f32 * 0.71).sin())
                    .collect();
                let modes: Vec<u32> = if ty == 39 {
                    vec![0, 1, 2, 3]
                } else {
                    vec![0, 3]
                };
                for mode in modes {
                    let mut got = vec![f32::NAN; rows * tokens];
                    let mut reference = got.clone();
                    let mut elapsed = 0.0;
                    assert_eq!(
                        unsafe {
                            run(
                                ty,
                                p.as_ptr().cast(),
                                rb,
                                x.as_ptr(),
                                cols as u32,
                                rows as u32,
                                tokens as u32,
                                mode,
                                2,
                                got.as_mut_ptr(),
                                reference.as_mut_ptr(),
                                &mut elapsed,
                            )
                        },
                        0
                    );
                    assert!(elapsed.is_finite() && elapsed >= 0.0);
                    for t in 0..tokens {
                        for r in 0..rows {
                            let i = t * rows + r;
                            assert_eq!(
                                got[i].to_bits(),
                                reference[i].to_bits(),
                                "bit order ty={ty} mode={mode} cols={cols} row={r} token={t}"
                            );
                            let expected = dense[r * cols..(r + 1) * cols]
                                .iter()
                                .zip(&x[t * cols..(t + 1) * cols])
                                .map(|(&w, &x)| w as f64 * x as f64)
                                .sum::<f64>();
                            let scale = dense[r * cols..(r + 1) * cols]
                                .iter()
                                .zip(&x[t * cols..(t + 1) * cols])
                                .map(|(&w, &x)| (w as f64 * x as f64).abs())
                                .sum::<f64>();
                            assert!((got[i] as f64-expected).abs()<=2e-6*scale.max(1e-12),"F64 ty={ty} mode={mode} cols={cols} row={r} token={t}: {} vs {expected}",got[i]);
                        }
                    }
                }
            }
        }
    }
}
