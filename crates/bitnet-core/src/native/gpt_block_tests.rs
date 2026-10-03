// Real device block/serial parity plus independent F64; no mocked GPU.
#[test]
fn gpt_block_all_positions_modes_tails_f64_and_prefix_continuation() {
    if !enabled() {
        return;
    }
    let rt = CudaRuntime::try_load().unwrap();
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let create = unsafe {
        *lib.get::<Create>(b"rbitnet_cuda_gpt_full_create\0")
            .unwrap()
    };
    let destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_gpt_full_destroy\0")
            .unwrap()
    };
    let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_gpt_full_step\0").unwrap() };
    let configure = unsafe {
        *lib.get::<ConfigureBlock>(b"rbitnet_cuda_gpt_configure_prefill\0")
            .unwrap()
    };
    let capacity = unsafe {
        *lib.get::<BlockCapacity>(b"rbitnet_cuda_gpt_prefill_capacity\0")
            .unwrap()
    };
    let prefill = unsafe {
        *lib.get::<BlockStep>(b"rbitnet_cuda_gpt_full_prefill\0")
            .unwrap()
    };
    let verify = unsafe {
        *lib.get::<BlockStep>(b"rbitnet_cuda_gpt_full_verify\0")
            .unwrap()
    };
    let snapshot = unsafe { *lib.get::<Snapshot>(b"rbitnet_cuda_gpt_snapshot\0").unwrap() };
    let restore = unsafe { *lib.get::<Restore>(b"rbitnet_cuda_gpt_restore\0").unwrap() };
    let snapshot_destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_gpt_snapshot_destroy\0")
            .unwrap()
    };
    let frequency: Vec<_> = (0..16)
        .map(|i| 10000f32.powf(-2.0 * i as f32 / 32.0) / 1.3)
        .collect();
    let embedding = |pos: usize, pass: usize| {
        (0..SEG_N)
            .map(|i| (i as f32 * 0.23 + pos as f32 * 0.31 + pass as f32 * 0.8).sin() * 0.9)
            .collect::<Vec<_>>()
    };
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32
    };
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        for (graphs, split, tile) in [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
        ] {
            let mut f = SegmentedFixture::new(&rt, ty, false, false);
            let cfg = NativeConfig {
                embd: SEG_N as u32,
                vocab: 257,
                layers: 2,
                heads: 6,
                kv_heads: 2,
                head_dim: 64,
                rotary: 32,
                capacity: 545,
                window: 12,
                experts: SEG_EXP as u32,
                used: SEG_USED as u32,
                graphs,
                split,
                ordered: crate::ggml::f32_accumulator_lanes().unwrap() as u32,
                epsilon: 1e-5,
                rope_magnitude: 1.13,
                weight_scale: 0.7,
            };
            let descriptors = f.descriptors.as_ptr();
            let head = f.head;
            let norm = f.norm.as_ptr();
            let make = || Handle {
                ptr: unsafe { create(&cfg, descriptors, &head, norm, frequency.as_ptr()) },
                destroy,
            };
            let serial = make();
            let block = make();
            assert!(!serial.ptr.is_null() && !block.ptr.is_null());
            assert_eq!(unsafe { capacity(block.ptr) }, 0);
            assert_ne!(unsafe { configure(block.ptr, 0, tile) }, 0);
            assert_ne!(unsafe { configure(block.ptr, 33, tile) }, 0);
            assert_ne!(unsafe { configure(block.ptr, 16, 2) }, 0);
            assert_eq!(unsafe { configure(block.ptr, 32, tile) }, 0);
            assert_eq!(unsafe { capacity(block.ptr) }, 32);
            assert_ne!(unsafe { configure(block.ptr, 32, tile) }, 0);
            let positions = if (ty == 0 || ty == 39) && graphs == 1 && split == 1 {
                263
            } else {
                43
            };
            let mut saved = None;
            for pass in 0..2 {
                let mut expected = Vec::new();
                let mut inputs = Vec::new();
                for pos in 0..positions {
                    let x = embedding(pos, pass);
                    inputs.extend(&x);
                    let mut out = vec![0.0; 257];
                    let mut token = 0;
                    assert_eq!(
                        unsafe {
                            step(
                                serial.ptr,
                                x.as_ptr(),
                                pos as u32,
                                1,
                                out.as_mut_ptr(),
                                &mut token,
                            )
                        },
                        0
                    );
                    let mut f64x: Vec<_> = x.iter().map(|&v| v as f64).collect();
                    for il in 0..2 {
                        f.oracles[il].forward(
                            &mut f64x,
                            pos,
                            if il == 0 { 12 } else { 0 },
                            &frequency,
                            1.13,
                        );
                    }
                    let oracle = dot(&f.head_dense, SEG_N, &rms(&f64x, &f.norm), &[]);
                    for (i, (&got, &reference)) in out.iter().zip(&oracle).enumerate() {
                        assert!(
                            (got as f64 - reference).abs() < 5e-5 * (1.0 + reference.abs()),
                            "serial F64 ty={ty} pos={pos} row={i}"
                        );
                    }
                    expected.push(out);
                }
                let mut pos = 0;
                let chunks = [1, 7, 16, 32, 3];
                let mut index = 0;
                while pos < positions {
                    let count = chunks[index % chunks.len()].min(positions - pos);
                    let mut got = vec![0.0; count * 257];
                    let mut token = 0;
                    assert_eq!(
                        unsafe {
                            verify(
                                block.ptr,
                                inputs[pos * SEG_N..].as_ptr(),
                                pos as u32,
                                count as u32,
                                1,
                                got.as_mut_ptr(),
                                &mut token,
                            )
                        },
                        0
                    );
                    for (local, row) in got.chunks_exact(257).enumerate() {
                        assert_eq!(row,expected[pos+local].as_slice(),"bit-exact block ty={ty} graph={graphs} split={split} tile={tile} pos={}",pos+local);
                    }
                    pos += count;
                    index += 1;
                    if pos == 24 && pass == 0 {
                        let cp = Handle {
                            ptr: unsafe { snapshot(block.ptr, 24) },
                            destroy: snapshot_destroy,
                        };
                        assert!(!cp.ptr.is_null());
                        saved = Some(cp);
                    }
                }
                // The output head follows Rust max_by (last token wins ties).
                // Expert-router ties have their separate lowest-ID rule.
                let mut pos = 0;
                while pos < positions {
                    let count = 32.min(positions - pos);
                    let mut tokens = vec![0; count];
                    assert_eq!(
                        unsafe {
                            verify(
                                block.ptr,
                                inputs[pos * SEG_N..].as_ptr(),
                                pos as u32,
                                count as u32,
                                2,
                                std::ptr::null_mut(),
                                tokens.as_mut_ptr(),
                            )
                        },
                        0
                    );
                    for (local, &token) in tokens.iter().enumerate() {
                        assert_eq!(token,argmax(&expected[pos+local]),"greedy ty={ty} graphs={graphs} split={split} tile={tile} pass={pass} pos={}",pos+local);
                    }
                    pos += count;
                }
                // Head-free blocks followed by a last-only block then serial decode.
                let mut token = 0;
                let mut out = vec![0.0; 257];
                assert_eq!(
                    unsafe {
                        prefill(
                            block.ptr,
                            inputs.as_ptr(),
                            0,
                            32,
                            0,
                            std::ptr::null_mut(),
                            &mut token,
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        prefill(
                            block.ptr,
                            inputs[32 * SEG_N..].as_ptr(),
                            32,
                            11,
                            1,
                            out.as_mut_ptr(),
                            &mut token,
                        )
                    },
                    0
                );
                assert_eq!(out, expected[42]);
                let x = embedding(43, pass);
                assert_eq!(
                    unsafe { step(block.ptr, x.as_ptr(), 43, 1, out.as_mut_ptr(), &mut token) },
                    0
                );
                let mut expected_out = vec![0.0; 257];
                // Reset serial and replay to the same position for every long/tail case.
                for p in 0..=43 {
                    let x = embedding(p, pass);
                    assert_eq!(
                        unsafe {
                            step(
                                serial.ptr,
                                x.as_ptr(),
                                p as u32,
                                1,
                                expected_out.as_mut_ptr(),
                                &mut token,
                            )
                        },
                        0
                    );
                }
                assert_eq!(out, expected_out);
            }
            let saved = saved.unwrap();
            assert_eq!(unsafe { restore(block.ptr, saved.ptr, 13) }, 0);
            let input: Vec<_> = (13..20).flat_map(|p| embedding(p, 0)).collect();
            let mut replay = vec![0.0; 7 * 257];
            let mut token = 0;
            assert_eq!(
                unsafe {
                    verify(
                        block.ptr,
                        input.as_ptr(),
                        13,
                        7,
                        1,
                        replay.as_mut_ptr(),
                        &mut token,
                    )
                },
                0
            );
            let mut serial_out = vec![0.0; 257];
            for p in 0..20 {
                let x = embedding(p, 0);
                assert_eq!(
                    unsafe {
                        step(
                            serial.ptr,
                            x.as_ptr(),
                            p as u32,
                            1,
                            serial_out.as_mut_ptr(),
                            &mut token,
                        )
                    },
                    0
                );
                if p >= 13 {
                    assert_eq!(
                        &replay[(p - 13) * 257..(p - 12) * 257],
                        serial_out.as_slice()
                    );
                }
            }
            assert_ne!(
                unsafe {
                    prefill(
                        block.ptr,
                        input.as_ptr(),
                        21,
                        1,
                        1,
                        serial_out.as_mut_ptr(),
                        &mut token,
                    )
                },
                0
            );
            assert_ne!(
                unsafe {
                    prefill(
                        block.ptr,
                        input.as_ptr(),
                        20,
                        0,
                        1,
                        serial_out.as_mut_ptr(),
                        &mut token,
                    )
                },
                0
            );
            assert_ne!(
                unsafe {
                    prefill(
                        block.ptr,
                        input.as_ptr(),
                        20,
                        1,
                        3,
                        serial_out.as_mut_ptr(),
                        &mut token,
                    )
                },
                0
            );
            assert_ne!(
                unsafe {
                    prefill(
                        block.ptr,
                        input.as_ptr(),
                        540,
                        7,
                        1,
                        serial_out.as_mut_ptr(),
                        &mut token,
                    )
                },
                0
            );
            // Refusals do not poison the next valid continuation.
            let x = embedding(20, 0);
            assert_eq!(
                unsafe {
                    prefill(
                        block.ptr,
                        x.as_ptr(),
                        20,
                        1,
                        2,
                        std::ptr::null_mut(),
                        &mut token,
                    )
                },
                0
            );
            assert_ne!(unsafe { configure(block.ptr, 8, tile) }, 0);
            drop(saved);
            drop(block);
            drop(serial);
            drop(f);
        }
    }
}
