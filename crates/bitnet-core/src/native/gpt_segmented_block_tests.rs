// Segmented GPT block prefill parity on device; fixed MoE banks for native FFN loop.
#[test]
fn gpt_segmented_block_prefill_matches_serial_segmented() {
    if !enabled() {
        return;
    }
    let rt = CudaRuntime::try_load().unwrap();
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let create = unsafe {
        *lib.get::<Create>(b"rbitnet_cuda_gpt_segmented_create\0")
            .unwrap()
    };
    let destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_gpt_full_destroy\0")
            .unwrap()
    };
    let begin = unsafe {
        *lib.get::<Begin>(b"rbitnet_cuda_gpt_segmented_begin\0")
            .unwrap()
    };
    let prepare = unsafe {
        *lib.get::<Prepare>(b"rbitnet_cuda_gpt_segmented_prepare\0")
            .unwrap()
    };
    let finish = unsafe {
        *lib.get::<Finish>(b"rbitnet_cuda_gpt_segmented_finish\0")
            .unwrap()
    };
    let end = unsafe {
        *lib.get::<End>(b"rbitnet_cuda_gpt_segmented_end\0")
            .unwrap()
    };
    let configure = unsafe {
        *lib.get::<ConfigureBlock>(b"rbitnet_cuda_gpt_segmented_configure_prefill\0")
            .unwrap()
    };
    let capacity = unsafe {
        *lib.get::<BlockCapacity>(b"rbitnet_cuda_gpt_segmented_prefill_capacity\0")
            .unwrap()
    };
    let block_prefill = unsafe {
        *lib.get::<BlockStep>(b"rbitnet_cuda_gpt_segmented_prefill\0")
            .unwrap()
    };
    let frequency: Vec<_> = (0..16)
        .map(|i| 10000f32.powf(-2.0 * i as f32 / 32.0) / 1.3)
        .collect();
    let ty = 0;
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
        graphs: 0,
        split: 0,
        ordered: crate::ggml::f32_accumulator_lanes().unwrap() as u32,
        epsilon: 1e-5,
        rope_magnitude: 1.13,
        weight_scale: 0.7,
    };
    let serial = Handle {
        ptr: unsafe {
            create(
                &cfg,
                f.descriptors.as_ptr(),
                &f.head,
                f.norm.as_ptr(),
                frequency.as_ptr(),
            )
        },
        destroy,
    };
    let block = Handle {
        ptr: unsafe {
            create(
                &cfg,
                f.descriptors.as_ptr(),
                &f.head,
                f.norm.as_ptr(),
                frequency.as_ptr(),
            )
        },
        destroy,
    };
    assert!(!serial.ptr.is_null() && !block.ptr.is_null());
    assert_eq!(unsafe { configure(block.ptr, 16, 0) }, 0);
    assert_eq!(unsafe { capacity(block.ptr) }, 16);
    let positions = 24;
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for pos in 0..positions {
        let x: Vec<f32> = (0..SEG_N)
            .map(|i| (i as f32 * 0.23 + pos as f32 * 0.31).sin() * 0.9)
            .collect();
        inputs.extend(&x);
        assert_eq!(unsafe { begin(serial.ptr, x.as_ptr(), pos as u32) }, 0);
        for il in 0..2 {
            let mut ids = vec![0u32; SEG_USED];
            let mut probs = vec![0.0f32; SEG_USED];
            assert_eq!(
                unsafe { prepare(serial.ptr, il as u32, ids.as_mut_ptr(), probs.as_mut_ptr()) },
                0
            );
            assert_eq!(
                unsafe { finish(serial.ptr, il as u32, std::ptr::null(), std::ptr::null()) },
                0
            );
        }
        let mut logits = vec![0.0; 257];
        assert_eq!(unsafe { end(serial.ptr, 1, logits.as_mut_ptr(), std::ptr::null_mut()) }, 0);
        expected.push(logits);
    }
    let mut pos = 0;
    while pos < positions {
        let count = 8.min(positions - pos);
        let mut got = vec![0.0; count * 257];
        assert_eq!(
            unsafe {
                block_prefill(
                    block.ptr,
                    inputs[pos * SEG_N..].as_ptr(),
                    pos as u32,
                    count as u32,
                    1,
                    got.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            0
        );
        for (local, row) in got.chunks_exact(257).enumerate() {
            assert_eq!(
                row,
                expected[pos + local].as_slice(),
                "segmented block ty={ty} pos={}",
                pos + local
            );
        }
        pos += count;
    }
    drop(block);
    drop(serial);
    eprintln!("GPT segmented block prefill F64 passed format {ty}");
}
