// Independent F64 numerical and state lifecycle coverage for host-admitted GPT.
// Included in gpt_full.rs::tests to reuse the existing quant fixture decoder.
type SegmentedHidden = unsafe extern "C" fn(*mut c_void, *mut f32) -> i32;
const SEG_N: usize = 256;
const SEG_EXP: usize = 5;
const SEG_USED: usize = 3;

struct SegmentedFixture {
    // Borrowed MoE contexts must drop before their backing allocations.
    _moes: Vec<Handle>,
    _owned: Vec<CudaDeviceQuantMatrix>,
    _output: CudaDeviceQuantMatrix,
    descriptors: Vec<Layer>,
    banks: Vec<[Matrix; 3]>,
    oracles: Vec<OracleLayer>,
    head: Matrix,
    head_dense: Vec<f32>,
    norm: Vec<f32>,
}
impl SegmentedFixture {
    fn new(rt: &std::sync::Arc<CudaRuntime>, ty: u32, dynamic: bool, no_moe: bool) -> Self {
        let lib = crate::ggml::load_cuda_quant_library().unwrap();
        let create = unsafe {
            *lib.get::<MoeCreate>(if dynamic {
                b"rbitnet_cuda_moe_dynamic_create\0"
            } else {
                b"rbitnet_cuda_moe_create\0"
            })
            .unwrap()
        };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").unwrap() };
        let mut result = Self {
            _moes: Vec::new(),
            _owned: Vec::new(),
            _output: CudaDeviceQuantMatrix::from_payload(
                Some(rt),
                ty,
                payload(ty, SEG_N, 257, 29),
                257,
                SEG_N,
            )
            .unwrap(),
            descriptors: Vec::new(),
            banks: Vec::new(),
            oracles: Vec::new(),
            head: Matrix {
                weights: std::ptr::null(),
                row_bytes: 0,
                ty: 0,
                cols: 0,
                rows: 0,
            },
            head_dense: crate::ggml::tensor_to_f32(
                &payload(ty, SEG_N, 257, 29),
                ty,
                &[SEG_N as u64, 257],
            )
            .unwrap(),
            norm: vec![1.0; SEG_N],
        };
        result.head = Matrix::from_device(&result._output).unwrap();
        for il in 0..2 {
            let cols = [SEG_N, SEG_N, SEG_N, 384, SEG_N, SEG_N, SEG_N, SEG_N];
            let rows = [
                384,
                128,
                128,
                SEG_N,
                SEG_EXP,
                SEG_N * SEG_EXP,
                SEG_N * SEG_EXP,
                SEG_N * SEG_EXP,
            ];
            let mut matrices = Vec::new();
            let mut weights = Vec::new();
            let mut biases = Vec::new();
            for p in 0..8 {
                let format = if p == 3 || p == 4 { 0 } else { ty };
                let bytes = payload(format, cols[p], rows[p], p + il * 11);
                weights.push(
                    crate::ggml::tensor_to_f32(&bytes, format, &[cols[p] as u64, rows[p] as u64])
                        .unwrap(),
                );
                let m =
                    CudaDeviceQuantMatrix::from_payload(Some(rt), format, bytes, rows[p], cols[p])
                        .unwrap();
                matrices.push(Matrix::from_device(&m).unwrap());
                result._owned.push(m);
                biases.push(
                    (0..rows[p])
                        .map(|i| {
                            if p == 4 {
                                i as f32 * 0.41
                            } else {
                                (i as f32 * 0.17 + p as f32).sin()
                                    * if p == 5 || p == 6 { 8.0 } else { 0.07 }
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let norm = |phase: f32| {
                (0..SEG_N)
                    .map(|i| 1.0 + (i as f32 * 0.13 + phase).sin() * 0.04)
                    .collect::<Vec<_>>()
            };
            let oracle = OracleLayer {
                weights,
                biases,
                an: norm(il as f32),
                fnorm: norm(il as f32 + 2.0),
                sinks: (0..6).map(|i| i as f32 * 0.9 - 2.0).collect(),
                selection: vec![0.1, -0.2, 0.3, 0.0, -0.1],
                keys: Vec::new(),
                values: Vec::new(),
            };
            let banks = [matrices[5], matrices[6], matrices[7]];
            let mut borrowed = banks;
            if dynamic {
                for m in &mut borrowed {
                    m.weights = std::ptr::null();
                }
            }
            let mc = MoeConfig {
                embd: SEG_N as u32,
                ffn: SEG_N as u32,
                experts: SEG_EXP as u32,
                used: SEG_USED as u32,
                oai: 1,
            };
            let ptr = if no_moe {
                std::ptr::null_mut()
            } else {
                unsafe {
                    create(
                        &mc,
                        &borrowed[0],
                        &borrowed[1],
                        &borrowed[2],
                        oracle.biases[5].as_ptr(),
                        oracle.biases[6].as_ptr(),
                        oracle.biases[7].as_ptr(),
                    )
                }
            };
            if !no_moe {
                assert!(!ptr.is_null());
                result._moes.push(Handle { ptr, destroy });
            }
            result.descriptors.push(Layer {
                q: matrices[0],
                k: matrices[1],
                v: matrices[2],
                out: matrices[3],
                router: matrices[4],
                attn_norm: oracle.an.as_ptr(),
                ffn_norm: oracle.fnorm.as_ptr(),
                q_bias: oracle.biases[0].as_ptr(),
                k_bias: oracle.biases[1].as_ptr(),
                v_bias: oracle.biases[2].as_ptr(),
                out_bias: oracle.biases[3].as_ptr(),
                router_bias: oracle.biases[4].as_ptr(),
                selection_bias: oracle.selection.as_ptr(),
                sinks: oracle.sinks.as_ptr(),
                moe: ptr,
            });
            result.banks.push(banks);
            result.oracles.push(oracle);
        }
        result
    }
    fn pointers(&self, layer: usize, ids: &[u32]) -> Vec<*const c_void> {
        self.banks[layer]
            .iter()
            .flat_map(|matrix| {
                ids.iter().map(move |&id| unsafe {
                    (matrix.weights as *const u8).add(id as usize * SEG_N * matrix.row_bytes)
                        as *const c_void
                })
            })
            .collect()
    }
    fn cpu(&self, layer: usize, input: &[f32], ids: &[u32], probs: &[f32]) -> Vec<f32> {
        let o = &self.oracles[layer];
        let x: Vec<_> = input.iter().map(|&x| x as f64).collect();
        let mut out = vec![0.0; SEG_N];
        for (&expert, &p) in ids.iter().zip(probs) {
            let e = expert as usize;
            let span = e * SEG_N * SEG_N..(e + 1) * SEG_N * SEG_N;
            let bias = e * SEG_N..(e + 1) * SEG_N;
            let gate = dot(
                &o.weights[5][span.clone()],
                SEG_N,
                &x,
                &o.biases[5][bias.clone()],
            );
            let up = dot(
                &o.weights[6][span.clone()],
                SEG_N,
                &x,
                &o.biases[6][bias.clone()],
            );
            let hidden: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| {
                    let g = g.min(7.0);
                    g / (1.0 + (-1.702f32 as f64 * g).exp()) * (u.clamp(-7.0, 7.0) + 1.0)
                })
                .collect();
            let down = dot(&o.weights[7][span], SEG_N, &hidden, &o.biases[7][bias]);
            for (out, d) in out.iter_mut().zip(down) {
                *out += p as f64 * d;
            }
        }
        out.into_iter().map(|x| x as f32).collect()
    }
}

#[test]
fn segmented_gpt_f64_dynamic_and_cpu_graphs_prefix_generation_and_cancel() {
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
    let input = unsafe {
        *lib.get::<FfnInput>(b"rbitnet_cuda_gpt_segmented_ffn_input\0")
            .unwrap()
    };
    let finish = unsafe {
        *lib.get::<Finish>(b"rbitnet_cuda_gpt_segmented_finish\0")
            .unwrap()
    };
    let end = unsafe { *lib.get::<End>(b"rbitnet_cuda_gpt_segmented_end\0").unwrap() };
    let hidden = unsafe {
        *lib.get::<SegmentedHidden>(b"rbitnet_cuda_gpt_segmented_hidden_check\0")
            .unwrap()
    };
    let snapshot = unsafe { *lib.get::<Snapshot>(b"rbitnet_cuda_gpt_snapshot\0").unwrap() };
    let snapshot_destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_gpt_snapshot_destroy\0")
            .unwrap()
    };
    let restore = unsafe { *lib.get::<Restore>(b"rbitnet_cuda_gpt_restore\0").unwrap() };
    let frequency: Vec<_> = (0..16)
        .map(|i| 10000f32.powf(-2.0 * i as f32 / 32.0) / 1.3)
        .collect();
    let embedding = |pos: usize, pass: usize| {
        (0..SEG_N)
            .map(|i| (i as f32 * 0.23 + pos as f32 * 0.31 + pass as f32 * 0.8).sin() * 0.9)
            .collect::<Vec<_>>()
    };
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        for (graphs, split, dynamic, force_cpu, no_moe) in [
            (0, 0, false, false, false),
            (1, 1, false, false, false),
            (1, 0, true, false, false),
            (1, 1, true, false, false),
            (1, 1, true, true, false),
            (1, 1, false, true, true),
        ] {
            let mut f = SegmentedFixture::new(&rt, ty, dynamic, no_moe);
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
            // Heap buffers stay owned by f; capture raw descriptors so the
            // independent oracle's K/V vectors can mutate during execution.
            let descriptor_ptr = f.descriptors.as_ptr();
            let head = f.head;
            let norm_ptr = f.norm.as_ptr();
            let make = || Handle {
                ptr: unsafe { create(&cfg, descriptor_ptr, &head, norm_ptr, frequency.as_ptr()) },
                destroy,
            };
            let context = make();
            assert!(!context.ptr.is_null());
            let ptr = context.ptr;
            let mut ids = vec![0; SEG_USED];
            let mut probs = vec![0.0; SEG_USED];
            let mut id = 0;
            let mut hidden_buffer = vec![0.0; SEG_N];
            assert_ne!(unsafe { hidden(ptr, hidden_buffer.as_mut_ptr()) }, 0);
            // No preparation or output can consume an uninitialized token.
            assert_ne!(
                unsafe { prepare(ptr, 0, ids.as_mut_ptr(), probs.as_mut_ptr()) },
                0
            );
            assert_ne!(unsafe { end(ptr, 0, std::ptr::null_mut(), &mut id) }, 0);
            let mut saved = None;
            let mut saved_logits = Vec::new();
            let positions = if ty == 0 && split == 1 && dynamic && !force_cpu {
                263
            } else {
                16
            };
            for pass in 0..2 {
                for pos in 0..positions {
                    let x = embedding(pos, pass);
                    let mut expected: Vec<_> = x.iter().map(|&x| x as f64).collect();
                    assert_eq!(unsafe { begin(ptr, x.as_ptr(), pos as u32) }, 0);
                    assert_ne!(unsafe { hidden(ptr, hidden_buffer.as_mut_ptr()) }, 0);
                    for il in 0..2 {
                        let (expected_ids, expected_p, expected_h) = f.oracles[il].forward(
                            &mut expected,
                            pos,
                            if il == 0 { 12 } else { 0 },
                            &frequency,
                            1.13,
                        );
                        assert_ne!(
                            unsafe { finish(ptr, il as u32, std::ptr::null(), std::ptr::null()) },
                            0
                        );
                        assert_eq!(
                            unsafe {
                                prepare(ptr, il as u32, ids.as_mut_ptr(), probs.as_mut_ptr())
                            },
                            0
                        );
                        assert_ne!(
                            unsafe {
                                prepare(ptr, il as u32, ids.as_mut_ptr(), probs.as_mut_ptr())
                            },
                            0
                        );
                        assert_eq!(
                            ids,
                            expected_ids.iter().map(|&i| i as u32).collect::<Vec<_>>()
                        );
                        for (&a, &b) in probs.iter().zip(&expected_p) {
                            assert!(
                                (a as f64 - b).abs() < 2e-6,
                                "router ty={ty} pos={pos} layer={il}: {a} vs {b}"
                            );
                        }
                        let mut h = vec![0.0; SEG_N];
                        assert_eq!(unsafe { input(ptr, h.as_mut_ptr()) }, 0);
                        for (&a, &b) in h.iter().zip(&expected_h) {
                            assert!(
                                (a as f64 - b).abs() < 5e-5 * (1.0 + b.abs()),
                                "FFN input ty={ty} pos={pos} layer={il}: {a} vs {b}"
                            );
                        }
                        let cpu = force_cpu || pass == 1 && (pos + il) % 2 == 1;
                        let routed = if cpu {
                            f.cpu(il, &h, &ids, &probs)
                        } else {
                            Vec::new()
                        };
                        let pointers = if !cpu && dynamic {
                            f.pointers(il, &ids)
                        } else {
                            Vec::new()
                        };
                        assert_eq!(
                            unsafe {
                                finish(
                                    ptr,
                                    il as u32,
                                    if pointers.is_empty() {
                                        std::ptr::null()
                                    } else {
                                        pointers.as_ptr()
                                    },
                                    if cpu {
                                        routed.as_ptr()
                                    } else {
                                        std::ptr::null()
                                    },
                                )
                            },
                            0
                        );
                    }
                    let mut logits = vec![0.0; 257];
                    assert_eq!(unsafe { end(ptr, 1, logits.as_mut_ptr(), &mut id) }, 0);
                    let mut got = vec![0.0; SEG_N];
                    assert_eq!(unsafe { hidden(ptr, got.as_mut_ptr()) }, 0);
                    for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                        assert!((a as f64-b).abs()<5e-5*(1.0+b.abs()),"hidden ty={ty} graphs={graphs} split={split} CPU={force_cpu} pos={pos} pass={pass} row={i}: {a} vs {b}");
                    }
                    let expected = dot(&f.head_dense, SEG_N, &rms(&expected, &f.norm), &[]);
                    for (i, (&a, &b)) in logits.iter().zip(&expected).enumerate() {
                        assert!(
                            (a as f64 - b).abs() < 5e-5 * (1.0 + b.abs()),
                            "logits ty={ty} pos={pos} row={i}: {a} vs {b}"
                        );
                    }
                    if pos == 5 && pass == 0 {
                        saved_logits = logits;
                    }
                    if pos == 7 && pass == 0 {
                        let cp = Handle {
                            ptr: unsafe { snapshot(ptr, 8) },
                            destroy: snapshot_destroy,
                        };
                        assert!(!cp.ptr.is_null());
                        saved = Some(cp);
                    }
                }
            }
            let cp = saved.unwrap();
            assert_ne!(unsafe { restore(ptr, cp.ptr, 9) }, 0);
            assert_eq!(unsafe { restore(ptr, cp.ptr, 5) }, 0);
            assert_ne!(unsafe { hidden(ptr, hidden_buffer.as_mut_ptr()) }, 0);
            // A restored prefix is valid KV even though it has no output yet.
            let restored = Handle {
                ptr: unsafe { snapshot(ptr, 5) },
                destroy: snapshot_destroy,
            };
            assert!(!restored.ptr.is_null());
            drop(restored);
            let x = embedding(5, 0);
            assert_ne!(unsafe { begin(ptr, x.as_ptr(), 6) }, 0);
            assert_eq!(unsafe { begin(ptr, x.as_ptr(), 5) }, 0);
            let mut logits = vec![0.0; 257];
            // Replaying the immutable truncated prefix after overwrite is exact.
            for mode in [1, 2, 0] {
                if mode != 1 {
                    assert_eq!(unsafe { begin(ptr, x.as_ptr(), 5) }, 0);
                }
                for il in 0..2 {
                    assert_eq!(
                        unsafe { prepare(ptr, il as u32, ids.as_mut_ptr(), probs.as_mut_ptr()) },
                        0
                    );
                    let pointers = if dynamic && !force_cpu {
                        f.pointers(il, &ids)
                    } else {
                        Vec::new()
                    };
                    let routed = if force_cpu {
                        let mut h = vec![0.0; SEG_N];
                        assert_eq!(unsafe { input(ptr, h.as_mut_ptr()) }, 0);
                        f.cpu(il, &h, &ids, &probs)
                    } else {
                        Vec::new()
                    };
                    assert_eq!(
                        unsafe {
                            finish(
                                ptr,
                                il as u32,
                                if pointers.is_empty() {
                                    std::ptr::null()
                                } else {
                                    pointers.as_ptr()
                                },
                                if routed.is_empty() {
                                    std::ptr::null()
                                } else {
                                    routed.as_ptr()
                                },
                            )
                        },
                        0
                    );
                }
                assert_eq!(unsafe { end(ptr, mode, logits.as_mut_ptr(), &mut id) }, 0);
                assert_eq!(unsafe { hidden(ptr, hidden_buffer.as_mut_ptr()) }, 0);
                if mode == 1 {
                    assert_eq!(logits, saved_logits);
                }
                if mode == 2 {
                    assert_eq!(
                        id,
                        saved_logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0 as u32
                    );
                }
            }
            assert_eq!(unsafe { begin(ptr, x.as_ptr(), 0) }, 0);
            assert_eq!(
                unsafe { prepare(ptr, 0, ids.as_mut_ptr(), probs.as_mut_ptr()) },
                0
            );
            assert!(unsafe { snapshot(ptr, 1) }.is_null());
            assert_ne!(unsafe { end(ptr, 1, logits.as_mut_ptr(), &mut id) }, 0);
            // Restore safely abandons a cancelled prepared token; begin zero also resets.
            assert_eq!(unsafe { restore(ptr, cp.ptr, 5) }, 0);
            assert_eq!(unsafe { begin(ptr, x.as_ptr(), 0) }, 0);
            assert_eq!(
                unsafe { prepare(ptr, 0, ids.as_mut_ptr(), probs.as_mut_ptr()) },
                0
            );
            assert_eq!(unsafe { begin(ptr, x.as_ptr(), 0) }, 0);
            let alien = make();
            assert!(!alien.ptr.is_null());
            assert_ne!(unsafe { restore(alien.ptr, cp.ptr, 5) }, 0);
            drop(alien);
            // The original snapshot survives context destruction, but must never
            // become valid for a new context even if malloc reuses the address.
            drop(context);
            let successor = make();
            assert!(!successor.ptr.is_null());
            assert_ne!(unsafe { restore(successor.ptr, cp.ptr, 5) }, 0);
            drop(successor);
            drop(cp);
        }
        eprintln!("GPT segmented F64 passed format {ty}: fixed/dynamic/CPU/absent MoE, graph/eager, split, prefix and generation identity");
    }
}
