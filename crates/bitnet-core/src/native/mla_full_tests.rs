//! Independent F64 oracles for compressed MLA and unchanged quantized FFNs.
use super::*;
type Router = unsafe extern "C" fn(
    *const f32,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    *mut u32,
    *mut f32,
) -> i32;
type Attention = unsafe extern "C" fn(
    *const f32,
    *const f32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
    u32,
    *mut f32,
) -> i32;
type Hidden = unsafe extern "C" fn(*mut c_void, *mut f32) -> i32;
#[repr(C)]
struct MoeConfig {
    embd: u32,
    ffn: u32,
    experts: u32,
    used: u32,
    oai: u32,
}
type MoeCreate = unsafe extern "C" fn(
    *const MoeConfig,
    *const Matrix,
    *const Matrix,
    *const Matrix,
    *const f32,
    *const f32,
    *const f32,
) -> *mut c_void;
struct Handle {
    ptr: *mut c_void,
    destroy: Destroy,
}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.ptr) }
    }
}
fn enabled() -> bool {
    std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() == Ok("1")
}
fn payload(ty: u32, cols: usize, rows: usize, seed: usize) -> Vec<u8> {
    let rb = crate::ggml::ggml_row_size(ty, cols as u64).unwrap();
    let mut p: Vec<_> = (0..rb * rows)
        .map(|i| (i * 37 + seed * 53 + 19) as u8)
        .collect();
    if ty == 0 {
        for (i, b) in p.chunks_exact_mut(4).enumerate() {
            b.copy_from_slice(&((i as f32 * 0.031 + seed as f32).cos() * 0.015).to_le_bytes());
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
        for b in p.chunks_exact_mut(size) {
            if ty == 39 {
                b[0] = 119;
            } else {
                let offset = if ty == 14 { 208 } else { 0 };
                b[offset..offset + 2].copy_from_slice(
                    &half::f16::from_f32(if ty == 14 { 0.00003 } else { 0.00007 })
                        .to_bits()
                        .to_le_bytes(),
                );
                if ty == 12 || ty == 13 {
                    b[2..4].copy_from_slice(&half::f16::from_f32(0.00003).to_bits().to_le_bytes());
                }
            }
        }
    }
    p
}
fn dot(w: &[f32], cols: usize, x: &[f64]) -> Vec<f64> {
    w.chunks_exact(cols)
        .map(|r| r.iter().zip(x).map(|(&w, &x)| w as f64 * x).sum())
        .collect()
}
fn rms(x: &[f64], w: &[f32]) -> Vec<f64> {
    let inv = (x.iter().map(|x| x * x).sum::<f64>() / x.len() as f64 + 1e-5f32 as f64)
        .sqrt()
        .recip();
    x.iter().zip(w).map(|(&x, &w)| x * inv * w as f64).collect()
}
fn selected(
    raw: &[f64],
    bias: &[f32],
    sigmoid: bool,
    groups: usize,
    groups_used: usize,
    used: usize,
    normalize: bool,
    scale: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut p: Vec<_> = if sigmoid {
        raw.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    } else {
        let m = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p: Vec<_> = raw.iter().map(|&x| (x - m).exp()).collect();
        let sum = p.iter().sum::<f64>();
        for p in &mut p {
            *p /= sum;
        }
        p
    };
    let mut scores: Vec<_> = p
        .iter()
        .enumerate()
        .map(|(i, &p)| p + bias.get(i).copied().unwrap_or(0.0) as f64)
        .collect();
    if groups > 1 {
        let width = p.len() / groups;
        let mut ranked: Vec<_> = (0..groups)
            .map(|g| {
                let mut s = scores[g * width..(g + 1) * width].to_vec();
                s.sort_by(|a, b| b.total_cmp(a));
                (g, s[0] + s.get(1).copied().unwrap_or(0.0))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for g in 0..groups {
            if !ranked[..groups_used].iter().any(|&(i, _)| i == g) {
                scores[g * width..(g + 1) * width].fill(f64::NEG_INFINITY);
            }
        }
    }
    let mut ids: Vec<_> = (0..p.len()).collect();
    ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.truncate(used);
    let sum = ids.iter().map(|&i| p[i]).sum::<f64>().max(1.0 / 16384.0);
    for p in &mut p {
        *p = if normalize {
            *p / sum * scale
        } else {
            *p * scale
        };
    }
    let weights = ids.iter().map(|&i| p[i]).collect();
    (ids, weights)
}
#[test]
fn compressed_attention_matches_f64_with_different_key_value_widths_at_tile_boundaries() {
    if !enabled() {
        return;
    }
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let run = unsafe {
        *lib.get::<Attention>(b"rbitnet_cuda_mla_attention_check\0")
            .unwrap()
    };
    const CAP: usize = 513;
    const HEADS: usize = 3;
    for (rank, rotary) in [(32, 16), (128, 32), (512, 64)] {
        let width = rank + rotary;
        let cache: Vec<f32> = (0..CAP * width)
            .map(|i| (i as f32 * 0.071).sin() * 0.3)
            .collect();
        let query: Vec<f32> = (0..HEADS * width)
            .map(|i| (i as f32 * 0.037).cos() * 0.2)
            .collect();
        let scale = 1.0 / (width as f32).sqrt();
        for pos in [0, 1, 31, 255, 256, 511, 512] {
            let expected: Vec<f64> = (0..HEADS)
                .flat_map(|h| {
                    let q = &query[h * width..(h + 1) * width];
                    let scores: Vec<f64> = (0..=pos)
                        .map(|p| {
                            q.iter()
                                .zip(&cache[p * width..(p + 1) * width])
                                .map(|(&q, &k)| q as f64 * k as f64)
                                .sum::<f64>()
                                * scale as f64
                        })
                        .collect();
                    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities: Vec<_> = scores.iter().map(|s| (s - m).exp()).collect();
                    let total = probabilities.iter().sum::<f64>();
                    (0..rank)
                        .map(|i| {
                            probabilities
                                .iter()
                                .enumerate()
                                .map(|(p, &s)| s / total * cache[p * width + i] as f64)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            for split in [0, 1] {
                let mut out = vec![0.0; HEADS * rank];
                assert_eq!(
                    unsafe {
                        run(
                            cache.as_ptr(),
                            query.as_ptr(),
                            CAP as u32,
                            HEADS as u32,
                            rank as u32,
                            rotary as u32,
                            scale,
                            pos as u32,
                            split,
                            out.as_mut_ptr(),
                        )
                    },
                    0
                );
                for (i, (&a, &b)) in out.iter().zip(&expected).enumerate() {
                    assert!(
                        (a as f64 - b).abs() < 2e-6 * (1.0 + b.abs()),
                        "rank={rank} rot={rotary} pos={pos} split={split} row={i}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
#[test]
fn sigmoid_softmax_group_selection_preserves_ids_ties_bias_and_normalization() {
    if !enabled() {
        return;
    }
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let run = unsafe {
        *lib.get::<Router>(b"rbitnet_cuda_mla_router_check\0")
            .unwrap()
    };
    let cases = [
        vec![3.0, 3.0, 1.0, 2.0, -2.0, -1.0, 2.0, 2.0],
        vec![-0.0, 0.0, -0.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        vec![10.0, -8.0, 0.5, -0.25, 1.0, 5.0, -3.0, -4.0],
    ];
    for raw in cases {
        for sigmoid in [0, 1] {
            for groups in [1, 2, 4] {
                for normalize in [0, 1] {
                    for biased in [false, true] {
                        let bias: Vec<f32> = (0..raw.len())
                            .map(|i| if i % 3 == 0 { 0.2 } else { -0.1 })
                            .collect();
                        let (ids, p) = selected(
                            &raw.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                            if biased { &bias } else { &[] },
                            sigmoid == 1,
                            groups,
                            1,
                            2,
                            normalize == 1,
                            0.7f32 as f64,
                        );
                        let mut actual = vec![0; 2];
                        let mut probabilities = vec![0.0; 2];
                        assert_eq!(
                            unsafe {
                                run(
                                    raw.as_ptr(),
                                    if biased {
                                        bias.as_ptr()
                                    } else {
                                        std::ptr::null()
                                    },
                                    raw.len() as u32,
                                    2,
                                    groups as u32,
                                    1,
                                    sigmoid,
                                    normalize,
                                    0.7,
                                    actual.as_mut_ptr(),
                                    probabilities.as_mut_ptr(),
                                )
                            },
                            0
                        );
                        assert_eq!(actual,ids.iter().map(|&i|i as u32).collect::<Vec<_>>(),"raw={raw:?} sigmoid={sigmoid} groups={groups} normalize={normalize} biased={biased}");
                        for (a, b) in probabilities.iter().zip(p) {
                            assert!((*a as f64 - b).abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }
}
struct Oracle {
    weights: Vec<Vec<f32>>,
    an: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    fnorm: Vec<f32>,
    bias: Vec<f32>,
    cache: Vec<f64>,
}
const N: usize = 256;
const HEADS: usize = 2;
const RANK: usize = 256;
const ROT: usize = 64;
const DIM: usize = 320;
const EXP: usize = 4;
const USED: usize = 2;
impl Oracle {
    fn routed(&self, h: &[f64], ids: &[usize], weights: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; N];
        for (&e, &p) in ids.iter().zip(weights) {
            let gate = dot(&self.weights[10][e * N * N..(e + 1) * N * N], N, h);
            let up = dot(&self.weights[11][e * N * N..(e + 1) * N * N], N, h);
            let hidden: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            for (o, v) in out.iter_mut().zip(dot(
                &self.weights[12][e * N * N..(e + 1) * N * N],
                N,
                &hidden,
            )) {
                *o += p * v;
            }
        }
        out
    }
    fn forward(
        &mut self,
        x: &mut [f64],
        pos: usize,
        phases: &[f32],
        dense: bool,
        sigmoid: bool,
        groups: usize,
        normalize: bool,
    ) -> (Vec<usize>, Vec<f64>) {
        let h = rms(x, &self.an);
        let qa = dot(&self.weights[0], N, &h);
        let qa = rms(&qa, &self.qn);
        let mut q = dot(&self.weights[1], N, &qa);
        let kv = dot(&self.weights[2], N, &h);
        let latent = rms(&kv[..RANK], &self.kn);
        let mut key = latent;
        for i in 0..ROT / 2 {
            let s = phases[pos * ROT + 2 * i] as f64;
            let c = phases[pos * ROT + 2 * i + 1] as f64;
            let a = kv[RANK + 2 * i];
            let b = kv[RANK + 2 * i + 1];
            key.extend([
                (a * c - b * s) * 1.13f32 as f64,
                (a * s + b * c) * 1.13f32 as f64,
            ]);
            for h in 0..HEADS {
                let j = h * DIM + DIM - ROT + 2 * i;
                let a = q[j];
                let b = q[j + 1];
                q[j] = (a * c - b * s) * 1.13f32 as f64;
                q[j + 1] = (a * s + b * c) * 1.13f32 as f64;
            }
        }
        self.cache.truncate(pos * (RANK + ROT));
        self.cache.extend(key);
        let mut attended = Vec::new();
        for h in 0..HEADS {
            let mut query = dot(
                &self.weights[3][h * RANK * N..(h + 1) * RANK * N],
                N,
                &q[h * DIM..h * DIM + N],
            );
            query.extend_from_slice(&q[h * DIM + N..(h + 1) * DIM]);
            let scores: Vec<_> = (0..=pos)
                .map(|p| {
                    query
                        .iter()
                        .zip(&self.cache[p * (RANK + ROT)..(p + 1) * (RANK + ROT)])
                        .map(|(q, k)| q * k)
                        .sum::<f64>()
                        / (DIM as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probs: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let total = probs.iter().sum::<f64>();
            let value: Vec<_> = (0..RANK)
                .map(|i| {
                    probs
                        .iter()
                        .enumerate()
                        .map(|(p, s)| s / total * self.cache[p * (RANK + ROT) + i])
                        .sum()
                })
                .collect();
            attended.extend(dot(
                &self.weights[4][h * N * RANK..(h + 1) * N * RANK],
                RANK,
                &value,
            ));
        }
        for (x, a) in x
            .iter_mut()
            .zip(dot(&self.weights[5], HEADS * N, &attended))
        {
            *x += a;
        }
        let h = rms(x, &self.fnorm);
        let gate = dot(&self.weights[7], N, &h);
        let up = dot(&self.weights[8], N, &h);
        let hidden: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let shared = dot(&self.weights[9], N, &hidden);
        let (ids, p) = if dense {
            (vec![], vec![])
        } else {
            selected(
                &dot(&self.weights[6], N, &h),
                &self.bias,
                sigmoid,
                groups,
                1,
                USED,
                normalize,
                0.7f32 as f64,
            )
        };
        let mut routed = if dense {
            vec![0.0; N]
        } else {
            self.routed(&h, &ids, &p)
        };
        for ((x, r), s) in x.iter_mut().zip(&mut routed).zip(shared) {
            *r += s;
            *x += *r;
        }
        (ids, p)
    }
}
#[test]
fn quantized_full_mla_matches_f64_fixed_dynamic_cpu_fallback_graphs_reset_and_prefix() {
    if !enabled() {
        return;
    }
    let rt = CudaRuntime::try_load().unwrap();
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    let create = unsafe {
        *lib.get::<Create>(b"rbitnet_cuda_mla_full_create\0")
            .unwrap()
    };
    let destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_mla_full_destroy\0")
            .unwrap()
    };
    let begin = unsafe { *lib.get::<Begin>(b"rbitnet_cuda_mla_full_begin\0").unwrap() };
    let prepare = unsafe {
        *lib.get::<Prepare>(b"rbitnet_cuda_mla_full_prepare\0")
            .unwrap()
    };
    let input = unsafe {
        *lib.get::<FfnInput>(b"rbitnet_cuda_mla_full_ffn_input\0")
            .unwrap()
    };
    let finish = unsafe {
        *lib.get::<Finish>(b"rbitnet_cuda_mla_full_finish\0")
            .unwrap()
    };
    let end = unsafe { *lib.get::<End>(b"rbitnet_cuda_mla_full_end\0").unwrap() };
    let hidden = unsafe {
        *lib.get::<Hidden>(b"rbitnet_cuda_mla_hidden_check\0")
            .unwrap()
    };
    let snapshot = unsafe { *lib.get::<Snapshot>(b"rbitnet_cuda_mla_snapshot\0").unwrap() };
    let snapshot_destroy = unsafe {
        *lib.get::<Destroy>(b"rbitnet_cuda_mla_snapshot_destroy\0")
            .unwrap()
    };
    let restore = unsafe { *lib.get::<Restore>(b"rbitnet_cuda_mla_restore\0").unwrap() };
    let moe_destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_moe_destroy\0").unwrap() };
    let norm = vec![1.0; N];
    let kvnorm = vec![1.0; RANK];
    let bias = vec![0.25, -0.4, 0.1, 0.35];
    let phases: Vec<_> = (0..32)
        .flat_map(|pos| {
            (0..ROT / 2).flat_map(move |i| {
                let (s, c) = (pos as f32 * 10000f32.powf(-2.0 * i as f32 / ROT as f32)).sin_cos();
                [s, c]
            })
        })
        .collect();
    for ty in [0, 2, 6, 8, 12, 13, 14, 39] {
        let mut owned = Vec::new();
        let mut descriptors = Vec::new();
        let mut dense = Vec::new();
        let shapes = [
            (N, N),
            (N, HEADS * DIM),
            (N, RANK + ROT),
            (N, HEADS * RANK),
            (RANK, HEADS * N),
            (HEADS * N, N),
            (N, EXP),
            (N, N),
            (N, N),
            (N, N),
            (N, EXP * N),
            (N, EXP * N),
            (N, EXP * N),
            (N, 257),
        ];
        for (i, (cols, rows)) in shapes.into_iter().enumerate() {
            let t = if i == 6 { 0 } else { ty };
            let p = payload(t, cols, rows, i);
            dense.push(crate::ggml::tensor_to_f32(&p, t, &[cols as u64, rows as u64]).unwrap());
            let m = CudaDeviceQuantMatrix::from_payload(Some(&rt), t, p, rows, cols).unwrap();
            descriptors.push(Matrix::device(&m).unwrap());
            owned.push(m);
        }
        for (graphs, split, dynamic, sigmoid, groups, normalize) in [
            (0, 0, false, true, 2, true),
            (1, 0, true, false, 1, false),
            (1, 1, true, true, 2, true),
        ] {
            let moe_create = unsafe {
                *lib.get::<MoeCreate>(if dynamic {
                    b"rbitnet_cuda_moe_dynamic_create\0"
                } else {
                    b"rbitnet_cuda_moe_create\0"
                })
                .unwrap()
            };
            let mut banks = [descriptors[10], descriptors[11], descriptors[12]];
            if dynamic {
                for m in &mut banks {
                    m.weights = std::ptr::null();
                }
            }
            let mc = MoeConfig {
                embd: N as u32,
                ffn: N as u32,
                experts: EXP as u32,
                used: USED as u32,
                oai: 0,
            };
            let moe = Handle {
                ptr: unsafe {
                    moe_create(
                        &mc,
                        &banks[0],
                        &banks[1],
                        &banks[2],
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                },
                destroy: moe_destroy,
            };
            assert!(!moe.ptr.is_null());
            let layers: Vec<_> = (0..2)
                .map(|il| Layer {
                    qa: descriptors[0],
                    qb: descriptors[1],
                    kva: descriptors[2],
                    kb: descriptors[3],
                    vb: descriptors[4],
                    out: descriptors[5],
                    router: descriptors[6],
                    shared_gate: descriptors[7],
                    shared_up: descriptors[8],
                    shared_down: descriptors[9],
                    attn_norm: norm.as_ptr(),
                    qa_norm: norm.as_ptr(),
                    kv_norm: kvnorm.as_ptr(),
                    ffn_norm: norm.as_ptr(),
                    selection_bias: bias.as_ptr(),
                    moe: if il == 0 {
                        std::ptr::null_mut()
                    } else {
                        moe.ptr
                    },
                })
                .collect();
            let cfg = NativeConfig {
                embd: N as u32,
                vocab: 257,
                layers: 2,
                heads: HEADS as u32,
                head: DIM as u32,
                value: N as u32,
                rotary: ROT as u32,
                rank: RANK as u32,
                capacity: 32,
                experts: EXP as u32,
                used: USED as u32,
                groups,
                groups_used: 1,
                sigmoid: u32::from(sigmoid),
                weight_norm: u32::from(normalize),
                ordered: crate::ggml::f32_accumulator_lanes().unwrap() as u32,
                graphs,
                split,
                dense: 1,
                epsilon: 1e-5,
                rope_magnitude: 1.13,
                weight_scale: 0.7,
            };
            let handle = Handle {
                ptr: unsafe {
                    create(
                        &cfg,
                        layers.as_ptr(),
                        &descriptors[13],
                        norm.as_ptr(),
                        phases.as_ptr(),
                    )
                },
                destroy,
            };
            assert!(!handle.ptr.is_null());
            let mut oracles: Vec<_> = (0..2)
                .map(|_| Oracle {
                    weights: dense.clone(),
                    an: norm.clone(),
                    qn: norm.clone(),
                    kn: kvnorm.clone(),
                    fnorm: norm.clone(),
                    bias: bias.clone(),
                    cache: Vec::new(),
                })
                .collect();
            let mut saved = None;
            let mut prefix_logits = Vec::new();
            for pass in 0..2 {
                for pos in 0..12 {
                    let embedding: Vec<_> = (0..N)
                        .map(|i| ((i + pos * 3) as f32 * 0.041).cos() * 0.6)
                        .collect();
                    let mut expected: Vec<_> = embedding.iter().map(|&x| x as f64).collect();
                    assert_eq!(
                        unsafe { begin(handle.ptr, embedding.as_ptr(), pos as u32) },
                        0
                    );
                    for il in 0..2 {
                        let (expected_ids, expected_p) = oracles[il].forward(
                            &mut expected,
                            pos,
                            &phases,
                            il == 0,
                            sigmoid,
                            groups as usize,
                            normalize,
                        );
                        let mut ids = vec![0; USED];
                        let mut p = vec![0.0; USED];
                        assert_eq!(
                            unsafe {
                                prepare(handle.ptr, il as u32, ids.as_mut_ptr(), p.as_mut_ptr())
                            },
                            0
                        );
                        if il == 1 {
                            assert_eq!(
                                ids,
                                expected_ids.iter().map(|&i| i as u32).collect::<Vec<_>>()
                            );
                            for (a, b) in p.iter().zip(expected_p) {
                                assert!((*a as f64 - b).abs() < 2e-6);
                            }
                        }
                        let mut pointers = Vec::new();
                        let mut cpu = Vec::new();
                        if il == 1 && pass == 1 {
                            let mut h = vec![0.0; N];
                            assert_eq!(unsafe { input(handle.ptr, h.as_mut_ptr()) }, 0);
                            cpu = oracles[il]
                                .routed(
                                    &h.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                                    &ids.iter().map(|&i| i as usize).collect::<Vec<_>>(),
                                    &p.iter().map(|&p| p as f64).collect::<Vec<_>>(),
                                )
                                .into_iter()
                                .map(|x| x as f32)
                                .collect();
                        } else if il == 1 && dynamic {
                            for matrix in &descriptors[10..13] {
                                for &id in &ids {
                                    pointers.push(unsafe {
                                        (matrix.weights as *const u8)
                                            .add(id as usize * N * matrix.row_bytes)
                                    }
                                        as *const c_void);
                                }
                            }
                        }
                        assert_eq!(
                            unsafe {
                                finish(
                                    handle.ptr,
                                    il as u32,
                                    if pointers.is_empty() {
                                        std::ptr::null()
                                    } else {
                                        pointers.as_ptr()
                                    },
                                    if cpu.is_empty() {
                                        std::ptr::null()
                                    } else {
                                        cpu.as_ptr()
                                    },
                                )
                            },
                            0
                        );
                    }
                    let mut got = vec![0.0; N];
                    assert_eq!(unsafe { hidden(handle.ptr, got.as_mut_ptr()) }, 0);
                    for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                        assert!((a as f64-b).abs()<5e-5*(1.0+b.abs()),"format={ty} graph={graphs} split={split} CPU={pass} pos={pos} row={i}: {a} vs {b}");
                    }
                    let mut logits = vec![0.0; 257];
                    let mut id = 0;
                    assert_eq!(
                        unsafe { end(handle.ptr, 1, logits.as_mut_ptr(), &mut id) },
                        0
                    );
                    let expected = dot(&dense[13], N, &rms(&expected, &norm));
                    for (i, (&a, b)) in logits.iter().zip(expected).enumerate() {
                        assert!(
                            (a as f64 - b).abs() < 5e-5 * (1.0 + b.abs()),
                            "logits format={ty} pos={pos} row={i}: {a} vs {b}"
                        );
                    }
                    assert_eq!(
                        unsafe { end(handle.ptr, 2, std::ptr::null_mut(), &mut id) },
                        0
                    );
                    let maximum = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    assert_eq!(id, maximum);
                    if pos == 5 && pass == 0 {
                        prefix_logits = logits;
                    }
                    if pos == 7 && pass == 0 {
                        let s = Handle {
                            ptr: unsafe { snapshot(handle.ptr, 8) },
                            destroy: snapshot_destroy,
                        };
                        assert!(!s.ptr.is_null());
                        saved = Some(s);
                    }
                }
            }
            let checkpoint = saved.unwrap();
            assert_eq!(unsafe { restore(handle.ptr, checkpoint.ptr, 5) }, 0);
            // A restored truncated prefix must reproduce the exact old result,
            // after another sequence has overwritten the live native cache.
            let embedding: Vec<_> = (0..N)
                .map(|i| ((i + 5 * 3) as f32 * 0.041).cos() * 0.6)
                .collect();
            assert_ne!(unsafe { begin(handle.ptr, embedding.as_ptr(), 6) }, 0);
            assert_eq!(unsafe { begin(handle.ptr, embedding.as_ptr(), 5) }, 0);
            for il in 0..2 {
                let mut ids = vec![0; USED];
                let mut p = vec![0.0; USED];
                assert_eq!(
                    unsafe { prepare(handle.ptr, il, ids.as_mut_ptr(), p.as_mut_ptr()) },
                    0
                );
                let mut pointers = Vec::new();
                if il == 1 && dynamic {
                    for m in &descriptors[10..13] {
                        for &e in &ids {
                            pointers.push(unsafe {
                                (m.weights as *const u8).add(e as usize * N * m.row_bytes)
                            } as *const c_void);
                        }
                    }
                }
                assert_eq!(
                    unsafe {
                        finish(
                            handle.ptr,
                            il,
                            if pointers.is_empty() {
                                std::ptr::null()
                            } else {
                                pointers.as_ptr()
                            },
                            std::ptr::null(),
                        )
                    },
                    0
                );
            }
            let mut restored_logits = vec![0.0; 257];
            let mut id = 0;
            assert_eq!(
                unsafe { end(handle.ptr, 1, restored_logits.as_mut_ptr(), &mut id) },
                0
            );
            assert_eq!(restored_logits, prefix_logits);
            let alien = Handle {
                ptr: unsafe {
                    create(
                        &cfg,
                        layers.as_ptr(),
                        &descriptors[13],
                        norm.as_ptr(),
                        phases.as_ptr(),
                    )
                },
                destroy,
            };
            assert!(!alien.ptr.is_null());
            assert_ne!(unsafe { restore(alien.ptr, checkpoint.ptr, 5) }, 0);
            drop(alien);
            assert_eq!(unsafe { begin(handle.ptr, embedding.as_ptr(), 0) }, 0);
            let mut ids = vec![0; USED];
            let mut p = vec![0.0; USED];
            assert_eq!(
                unsafe { prepare(handle.ptr, 0, ids.as_mut_ptr(), p.as_mut_ptr()) },
                0
            );
            // Reset abandons this partly completed token without reading its tail.
            assert_eq!(unsafe { begin(handle.ptr, embedding.as_ptr(), 0) }, 0);
            drop(checkpoint);
            drop(handle);
            drop(moe);
        }
        drop(owned);
        eprintln!("MLA F64 oracle passed format {ty}: fixed/dynamic/CPU fallback, eager/graphs/split, reset and prefix bounds");
    }
}
