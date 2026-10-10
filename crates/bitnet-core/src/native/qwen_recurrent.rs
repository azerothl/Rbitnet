//! Complete dense recurrent Qwen3.5 layer: immutable weights and private GPU state.
use super::weights::Weights;
use crate::backend::{CudaDeviceQuantMatrix, CudaRuntime};
use crate::error::{BitNetError, Result};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Matrix {
    weights: *const c_void,
    row_bytes: usize,
    ty: u32,
    cols: u32,
    rows: u32,
}
#[repr(C)]
struct Config {
    embd: u32,
    ffn: u32,
    head: u32,
    num_k: u32,
    num_v: u32,
    conv: u32,
    graphs: u32,
    epsilon: f32,
}
type Create = unsafe extern "C" fn(
    *const Config,
    *const Matrix,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, *mut f32) -> i32;
type Snapshot = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;
struct SnapshotApi {
    create: Snapshot,
    restore: Restore,
    destroy: Destroy,
}
pub(crate) struct SavedRecurrent {
    context: usize,
    destroy: Destroy,
}
impl Drop for SavedRecurrent {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

pub(crate) struct GpuRecurrent {
    context: usize,
    destroy: Destroy,
    step: Step,
    _weights: Vec<CudaDeviceQuantMatrix>,
    embd: usize,
    graphs: bool,
    snapshots: Option<SnapshotApi>,
    pub state_bytes: usize,
    pub extra_weights_bytes: usize,
}
impl GpuRecurrent {
    // Only the exclusive owning runtime may lend this context to a pipeline.
    pub(crate) fn context_address(&self) -> usize {
        self.context
    }
    pub fn new(
        weights: &Weights,
        layer: usize,
        head: usize,
        num_k: usize,
        num_v: usize,
        epsilon: f32,
        extra_budget: usize,
    ) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_QWEN_RECURRENT").as_deref() == Ok("0") {
            return None;
        }
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe {
            *lib.get::<Create>(b"rbitnet_cuda_qwen_recurrent_create\0")
                .ok()?
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_recurrent_destroy\0")
                .ok()?
        };
        let step = unsafe {
            *lib.get::<Step>(b"rbitnet_cuda_qwen_recurrent_step\0")
                .ok()?
        };
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        let mut owned = Vec::new();
        let mut matrices = Vec::new();
        let mut extra_weights_bytes = 0usize;
        for suffix in [
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
            "ssm_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let tensor = name(suffix);
            let m = if let Some(m) = weights.device_matrix(&tensor) {
                m
            } else if suffix == "ssm_beta.weight" || suffix == "ssm_alpha.weight" {
                // These small head projections are deliberately excluded by the
                // shared weight planner's 128-row threshold. Own their copies.
                let t = weights.tensor(&tensor).ok()?;
                if t.dimensions.len() != 2 || t.dimensions[1] as usize != num_v {
                    return None;
                }
                let runtime = CudaRuntime::try_load()?;
                let payload = weights.archive.tensor_payload(t).ok()?;
                if extra_weights_bytes.checked_add(payload.len())? > extra_budget {
                    return None;
                }
                let m = CudaDeviceQuantMatrix::from_payload(
                    Some(&runtime),
                    t.ggml_type,
                    payload.to_vec(),
                    t.dimensions[1].try_into().ok()?,
                    t.dimensions[0].try_into().ok()?,
                )
                .ok()?;
                extra_weights_bytes += m.bytes();
                m
            } else {
                return None;
            };
            matrices.push(Matrix {
                weights: m.device_address()? as *const c_void,
                row_bytes: m.bytes().checked_div(m.out_rows())?,
                ty: m.ggml_type(),
                cols: m.in_cols().try_into().ok()?,
                rows: m.out_rows().try_into().ok()?,
            });
            owned.push(m);
        }
        let conv = weights.tensor(&name("ssm_conv1d.weight")).ok()?;
        if conv.dimensions.len() != 2 {
            return None;
        }
        let conv_weights = crate::ggml::tensor_to_f32(
            weights.archive.tensor_payload(conv).ok()?,
            conv.ggml_type,
            &conv.dimensions,
        )
        .ok()?;
        let cfg = Config {
            embd: matrices[0].cols,
            ffn: matrices[5].rows,
            head: head.try_into().ok()?,
            num_k: num_k.try_into().ok()?,
            num_v: num_v.try_into().ok()?,
            conv: conv.dimensions[0].try_into().ok()?,
            graphs: u32::from(
                std::env::var("RBITNET_CUDA_QWEN_RECURRENT_GRAPH").as_deref() != Ok("0"),
            ),
            epsilon,
        };
        let values = head.checked_mul(num_v)?;
        let inner = head.checked_mul(num_k.checked_mul(2)?.checked_add(num_v)?)?;
        if conv.dimensions[1] as usize != inner {
            return None;
        }
        let attn_norm = weights.dense(&name("attn_norm.weight")).ok()?;
        let ffn_norm = weights.dense(&name("post_attention_norm.weight")).ok()?;
        let dt = ["ssm_dt.bias", "ssm_dt"]
            .iter()
            .find_map(|suffix| weights.dense(&name(suffix)).ok())?;
        let a = ["ssm_a_noscan.weight", "ssm_a.weight", "ssm_a"]
            .iter()
            .find_map(|s| weights.dense(&name(s)).ok())?;
        let norm = weights.dense(&name("ssm_norm.weight")).ok()?;
        if attn_norm.len() != cfg.embd as usize
            || ffn_norm.len() != cfg.embd as usize
            || dt.len() != num_v
            || a.len() != num_v
            || (norm.len() != head && norm.len() != values)
        {
            return None;
        }
        let expanded_norm = if norm.len() == head {
            norm.repeat(num_v)
        } else {
            norm.to_vec()
        };
        let context = unsafe {
            create(
                &cfg,
                matrices.as_ptr(),
                attn_norm.as_ptr(),
                ffn_norm.as_ptr(),
                conv_weights.as_ptr(),
                dt.as_ptr(),
                a.as_ptr(),
                expanded_norm.as_ptr(),
            )
        } as usize;
        if context == 0 {
            return None;
        }
        let snapshots = unsafe {
            lib.get::<Snapshot>(b"rbitnet_cuda_qwen_recurrent_snapshot\0")
                .ok()
                .zip(
                    lib.get::<Restore>(b"rbitnet_cuda_qwen_recurrent_restore\0")
                        .ok(),
                )
                .zip(
                    lib.get::<Destroy>(b"rbitnet_cuda_qwen_recurrent_snapshot_destroy\0")
                        .ok(),
                )
                .map(|((create, restore), destroy)| SnapshotApi {
                    create: *create,
                    restore: *restore,
                    destroy: *destroy,
                })
        };
        Some(Self {
            context,
            destroy,
            step,
            _weights: owned,
            embd: cfg.embd as usize,
            graphs: cfg.graphs != 0,
            snapshots,
            state_bytes: (num_v * head * head + inner * cfg.conv as usize) * 4,
            extra_weights_bytes,
        })
    }
    pub fn supports_snapshot(&self) -> bool {
        self.snapshots.is_some()
    }
    pub fn snapshot(&self) -> Option<SavedRecurrent> {
        let api = self.snapshots.as_ref()?;
        let context = unsafe { (api.create)(self.context as *mut c_void) } as usize;
        (context != 0).then_some(SavedRecurrent {
            context,
            destroy: api.destroy,
        })
    }
    pub fn restore(&mut self, snapshot: &SavedRecurrent, length: usize) -> Result<()> {
        let api = self
            .snapshots
            .as_ref()
            .ok_or_else(|| BitNetError::Inference("Qwen state snapshots unavailable".into()))?;
        let length = u32::try_from(length)
            .map_err(|_| BitNetError::Inference("Qwen snapshot position overflow".into()))?;
        let status = unsafe {
            (api.restore)(
                self.context as *mut c_void,
                snapshot.context as *const c_void,
                length,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "Qwen state restore failed: {status}"
            )));
        }
        Ok(())
    }
    pub fn run(&mut self, input: &[f32], pos: usize) -> Result<Vec<f32>> {
        if input.len() != self.embd {
            return Err(BitNetError::Inference(
                "Qwen CUDA block input shape mismatch".into(),
            ));
        }
        let position = u32::try_from(pos)
            .map_err(|_| BitNetError::Inference("Qwen CUDA position out of bounds".into()))?;
        let mut output = vec![0.0; self.embd];
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                input.as_ptr(),
                position,
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "resident Qwen CUDA block failed: {status}"
            )));
        }
        crate::perf::record_gpu_transfer((self.embd * 4) as u64, (self.embd * 4) as u64, 8);
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        Ok(output)
    }
}
impl Drop for GpuRecurrent {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context as *mut c_void) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Oracle {
        head: usize,
        nk: usize,
        nv: usize,
        taps: usize,
        weights: Vec<Vec<f32>>,
        cols: Vec<usize>,
        an: Vec<f32>,
        fnorm: Vec<f32>,
        conv: Vec<f32>,
        dt: Vec<f32>,
        a: Vec<f32>,
        sn: Vec<f32>,
        history: Vec<f64>,
        state: Vec<f64>,
    }
    impl Oracle {
        fn norm(x: &[f64], w: &[f32]) -> Vec<f64> {
            let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-5)
                .sqrt()
                .recip();
            x.iter().zip(w).map(|(&x, &w)| x * w as f64 * inv).collect()
        }
        fn projection(&self, p: usize, x: &[f64]) -> Vec<f64> {
            self.weights[p]
                .chunks_exact(self.cols[p])
                .map(|row| row.iter().zip(x).map(|(&w, &x)| w as f64 * x).sum())
                .collect()
        }
        fn reset(&mut self) {
            self.history.fill(0.0);
            self.state.fill(0.0);
        }
        fn layer(&mut self, input: &[f32]) -> Vec<f64> {
            let mut x: Vec<f64> = input.iter().map(|&x| x as f64).collect();
            let h = Self::norm(&x, &self.an);
            let strip = self.projection(0, &h);
            let z = self.projection(1, &h);
            let beta = self.projection(2, &h);
            let alpha = self.projection(3, &h);
            let inner = strip.len();
            let value = self.nv * self.head;
            let key = self.nk * self.head;
            let mut activated = vec![0.0; inner];
            for i in 0..inner {
                let sum = (0..self.taps)
                    .map(|tap| {
                        let v = if tap + 1 == self.taps {
                            strip[i]
                        } else {
                            self.history[tap * inner + i]
                        };
                        v * self.conv[i * self.taps + tap] as f64
                    })
                    .sum::<f64>();
                activated[i] = sum / (1.0 + (-sum).exp());
            }
            if self.taps > 1 {
                self.history.copy_within(inner.., 0);
                let start = (self.taps - 2) * inner;
                self.history[start..start + inner].copy_from_slice(&strip);
            }
            for chunk in activated[..2 * key].chunks_exact_mut(self.head) {
                let inv = chunk
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .max(1e-5)
                    .sqrt()
                    .recip();
                for x in chunk {
                    *x *= inv;
                }
            }
            let mut attn = vec![0.0; value];
            for h in 0..self.nv {
                let kh = h % self.nk;
                let q = &activated[kh * self.head..(kh + 1) * self.head];
                let k = &activated[key + kh * self.head..key + (kh + 1) * self.head];
                let t = alpha[h] + self.dt[h] as f64;
                let sp = if t > 35.0 {
                    t
                } else if t < -35.0 {
                    0.0
                } else {
                    (1.0 + t.exp()).ln()
                };
                let decay = (sp * self.a[h] as f64).exp();
                let b = 1.0 / (1.0 + (-beta[h]).exp());
                for row in 0..self.head {
                    let start = (h * self.head + row) * self.head;
                    let s = &mut self.state[start..start + self.head];
                    for s in s.iter_mut() {
                        *s *= decay;
                    }
                    let prediction = s.iter().zip(k).map(|(&s, &k)| s * k).sum::<f64>();
                    let delta = (activated[2 * key + h * self.head + row] - prediction) * b;
                    for (s, &k) in s.iter_mut().zip(k) {
                        *s += k * delta;
                    }
                    attn[h * self.head + row] = s.iter().zip(q).map(|(&s, &q)| s * q).sum::<f64>()
                        / (self.head as f64).sqrt();
                }
            }
            let mut normed = Vec::new();
            for h in 0..self.nv {
                let start = h * self.head;
                let n = Self::norm(
                    &attn[start..start + self.head],
                    &self.sn[start..start + self.head],
                );
                normed.extend(
                    n.iter()
                        .zip(&z[start..start + self.head])
                        .map(|(&n, &z)| n * z / (1.0 + (-z).exp())),
                );
            }
            let attention = self.projection(4, &normed);
            for (x, &a) in x.iter_mut().zip(&attention) {
                *x += a;
            }
            let h = Self::norm(&x, &self.fnorm);
            let g = self.projection(5, &h);
            let u = self.projection(6, &h);
            let hidden: Vec<_> = g
                .iter()
                .zip(u)
                .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let ffn = self.projection(7, &hidden);
            for (x, &f) in x.iter_mut().zip(&ffn) {
                *x += f;
            }
            x
        }
    }

    #[test]
    fn opt_in_dense_recurrent_layer_matches_f64_and_restarts_sequence() {
        if std::env::var("RBITNET_CUDA_QUANT_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let rt = CudaRuntime::try_load().expect("CUDA required");
        let lib = crate::ggml::load_cuda_quant_library().expect("native CUDA DLL required");
        let create = unsafe {
            *lib.get::<Create>(b"rbitnet_cuda_qwen_recurrent_create\0")
                .unwrap()
        };
        let destroy = unsafe {
            *lib.get::<Destroy>(b"rbitnet_cuda_qwen_recurrent_destroy\0")
                .unwrap()
        };
        let step = unsafe {
            *lib.get::<Step>(b"rbitnet_cuda_qwen_recurrent_step\0")
                .unwrap()
        };
        type Block = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32) -> i32;
        let block = unsafe {
            *lib.get::<Block>(b"rbitnet_cuda_qwen_recurrent_prefill_check\0")
                .unwrap()
        };
        for graphs in [0, 1] {
            for format in [0, 2, 6, 8, 12, 13, 14, 39] {
                for (head, nk, nv, taps) in [(32, 1, 2, 1), (128, 2, 4, 4), (256, 1, 2, 16)] {
                    let embd = 256;
                    let ffn = 512;
                    let inner = (2 * nk + nv) * head;
                    let value = nv * head;
                    let cols = vec![embd, embd, embd, embd, value, embd, embd, ffn];
                    let rows = [inner, value, nv, nv, embd, ffn, ffn, embd];
                    let mut owned = Vec::new();
                    let mut dense = Vec::new();
                    let mut matrices = Vec::new();
                    for p in 0..8 {
                        let ty = if [12, 13, 14].contains(&format) && cols[p] % 256 != 0 {
                            0
                        } else {
                            format
                        };
                        let rb = crate::ggml::ggml_row_size(ty, cols[p] as u64).unwrap();
                        let mut payload: Vec<u8> = (0..rb * rows[p])
                            .map(|i| (i * 37 + p * 71 + 3) as u8)
                            .collect();
                        if ty == 0 {
                            for (i, v) in payload.chunks_exact_mut(4).enumerate() {
                                v.copy_from_slice(
                                    &((i as f32 * 0.017 + p as f32 * 0.7).sin() * 0.045)
                                        .to_le_bytes(),
                                );
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
                            for block in payload.chunks_exact_mut(bytes) {
                                if ty == 39 {
                                    block[0] = 119;
                                    continue;
                                }
                                let offset = if ty == 14 { 208 } else { 0 };
                                block[offset..offset + 2].copy_from_slice(
                                    &half::f16::from_f32(if ty == 14 {
                                        0.000002
                                    } else {
                                        0.000007
                                    })
                                    .to_bits()
                                    .to_le_bytes(),
                                );
                                if ty == 12 || ty == 13 {
                                    block[2..4].copy_from_slice(
                                        &half::f16::from_f32(0.000003).to_bits().to_le_bytes(),
                                    );
                                }
                            }
                        }
                        dense.push(
                            crate::ggml::tensor_to_f32(
                                &payload,
                                ty,
                                &[cols[p] as u64, rows[p] as u64],
                            )
                            .unwrap(),
                        );
                        let m = CudaDeviceQuantMatrix::from_payload(
                            Some(&rt),
                            ty,
                            payload,
                            rows[p],
                            cols[p],
                        )
                        .unwrap();
                        matrices.push(Matrix {
                            weights: m.device_address().unwrap() as *const c_void,
                            row_bytes: rb,
                            ty,
                            cols: cols[p] as u32,
                            rows: rows[p] as u32,
                        });
                        owned.push(m);
                    }
                    let norm = |n| {
                        (0..n)
                            .map(|i| 1.0 + (i as f32 * 0.13).sin() * 0.08)
                            .collect::<Vec<_>>()
                    };
                    let mut oracle = Oracle {
                        head,
                        nk,
                        nv,
                        taps,
                        weights: dense,
                        cols,
                        an: norm(embd),
                        fnorm: norm(embd),
                        conv: (0..taps * inner)
                            .map(|i| (i as f32 * 0.11).cos() * 0.4)
                            .collect(),
                        dt: (0..nv)
                            .map(|i| {
                                if i == 0 {
                                    40.0
                                } else if i == 1 {
                                    -40.0
                                } else {
                                    i as f32 * 0.17
                                }
                            })
                            .collect(),
                        a: (0..nv).map(|i| -0.02 - i as f32 * 0.01).collect(),
                        sn: norm(value),
                        history: vec![0.0; (taps - 1) * inner],
                        state: vec![0.0; value * head],
                    };
                    let cfg = Config {
                        embd: embd as u32,
                        ffn: ffn as u32,
                        head: head as u32,
                        num_k: nk as u32,
                        num_v: nv as u32,
                        conv: taps as u32,
                        graphs,
                        epsilon: 1e-5,
                    };
                    let context = unsafe {
                        create(
                            &cfg,
                            matrices.as_ptr(),
                            oracle.an.as_ptr(),
                            oracle.fnorm.as_ptr(),
                            oracle.conv.as_ptr(),
                            oracle.dt.as_ptr(),
                            oracle.a.as_ptr(),
                            oracle.sn.as_ptr(),
                        )
                    };
                    assert!(!context.is_null(), "resident Qwen block required");
                    let serial = unsafe {
                        create(
                            &cfg,
                            matrices.as_ptr(),
                            oracle.an.as_ptr(),
                            oracle.fnorm.as_ptr(),
                            oracle.conv.as_ptr(),
                            oracle.dt.as_ptr(),
                            oracle.a.as_ptr(),
                            oracle.sn.as_ptr(),
                        )
                    };
                    assert!(!serial.is_null());
                    let mut output = vec![0.0; embd];
                    let mut maximum = 0.0f64;
                    for sequence in 0..2 {
                        oracle.reset();
                        for pos in 0..17 {
                            let input: Vec<_> = (0..embd)
                                .map(|i| {
                                    (i as f32 * 0.73 + pos as f32 * 0.31 + sequence as f32 * 1.17)
                                        .sin()
                                        * 2.0
                                })
                                .collect();
                            if pos == 0 {
                                assert_eq!(
                                    unsafe {
                                        step(context, input.as_ptr(), 7, output.as_mut_ptr())
                                    },
                                    2
                                );
                            }
                            assert_eq!(
                                unsafe { step(context, input.as_ptr(), pos, output.as_mut_ptr()) },
                                0
                            );
                            let expected = oracle.layer(&input);
                            for (i, (&got, &expected)) in output.iter().zip(&expected).enumerate() {
                                let error = (got as f64 - expected).abs();
                                maximum = maximum.max(error);
                                assert!(error<2e-5*(1.0+expected.abs()),"head={head} graphs={graphs} sequence={sequence} pos={pos} element={i} got={got} expected={expected}");
                            }
                        }
                    }
                    oracle.reset();
                    for (pos, count) in [(0, 7), (7, 16), (23, 33), (56, 64), (120, 128), (0, 17)] {
                        if pos == 0 {
                            oracle.reset();
                        }
                        let input: Vec<_> = (0..embd * count)
                            .map(|i| (i as f32 * 0.037 + pos as f32 * 0.21).sin() * 1.7)
                            .collect();
                        let mut output = vec![0.0; input.len()];
                        assert_eq!(
                            unsafe {
                                block(
                                    context,
                                    input.as_ptr(),
                                    pos as u32,
                                    count as u32,
                                    output.as_mut_ptr(),
                                )
                            },
                            0
                        );
                        for (t, (input, output)) in input
                            .chunks_exact(embd)
                            .zip(output.chunks_exact(embd))
                            .enumerate()
                        {
                            let expected = oracle.layer(input);
                            let mut scalar = vec![0.0; embd];
                            assert_eq!(
                                unsafe {
                                    step(
                                        serial,
                                        input.as_ptr(),
                                        (pos + t) as u32,
                                        scalar.as_mut_ptr(),
                                    )
                                },
                                0
                            );
                            for (i, (&got, &e)) in output.iter().zip(&expected).enumerate() {
                                assert!((got as f64-e).abs()<5e-5*(1.0+e.abs()),"format={format} head={head} graphs={graphs} block={count} pos={} element={i}: block={got}, scalar={}, F64={e}",pos+t,scalar[i]);
                                assert!((got-scalar[i]).abs()<5e-5*(1.0+scalar[i].abs()),"block/scalar differ format={format} head={head} pos={} element={i}",pos+t);
                            }
                        }
                    }
                    unsafe {
                        destroy(context);
                        destroy(serial);
                    }
                    println!("Qwen block format={format} head={head} graphs={graphs}: serial and blocks 7/16/33/64/128/reset passed, serial max abs error={maximum}");
                }
            }
        }
    }
}
