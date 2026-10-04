//! Fully GPU-resident dense Llama graph, using the same owned quantized weights.
use super::model::{LlamaModel, MatrixWeights};
use crate::backend::CudaDeviceQuantMatrix;
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
struct Layer {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    out: Matrix,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    attn_norm: *const f32,
    ffn_norm: *const f32,
}
#[repr(C)]
struct Config {
    embd: u32,
    ffn: u32,
    vocab: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary: u32,
    capacity: u32,
    window: u32,
    graphs: u32,
    epsilon: f32,
}
type Create = unsafe extern "C" fn(
    *const Config,
    *const Layer,
    *const Matrix,
    *const f32,
    *const f32,
) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32, *mut u32) -> i32;
type Prefill =
    unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, u32, *mut f32, *mut u32) -> i32;
type Snapshot = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;

struct SavedPrefix {
    context: usize,
    destroy: Destroy,
}
impl Drop for SavedPrefix {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context as *mut c_void);
        }
    }
}

struct SnapshotApi {
    create: Snapshot,
    restore: Restore,
    destroy: Destroy,
}

pub(super) struct Resident {
    context: usize,
    step: Step,
    prefill: Option<Prefill>,
    prefill_embeddings: Vec<f32>,
    destroy: Destroy,
    _weights: Vec<CudaDeviceQuantMatrix>,
    embedding: Vec<f32>,
    vocab: usize,
    capacity: usize,
    matrix_count: u64,
    layers: u64,
    graphs: bool,
    snapshots: Option<SnapshotApi>,
    prefixes: crate::native::prefix::PrefixStore<SavedPrefix>,
    kv_bytes_per_token: usize,
}
impl Resident {
    pub fn new(model: &LlamaModel) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_RESIDENT").as_deref() == Ok("0") {
            return None;
        }
        let c = &model.cfg;
        let kv_bytes_per_token = c
            .n_layer
            .checked_mul(c.n_kv)?
            .checked_mul(c.head_dim)?
            .checked_mul(8)?;
        let mut weights = Vec::new();
        let mut matrix = |m: &MatrixWeights, cols: usize, rows: usize| -> Option<Matrix> {
            let MatrixWeights::CudaQuant { device, .. } = m else {
                return None;
            };
            if device.in_cols() != cols || device.out_rows() != rows {
                return None;
            }
            let result = Matrix {
                weights: device.device_address()? as *const c_void,
                row_bytes: device.bytes() / rows,
                ty: device.ggml_type(),
                cols: cols.try_into().ok()?,
                rows: rows.try_into().ok()?,
            };
            weights.push(device.clone());
            Some(result)
        };
        let output = matrix(&model.output, c.n_embd, c.n_vocab)?;
        let mut layers = Vec::new();
        for l in &model.layers {
            // Alternate per-head RMSNorm needs its own fused graph variant.
            if l.attn_q_norm.is_some() || l.attn_k_norm.is_some() {
                return None;
            }
            layers.push(Layer {
                q: matrix(&l.wq, c.n_embd, c.n_embd)?,
                k: matrix(&l.wk, c.n_embd, c.n_kv * c.head_dim)?,
                v: matrix(&l.wv, c.n_embd, c.n_kv * c.head_dim)?,
                out: matrix(&l.wo, c.n_embd, c.n_embd)?,
                gate: matrix(&l.ffn_gate, c.n_embd, c.n_ff)?,
                up: matrix(&l.ffn_up, c.n_embd, c.n_ff)?,
                down: matrix(&l.ffn_down, c.n_ff, c.n_embd)?,
                attn_norm: l.attn_norm.as_ptr(),
                ffn_norm: l.ffn_norm.as_ptr(),
            });
        }
        let cfg = Config {
            embd: c.n_embd.try_into().ok()?,
            ffn: c.n_ff.try_into().ok()?,
            vocab: c.n_vocab.try_into().ok()?,
            layers: c.n_layer.try_into().ok()?,
            heads: c.n_head.try_into().ok()?,
            kv_heads: c.n_kv.try_into().ok()?,
            head_dim: c.head_dim.try_into().ok()?,
            rotary: c.rope_rot_dims.try_into().ok()?,
            capacity: c.max_seq.try_into().ok()?,
            window: c.sliding_window.unwrap_or(0).try_into().ok()?,
            graphs: u32::from(std::env::var("RBITNET_CUDA_RESIDENT_GRAPH").as_deref() != Ok("0")),
            epsilon: c.norm_eps,
        };
        let frequency: Vec<f32> = (0..c.rope_rot_dims / 2)
            .map(|i| {
                model
                    .rope_inv_freq
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .copied()
                    .unwrap_or_else(|| {
                        1.0 / c.rope_theta.powf(2.0 * i as f32 / c.rope_rot_dims as f32)
                    })
            })
            .collect();
        let lib = crate::ggml::load_cuda_quant_library()?;
        let create = unsafe { lib.get::<Create>(b"rbitnet_cuda_llama_create\0").ok()? };
        let destroy = unsafe { *lib.get::<Destroy>(b"rbitnet_cuda_llama_destroy\0").ok()? };
        let step = unsafe { *lib.get::<Step>(b"rbitnet_cuda_llama_step\0").ok()? };
        let prefill = if std::env::var("RBITNET_CUDA_PREFILL").as_deref() == Ok("1") {
            unsafe {
                lib.get::<Prefill>(b"rbitnet_cuda_llama_prefill\0")
                    .ok()
                    .map(|p| *p)
            }
        } else {
            None
        };
        if std::env::var("RBITNET_CUDA_PREFILL").as_deref() == Ok("1") && prefill.is_none() {
            tracing::warn!("CUDA block prefill API unavailable; using resident token graphs");
        }
        let snapshots = unsafe {
            lib.get::<Snapshot>(b"rbitnet_cuda_llama_snapshot\0")
                .ok()
                .zip(lib.get::<Restore>(b"rbitnet_cuda_llama_restore\0").ok())
                .zip(
                    lib.get::<Destroy>(b"rbitnet_cuda_llama_snapshot_destroy\0")
                        .ok(),
                )
                .map(|((create, restore), destroy)| SnapshotApi {
                    create: *create,
                    restore: *restore,
                    destroy: *destroy,
                })
        };
        if crate::native::prefix::enabled() && snapshots.is_none() {
            return None;
        }
        let context = unsafe {
            create(
                &cfg,
                layers.as_ptr(),
                &output,
                model.output_norm.as_ptr(),
                frequency.as_ptr(),
            )
        } as usize;
        if context == 0 {
            return None;
        }
        tracing::info!(
            graphs = cfg.graphs != 0,
            "fully resident Llama CUDA token graph enabled"
        );
        Some(Self {
            context,
            step,
            prefill,
            prefill_embeddings: Vec::new(),
            destroy,
            _weights: weights,
            embedding: vec![0.0; c.n_embd],
            vocab: c.n_vocab,
            capacity: c.max_seq,
            matrix_count: 7 * c.n_layer as u64,
            layers: c.n_layer as u64,
            graphs: cfg.graphs != 0,
            snapshots,
            prefixes: crate::native::prefix::PrefixStore::from_env(),
            kv_bytes_per_token,
        })
    }
    pub fn supports_prefix_cache(&self) -> bool {
        self.snapshots.is_some()
    }
    pub fn prefill(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
        base: usize,
        greedy: bool,
    ) -> Result<(Vec<f32>, u32)> {
        let Some(prefill) = self.prefill else {
            let mut logits = Vec::new();
            let mut next = 0;
            for (i, &token) in tokens.iter().enumerate() {
                if crate::cancel::inference_cancelled() {
                    return Err(BitNetError::Inference("inference cancelled".into()));
                }
                if greedy {
                    next = self.greedy(model, token, base + i, i + 1 == tokens.len())?;
                } else {
                    logits = self.forward(model, token, base + i, i + 1 == tokens.len())?;
                }
            }
            return Ok((logits, next));
        };
        if tokens.is_empty()
            || base
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.capacity)
        {
            return Err(BitNetError::Inference(
                "resident prefill context out of bounds".into(),
            ));
        }
        let chunk_size = std::env::var("RBITNET_CUDA_PREFILL_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(128)
            .clamp(1, 128);
        let mut logits = if greedy {
            Vec::new()
        } else {
            vec![0.0; self.vocab]
        };
        let mut next = 0;
        for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
            if crate::cancel::inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            self.prefill_embeddings
                .resize(chunk.len() * model.cfg.n_embd, 0.0);
            for (&token, row) in chunk
                .iter()
                .zip(self.prefill_embeddings.chunks_exact_mut(model.cfg.n_embd))
            {
                if token as usize >= self.vocab {
                    return Err(BitNetError::Inference(
                        "resident prefill token out of bounds".into(),
                    ));
                }
                model
                    .token_embd
                    .embed_row(token as usize, model.cfg.n_embd, self.vocab, row)?;
            }
            let position = base + index * chunk_size;
            let mode = if index * chunk_size + chunk.len() < tokens.len() {
                0
            } else if greedy {
                2
            } else {
                1
            };
            let status = unsafe {
                prefill(
                    self.context as *mut c_void,
                    self.prefill_embeddings.as_ptr(),
                    position as u32,
                    chunk.len() as u32,
                    mode,
                    logits.as_mut_ptr(),
                    &mut next,
                )
            };
            if status != 0 {
                return Err(BitNetError::Inference(format!(
                    "CUDA block prefill failed: {status}"
                )));
            }
            crate::perf::record_gpu_prefill(
                chunk.len(),
                if chunk.len() > 1 {
                    self.matrix_count
                } else {
                    0
                },
            );
            for _ in 0..self.layers {
                crate::perf::record_gpu_attention();
            }
            crate::perf::record_gpu_transfer(
                (self.prefill_embeddings.len() * 4 + 4) as u64,
                if mode == 1 {
                    (self.vocab * 4) as u64
                } else if mode == 2 {
                    4
                } else {
                    0
                },
                u64::from(mode > 0)
                    + if chunk.len() == 1 {
                        self.matrix_count
                    } else {
                        0
                    },
            );
        }
        Ok((logits, next))
    }

    pub fn restore_prefix(&mut self, tokens: &[u32]) -> Result<usize> {
        if !crate::native::prefix::enabled() {
            return Ok(0);
        }
        let Some(api) = &self.snapshots else {
            return Ok(0);
        };
        // Recompute the last matching token to obtain its logits, also for a full hit.
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        if let Some((snapshot, matched)) =
            self.prefixes
                .lookup(reusable, true, crate::native::prefix::minimum_tokens())
        {
            let status = unsafe {
                (api.restore)(
                    self.context as *mut c_void,
                    snapshot.context as *const c_void,
                    matched as u32,
                )
            };
            if status != 0 {
                return Err(BitNetError::Inference(format!(
                    "CUDA prefix restore failed: {status}"
                )));
            }
            crate::perf::record_prefix_hit(matched.saturating_mul(self.kv_bytes_per_token));
            return Ok(matched);
        }
        crate::perf::record_prefix_cache_miss();
        Ok(0)
    }

    pub fn save_prefix(&mut self, tokens: &[u32]) {
        if !crate::native::prefix::enabled()
            || tokens.len() < crate::native::prefix::minimum_tokens()
        {
            return;
        }
        let Some(api) = &self.snapshots else {
            return;
        };
        if self.prefixes.contains(tokens) {
            return;
        }
        let bytes = tokens.len().saturating_mul(self.kv_bytes_per_token);
        if !self.prefixes.reserve(tokens, bytes) {
            return;
        }
        let context =
            unsafe { (api.create)(self.context as *mut c_void, tokens.len() as u32) } as usize;
        if context == 0 {
            tracing::warn!(
                bytes,
                "CUDA prefix snapshot allocation/copy failed; skipping cache insert"
            );
            return;
        }
        self.prefixes.insert(
            tokens.to_vec(),
            SavedPrefix {
                context,
                destroy: api.destroy,
            },
            bytes,
        );
    }
    fn run(
        &mut self,
        model: &LlamaModel,
        token: u32,
        pos: usize,
        mode: u32,
        logits: &mut [f32],
    ) -> Result<u32> {
        if pos >= self.capacity
            || token as usize >= self.vocab
            || (mode == 1 && logits.len() != self.vocab)
        {
            return Err(BitNetError::Inference(
                "resident Llama token/context out of bounds".into(),
            ));
        }
        model.token_embd.embed_row(
            token as usize,
            model.cfg.n_embd,
            self.vocab,
            &mut self.embedding,
        )?;
        let mut next = 0;
        let status = unsafe {
            (self.step)(
                self.context as *mut c_void,
                self.embedding.as_ptr(),
                pos as u32,
                mode,
                logits.as_mut_ptr(),
                &mut next,
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "resident CUDA Llama failed: {status}"
            )));
        }
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        crate::perf::record_gpu_transfer(
            (self.embedding.len() * 4 + 4) as u64,
            if mode == 1 {
                (self.vocab * 4) as u64
            } else if mode == 2 {
                4
            } else {
                0
            },
            self.matrix_count + u64::from(mode != 0),
        );
        for _ in 0..self.layers {
            crate::perf::record_gpu_attention();
        }
        Ok(next)
    }
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        token: u32,
        pos: usize,
        logits: bool,
    ) -> Result<Vec<f32>> {
        let mut out = if logits {
            vec![0.0; self.vocab]
        } else {
            Vec::new()
        };
        self.run(model, token, pos, u32::from(logits), &mut out)?;
        Ok(out)
    }
    pub fn greedy(
        &mut self,
        model: &LlamaModel,
        token: u32,
        pos: usize,
        last: bool,
    ) -> Result<u32> {
        self.run(model, token, pos, if last { 2 } else { 0 }, &mut [])
    }
}
impl Drop for Resident {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context as *mut c_void);
        }
    }
}
