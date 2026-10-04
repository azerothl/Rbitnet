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
type CreatePaged = unsafe extern "C" fn(
    *const Config,
    *const Layer,
    *const Matrix,
    *const f32,
    *const f32,
    u32,
    *const c_void,
    u32,
) -> *mut c_void;
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub(crate) struct PageStats {
    pub allocated_pages: u64,
    pub peak_pages: u64,
    pub limit_pages: u64,
    pub bytes_per_page: u64,
    pub referenced_pages: u64,
    pub references: u64,
    pub active_pages: u64,
    pub tokens: u64,
    pub allocations: u64,
    pub reuses: u64,
    pub cow_pages: u64,
    pub refusals: u64,
}
type QueryPages = unsafe extern "C" fn(*const c_void, *mut PageStats) -> i32;
type TrimPages = unsafe extern "C" fn(*mut c_void) -> i32;
struct Pages {
    limit: u32,
    query: QueryPages,
    trim: TrimPages,
}
pub(crate) fn configured_page_limit() -> Result<Option<u32>> {
    match std::env::var("RBITNET_CUDA_KV_PAGE_LIMIT") {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(s) if s == "0" => Ok(None),
        Ok(s) => s
            .parse::<u32>()
            .ok()
            .filter(|n| (1..=65536).contains(n))
            .map(Some)
            .ok_or_else(|| {
                BitNetError::Inference(
                    "RBITNET_CUDA_KV_PAGE_LIMIT must be 0 or 1..65536 pages".into(),
                )
            }),
        Err(_) => Err(BitNetError::Inference(
            "RBITNET_CUDA_KV_PAGE_LIMIT is not Unicode".into(),
        )),
    }
}
type CreateKv = unsafe extern "C" fn(
    *const Config,
    *const Layer,
    *const Matrix,
    *const f32,
    *const f32,
    u32,
    *const c_void,
    u32,
    u32,
) -> *mut c_void;
pub(crate) fn configured_kv_format() -> Result<u32> {
    match std::env::var("RBITNET_CUDA_KV_FORMAT") {
        Err(std::env::VarError::NotPresent) => Ok(0),
        Ok(s) if s == "f32" => Ok(0),
        Ok(s) if s == "f16" => Ok(1),
        Ok(s) if s == "q8" => Ok(2),
        _ => Err(BitNetError::Inference(
            "RBITNET_CUDA_KV_FORMAT must be f32, f16 or q8".into(),
        )),
    }
}
type Step = unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, *mut f32, *mut u32) -> i32;
type Prefill =
    unsafe extern "C" fn(*mut c_void, *const f32, u32, u32, u32, *mut f32, *mut u32) -> i32;
type Snapshot = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;
type Restore = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;
type Truncate = unsafe extern "C" fn(*mut c_void, u32) -> i32;

pub(super) enum Verified {
    Greedy(Vec<u32>),
    Logits(Vec<f32>, usize),
}
impl Verified {
    pub fn sample(
        &self,
        index: usize,
        options: &crate::sampling::SamplingOptions,
        history: &[u32],
        rng: &mut impl rand::Rng,
    ) -> u32 {
        match self {
            Self::Greedy(tokens) => tokens[index],
            Self::Logits(logits, vocab) => crate::sampling::sample_token(
                &logits[index * vocab..(index + 1) * vocab],
                options,
                history,
                rng,
            ),
        }
    }
}

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
    verify: Option<Prefill>,
    truncate: Option<Truncate>,
    prefill_embeddings: Vec<f32>,
    destroy: Destroy,
    _weights: Vec<CudaDeviceQuantMatrix>,
    embedding: Vec<f32>,
    vocab: usize,
    capacity: usize,
    matrix_count: u64,
    layers: u64,
    graphs: bool,
    split_layers: u64,
    tensor_gemm_calls: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    snapshots: Option<SnapshotApi>,
    prefixes: crate::native::prefix::PrefixStore<SavedPrefix>,
    kv_bytes_per_token: usize,
    pages: Option<Pages>,
    kv_format: u32,
}
impl Resident {
    pub fn new(model: &LlamaModel) -> Option<Self> {
        Self::new_with_pages(model, configured_page_limit().ok()?, None)
    }
    pub(super) fn new_with_pages(
        model: &LlamaModel,
        page_limit: Option<u32>,
        peer: Option<&Self>,
    ) -> Option<Self> {
        if peer.is_some() && (page_limit.is_none() || peer?.pages.as_ref()?.limit != page_limit?) {
            return None;
        }
        if std::env::var("RBITNET_CUDA_RESIDENT").as_deref() == Ok("0") {
            return None;
        }
        let kv_format = configured_kv_format().ok()?;
        if page_limit.is_some_and(|n| n == 0 || n > 65536) {
            return None;
        }
        if peer.is_some_and(|p| p.kv_format != kv_format) {
            return None;
        }
        let c = &model.cfg;
        let vector_bytes = match kv_format {
            0 => c.head_dim.checked_mul(4)?,
            1 => c.head_dim.checked_mul(2)?,
            2 => c.head_dim.checked_add(4)?,
            _ => return None,
        };
        let kv_bytes_per_token = c
            .n_layer
            .checked_mul(c.n_kv)?
            .checked_mul(vector_bytes)?
            .checked_mul(2)?;
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
        let verify = unsafe {
            lib.get::<Prefill>(b"rbitnet_cuda_llama_verify\0")
                .ok()
                .map(|p| *p)
        };
        let truncate = unsafe {
            lib.get::<Truncate>(b"rbitnet_cuda_llama_truncate\0")
                .ok()
                .map(|p| *p)
        };
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
        let pages = if let Some(limit) = page_limit {
            let query = unsafe {
                *lib.get::<QueryPages>(b"rbitnet_cuda_llama_paged_stats\0")
                    .ok()?
            };
            let trim = unsafe {
                *lib.get::<TrimPages>(b"rbitnet_cuda_llama_paged_trim\0")
                    .ok()?
            };
            Some(Pages { limit, query, trim })
        } else {
            None
        };
        let variants = u32::from(std::env::var("RBITNET_CUDA_SPLIT_KV").as_deref() == Ok("1"))
            | (u32::from(std::env::var("RBITNET_CUDA_PREFILL_TF32X3").as_deref() == Ok("1")) << 1);
        let context = if kv_format != 0 {
            let create_kv = unsafe {
                *lib.get::<CreateKv>(b"rbitnet_cuda_llama_create_kv\0")
                    .ok()?
            };
            unsafe {
                create_kv(
                    &cfg,
                    layers.as_ptr(),
                    &output,
                    model.output_norm.as_ptr(),
                    frequency.as_ptr(),
                    page_limit.unwrap_or(0),
                    peer.map_or(std::ptr::null(), |p| p.context as *const c_void),
                    variants,
                    kv_format,
                )
            }
        } else if let Some(limit) = page_limit {
            let create_paged = unsafe {
                *lib.get::<CreatePaged>(b"rbitnet_cuda_llama_create_paged\0")
                    .ok()?
            };
            let variants = u32::from(std::env::var("RBITNET_CUDA_SPLIT_KV").as_deref() == Ok("1"))
                | (u32::from(std::env::var("RBITNET_CUDA_PREFILL_TF32X3").as_deref() == Ok("1"))
                    << 1);
            unsafe {
                create_paged(
                    &cfg,
                    layers.as_ptr(),
                    &output,
                    model.output_norm.as_ptr(),
                    frequency.as_ptr(),
                    limit,
                    peer.map_or(std::ptr::null(), |p| p.context as *const c_void),
                    variants,
                )
            }
        } else {
            unsafe {
                create(
                    &cfg,
                    layers.as_ptr(),
                    &output,
                    model.output_norm.as_ptr(),
                    frequency.as_ptr(),
                )
            }
        } as usize;
        if context == 0 {
            return None;
        }
        type SplitLayers = unsafe extern "C" fn(*mut c_void) -> u32;
        let split_layers = unsafe {
            lib.get::<SplitLayers>(b"rbitnet_cuda_llama_split_attention_layers\0")
                .map_or(0, |query| query(context as *mut c_void) as u64)
        };
        type ConfigureTensor = unsafe extern "C" fn(*mut c_void, u32) -> i32;
        if let Ok(configure) =
            unsafe { lib.get::<ConfigureTensor>(b"rbitnet_cuda_llama_configure_tensor_prefill\0") }
        {
            let enabled =
                u32::from(std::env::var("RBITNET_CUDA_PREFILL_TF32X3").as_deref() == Ok("1"));
            let status = unsafe { configure(context as *mut c_void, enabled) };
            if status != 0 {
                unsafe { destroy(context as *mut c_void) };
                return None;
            }
        }
        let tensor_gemm_calls = unsafe {
            lib.get::<SplitLayers>(b"rbitnet_cuda_llama_tensor_gemm_calls\0")
                .ok()
                .map(|query| *query)
        };
        tracing::info!(
            graphs = cfg.graphs != 0,
            split_layers,
            "fully resident Llama CUDA token graph enabled"
        );
        Some(Self {
            context,
            step,
            prefill,
            verify,
            truncate,
            prefill_embeddings: Vec::new(),
            destroy,
            _weights: weights,
            embedding: vec![0.0; c.n_embd],
            vocab: c.n_vocab,
            capacity: c.max_seq,
            matrix_count: 7 * c.n_layer as u64,
            layers: c.n_layer as u64,
            graphs: cfg.graphs != 0,
            split_layers,
            snapshots,
            tensor_gemm_calls,
            prefixes: crate::native::prefix::PrefixStore::from_env(),
            kv_bytes_per_token,
            pages,
            kv_format,
        })
    }
    pub(crate) fn page_stats(&self) -> Result<Option<PageStats>> {
        let Some(pages) = &self.pages else {
            return Ok(None);
        };
        let mut out = PageStats::default();
        let status = unsafe { (pages.query)(self.context as *const c_void, &mut out) };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA physical KV page stats failed: {status}"
            )));
        }
        Ok(Some(out))
    }
    pub(crate) fn trim_pages(&mut self) -> Result<()> {
        if let Some(pages) = &self.pages {
            let status = unsafe { (pages.trim)(self.context as *mut c_void) };
            if status != 0 {
                return Err(BitNetError::Inference(format!(
                    "CUDA physical KV page trim failed: {status}"
                )));
            }
        }
        Ok(())
    }
    pub fn supports_prefix_cache(&self) -> bool {
        self.snapshots.is_some()
    }
    fn record_tensor_gemm(&self) {
        let calls = self
            .tensor_gemm_calls
            .map_or(0, |query| unsafe { query(self.context as *mut c_void) });
        crate::perf::record_gpu_tensor_gemm(calls as u64);
    }
    pub fn supports_verification(&self) -> bool {
        self.verify.is_some() && self.truncate.is_some()
    }
    pub fn remaining_capacity(&self, position: usize) -> usize {
        self.capacity.saturating_sub(position)
    }
    pub fn verify(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
        position: usize,
        greedy: bool,
    ) -> Result<Verified> {
        if tokens.is_empty()
            || tokens.len() > 16
            || tokens.len() > self.remaining_capacity(position)
        {
            return Err(BitNetError::Inference(
                "verification block/context out of bounds".into(),
            ));
        }
        let verify = self
            .verify
            .ok_or_else(|| BitNetError::Inference("CUDA verification API unavailable".into()))?;
        self.prefill_embeddings
            .resize(tokens.len() * model.cfg.n_embd, 0.0);
        for (&token, row) in tokens
            .iter()
            .zip(self.prefill_embeddings.chunks_exact_mut(model.cfg.n_embd))
        {
            if token as usize >= self.vocab {
                return Err(BitNetError::Inference(
                    "verification token out of bounds".into(),
                ));
            }
            model
                .token_embd
                .embed_row(token as usize, model.cfg.n_embd, self.vocab, row)?;
        }
        let mut logits = if greedy {
            Vec::new()
        } else {
            vec![0.0; tokens.len() * self.vocab]
        };
        let mut next = if greedy {
            vec![0; tokens.len()]
        } else {
            Vec::new()
        };
        let status = unsafe {
            verify(
                self.context as *mut c_void,
                self.prefill_embeddings.as_ptr(),
                position as u32,
                tokens.len() as u32,
                if greedy { 2 } else { 1 },
                logits.as_mut_ptr(),
                next.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA verification failed: {status}"
            )));
        }
        crate::perf::record_gpu_verification(
            tokens.len(),
            if tokens.len() > 1 {
                self.matrix_count + 1
            } else {
                0
            },
        );
        if self.graphs {
            crate::perf::record_cuda_graph_replay();
        }
        crate::perf::record_gpu_transfer(
            (self.prefill_embeddings.len() * 4 + 4) as u64,
            (tokens.len() * if greedy { 4 } else { self.vocab * 4 }) as u64,
            if tokens.len() == 1 {
                self.matrix_count + 1
            } else {
                0
            },
        );
        for _ in 0..self.layers {
            crate::perf::record_gpu_attention();
        }
        crate::perf::record_split_attention(self.split_layers * tokens.len() as u64);
        self.record_tensor_gemm();
        Ok(if greedy {
            Verified::Greedy(next)
        } else {
            Verified::Logits(logits, self.vocab)
        })
    }
    pub fn truncate(&mut self, length: usize) -> Result<()> {
        let truncate = self
            .truncate
            .ok_or_else(|| BitNetError::Inference("CUDA truncate API unavailable".into()))?;
        let length = u32::try_from(length)
            .map_err(|_| BitNetError::Inference("CUDA truncate length overflow".into()))?;
        let status = unsafe { truncate(self.context as *mut c_void, length) };
        if status != 0 {
            return Err(BitNetError::Inference(format!(
                "CUDA truncate failed: {status}"
            )));
        }
        Ok(())
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
            crate::perf::record_split_attention(self.split_layers * chunk.len() as u64);
            self.record_tensor_gemm();
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
        crate::perf::record_split_attention(self.split_layers);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_real_tensor_prefill_teacher_forcing_matches_simt() {
        if std::env::var("RBITNET_CUDA_TENSOR_TEST").as_deref() != Ok("1") {
            return;
        }
        let gguf = std::env::var("RBITNET_TEST_GGUF").expect("real GGUF required");
        let tokenizer = tokenizers::Tokenizer::from_file(
            std::env::var("RBITNET_TOKENIZER").expect("tokenizer required"),
        )
        .unwrap();
        let archive = std::sync::Arc::new(
            crate::gguf::GgufArchive::mmap_path(std::path::Path::new(&gguf)).unwrap(),
        );
        let model =
            LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda)
                .unwrap();
        std::env::set_var("RBITNET_CUDA_PREFILL", "1");
        std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
        let mut reference = Resident::new(&model).unwrap();
        std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "1");
        let mut tensor = Resident::new(&model).unwrap();
        let before = crate::perf::snapshot().gpu_tensor_gemm_calls;
        let mut checked = 0;
        let mut worst_kl: f64 = 0.0;
        let mut worst_nll_delta: f64 = 0.0;
        for content in [
            "Le robot entre dans une bibliothèque et cherche un livre sur les étoiles. Il découvre un jardin calme derrière le bâtiment. ",
            "fn sum(values: &[i32]) -> i32 { values.iter().sum() } // Explain the time complexity and provide a checked version. ",
        ] {
            let prompt=format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",content.repeat(20));
            let ids=tokenizer.encode(prompt,false).unwrap();let ids=ids.get_ids();
            for count in [63,64,129,257] {
                let (mut expected,_)=reference.prefill(&model,&ids[..count],0,false).unwrap();
                let (mut actual,_)=tensor.prefill(&model,&ids[..count],0,false).unwrap();
                for pos in count..count+6 {
                    let log_probs=|x:&[f32]| {
                        let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
                        let log_z=x.iter().map(|&v|(v as f64-max).exp()).sum::<f64>().ln()+max;
                        x.iter().map(|&v|v as f64-log_z).collect::<Vec<_>>()
                    };
                    let p=log_probs(&expected);let q=log_probs(&actual);
                    let kl=p.iter().zip(&q).map(|(&p,&q)|p.exp()*(p-q)).sum::<f64>();
                    let nll_delta=(p[ids[pos] as usize]-q[ids[pos] as usize]).abs();
                    worst_kl=worst_kl.max(kl);worst_nll_delta=worst_nll_delta.max(nll_delta);
                    assert!(kl.is_finite() && kl<=1e-5 && nll_delta<=1e-3,"count={count} pos={pos} KL={kl} NLL delta={nll_delta}");
                    let max_id=|x:&[f32]|x.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
                    assert_eq!(max_id(&actual),max_id(&expected));
                    for (&a,&b) in actual.iter().zip(&expected) {assert!(a.is_finite() && (a-b).abs()<=0.003*(1.0+b.abs()));}
                    checked+=1;
                    if pos+1<count+6 {
                        expected=reference.forward(&model,ids[pos],pos,true).unwrap();
                        actual=tensor.forward(&model,ids[pos],pos,true).unwrap();
                    }
                }
            }
        }
        assert!(crate::perf::snapshot().gpu_tensor_gemm_calls > before);
        eprintln!("Teacher-forced positions={checked}, worst KL={worst_kl:.3e}, worst abs NLL delta={worst_nll_delta:.3e}; argmax and logits within tolerance");
    }
    #[test]
    fn optional_real_verification_logits_argmax_and_rollback_match_serial() {
        if std::env::var("RBITNET_CUDA_VERIFY_TEST").as_deref() != Ok("1") {
            return;
        }
        let gguf = std::env::var("RBITNET_TEST_GGUF").expect("real model required");
        let archive = std::sync::Arc::new(
            crate::gguf::GgufArchive::mmap_path(std::path::Path::new(&gguf)).unwrap(),
        );
        let model =
            LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda)
                .unwrap();
        let mut block = Resident::new(&model).expect("resident required");
        let mut serial = Resident::new(&model).expect("resident required");
        assert!(block.supports_verification());
        let prefix = [128000, 791, 2735, 374, 264, 1296];
        let inputs = [
            13, 791, 649, 1296, 2735, 596, 128, 42, 293, 10, 31, 400, 128, 81, 237, 13,
        ];
        let close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (&a, &e) in actual.iter().zip(expected) {
                assert!(
                    a.is_finite() && e.is_finite() && (a - e).abs() <= 0.003 * (1.0 + e.abs()),
                    "logits diverged: {a} vs {e}"
                );
            }
            let max = |x: &[f32]| {
                x.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            assert_eq!(max(actual), max(expected));
        };
        for count in [1, 2, 7, 15, 16] {
            block.prefill(&model, &prefix, 0, false).unwrap();
            serial.prefill(&model, &prefix, 0, false).unwrap();
            let Verified::Logits(logits, vocab) = block
                .verify(&model, &inputs[..count], prefix.len(), false)
                .unwrap()
            else {
                panic!()
            };
            let mut expected_ids = Vec::new();
            for (i, &token) in inputs[..count].iter().enumerate() {
                let expected = serial
                    .forward(&model, token, prefix.len() + i, true)
                    .unwrap();
                close(&logits[i * vocab..(i + 1) * vocab], &expected);
                expected_ids.push(
                    expected
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32,
                );
            }
            block.prefill(&model, &prefix, 0, true).unwrap();
            let Verified::Greedy(ids) = block
                .verify(&model, &inputs[..count], prefix.len(), true)
                .unwrap()
            else {
                panic!()
            };
            assert_eq!(ids, expected_ids);
            // The rejected half remains physically allocated but must not affect a new tail.
            let kept = count / 2;
            block.truncate(prefix.len() + kept).unwrap();
            serial.prefill(&model, &prefix, 0, false).unwrap();
            for (i, &token) in inputs[..kept].iter().enumerate() {
                serial
                    .forward(&model, token, prefix.len() + i, false)
                    .unwrap();
            }
            // Replay the same logits graph at a different position and state.
            let Verified::Logits(replayed, vocab) = block
                .verify(&model, &inputs[..count], prefix.len() + kept, false)
                .unwrap()
            else {
                panic!()
            };
            for (i, &token) in inputs[..count].iter().enumerate() {
                close(
                    &replayed[i * vocab..(i + 1) * vocab],
                    &serial
                        .forward(&model, token, prefix.len() + kept + i, true)
                        .unwrap(),
                );
            }
            block.truncate(prefix.len() + kept).unwrap();
            serial.prefill(&model, &prefix, 0, false).unwrap();
            for (i, &token) in inputs[..kept].iter().enumerate() {
                serial
                    .forward(&model, token, prefix.len() + i, false)
                    .unwrap();
            }
            for (i, token) in [400, 2735, 791].into_iter().enumerate() {
                close(
                    &block
                        .forward(&model, token, prefix.len() + kept + i, true)
                        .unwrap(),
                    &serial
                        .forward(&model, token, prefix.len() + kept + i, true)
                        .unwrap(),
                );
            }
        }
        assert!(block.truncate(model.cfg.max_seq + 1).is_err());
        assert!(block.verify(&model, &[], 0, true).is_err());
    }
}

#[cfg(test)]
mod paged_tests;

#[cfg(test)]
mod quantized_tests;

#[cfg(test)]
#[path="resident/canonical_tests.rs"]
mod canonical_tests;

#[cfg(test)]
#[path="resident/quantized_long_tests.rs"]
mod quantized_long_tests;
