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

pub(super) struct Resident {
    context: usize,
    step: Step,
    destroy: Destroy,
    _weights: Vec<CudaDeviceQuantMatrix>,
    embedding: Vec<f32>,
    vocab: usize,
    capacity: usize,
    matrix_count: u64,
    layers: u64,
    graphs: bool,
}
impl Resident {
    pub fn new(model: &LlamaModel) -> Option<Self> {
        if std::env::var("RBITNET_CUDA_RESIDENT").as_deref() == Ok("0") {
            return None;
        }
        let c = &model.cfg;
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
            destroy,
            _weights: weights,
            embedding: vec![0.0; c.n_embd],
            vocab: c.n_vocab,
            capacity: c.max_seq,
            matrix_count: 7 * c.n_layer as u64,
            layers: c.n_layer as u64,
            graphs: cfg.graphs != 0,
        })
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
