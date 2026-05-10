//! Llama forward: dense `f32` weights or mmap-backed quantized tensors (GEMV without full dequant).

use std::sync::Arc;

use crate::backend::{ComputeBackend, CpuBackend};
use crate::error::{BitNetError, Result};
use crate::ggml::{
    embedding_row_mmap, matvec_embd_out_mmap, matvec_ff_mmap, tensor_to_f32,
    ggml_type_supported_mmap_matvec,
};
use crate::gguf::{GgufArchive, GgufTensorInfo};

use super::config::LlamaConfig;
use super::kv_storage::{KvCache, KvStorage};

/// How Llama matrices are stored / executed (`RBITNET_LLAMA_WEIGHT_MODE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlamaWeightMode {
    /// Legacy: full `tensor_to_f32` at load (high RAM).
    Dense,
    /// mmap GGUF + row-wise quant GEMV (`mmap_quant`).
    MmapQuant,
    /// Use mmap quant when all weight tensors use supported GGML types; else dense.
    Auto,
}

pub fn llama_weight_mode_from_env() -> LlamaWeightMode {
    match std::env::var("RBITNET_LLAMA_WEIGHT_MODE").as_deref() {
        Ok(s) if s.eq_ignore_ascii_case("dense") => LlamaWeightMode::Dense,
        Ok(s) if s.eq_ignore_ascii_case("mmap_quant") => LlamaWeightMode::MmapQuant,
        Ok(s) if s.eq_ignore_ascii_case("auto") => LlamaWeightMode::Auto,
        Ok(other) => {
            tracing::warn!("unknown RBITNET_LLAMA_WEIGHT_MODE={other}, using auto");
            LlamaWeightMode::Auto
        }
        Err(_) => LlamaWeightMode::Auto,
    }
}

/// Large linear weights: either dense `f32` or a mmap tensor view.
#[derive(Clone)]
pub enum MatrixWeights {
    Dense(Vec<f32>),
    Quant {
        archive: Arc<GgufArchive>,
        tensor: GgufTensorInfo,
    },
}

impl MatrixWeights {
    fn matvec_embd_out(&self, x: &[f32], ne0: usize, ne1: usize) -> Result<Vec<f32>> {
        match self {
            Self::Dense(w) => Ok(matvec_embd_out_dense(w, x, ne0, ne1)),
            Self::Quant { archive, tensor } => {
                matvec_embd_out_mmap(archive.as_ref(), tensor, x, ne0, ne1)
            }
        }
    }

    fn matvec_ff(&self, x: &[f32], n_ff: usize, n_embd: usize) -> Result<Vec<f32>> {
        match self {
            Self::Dense(w) => Ok(matvec_ff_embd_dense(w, x, n_ff, n_embd)),
            Self::Quant { archive, tensor } => {
                matvec_ff_mmap(archive.as_ref(), tensor, x, n_ff, n_embd)
            }
        }
    }

    fn embed_row(&self, tok: usize, n_embd: usize, n_vocab: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::Dense(v) => {
                for j in 0..n_embd {
                    out[j] = v[j + tok * n_embd];
                }
                Ok(())
            }
            Self::Quant { archive, tensor } => embedding_row_mmap(
                archive.as_ref(),
                tensor,
                tok,
                n_embd,
                n_vocab,
                out,
            ),
        }
    }
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: MatrixWeights,
    pub wk: MatrixWeights,
    pub wv: MatrixWeights,
    pub wo: MatrixWeights,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: MatrixWeights,
    pub ffn_up: MatrixWeights,
    pub ffn_down: MatrixWeights,
}

pub struct LlamaModel {
    pub cfg: LlamaConfig,
    pub token_embd: MatrixWeights,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: MatrixWeights,
}

fn load_tensor_dense(archive: &GgufArchive, names: &[&str]) -> Result<Vec<f32>> {
    let t = archive.tensor_first_of(names).ok_or_else(|| {
        BitNetError::Inference(format!("missing tensor (tried {:?})", names))
    })?;
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn load_tensor_strings_dense(archive: &GgufArchive, names: &[String]) -> Result<Vec<f32>> {
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    load_tensor_dense(archive, &refs)
}

fn tensor_info_first(archive: &GgufArchive, names: &[&str]) -> Result<GgufTensorInfo> {
    let t = archive.tensor_first_of(names).ok_or_else(|| {
        BitNetError::Inference(format!("missing tensor (tried {:?})", names))
    })?;
    Ok(t.clone())
}

fn tensor_info_strings(archive: &GgufArchive, names: &[String]) -> Result<GgufTensorInfo> {
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    tensor_info_first(archive, &refs)
}

fn matrix_mmap_supported(t: &GgufTensorInfo) -> bool {
    ggml_type_supported_mmap_matvec(t.ggml_type)
}

/// Returns `Ok(())` if every Llama weight matrix uses a GGML type we can mmap-GEMV.
pub fn llama_mmap_quant_supported(archive: &GgufArchive) -> Result<()> {
    let cfg = LlamaConfig::from_gguf(archive)?;
    let check = |names: &[&str]| -> Result<()> {
        let t = archive.tensor_first_of(names).ok_or_else(|| {
            BitNetError::Inference(format!("mmap check: missing tensor {:?}", names))
        })?;
        if !matrix_mmap_supported(t) {
            return Err(BitNetError::UnsupportedGgmlType(t.ggml_type));
        }
        Ok(())
    };

    check(&["token_embd.weight", "token_embd"])?;
    check(&["output.weight", "lm_head.weight"])?;

    for i in 0..cfg.n_layer {
        let p = format!("blk.{i}");
        check(&[&format!("{p}.attn_q.weight")])?;
        check(&[&format!("{p}.attn_k.weight")])?;
        check(&[&format!("{p}.attn_v.weight")])?;
        check(&[&format!("{p}.attn_output.weight"), &format!("{p}.attn_out.weight")])?;
        check(&[&format!("{p}.ffn_gate.weight")])?;
        check(&[&format!("{p}.ffn_up.weight")])?;
        check(&[&format!("{p}.ffn_down.weight")])?;
    }
    Ok(())
}

fn llama_mmap_quant_supported_ok(archive: &GgufArchive) -> bool {
    llama_mmap_quant_supported(archive).is_ok()
}

/// `y[out] = sum_i W[i + out * n_embd] * x[i]` — GGUF layout `ne[0]=n_embd`, `ne[1]=out`.
fn matvec_embd_out_dense(w: &[f32], x: &[f32], n_embd: usize, n_out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_out];
    for o in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n_embd {
            acc += w[i + o * n_embd] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// `ffn_down`: `ne[0]=n_ff`, `ne[1]=n_embd` — `y[out] = sum_i W[i + out * n_ff] * x[i]`.
fn matvec_ff_embd_dense(w: &[f32], x: &[f32], n_ff: usize, n_embd: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_embd];
    for o in 0..n_embd {
        let mut acc = 0.0f32;
        for i in 0..n_ff {
            acc += w[i + o * n_ff] * x[i];
        }
        y[o] = acc;
    }
    y
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
    let scale = 1.0 / (s + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * scale)
        .collect()
}

fn softmax_inplace(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for z in s.iter_mut() {
        *z = (*z - m).exp();
        sum += *z;
    }
    if sum > 0.0 {
        for z in s.iter_mut() {
            *z /= sum;
        }
    }
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v / (1.0 + (-v).exp()))
        .collect()
}

fn rope_inplace(slice: &mut [f32], pos: usize, theta: f32) {
    let h = slice.len();
    assert!(h % 2 == 0);
    let half = h / 2;
    for i in 0..half {
        let inv_freq = 1.0 / theta.powf(2.0 * (i as f32) / (h as f32));
        let angle = pos as f32 * inv_freq;
        let c = angle.cos();
        let s = angle.sin();
        let x0 = slice[2 * i];
        let x1 = slice[2 * i + 1];
        slice[2 * i] = x0 * c - x1 * s;
        slice[2 * i + 1] = x0 * s + x1 * c;
    }
}

impl LlamaModel {
    /// Load from mmap archive using `RBITNET_LLAMA_WEIGHT_MODE`.
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        Self::from_gguf_arc(Arc::new(archive.clone()))
    }

    /// Preferred entry: keeps a single `Arc` to the mmap-backed archive for quant weights.
    pub fn from_gguf_arc(archive: Arc<GgufArchive>) -> Result<Self> {
        let mode = llama_weight_mode_from_env();
        match mode {
            LlamaWeightMode::Dense => Self::from_gguf_dense_internal(archive),
            LlamaWeightMode::MmapQuant => {
                if llama_mmap_quant_supported(archive.as_ref()).is_err() {
                    return Err(BitNetError::Inference(
                        "mmap_quant: unsupported ggml_type on one or more Llama matrices \
                         (see ggml::ggml_type_supported_mmap_matvec)"
                            .into(),
                    ));
                }
                Self::from_gguf_mmap_internal(archive)
            }
            LlamaWeightMode::Auto => {
                if llama_mmap_quant_supported_ok(archive.as_ref()) {
                    Self::from_gguf_mmap_internal(archive)
                } else {
                    Self::from_gguf_dense_internal(archive)
                }
            }
        }
    }

    fn from_gguf_dense_internal(archive: Arc<GgufArchive>) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        let n_embd = cfg.n_embd;
        let n_vocab = cfg.n_vocab;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = MatrixWeights::Dense(load_tensor_dense(
            archive.as_ref(),
            &["token_embd.weight", "token_embd"],
        )?);
        if let MatrixWeights::Dense(ref v) = token_embd {
            if v.len() != n_embd * n_vocab {
                return Err(BitNetError::Inference(
                    "token_embd.weight element count mismatch".into(),
                ));
            }
        }

        let output_norm = load_tensor_dense(archive.as_ref(), &["output_norm.weight"])?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = MatrixWeights::Dense(load_tensor_dense(
            archive.as_ref(),
            &["output.weight", "lm_head.weight"],
        )?);
        if let MatrixWeights::Dense(ref v) = output {
            if v.len() != n_embd * n_vocab {
                return Err(BitNetError::Inference("output.weight shape mismatch".into()));
            }
        }

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.attn_norm.weight")])?;
            let wq = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_q.weight")],
            )?);
            let wk = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_k.weight")],
            )?);
            let wv = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.attn_v.weight")],
            )?);
            let wo = MatrixWeights::Dense(load_tensor_strings_dense(archive.as_ref(), &[
                format!("{p}.attn_output.weight"),
                format!("{p}.attn_out.weight"),
            ])?);
            let ffn_norm = load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_norm.weight")])?;
            let ffn_gate = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_gate.weight")],
            )?);
            let ffn_up = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_up.weight")],
            )?);
            let ffn_down = MatrixWeights::Dense(load_tensor_strings_dense(
                archive.as_ref(),
                &[format!("{p}.ffn_down.weight")],
            )?);

            Self::validate_layer_dense(
                i,
                &attn_norm,
                &wq,
                &wk,
                &wv,
                &wo,
                &ffn_norm,
                &ffn_gate,
                &ffn_up,
                &ffn_down,
                n_embd,
                n_embd_kv,
                n_ff,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    fn validate_layer_dense(
        i: usize,
        attn_norm: &[f32],
        wq: &MatrixWeights,
        wk: &MatrixWeights,
        wv: &MatrixWeights,
        wo: &MatrixWeights,
        ffn_norm: &[f32],
        ffn_gate: &MatrixWeights,
        ffn_up: &MatrixWeights,
        ffn_down: &MatrixWeights,
        n_embd: usize,
        n_embd_kv: usize,
        n_ff: usize,
    ) -> Result<()> {
        let len = |m: &MatrixWeights| -> Result<usize> {
            match m {
                MatrixWeights::Dense(v) => Ok(v.len()),
                MatrixWeights::Quant { .. } => Err(BitNetError::Inference(
                    "validate_layer_dense: expected dense matrix".into(),
                )),
            }
        };
        if attn_norm.len() != n_embd
            || len(wq)? != n_embd * n_embd
            || len(wk)? != n_embd * n_embd_kv
            || len(wv)? != n_embd * n_embd_kv
            || len(wo)? != n_embd * n_embd
            || ffn_norm.len() != n_embd
            || len(ffn_gate)? != n_embd * n_ff
            || len(ffn_up)? != n_embd * n_ff
            || len(ffn_down)? != n_ff * n_embd
        {
            return Err(BitNetError::Inference(format!(
                "layer {i} weight shape mismatch"
            )));
        }
        Ok(())
    }

    fn from_gguf_mmap_internal(archive: Arc<GgufArchive>) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(archive.as_ref())?;
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let n_embd_kv = cfg.n_kv * cfg.head_dim;

        let token_embd = MatrixWeights::Quant {
            archive: Arc::clone(&archive),
            tensor: tensor_info_first(archive.as_ref(), &["token_embd.weight", "token_embd"])?,
        };

        let output_norm = load_tensor_dense(archive.as_ref(), &["output_norm.weight"])?;
        if output_norm.len() != n_embd {
            return Err(BitNetError::Inference(
                "output_norm.weight shape mismatch".into(),
            ));
        }

        let output = MatrixWeights::Quant {
            archive: Arc::clone(&archive),
            tensor: tensor_info_first(archive.as_ref(), &["output.weight", "lm_head.weight"])?,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.attn_norm.weight")])?;
            let wq = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_q.weight")])?,
            };
            let wk = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_k.weight")])?,
            };
            let wv = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.attn_v.weight")])?,
            };
            let wo = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[
                    format!("{p}.attn_output.weight"),
                    format!("{p}.attn_out.weight"),
                ])?,
            };
            let ffn_norm = load_tensor_strings_dense(archive.as_ref(), &[format!("{p}.ffn_norm.weight")])?;
            let ffn_gate = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_gate.weight")])?,
            };
            let ffn_up = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_up.weight")])?,
            };
            let ffn_down = MatrixWeights::Quant {
                archive: Arc::clone(&archive),
                tensor: tensor_info_strings(archive.as_ref(), &[format!("{p}.ffn_down.weight")])?,
            };

            Self::validate_layer_mmap(
                archive.as_ref(),
                i,
                &attn_norm,
                &wq,
                &wk,
                &wv,
                &wo,
                &ffn_norm,
                &ffn_gate,
                &ffn_up,
                &ffn_down,
                n_embd,
                n_embd_kv,
                n_ff,
            )?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    fn validate_layer_mmap(
        _archive: &GgufArchive,
        i: usize,
        attn_norm: &[f32],
        wq: &MatrixWeights,
        wk: &MatrixWeights,
        wv: &MatrixWeights,
        wo: &MatrixWeights,
        ffn_norm: &[f32],
        ffn_gate: &MatrixWeights,
        ffn_up: &MatrixWeights,
        ffn_down: &MatrixWeights,
        n_embd: usize,
        n_embd_kv: usize,
        n_ff: usize,
    ) -> Result<()> {
        let nelements = |m: &MatrixWeights| -> Result<usize> {
            match m {
                MatrixWeights::Dense(_) => Err(BitNetError::Inference(
                    "validate mmap: unexpected dense".into(),
                )),
                MatrixWeights::Quant { tensor, .. } => Ok(tensor
                    .dimensions
                    .iter()
                    .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
                    .ok_or_else(|| BitNetError::InvalidGguf("tensor dims".into()))?),
            }
        };
        if attn_norm.len() != n_embd
            || nelements(wq)? != n_embd * n_embd
            || nelements(wk)? != n_embd * n_embd_kv
            || nelements(wv)? != n_embd * n_embd_kv
            || nelements(wo)? != n_embd * n_embd
            || ffn_norm.len() != n_embd
            || nelements(ffn_gate)? != n_embd * n_ff
            || nelements(ffn_up)? != n_embd * n_ff
            || nelements(ffn_down)? != n_ff * n_embd
        {
            return Err(BitNetError::Inference(format!(
                "layer {i} weight shape mismatch (mmap)"
            )));
        }
        // Touch payload bounds once per matrix
        let touch = |m: &MatrixWeights| -> Result<()> {
            if let MatrixWeights::Quant { archive, tensor } = m {
                archive.tensor_payload(tensor)?;
            }
            Ok(())
        };
        touch(wq)?;
        touch(wk)?;
        touch(wv)?;
        touch(wo)?;
        touch(ffn_gate)?;
        touch(ffn_up)?;
        touch(ffn_down)?;
        Ok(())
    }

    /// Run one forward step: token embedding + all layers + output matmul. Returns logits `[n_vocab]`.
    #[allow(dead_code)]
    pub fn forward(&self, kv: &mut KvCache, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cpu = CpuBackend;
        let mut wrap = KvStorage::Dense(std::mem::replace(kv, KvCache::new(&self.cfg)));
        let out = self.forward_with_backend(&mut wrap, token, pos, &cpu);
        if let KvStorage::Dense(d) = wrap {
            *kv = d;
        }
        out
    }

    pub fn forward_with_backend(
        &self,
        kv: &mut KvStorage,
        token: u32,
        pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference("sequence position >= max_seq".into()));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let n_embd = cfg.n_embd;
        let mut x = vec![0.0f32; n_embd];
        self.token_embd
            .embed_row(tok, n_embd, cfg.n_vocab, &mut x)?;

        let n_rep = cfg.n_head / cfg.n_kv;

        for (il, layer) in self.layers.iter().enumerate() {
            let h = rmsnorm(&x, &layer.attn_norm, cfg.norm_eps);
            let q = layer.wq.matvec_embd_out(&h, n_embd, n_embd)?;
            let k = layer
                .wk
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;
            let v = layer
                .wv
                .matvec_embd_out(&h, n_embd, cfg.n_kv * cfg.head_dim)?;

            let mut q_heads = q;
            for h in 0..cfg.n_head {
                let s = &mut q_heads[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                rope_inplace(s, pos, cfg.rope_theta);
            }

            let mut k_heads = k;
            for h in 0..cfg.n_kv {
                let s = &mut k_heads[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                rope_inplace(s, pos, cfg.rope_theta);
            }

            let stride = cfg.n_kv * cfg.head_dim;
            kv.write_layer_kv(il, pos, &k_heads, &v, stride)?;

            let mut attn_out = vec![0.0f32; n_embd];
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();

            for qh in 0..cfg.n_head {
                let kv_h = qh / n_rep;
                let q_slice = &q_heads[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                let mut scores: Vec<f32> = if backend.kind() == crate::backend::BackendKind::Cpu {
                    (0..=pos)
                        .map(|p| {
                            let k_slice =
                                kv.k_head_slice(il, p, kv_h, cfg.head_dim, stride);
                            let dot: f32 =
                                q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
                            dot * scale
                        })
                        .collect()
                } else {
                    let mut k_mat = vec![0.0f32; (pos + 1) * cfg.head_dim];
                    kv.fill_k_rows_gpu(il, pos, kv_h, cfg.head_dim, stride, &mut k_mat);
                    let mut s = backend.matvec(&k_mat, q_slice, pos + 1, cfg.head_dim)?;
                    for v in &mut s {
                        *v *= scale;
                    }
                    s
                };
                softmax_inplace(&mut scores);
                let mut comb = vec![0.0f32; cfg.head_dim];
                for p in 0..=pos {
                    let v_slice = kv.v_head_slice(il, p, kv_h, cfg.head_dim, stride);
                    let sp = scores[p];
                    for i in 0..cfg.head_dim {
                        comb[i] += sp * v_slice[i];
                    }
                }
                let dst = qh * cfg.head_dim;
                attn_out[dst..dst + cfg.head_dim].copy_from_slice(&comb);
            }

            let y = layer.wo.matvec_embd_out(&attn_out, n_embd, n_embd)?;
            for i in 0..n_embd {
                x[i] += y[i];
            }

            let h2 = rmsnorm(&x, &layer.ffn_norm, cfg.norm_eps);
            let gate = silu(&layer.ffn_gate.matvec_embd_out(&h2, n_embd, cfg.n_ff)?);
            let up = layer.ffn_up.matvec_embd_out(&h2, n_embd, cfg.n_ff)?;
            let mut tmp = vec![0.0f32; cfg.n_ff];
            for i in 0..cfg.n_ff {
                tmp[i] = gate[i] * up[i];
            }
            let y2 = layer.ffn_down.matvec_ff(&tmp, cfg.n_ff, n_embd)?;
            for i in 0..n_embd {
                x[i] += y2[i];
            }
        }

        let xn = rmsnorm(&x, &self.output_norm, cfg.norm_eps);
        self.output
            .matvec_embd_out(&xn, n_embd, cfg.n_vocab)
    }
}
