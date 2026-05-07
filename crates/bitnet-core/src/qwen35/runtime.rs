//! One-token Transformer stack for hybrid Qwen3 MoE GGUF checkpoints.

use std::path::Path;
use std::sync::Arc;

use rand::Rng;
use tokenizers::Tokenizer;

use crate::backend::BackendKind;
use crate::error::{BitNetError, Result};
use crate::ggml::{ggml_nbytes, tensor_to_f32};
use crate::gguf::{GgufArchive, GgufTensorInfo};

use super::attention::{block_full_attention, AttnKvCache};
use super::config::Qwen35Config;
use super::cuda_ctx::QwenCudaContext;
use super::moe::{moe_forward, shared_expert_forward};
use super::qmatvec::token_embedding_row;
use super::recurrent::{recurrent_forward, RecurrentState};

pub struct Qwen35Runtime {
    cfg: Qwen35Config,
    archive: Arc<GgufArchive>,
    tokenizer: Tokenizer,
    attn_kv: AttnKvCache,
    rec: Vec<RecurrentState>,
    cuda: QwenCudaContext,
    tok_embd: GgufTensorInfo,
    out_norm: GgufTensorInfo,
    out_head: GgufTensorInfo,
}

fn must_tensor(archive: &GgufArchive, name: &str) -> Result<GgufTensorInfo> {
    archive
        .tensor_by_name(name)
        .cloned()
        .ok_or_else(|| BitNetError::Inference(format!("missing GGUF tensor `{name}`")))
}

fn tensor_f32_flat(archive: &GgufArchive, t: &GgufTensorInfo) -> Result<Vec<f32>> {
    let payload = archive.tensor_payload(t)?;
    tensor_to_f32(payload, t.ggml_type, &t.dimensions)
}

fn resolve_lm_head(archive: &GgufArchive, tie: &GgufTensorInfo) -> Result<GgufTensorInfo> {
    for &n in &["output.weight", "lm_head.weight"] {
        if archive.tensor_by_name(n).is_some() {
            return must_tensor(archive, n);
        }
    }
    Ok(tie.clone())
}

fn rms_combine(x: &[f32], w_info: &GgufTensorInfo, archive: &GgufArchive, eps: f32) -> Result<Vec<f32>> {
    let w = tensor_f32_flat(archive, w_info)?;
    if w.len() != x.len() {
        return Err(BitNetError::Inference("rmsnorm width mismatch vs hidden".into()));
    }
    let s = x.iter().map(|v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let sc = 1.0 / (s + eps).sqrt();
    Ok(x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi * wi * sc)
        .collect())
}

fn logits_project(
    archive: &GgufArchive,
    cuda: &QwenCudaContext,
    cfg: &Qwen35Config,
    head: &GgufTensorInfo,
    x_norm: &[f32],
    chunk_vocab: usize,
) -> Result<Vec<f32>> {
    let n_embd_w = usize::try_from(head.dimensions[0]).map_err(|_| BitNetError::Inference("out ne0".into()))?;
    let n_vocab_w = usize::try_from(head.dimensions[1]).map_err(|_| BitNetError::Inference("out ne1".into()))?;
    let py = archive.tensor_payload(head)?;
    let ggml_ty = head.ggml_type;
    let row_stride = ggml_nbytes(&[head.dimensions[0], 1], ggml_ty)
        .map_err(|_| BitNetError::Inference("output row stride".into()))?;

    if x_norm.len() != cfg.n_embd || n_embd_w != cfg.n_embd {
        return Err(BitNetError::Inference("embedding vs output weight mismatch".into()));
    }

    let mut logits = vec![0f32; cfg.n_vocab];
    let mut vocab_off = 0usize;
    while vocab_off < n_vocab_w.min(cfg.n_vocab) {
        let hi = chunk_vocab.min(n_vocab_w - vocab_off).max(1);
        let slice_off = vocab_off
            .checked_mul(row_stride)
            .ok_or_else(|| BitNetError::Inference("logits offset overflow".into()))?;
        let chunk_w = hi * row_stride;
        if slice_off.checked_add(chunk_w).filter(|e| *e <= py.len()).is_none() {
            return Err(BitNetError::Inference("output weight truncated".into()));
        }

        let wchunk = tensor_to_f32(
            &py[slice_off..slice_off + chunk_w],
            ggml_ty,
            &[n_embd_w as u64, hi as u64],
        )?;

        let y = cuda.logits_gemv_maybe(
            &wchunk,
            x_norm,
            hi,
            n_embd_w,
            || {
                let mut out = vec![0f32; hi];
                for j in 0..hi {
                    let mut sum = 0f32;
                    let base = j * cfg.n_embd;
                    for i in 0..cfg.n_embd {
                        sum += wchunk[base + i] * x_norm[i];
                    }
                    out[j] = sum;
                }
                out
            },
        );
        for i in 0..y.len() {
            logits[vocab_off + i] = y[i];
        }
        vocab_off += hi;
    }
    Ok(logits)
}

impl Qwen35Runtime {
    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer_path: &Path,
        cuda: QwenCudaContext,
        _backend_kind: BackendKind,
    ) -> Result<Self> {
        let cfg = Qwen35Config::from_gguf(archive.as_ref())?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BitNetError::Inference(format!("tokenizer load: {e}")))?;

        let tok_embd = must_tensor(
            archive.as_ref(),
            if archive.tensor_by_name("token_embd.weight").is_some() {
                "token_embd.weight"
            } else {
                "token_embd"
            },
        )?;
        let out_norm = must_tensor(archive.as_ref(), "output_norm.weight")?;
        let out_head = resolve_lm_head(archive.as_ref(), &tok_embd)?;

        let attn_kv = AttnKvCache::new(&cfg, cfg.max_seq);
        let d_conv = cfg.ssm_d_conv.max(2);
        let d_inner = cfg.ssm_d_inner.max(1);
        let sv_state = (cfg.ssm_d_inner / cfg.ssm_dt_rank.max(1)).max(1);
        let mut rec = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            rec.push(RecurrentState::new(
                d_conv,
                d_inner,
                sv_state,
                cfg.ssm_dt_rank.max(1),
            ));
        }

        Ok(Self {
            cfg,
            archive,
            tokenizer,
            attn_kv,
            rec,
            cuda,
            tok_embd,
            out_norm,
            out_head,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        self.attn_kv.clear();
        for st in &mut self.rec {
            st.conv_hist.fill(0f32);
            st.ssm_state.fill(0f32);
        }
        let arch = Arc::clone(&self.archive);
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| BitNetError::Inference(format!("encode: {e}")))?;
        let prompt_ids: Vec<u32> = enc.get_ids().iter().copied().collect();
        if prompt_ids.is_empty() {
            return Ok(String::new());
        }

        let mut logits = Vec::new();
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.forward_one(tid, pos, &arch)?;
        }

        let eos_id = self
            .tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"));

        let mut gen = Vec::new();
        let mut rng = rand::thread_rng();
        let mut pos = prompt_ids.len();

        for _ in 0..max_tokens {
            let next_id = sample_token(&logits, temperature, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            gen.push(next_id);
            logits = self.forward_one(next_id, pos, &arch)?;
            pos += 1;
        }

        self.tokenizer
            .decode(&gen, true)
            .map_err(|e| BitNetError::Inference(format!("decode: {e}")))
    }

    fn forward_one(&mut self, token: u32, pos: usize, archive: &Arc<GgufArchive>) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(BitNetError::Inference("sequence position >= max_seq".into()));
        }
        let tok = token as usize;
        if tok >= cfg.n_vocab {
            return Err(BitNetError::Inference("token id out of range".into()));
        }

        let mut x = token_embedding_row(archive, &self.tok_embd, tok, cfg.n_embd, cfg.n_vocab)?;

        for il in 0..cfg.n_layer {
            let residual = x.clone();
            let h = rms_combine(
                &x,
                &must_tensor(archive, &format!("blk.{il}.attn_norm.weight"))?,
                archive,
                cfg.norm_eps,
            )?;

            let y = if cfg.is_recurrent_layer(il) {
                let wqkv = must_tensor(archive, &format!("blk.{il}.attn_qkv.weight"))?;
                let wgate = must_tensor(archive, &format!("blk.{il}.attn_gate.weight"))?;
                let conv = must_tensor(archive, &format!("blk.{il}.ssm_conv1d.weight"))?;
                let conv_f = tensor_f32_flat(archive, &conv)?;
                let d0 = usize::try_from(conv.dimensions[0]).unwrap_or(0);
                let d1 = usize::try_from(conv.dimensions[1]).unwrap_or(0);
                let ssm_b = must_tensor(archive, &format!("blk.{il}.ssm_beta.weight"))?;
                let ssm_al = must_tensor(archive, &format!("blk.{il}.ssm_alpha.weight"))?;
                let ssm_dt = must_tensor(archive, &format!("blk.{il}.ssm_dt.bias"))?;
                let ssm_a = archive
                    .tensor_first_of(&[
                        &format!("blk.{il}.ssm_a_noscan.weight"),
                        &format!("blk.{il}.ssm_a.weight"),
                    ])
                    .cloned()
                    .ok_or_else(|| {
                        BitNetError::Inference(format!("layer {il}: missing ssm_a / ssm_a_noscan tensor"))
                    })?;
                let ssm_no = must_tensor(archive, &format!("blk.{il}.ssm_norm.weight"))?;
                let ssm_out = must_tensor(archive, &format!("blk.{il}.ssm_out.weight"))?;
                let dt_b = tensor_f32_flat(archive, &ssm_dt)?;
                let a_b = tensor_f32_flat(archive, &ssm_a)?;
                let norm_b = tensor_f32_flat(archive, &ssm_no)?;
                let wqkv_p = archive.tensor_payload(&wqkv)?;
                let wg_p = archive.tensor_payload(&wgate)?;
                let sb_p = archive.tensor_payload(&ssm_b)?;
                let sa_p = archive.tensor_payload(&ssm_al)?;
                let so_p = archive.tensor_payload(&ssm_out)?;
                recurrent_forward(
                    archive,
                    cfg,
                    &mut self.rec[il],
                    il,
                    &h,
                    (
                        wqkv_p,
                        wqkv.ggml_type,
                        usize::try_from(wqkv.dimensions[0]).unwrap_or(0),
                        usize::try_from(wqkv.dimensions[1]).unwrap_or(0),
                    ),
                    (
                        wg_p,
                        wgate.ggml_type,
                        usize::try_from(wgate.dimensions[0]).unwrap_or(0),
                        usize::try_from(wgate.dimensions[1]).unwrap_or(0),
                    ),
                    &conv_f,
                    d0,
                    d1,
                    (
                        sb_p,
                        ssm_b.ggml_type,
                        usize::try_from(ssm_b.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_b.dimensions[1]).unwrap_or(0),
                    ),
                    (
                        sa_p,
                        ssm_al.ggml_type,
                        usize::try_from(ssm_al.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_al.dimensions[1]).unwrap_or(0),
                    ),
                    &dt_b,
                    &a_b,
                    &norm_b,
                    (
                        so_p,
                        ssm_out.ggml_type,
                        usize::try_from(ssm_out.dimensions[0]).unwrap_or(0),
                        usize::try_from(ssm_out.dimensions[1]).unwrap_or(0),
                    ),
                )?
            } else {
                let wq = must_tensor(archive, &format!("blk.{il}.attn_q.weight"))?;
                let wk = must_tensor(archive, &format!("blk.{il}.attn_k.weight"))?;
                let wv = must_tensor(archive, &format!("blk.{il}.attn_v.weight"))?;
                let wo = must_tensor(archive, &format!("blk.{il}.attn_output.weight"))?;
                let qn = must_tensor(archive, &format!("blk.{il}.attn_q_norm.weight"))?;
                let kn = must_tensor(archive, &format!("blk.{il}.attn_k_norm.weight"))?;
                let qnw = tensor_f32_flat(archive, &qn)?;
                let knw = tensor_f32_flat(archive, &kn)?;
                block_full_attention(
                    cfg,
                    &mut self.attn_kv,
                    il,
                    pos,
                    &h,
                    archive.tensor_payload(&wq)?,
                    wq.ggml_type,
                    usize::try_from(wq.dimensions[0]).unwrap_or(0),
                    usize::try_from(wq.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wk)?,
                    wk.ggml_type,
                    usize::try_from(wk.dimensions[0]).unwrap_or(0),
                    usize::try_from(wk.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wv)?,
                    wv.ggml_type,
                    usize::try_from(wv.dimensions[0]).unwrap_or(0),
                    usize::try_from(wv.dimensions[1]).unwrap_or(0),
                    archive.tensor_payload(&wo)?,
                    wo.ggml_type,
                    usize::try_from(wo.dimensions[0]).unwrap_or(0),
                    usize::try_from(wo.dimensions[1]).unwrap_or(0),
                    &qnw,
                    &knw,
                )?
            };

            for i in 0..cfg.n_embd {
                x[i] = residual[i] + y[i];
            }

            let ffn_residual = x.clone();

            let post_name = archive
                .tensor_by_name(&format!("blk.{il}.post_attention_norm.weight"))
                .or_else(|| archive.tensor_by_name(&format!("blk.{il}.attn_post_norm.weight")))
                .or_else(|| archive.tensor_by_name(&format!("blk.{il}.attention_norm_after.weight")))
                .ok_or_else(|| {
                    BitNetError::Inference(format!("layer {il}: missing post-attention RMS norm tensor"))
                })?;

            let h2 = rms_combine(&x, post_name, archive, cfg.norm_eps)?;

            let gate_in = must_tensor(archive, &format!("blk.{il}.ffn_gate_inp.weight"))?;
            let up = must_tensor(archive, &format!("blk.{il}.ffn_up_exps.weight"))?;
            let gate = must_tensor(archive, &format!("blk.{il}.ffn_gate_exps.weight"))?;
            let down = must_tensor(archive, &format!("blk.{il}.ffn_down_exps.weight"))?;

            let mut moe_delta = moe_forward(archive, cfg, &h2, &gate_in, &up, &gate, &down, None)?;

            if let Some(gs) = archive.tensor_first_of(&[
                &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
                &format!("blk.{il}.ffn_gate_inp_shexp"),
            ]) {
                let gate_w = must_tensor(archive, &format!("blk.{il}.ffn_gate_shexp.weight"))?;
                let up_w = must_tensor(archive, &format!("blk.{il}.ffn_up_shexp.weight"))?;
                let down_w = must_tensor(archive, &format!("blk.{il}.ffn_down_shexp.weight"))?;
                let sh = shared_expert_forward(archive, cfg, &h2, gs, &gate_w, &up_w, &down_w)?;
                for i in 0..cfg.n_embd {
                    moe_delta[i] += sh[i];
                }
            }

            for i in 0..cfg.n_embd {
                x[i] = ffn_residual[i] + moe_delta[i];
            }
        }

        let xn = rms_combine(&x, &self.out_norm, archive, cfg.norm_eps)?;
        logits_project(archive, &self.cuda, cfg, &self.out_head, &xn, 4096)
    }
}

fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    if temperature <= 0.0 {
        let (i, _) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(pi, pb), (i, v)| {
                let v = if v.is_nan() { f32::NEG_INFINITY } else { *v };
                if v > pb {
                    (i, v)
                } else {
                    (pi, pb)
                }
            });
        return i as u32;
    }
    let scaled: Vec<f32> = logits.iter().map(|z| z / temperature).collect();
    let m = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|z| (z - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * s;
    let mut c = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        c += e;
        if c >= r {
            return i as u32;
        }
    }
    (exps.len().saturating_sub(1)) as u32
}
