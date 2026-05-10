//! Hyperparameters parsed from GGUF metadata (`{arch}.{key}`, same conventions as llama.cpp).

use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};

/// Parsed numeric configuration plus runtime flags inferred from GGUF KV + tensors.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Qwen35Config {
    pub arch_label: String,
    pub kv_prefix: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub max_seq: usize,
    pub n_layer: usize,
    pub rope_freq_base: f32,
    /// Sum of rotary pairs or upper bound derived from KV / attention dims.
    pub rope_dim_pairs: usize,
    /// RMS norm epsilon (`{arch}.attention.layer_norm_rms_epsilon`).
    pub norm_eps: f32,
    /// MoE routers.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub feed_forward_length: usize,

    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub attn_scale: f32,

    pub ssm_d_conv: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,

    pub full_attn_interval: u32,

    pub recurrent_layers: Vec<bool>,
}

fn metadata_i64(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<i64> {
    md.get(key).and_then(|v| match v {
        GgufValue::U8(x) => Some(*x as i64),
        GgufValue::I8(x) => Some(*x as i64),
        GgufValue::U16(x) => Some(*x as i64),
        GgufValue::I16(x) => Some(*x as i64),
        GgufValue::U32(x) => Some(*x as i64),
        GgufValue::I32(x) => Some(*x as i64),
        GgufValue::U64(x) => i64::try_from(*x).ok(),
        GgufValue::I64(x) => Some(*x),
        _ => None,
    })
}

fn meta_req_i64(md: &std::collections::HashMap<String, GgufValue>, key: &str, label: &str) -> Result<i64> {
    metadata_i64(md, key)
        .ok_or_else(|| BitNetError::Inference(format!("missing or invalid GGUF metadata `{key}` for {label}")))
}

fn meta_opt_f32(md: &std::collections::HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    md.get(key).and_then(|v| match v {
        GgufValue::F32(x) => Some(*x),
        GgufValue::F64(x) => Some(*x as f32),
        _ => None,
    })
}

pub fn kv_prefix_from_arch(arch_key: &str) -> Result<String> {
    let trimmed = arch_key.trim().to_ascii_lowercase();
    match trimmed.as_str() {
        "qwen35moe" | "qwen35" => Ok(format!("{}.", trimmed)),
        _ => Err(BitNetError::Inference(
            "unsupported qwen GGUF architecture key — expected qwen35moe".into(),
        )),
    }
}

fn infer_blk_count(archive: &GgufArchive) -> usize {
    let mut m = None;
    let mut max_i: i64 = -1;
    for t in &archive.tensors {
        let parts: Vec<&str> = t.name.split('.').collect();
        if parts.len() >= 2 && parts[0] == "blk" && parts[1].bytes().all(|b| b.is_ascii_digit()) {
            let i: i64 = parts[1].parse().unwrap_or(-1);
            if i >= 0 {
                max_i = max_i.max(i);
                m.get_or_insert(0);
            }
        }
    }
    if max_i >= 0 {
        (max_i + 1) as usize
    } else {
        m.unwrap_or(0)
    }
}

impl Qwen35Config {
    pub fn from_gguf(archive: &GgufArchive) -> Result<Self> {
        let raw_arch = archive
            .normalized_architecture()
            .unwrap_or_else(|| "qwen35moe".into())
            .to_ascii_lowercase();
        let kv_prefix = kv_prefix_from_arch(&raw_arch)?;
        let md = &archive.metadata;

        let n_embd_i = meta_req_i64(
            md,
            &format!("{}{}", kv_prefix, "embedding_length"),
            "embedding",
        )?;

        let n_vocab_i =
            match meta_req_i64(md, &format!("{}{}", kv_prefix, "vocab_size"), "vocab") {
                Ok(v) => v,
                Err(_) => infer_vocab_from_emb(archive, n_embd_i as usize)
                    .map(|v| v as i64)
                    .ok_or_else(|| BitNetError::Inference("missing vocab_size (no token_embd dims)".into()))?,
            };

        let blk_meta = meta_req_i64(
            md,
            &format!("{}{}", kv_prefix, "block_count"),
            "layers",
        )?;
        let blk_inf = infer_blk_count(archive) as i64;

        let n_layer_i = if blk_inf > 0 {
            std::cmp::max(blk_meta, blk_inf)
        } else {
            blk_meta
        };

        let ctx_len = meta_req_i64(
            md,
            &format!("{}{}", kv_prefix, "context_length"),
            "context",
        )?;

        let n_expert_i = meta_req_i64(md, &format!("{}{}", kv_prefix, "expert_count"), "moe experts")?;

        let n_expert_used_i = meta_req_i64(
            md,
            &format!("{}{}", kv_prefix, "expert_used_count"),
            "moe routed experts",
        )?;

        let n_embd_us = i64_to_usize(n_embd_i)?
            .ok_or_else(|| BitNetError::Inference("invalid embedding_length".into()))?;

        let n_ff_dense = match metadata_i64(md, &format!("{}{}", kv_prefix, "feed_forward_length")) {
            Some(v) => v,
            None => infer_ff_dense_from_tensors(archive, n_embd_us)?.ok_or_else(|| {
                BitNetError::Inference(
                    "missing `*.feed_forward_length` in GGUF metadata; expected keys like `qwen35moe.feed_forward_length`, \
                     or inferable shapes from `blk.*.ffn_up_shexp.weight` / `ffn_gate_shexp.weight`"
                        .into(),
                )
            })?,
        };

        let n_ff_exp = metadata_i64(md, &format!("{}{}", kv_prefix, "expert_feed_forward_length"))
            .unwrap_or(0);

        let n_ff_shexp = metadata_i64(md, &format!("{}{}", kv_prefix, "expert_shared_feed_forward_length"))
            .unwrap_or(0);

        let n_ff_exp_us = std::cmp::max(
            i64_to_usize(n_ff_exp)?
                .ok_or_else(|| BitNetError::Inference("negative expert_ff length".into()))?,
            infer_ff_exp_fallback(archive, n_expert_i as usize)?.unwrap_or(0),
        );
        let dense_us = i64_to_usize(n_ff_dense)?.unwrap_or(0);
        let n_ff_shexp_us =
            std::cmp::max(i64_to_usize(n_ff_shexp)?.unwrap_or(dense_us), 0);

        let n_head_i = meta_req_i64(md, &format!("{}{}", kv_prefix, "attention.head_count"), "heads")?;
        let n_head_kv_i = meta_req_i64(
            md,
            &format!("{}{}", kv_prefix, "attention.head_count_kv"),
            "kv_heads",
        )?;

        let key_len = meta_opt_f32(md, &format!("{}{}", kv_prefix, "attention.key_length"))
            .map(|v| v as i64)
            .or_else(|| metadata_i64(md, &format!("{}{}", kv_prefix, "attention.key_length")));
        let mut head_dim_est = metadata_i64(md, &format!("{}{}", kv_prefix, "rope.dimension_count")).map(|rc| rc as usize);
        if head_dim_est.is_none() {
            if let Some(kl) = key_len {
                if n_head_kv_i > 0 {
                    head_dim_est = usize::try_from(kl.max(1) / n_head_kv_i.max(1)).ok().filter(|&h| h > 0);
                }
            }
        }
        let head_dim = head_dim_est
            .or_else(|| {
                if n_head_i > 0 {
                    usize::checked_div(n_embd_i as usize, n_head_i as usize).filter(|&h| h > 0)
                } else {
                    None
                }
            })
            .ok_or_else(|| BitNetError::Inference("unable to derive attention head dimensions".into()))?;

        let mut ssm_conv = meta_req_i64(md, &format!("{}{}", kv_prefix, "ssm.conv_kernel"), "ssm conv")?;
        let mut ssm_inner = meta_req_i64(md, &format!("{}{}", kv_prefix, "ssm.inner_size"), "ssm inner")?;
        // Some GGUF exports carry wrong or stale SSM KV vs tensors; conv1d weight is authoritative (layout `[d_conv, conv_channels]`).
        if let Some(t) = archive.tensor_first_of(&["blk.0.ssm_conv1d.weight", "blk.0.ssm_conv1d"]) {
            if t.dimensions.len() >= 2 {
                if let Ok(d0) = i64::try_from(t.dimensions[0]) {
                    ssm_conv = d0;
                }
                if let Ok(d1) = i64::try_from(t.dimensions[1]) {
                    ssm_inner = d1;
                }
            }
        }
        let ssm_state = meta_req_i64(md, &format!("{}{}", kv_prefix, "ssm.state_size"), "ssm state")?;
        let ssm_dt_rank = meta_req_i64(md, &format!("{}{}", kv_prefix, "ssm.time_step_rank"), "ssm dt")?;
        let ssm_group = meta_req_i64(md, &format!("{}{}", kv_prefix, "ssm.group_count"), "ssm groups")?;

        let full_attn_interval = metadata_i64(md, &format!("{}{}", kv_prefix, "full_attention_interval"))
            .unwrap_or(4)
            .max(1) as u32;

        let n_layer_us = usize::try_from(n_layer_i)
            .map_err(|_| BitNetError::Inference("block_count out of usize range".into()))?;
        let mut recurrent_layers = vec![false; n_layer_us];
        for il in 0..n_layer_us {
            recurrent_layers[il] = ((il + 1) % (full_attn_interval as usize)) != 0;
        }

        let norm_eps =
            meta_opt_f32(md, &format!("{}{}", kv_prefix, "attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);

        let rope_base = meta_opt_f32(md, &format!("{}{}", kv_prefix, "rope.freq_base")).unwrap_or(1e6_f32);

        let rope_dim_pairs = parse_rope_dim_pairs(md, &kv_prefix, head_dim)?;

        let attn_scale = meta_opt_f32(md, &format!("{}{}", kv_prefix, "attention.scale")).unwrap_or_else(|| {
            1.0 / (head_dim as f32).sqrt()
        });

        Ok(Self {
            arch_label: raw_arch.clone(),
            kv_prefix,
            n_vocab: i64_to_usize(n_vocab_i)?
                .ok_or_else(|| BitNetError::Inference("invalid vocab size".into()))?,
            n_embd: i64_to_usize(n_embd_i)?
                .ok_or_else(|| BitNetError::Inference("invalid embedding size".into()))?,
            max_seq: usize::try_from(ctx_len).map_err(|_| BitNetError::Inference("context length OOB".into()))?,
            n_layer: n_layer_us,
            rope_freq_base: rope_base,
            rope_dim_pairs: rope_dim_pairs / 2 * 2, // pairs
            norm_eps,
            n_expert: i64_to_usize(n_expert_i)?
                .ok_or_else(|| BitNetError::Inference("invalid expert_count".into()))?,
            n_expert_used: i64_to_usize(n_expert_used_i)?
                .filter(|x| *x > 0)
                .ok_or_else(|| BitNetError::Inference("invalid expert_used_count".into()))?,
            n_ff_exp: n_ff_exp_us,
            n_ff_shexp: n_ff_shexp_us.max(1),
            feed_forward_length: i64_to_usize(n_ff_dense)?
                .ok_or_else(|| BitNetError::Inference("feed_forward_length invalid".into()))?,
            n_head: i64_to_usize(n_head_i)?
                .ok_or_else(|| BitNetError::Inference("invalid head_count".into()))?,
            n_head_kv: i64_to_usize(n_head_kv_i)?
                .filter(|x| *x > 0)
                .ok_or_else(|| BitNetError::Inference("invalid head_count_kv".into()))?,
            head_dim,
            attn_scale,
            ssm_d_conv: usize::try_from(ssm_conv).unwrap_or(0),
            ssm_d_inner: usize::try_from(ssm_inner).unwrap_or(0),
            ssm_d_state: usize::try_from(ssm_state).unwrap_or(0),
            ssm_dt_rank: usize::try_from(ssm_dt_rank).unwrap_or(0),
            ssm_n_group: usize::try_from(ssm_group).unwrap_or(0),
            full_attn_interval,
            recurrent_layers,
        })
    }

    pub fn is_recurrent_layer(&self, il: usize) -> bool {
        self.recurrent_layers.get(il).copied().unwrap_or(false)
    }
}

fn i64_to_usize(v: i64) -> Result<Option<usize>> {
    if v < 0 {
        return Ok(None);
    }
    Ok(usize::try_from(v).ok())
}

fn infer_vocab_from_emb(archive: &GgufArchive, _n_embd: usize) -> Option<usize> {
    let t = archive.tensor_first_of(&["token_embd.weight", "token_embd"])?;
    usize::try_from(*t.dimensions.get(1)?).ok()
}

/// Shared-expert FFN width `[n_embd, n_ff]` (llama.cpp tensor layout).
fn infer_ff_dense_from_tensors(archive: &GgufArchive, n_embd: usize) -> Result<Option<i64>> {
    for il in [0usize, 1] {
        for name in [
            format!("blk.{il}.ffn_up_shexp.weight"),
            format!("blk.{il}.ffn_gate_shexp.weight"),
        ] {
            let Some(t) = archive.tensor_by_name(&name) else {
                continue;
            };
            if t.dimensions.len() >= 2 {
                let d0 = usize::try_from(t.dimensions[0]).unwrap_or(0);
                let d1 = usize::try_from(t.dimensions[1]).unwrap_or(0);
                if d0 == n_embd && d1 > 0 {
                    return Ok(Some(d1 as i64));
                }
                if d1 == n_embd && d0 > 0 {
                    return Ok(Some(d0 as i64));
                }
            }
        }
    }
    Ok(None)
}

fn infer_ff_exp_fallback(
    archive: &GgufArchive,
    expected_expert: usize,
) -> Result<Option<usize>> {
    for i in [0usize, 1] {
        if let Some(t) =
            archive.tensor_by_name(&format!("blk.{i}.ffn_down_exps.weight"))
        {
            if t.dimensions.len() == 3 {
                let nexp = usize::try_from(t.dimensions[2]).unwrap_or(0);
                if nexp != expected_expert {
                    continue;
                }
                return Ok(Some(usize::try_from(t.dimensions[0]).unwrap_or(0)));
            }
        }
    }
    Ok(None)
}

fn parse_rope_dim_pairs(md: &std::collections::HashMap<String, GgufValue>, kv_prefix: &str, fallback: usize) -> Result<usize> {
    let key = format!("{}{}", kv_prefix, "rope.dimension_sections");
    if let Some(GgufValue::Array(arr)) = md.get(&key) {
        let mut sum = 0i64;
        for v in arr {
            sum += match v {
                GgufValue::U32(x) => *x as i64,
                GgufValue::U64(x) => *x as i64,
                GgufValue::I64(x) => *x,
                GgufValue::I32(x) => *x as i64,
                _ => 0,
            };
        }
        if sum <= 0 {
            return Err(BitNetError::Inference("`rope.dimension_sections` empty".into()));
        }
        let s = usize::try_from(sum).unwrap_or(fallback);
        Ok(std::cmp::min((s / 2) * 2, (fallback / 2) * 2))
    } else {
        Ok((fallback / 2) * 2)
    }
}
