//! Autoregressive GPT-OSS and DeepSeek/GLM MLA graphs over GGUF quantized weights.
//! Operations follow llama.cpp 631109b34 openai-moe.cpp and deepseek2.cpp.
use super::moe_cost::{Cost, Execution};
use crate::native::moe::GpuMoe;
use super::weights::Weights;
use crate::backend::BackendKind;
use crate::cancel::inference_cancelled;
use crate::error::{BitNetError, Result};
use crate::gguf::{GgufArchive, GgufValue};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::model::ModelExecutor;
use crate::sampling::{sample_token, SamplingOptions};
use crate::stream::{emit_text_delta, StreamEvent};
use crate::timings::PhaseTimings;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(test)]
#[path = "gpt_segmented_runtime_tests.rs"]
mod gpt_segmented_runtime_tests;
#[path = "gpt_full.rs"]
mod gpu_full;
#[path = "mla_full.rs"]
mod gpu_mla;
#[path = "state_budget.rs"]
mod state_budget;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Family {
    GptOss,
    Mla,
}
impl Family {
    fn name(self) -> &'static str {
        match self {
            Self::GptOss => "gpt-oss",
            Self::Mla => "deepseek2",
        }
    }
}

struct Config {
    family: Family,
    embd: usize,
    vocab: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head: usize,
    value: usize,
    rotary: usize,
    kv_rank: usize,
    max_seq: usize,
    eps: f32,
    theta: f32,
    yarn_factor: f32,
    yarn_orig: usize,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    experts: usize,
    used: usize,
    dense_layers: usize,
    groups: usize,
    groups_used: usize,
    sigmoid: bool,
    weight_norm: bool,
    weight_scale: f32,
    window: usize,
}

fn number(v: &GgufValue) -> Option<f64> {
    match v {
        GgufValue::U8(v) => Some(*v as f64),
        GgufValue::I8(v) => Some(*v as f64),
        GgufValue::U16(v) => Some(*v as f64),
        GgufValue::I16(v) => Some(*v as f64),
        GgufValue::U32(v) => Some(*v as f64),
        GgufValue::I32(v) => Some(*v as f64),
        GgufValue::U64(v) => Some(*v as f64),
        GgufValue::I64(v) => Some(*v as f64),
        GgufValue::F32(v) => Some(*v as f64),
        GgufValue::F64(v) => Some(*v),
        _ => None,
    }
}
impl Config {
    fn load(archive: &GgufArchive, family: Family) -> Result<Self> {
        let arch = archive
            .normalized_architecture()
            .unwrap_or_else(|| family.name().into());
        let prefix = if family == Family::GptOss
            && archive.metadata.contains_key("gpt-oss.embedding_length")
        {
            "gpt-oss"
        } else {
            arch.as_str()
        };
        let val = |key: &str| {
            archive
                .metadata
                .get(&format!("{prefix}.{key}"))
                .and_then(number)
        };
        let req = |key: &str| {
            val(key)
                .filter(|v| v.is_finite() && *v > 0.0 && v.fract() == 0.0)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    BitNetError::InvalidGguf(format!("missing or invalid {prefix}.{key}"))
                })
        };
        let embd = req("embedding_length")?;
        let heads = req("attention.head_count")?;
        let rotary = val("rope.dimension_count")
            .unwrap_or(val("attention.key_length").unwrap_or(64.0)) as usize;
        let head = if family == Family::Mla {
            req("attention.key_length_mla")?
        } else {
            req("attention.key_length")?
        };
        let value = if family == Family::Mla {
            req("attention.value_length_mla")?
        } else {
            req("attention.value_length")?
        };
        let token = archive
            .tensor_by_name("token_embd.weight")
            .ok_or_else(|| BitNetError::InvalidGguf("missing token embedding".into()))?;
        let experts = req("expert_count")?;
        let used = req("expert_used_count")?;
        let groups = val("expert_group_count").unwrap_or(1.0) as usize;
        let groups_used = val("expert_group_used_count").unwrap_or(1.0) as usize;
        let kv_heads = req("attention.head_count_kv")?;
        let max_seq = std::env::var("RBITNET_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8192)
            .min(req("context_length")?);
        if token.dimensions.len() != 2
            || token.dimensions[0] as usize != embd
            || used > experts
            || groups == 0
            || groups_used == 0
            || groups_used > groups
            || experts % groups != 0
            || heads % kv_heads != 0
            || rotary % 2 != 0
            || rotary > head
            || value == 0
        {
            return Err(BitNetError::InvalidGguf(
                "invalid native model dimensions/expert groups".into(),
            ));
        }
        let bool_val = |key: &str| {
            matches!(
                archive.metadata.get(&format!("{prefix}.{key}")),
                Some(GgufValue::Bool(true))
            )
        };
        Ok(Self {
            family,
            embd,
            vocab: token.dimensions[1] as usize,
            layers: req("block_count")?,
            heads,
            kv_heads,
            head,
            value,
            rotary,
            kv_rank: if family == Family::Mla {
                req("attention.kv_lora_rank")?
            } else {
                0
            },
            max_seq,
            eps: val("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            theta: val("rope.freq_base").unwrap_or(10000.0) as f32,
            yarn_factor: val("rope.scaling.factor").unwrap_or(1.0) as f32,
            yarn_orig: val("rope.scaling.original_context_length").unwrap_or(max_seq as f64)
                as usize,
            yarn_beta_fast: val("rope.scaling.yarn_beta_fast").unwrap_or(32.0) as f32,
            yarn_beta_slow: val("rope.scaling.yarn_beta_slow").unwrap_or(1.0) as f32,
            experts,
            used,
            dense_layers: val("leading_dense_block_count").unwrap_or(0.0) as usize,
            groups,
            groups_used,
            sigmoid: val("expert_gating_func").unwrap_or(1.0) == 2.0,
            weight_norm: bool_val("expert_weights_norm"),
            weight_scale: val("expert_weights_scale").unwrap_or(1.0) as f32,
            window: val("attention.sliding_window").unwrap_or(0.0) as usize,
        })
    }
}

fn norm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != w.len() || x.is_empty() {
        return Err(BitNetError::Inference("RMS norm shape mismatch".into()));
    }
    let scale = (x.iter().map(|x| x * x).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(x.iter().zip(w).map(|(&x, &w)| x * scale * w).collect())
}
fn add(x: &mut [f32], y: &[f32]) -> Result<()> {
    if x.len() != y.len() {
        return Err(BitNetError::Inference(
            "residual/bias shape mismatch".into(),
        ));
    }
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
    Ok(())
}
fn softmax(scores: &mut [f32], sink: Option<f32>) {
    let max = scores
        .iter()
        .copied()
        .chain(sink)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = sink.map(|s| (s - max).exp()).unwrap_or(0.0);
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        total += *s;
    }
    for s in scores {
        *s /= total;
    }
}
fn rope(x: &mut [f32], pos: usize, cfg: &Config) {
    let width = x.len();
    let half = width / 2;
    let factor = cfg.yarn_factor;
    let corr = |rot: f32| {
        width as f32 * (cfg.yarn_orig as f32 / (rot * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * cfg.theta.ln())
    };
    let low = corr(cfg.yarn_beta_fast).floor().max(0.0);
    let high = corr(cfg.yarn_beta_slow).ceil().min(width as f32 - 1.0);
    let magnitude = if factor > 1.0 {
        1.0 + 0.1 * factor.ln()
    } else {
        1.0
    };
    for i in 0..half {
        let frequency = cfg.theta.powf(-2.0 * i as f32 / width as f32);
        let ramp = 1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
        let frequency = if factor > 1.0 {
            frequency / factor * (1.0 - ramp) + frequency * ramp
        } else {
            frequency
        };
        let angle = pos as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        // DeepSeek/GLM GGUF stores consecutive rotary pairs; GPT-OSS uses NEOX halves.
        // Applying NEOX to MLA preserves shapes but corrupts position-dependent attention.
        let (left, right) = if cfg.family == Family::Mla {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + half)
        };
        let a = x[left];
        let b = x[right];
        x[left] = (a * cos - b * sin) * magnitude;
        x[right] = (a * sin + b * cos) * magnitude;
    }
}

struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}
pub(crate) struct Runtime {
    cfg: Config,
    // CUDA graphs borrow expert contexts and matrix addresses. Drop them first.
    gpu_full: Option<gpu_full::GpuFull>,
    gpu_mla: Option<gpu_mla::GpuMla>,
    weights: Weights,
    tokenizer: LoadedPromptTokenizer,
    kv: Vec<LayerKv>,
    gpu_attention: Vec<Option<super::attention::CudaAttention>>,
    gpu_moe: Vec<Option<super::moe::GpuMoe>>,
    gpu_head: Option<super::head::GpuHead>,
    use_gpu_attention: bool,
    moe_execution: Execution,
    moe_cost: Vec<Cost>,
}
impl Runtime {
    fn load(
        archive: Arc<GgufArchive>,
        tokenizer: &Path,
        kind: BackendKind,
        family: Family,
    ) -> Result<Self> {
        let cfg = Config::load(&archive, family)?;
        // Check required tensor names and row formats before ready becomes true.
        for il in 0..cfg.layers {
            let mut required = vec![
                "attn_norm.weight",
                if family == Family::GptOss {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                },
                "attn_output.weight",
            ];
            if family == Family::GptOss {
                required.extend([
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_sinks.weight",
                    "attn_q.bias",
                    "attn_k.bias",
                    "attn_v.bias",
                    "attn_output.bias",
                ]);
            } else {
                required.extend([
                    "attn_q_a.weight",
                    "attn_q_a_norm.weight",
                    "attn_q_b.weight",
                    "attn_kv_a_mqa.weight",
                    "attn_kv_a_norm.weight",
                    "attn_k_b.weight",
                    "attn_v_b.weight",
                ]);
            }
            if il < cfg.dense_layers {
                required.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
            } else {
                required.extend([
                    "ffn_gate_inp.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]);
            }
            for suffix in required {
                let name = format!("blk.{il}.{suffix}");
                let t = archive
                    .tensor_by_name(&name)
                    .ok_or_else(|| BitNetError::InvalidGguf(format!("missing `{name}`")))?;
                if t.dimensions.len() >= 2
                    && !crate::ggml::ggml_type_supported_mmap_matvec(t.ggml_type)
                {
                    return Err(BitNetError::UnsupportedGgmlType(t.ggml_type));
                }
            }
            let shape = |suffix: &str, expected: &[usize]| -> Result<()> {
                let name = format!("blk.{il}.{suffix}");
                let t = archive
                    .tensor_by_name(&name)
                    .ok_or_else(|| BitNetError::InvalidGguf(format!("missing `{name}`")))?;
                if t.dimensions
                    .iter()
                    .copied()
                    .ne(expected.iter().map(|&v| v as u64))
                {
                    return Err(BitNetError::InvalidGguf(format!(
                        "{name}: shape {:?}, expected {expected:?}",
                        t.dimensions
                    )));
                }
                Ok(())
            };
            shape("attn_norm.weight", &[cfg.embd])?;
            shape(
                if family == Family::GptOss {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                },
                &[cfg.embd],
            )?;
            shape("attn_output.weight", &[cfg.heads * cfg.value, cfg.embd])?;
            if family == Family::GptOss {
                if cfg.head != cfg.value {
                    return Err(BitNetError::InvalidGguf(
                        "GPT-OSS key/value head dimensions differ".into(),
                    ));
                }
                for (suffix, width) in [
                    ("attn_q", cfg.heads * cfg.head),
                    ("attn_k", cfg.kv_heads * cfg.head),
                    ("attn_v", cfg.kv_heads * cfg.head),
                ] {
                    shape(&format!("{suffix}.weight"), &[cfg.embd, width])?;
                    shape(&format!("{suffix}.bias"), &[width])?;
                }
                shape("attn_sinks.weight", &[cfg.heads])?;
                shape("attn_output.bias", &[cfg.embd])?;
            } else {
                let qa = archive
                    .tensor_by_name(&format!("blk.{il}.attn_q_a.weight"))
                    .unwrap();
                let qrank = qa.dimensions.get(1).copied().unwrap_or(0) as usize;
                if qrank == 0 {
                    return Err(BitNetError::InvalidGguf("invalid MLA query rank".into()));
                }
                shape("attn_q_a.weight", &[cfg.embd, qrank])?;
                shape("attn_q_a_norm.weight", &[qrank])?;
                shape("attn_q_b.weight", &[qrank, cfg.heads * cfg.head])?;
                shape(
                    "attn_kv_a_mqa.weight",
                    &[cfg.embd, cfg.kv_rank + cfg.rotary],
                )?;
                shape("attn_kv_a_norm.weight", &[cfg.kv_rank])?;
                shape(
                    "attn_k_b.weight",
                    &[cfg.head - cfg.rotary, cfg.kv_rank, cfg.heads],
                )?;
                shape("attn_v_b.weight", &[cfg.kv_rank, cfg.value, cfg.heads])?;
            }
            let check_ffn = |suffix: &str, experts: bool| -> Result<usize> {
                let gate = format!("ffn_gate{suffix}.weight");
                let t = archive
                    .tensor_by_name(&format!("blk.{il}.{gate}"))
                    .ok_or_else(|| BitNetError::InvalidGguf(format!("missing {gate}")))?;
                let ff = t.dimensions.get(1).copied().unwrap_or(0) as usize;
                if ff == 0 {
                    return Err(BitNetError::InvalidGguf(
                        "invalid feed-forward width".into(),
                    ));
                }
                let dims = if experts {
                    vec![cfg.embd, ff, cfg.experts]
                } else {
                    vec![cfg.embd, ff]
                };
                shape(&gate, &dims)?;
                shape(&format!("ffn_up{suffix}.weight"), &dims)?;
                let down = if experts {
                    vec![ff, cfg.embd, cfg.experts]
                } else {
                    vec![ff, cfg.embd]
                };
                shape(&format!("ffn_down{suffix}.weight"), &down)?;
                Ok(ff)
            };
            if il < cfg.dense_layers {
                check_ffn("", false)?;
            } else {
                let ff = check_ffn("_exps", true)?;
                shape("ffn_gate_inp.weight", &[cfg.embd, cfg.experts])?;
                if family == Family::GptOss {
                    shape("ffn_gate_inp.bias", &[cfg.experts])?;
                    for (suffix, width) in [
                        ("ffn_gate_exps.bias", ff),
                        ("ffn_up_exps.bias", ff),
                        ("ffn_down_exps.bias", cfg.embd),
                    ] {
                        shape(suffix, &[width, cfg.experts])?;
                    }
                }
                if archive
                    .tensor_by_name(&format!("blk.{il}.exp_probs_b.bias"))
                    .is_some()
                {
                    shape("exp_probs_b.bias", &[cfg.experts])?;
                }
                if archive
                    .tensor_by_name(&format!("blk.{il}.ffn_gate_shexp.weight"))
                    .is_some()
                {
                    check_ffn("_shexp", false)?;
                }
            }
        }
        for (name, expected) in [
            ("output_norm.weight", vec![cfg.embd]),
            ("output.weight", vec![cfg.embd, cfg.vocab]),
        ] {
            let t = archive
                .tensor_by_name(name)
                .ok_or_else(|| BitNetError::InvalidGguf(format!("missing {name}")))?;
            if t.dimensions
                .iter()
                .copied()
                .ne(expected.iter().map(|&d| d as u64))
            {
                return Err(BitNetError::InvalidGguf(format!("invalid {name} shape")));
            }
        }
        let state_reserve = if matches!(kind, BackendKind::Cuda | BackendKind::Hybrid) {
            state_budget::reserve(&archive, &cfg)?
        } else {
            0
        };
        let mut weights = Weights::new_with_state_reserve(archive, kind, state_reserve)?;
        let name = match weights.archive.metadata.get("general.name") {
            Some(GgufValue::String(name)) => name.clone(),
            _ => cfg.family.name().to_owned(),
        };
        weights.enable_moe_metrics(super::moe_metrics::Model::new(
            name,
            cfg.family.name().to_owned(),
            cfg.layers,
        ))?;
        let tokenizer = LoadedPromptTokenizer::from_path(tokenizer)?;
        let kv = (0..cfg.layers)
            .map(|_| LayerKv {
                k: Vec::new(),
                v: Vec::new(),
            })
            .collect();
        let gpu_attention = (0..cfg.layers).map(|_| None).collect();
        let gpu_moe: Vec<_> = (0..cfg.layers)
            .map(|il| {
                if il < cfg.dense_layers {
                    None
                } else {
                    super::moe::GpuMoe::new(
                        &weights,
                        il,
                        cfg.experts,
                        cfg.used,
                        cfg.family == Family::GptOss,
                    )
                }
            })
            .collect();
        tracing::info!(
            layers = gpu_moe.iter().filter(|m| m.is_some()).count(),
            "resident CUDA routed expert layers"
        );
        let use_gpu_attention = matches!(kind, BackendKind::Cuda | BackendKind::Hybrid)
            && weights.resident_bytes > 0
            && !matches!(
                std::env::var("RBITNET_CUDA_ATTENTION").as_deref(),
                Ok("0" | "false" | "no")
            );
        let fused= gpu_moe.iter().flatten().filter(|m|m.is_fused()).count();
        if std::env::var("RBITNET_REQUIRE_FUSED_MOE").as_deref()==Ok("1")
            && (fused==0 || fused!=gpu_moe.iter().flatten().count()) {
            return Err(BitNetError::Inference("requested fused MoE unavailable: compatible GPU contexts and native fusion ABI are required".into()));
        }
        let gpu_full = gpu_full::GpuFull::new(&weights, &cfg, &gpu_moe, kind);
        if std::env::var("RBITNET_REQUIRE_GPT_FULL").as_deref() == Ok("1") && gpu_full.is_none() {
            return Err(BitNetError::Inference("resident GPT-OSS unavailable: requires CUDA or hybrid, supported native DLL, CPU quant SIMD, compatible biased GPT backbone within budget and fixed or host-admitted expert execution".into()));
        }
        let gpu_mla = gpu_mla::GpuMla::new(&weights, &cfg, &gpu_moe, kind);
        if std::env::var("RBITNET_REQUIRE_MLA_FULL").as_deref() == Ok("1") && gpu_mla.is_none() {
            return Err(BitNetError::Inference("resident MLA unavailable: requires CUDA, supported native DLL, CPU quant SIMD, compatible unbiased MLA projections and backbone weights within budget".into()));
        }
        if std::env::var("RBITNET_REQUIRE_GPT_PREFILL").as_deref()==Ok("1")
            && gpu_full.as_ref().is_none_or(|f|f.prefill_capacity()==0) {
            return Err(BitNetError::Inference("requested GPT block prefill unavailable: fixed expert banks, compatible ABI and state budget are required".into()));
        }
        let gpu_head = if gpu_full.is_none() && gpu_mla.is_none() {
            super::head::GpuHead::new(&weights, "output.weight", cfg.eps)
        } else {
            None
        };
        let moe_execution = Execution::from_env();
        let moe_cost = (0..cfg.layers).map(|_| Cost::new(moe_execution)).collect();
        Ok(Self {
            moe_execution,
            moe_cost,
            cfg,
            gpu_full,
            gpu_mla,
            weights,
            tokenizer,
            kv,
            gpu_attention,
            gpu_moe,
            gpu_head,
            use_gpu_attention,
        })
    }
    fn choose_moe_cpu(&mut self, il: usize, ids: &[usize]) -> Result<bool> {
        // Default cache behavior preserves the existing fallback path. Explicit
        // CPU and adaptive policies always use the dedicated CPU payload API.
        if self.moe_execution == Execution::Cache {
            return Ok(false);
        }
        let (fits, missing) = match &self.gpu_moe[il] {
            Some(moe) => moe.estimate_selected(ids)?,
            None => (false, 0),
        };
        let cpu = self.moe_cost[il].choose_cpu(fits, missing);
        if let Some(layer) = self.weights.moe_metrics.as_ref().and_then(|m| m.layer(il)) {
            use std::sync::atomic::Ordering;
            if cpu {
                layer.cpu_decisions.fetch_add(1, Ordering::Relaxed);
            } else {
                layer.gpu_decisions.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(cpu)
    }
    fn linear(&self, il: usize, suffix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let name = format!("blk.{il}.{suffix}");
        let mut y = self.weights.matvec(&(name.clone() + ".weight"), x)?;
        if let Ok(bias) = self.weights.dense(&(name + ".bias")) {
            add(&mut y, bias)?;
        }
        Ok(y)
    }
    fn rms(&self, il: usize, suffix: &str, x: &[f32]) -> Result<Vec<f32>> {
        norm(
            x,
            self.weights.dense(&format!("blk.{il}.{suffix}.weight"))?,
            self.cfg.eps,
        )
    }
    fn expert(&self, il: usize, suffix: &str, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        let name = format!("blk.{il}.{suffix}");
        let mut y = self
            .weights
            .expert(&(name.clone() + ".weight"), expert, x)?;
        if let Ok(bias) = self.weights.dense(&(name + ".bias")) {
            let rows = y.len();
            let b = bias
                .get(expert * rows..(expert + 1) * rows)
                .ok_or_else(|| BitNetError::Inference("expert bias shape".into()))?;
            add(&mut y, b)?;
        }
        Ok(y)
    }
    fn expert_cpu(&self, il: usize, suffix: &str, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        let name = format!("blk.{il}.{suffix}");
        let mut y = self
            .weights
            .expert_cpu(&(name.clone() + ".weight"), expert, x)?;
        if let Ok(bias) = self.weights.dense(&(name + ".bias")) {
            let rows = y.len();
            let b = bias
                .get(expert * rows..(expert + 1) * rows)
                .ok_or_else(|| BitNetError::Inference("expert bias shape".into()))?;
            add(&mut y, b)?;
        }
        Ok(y)
    }
    fn ffn(&self, il: usize, x: &[f32], shared: bool) -> Result<Vec<f32>> {
        let suffix = if shared { "_shexp" } else { "" };
        let gate = self.linear(il, &format!("ffn_gate{suffix}"), x)?;
        let up = self.linear(il, &format!("ffn_up{suffix}"), x)?;
        let hidden: Vec<_> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        self.linear(il, &format!("ffn_down{suffix}"), &hidden)
    }
    fn moe(&mut self, il: usize, x: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let mut prob = self.linear(il, "ffn_gate_inp", x)?;
        if prob.len() != c.experts {
            return Err(BitNetError::Inference("router size mismatch".into()));
        }
        if c.family == Family::Mla {
            if c.sigmoid {
                for p in &mut prob {
                    *p = 1.0 / (1.0 + (-*p).exp());
                }
            } else {
                softmax(&mut prob, None);
            }
        }
        let bias = self
            .weights
            .dense(&format!("blk.{il}.exp_probs_b.bias"))
            .ok();
        let mut selection: Vec<f32> = prob
            .iter()
            .enumerate()
            .map(|(i, &p)| p + bias.map(|b| b[i]).unwrap_or(0.0))
            .collect();
        if c.groups > 1 {
            let group_size = c.experts / c.groups;
            let mut groups: Vec<_> = (0..c.groups)
                .map(|g| {
                    let mut p = selection[g * group_size..(g + 1) * group_size].to_vec();
                    p.sort_by(|a, b| b.total_cmp(a));
                    (g, p[0] + p.get(1).copied().unwrap_or(0.0))
                })
                .collect();
            groups.sort_by(|a, b| b.1.total_cmp(&a.1));
            for g in 0..c.groups {
                if !groups[..c.groups_used].iter().any(|&(i, _)| i == g) {
                    selection[g * group_size..(g + 1) * group_size].fill(f32::NEG_INFINITY);
                }
            }
        }
        let mut selected: Vec<_> = (0..c.experts).collect();
        selected.sort_by(|&a, &b| selection[b].total_cmp(&selection[a]).then(a.cmp(&b)));
        selected.truncate(c.used);
        let mut weights: Vec<f32> = selected.iter().map(|&i| prob[i]).collect();
        if c.family == Family::GptOss {
            softmax(&mut weights, None);
        } else if c.weight_norm {
            let sum = weights.iter().sum::<f32>().max(1.0 / 16384.0);
            for w in &mut weights {
                *w /= sum;
            }
        }
        let scaled: Vec<_> = weights.iter().map(|&w| w * c.weight_scale).collect();
        let force_cpu = self.choose_moe_cpu(il, &selected)?;
        let ffn_start = Instant::now();
        let gpu_result = if force_cpu {
            None
        } else if let Some(gpu) = &mut self.gpu_moe[il] {
            gpu.run_with_upload(x, &selected, &scaled)?
        } else {
            None
        };
        let fallback = gpu_result.is_none();
        let mut result = if let Some((output, upload)) = gpu_result {
            let elapsed = ffn_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            self.moe_cost[il].observe_gpu(elapsed, upload.bytes, upload.ns);
            output
        } else {
            let cpu_start = Instant::now();
            let output = if self.moe_execution == Execution::Cache {
                self.routed_ffn(il, x, &selected, &scaled)?
            } else {
                self.routed_ffn_cpu(il, x, &selected, &scaled)?
            };
            if self.moe_execution != Execution::Cache {
                self.moe_cost[il]
                    .observe_cpu(cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            }
            output
        };
        let elapsed = ffn_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        crate::perf::record_native_moe(!fallback, 1, elapsed);
        if let Some(metrics) = &self.weights.moe_metrics {
            metrics.ffn(il, !fallback, Some(elapsed));
            if !fallback && self.gpu_moe[il].as_ref().is_some_and(GpuMoe::is_fused){metrics.fused_ffn(il);}
        }
        if self
            .weights
            .archive
            .tensor_by_name(&format!("blk.{il}.ffn_gate_shexp.weight"))
            .is_some()
        {
            add(&mut result, &self.ffn(il, x, true)?)?;
        }
        Ok(result)
    }
    /// Routed contribution only, in selected router order; shared FFN is separate.
    fn routed_ffn(
        &self,
        il: usize,
        x: &[f32],
        selected: &[usize],
        scaled: &[f32],
    ) -> Result<Vec<f32>> {
        if selected.len() != self.cfg.used
            || scaled.len() != selected.len()
            || selected.iter().any(|&e| e >= self.cfg.experts)
        {
            return Err(BitNetError::Inference("routed FFN shape mismatch".into()));
        }
        let mut result = vec![0.0; self.cfg.embd];
        for (&expert, &weight) in selected.iter().zip(scaled) {
            let gate = self.expert(il, "ffn_gate_exps", expert, x)?;
            let up = self.expert(il, "ffn_up_exps", expert, x)?;
            let hidden: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    if self.cfg.family == Family::GptOss {
                        let g = g.min(7.0);
                        g / (1.0 + (-1.702 * g).exp()) * (u.clamp(-7.0, 7.0) + 1.0)
                    } else {
                        g / (1.0 + (-g).exp()) * u
                    }
                })
                .collect();
            let down = self.expert(il, "ffn_down_exps", expert, &hidden)?;
            for (out, d) in result.iter_mut().zip(down) {
                *out += weight * d;
            }
        }
        Ok(result)
    }
    fn routed_ffn_cpu(
        &self,
        il: usize,
        x: &[f32],
        selected: &[usize],
        scaled: &[f32],
    ) -> Result<Vec<f32>> {
        if selected.len() != self.cfg.used
            || scaled.len() != selected.len()
            || selected.iter().any(|&e| e >= self.cfg.experts)
        {
            return Err(BitNetError::Inference("routed FFN shape mismatch".into()));
        }
        let mut result = vec![0.0; self.cfg.embd];
        for (&expert, &weight) in selected.iter().zip(scaled) {
            let gate = self.expert_cpu(il, "ffn_gate_exps", expert, x)?;
            let up = self.expert_cpu(il, "ffn_up_exps", expert, x)?;
            let hidden: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    if self.cfg.family == Family::GptOss {
                        let g = g.min(7.0);
                        g / (1.0 + (-1.702 * g).exp()) * (u.clamp(-7.0, 7.0) + 1.0)
                    } else {
                        g / (1.0 + (-g).exp()) * u
                    }
                })
                .collect();
            let down = self.expert_cpu(il, "ffn_down_exps", expert, &hidden)?;
            for (out, d) in result.iter_mut().zip(down) {
                *out += weight * d;
            }
        }
        Ok(result)
    }
    fn resident_gpt_forward(
        &mut self,
        x: &[f32],
        pos: usize,
        logits: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        // Reinsert on every Result path, including cancellation. The native next
        // request begins at zero or restores an immutable, complete checkpoint.
        let mut full = self
            .gpu_full
            .take()
            .ok_or_else(|| BitNetError::Inference("resident GPT unavailable".into()))?;
        let result = (|| {
            full.begin(x, pos)?;
            for il in 0..self.cfg.layers {
                if inference_cancelled() {
                    return Err(BitNetError::Inference("inference cancelled".into()));
                }
                let (ids, weights) = full.prepare(il)?;
                let force_cpu = self.choose_moe_cpu(il, &ids)?;
                let start = Instant::now();
                let leased = if force_cpu {
                    None
                } else if let Some(moe) = &self.gpu_moe[il] {
                    moe.lease_selected(&ids)?
                } else {
                    None
                };
                let resident = leased.is_some();
                if let Some(leased) = leased {
                    let upload = leased.upload();
                    full.finish(il, leased.pointers(), None)?;
                    // Synchronous finish guarantees the selected slot is no
                    // longer being read before the lease owners are dropped.
                    drop(leased);
                    self.moe_cost[il].observe_gpu(
                        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        upload.bytes,
                        upload.ns,
                    );
                } else {
                    // Failed admission cost is recorded separately. It cannot
                    // contaminate the CPU calibration used for future choices.
                    let cpu_start = Instant::now();
                    let input = full.ffn_input()?;
                    let routed = if self.moe_execution == Execution::Cache {
                        self.routed_ffn(il, &input, &ids, &weights)?
                    } else {
                        self.routed_ffn_cpu(il, &input, &ids, &weights)?
                    };
                    full.finish(il, None, Some(&routed))?;
                    if self.moe_execution != Execution::Cache {
                        self.moe_cost[il].observe_cpu(
                            cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        );
                    }
                }
                let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                crate::perf::record_native_moe(resident, 1, elapsed);
                if let Some(metrics) = &self.weights.moe_metrics {
                    metrics.ffn(il, resident, Some(elapsed));
                    if resident && self.gpu_moe[il].as_ref().is_some_and(GpuMoe::is_fused){metrics.fused_ffn(il);}
                }
            }
            full.end(logits, greedy)
        })();
        self.gpu_full = Some(full);
        result
    }
    fn resident_mla_forward(
        &mut self,
        x: &[f32],
        pos: usize,
        logits: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        // Reinsert on every Result path, including cancellation. The native next
        // request begins at zero or restores an immutable, complete checkpoint.
        let mut full = self
            .gpu_mla
            .take()
            .ok_or_else(|| BitNetError::Inference("resident MLA unavailable".into()))?;
        let result = (|| {
            full.begin(x, pos)?;
            for il in 0..self.cfg.layers {
                if inference_cancelled() {
                    return Err(BitNetError::Inference("inference cancelled".into()));
                }
                let (ids, weights) = full.prepare(il)?;
                if il < self.cfg.dense_layers {
                    full.finish(il, None, None)?;
                    continue;
                }
                let force_cpu = self.choose_moe_cpu(il, &ids)?;
                let start = Instant::now();
                let leased = if force_cpu {
                    None
                } else if let Some(moe) = &self.gpu_moe[il] {
                    moe.lease_selected(&ids)?
                } else {
                    None
                };
                let resident = leased.is_some();
                if let Some(leased) = leased {
                    let upload = leased.upload();
                    full.finish(il, leased.pointers(), None)?;
                    // Synchronous finish guarantees the selected slot is no
                    // longer being read before the lease owners are dropped.
                    drop(leased);
                    self.moe_cost[il].observe_gpu(
                        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        upload.bytes,
                        upload.ns,
                    );
                } else {
                    // Failed admission cost is recorded separately. It cannot
                    // contaminate the CPU calibration used for future choices.
                    let cpu_start = Instant::now();
                    let input = full.ffn_input()?;
                    let routed = if self.moe_execution == Execution::Cache {
                        self.routed_ffn(il, &input, &ids, &weights)?
                    } else {
                        self.routed_ffn_cpu(il, &input, &ids, &weights)?
                    };
                    full.finish(il, None, Some(&routed))?;
                    if self.moe_execution != Execution::Cache {
                        self.moe_cost[il].observe_cpu(
                            cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        );
                    }
                }
                let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                crate::perf::record_native_moe(resident, 1, elapsed);
                if let Some(metrics) = &self.weights.moe_metrics {
                    metrics.ffn(il, resident, Some(elapsed));
                    if resident && self.gpu_moe[il].as_ref().is_some_and(GpuMoe::is_fused){metrics.fused_ffn(il);}
                }
            }
            full.end(logits, greedy)
        })();
        self.gpu_mla = Some(full);
        result
    }
    fn attention(&mut self, il: usize, pos: usize, x: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if c.family == Family::GptOss {
            let mut q = self.linear(il, "attn_q", x)?;
            let mut k = self.linear(il, "attn_k", x)?;
            let v = self.linear(il, "attn_v", x)?;
            for h in q.chunks_exact_mut(c.head) {
                rope(&mut h[..c.rotary], pos, c);
            }
            for h in k.chunks_exact_mut(c.head) {
                rope(&mut h[..c.rotary], pos, c);
            }
            let stride = c.kv_heads * c.head;
            self.kv[il].k.truncate(pos * stride);
            self.kv[il].k.extend(k);
            self.kv[il].v.truncate(pos * stride);
            self.kv[il].v.extend(v);
            let kv = &self.kv[il];
            let sinks = self.weights.dense(&format!("blk.{il}.attn_sinks.weight"))?;
            let first = if c.window > 0 && il % 2 == 0 {
                (pos + 1).saturating_sub(c.window)
            } else {
                0
            };
            if self.use_gpu_attention {
                let slot = &mut self.gpu_attention[il];
                if slot.is_none() {
                    *slot = super::attention::CudaAttention::new(
                        c.max_seq, c.kv_heads, c.head, c.value, c.heads,
                    );
                }
                if let Some(attention) = slot {
                    let mut output = vec![0.0; c.heads * c.value];
                    attention.run(
                        &q,
                        &kv.k,
                        &kv.v,
                        pos,
                        first,
                        (c.head as f32).sqrt().recip(),
                        Some(sinks),
                        &mut output,
                    )?;
                    return self.linear(il, "attn_output", &output);
                }
            }
            let heads: Vec<Vec<f32>> = (0..c.heads)
                .into_par_iter()
                .map(|h| {
                    let kv_head = h / (c.heads / c.kv_heads);
                    let query = &q[h * c.head..(h + 1) * c.head];
                    let mut scores: Vec<f32> = (first..=pos)
                        .map(|p| {
                            crate::ggml::simd::dot(
                                query,
                                &kv.k[p * stride + kv_head * c.head
                                    ..p * stride + (kv_head + 1) * c.head],
                            ) / (c.head as f32).sqrt()
                        })
                        .collect();
                    softmax(&mut scores, Some(sinks[h]));
                    let mut out = vec![0.0; c.head];
                    for (p, &s) in (first..=pos).zip(&scores) {
                        for i in 0..c.head {
                            out[i] += s * kv.v[p * stride + kv_head * c.head + i];
                        }
                    }
                    out
                })
                .collect();
            return self.linear(
                il,
                "attn_output",
                &heads.into_iter().flatten().collect::<Vec<_>>(),
            );
        }
        // Absorb K into Q and project V after attention: retain only the compressed MLA cache.
        let qa = self.linear(il, "attn_q_a", x)?;
        let qa = self.rms(il, "attn_q_a_norm", &qa)?;
        let mut q = self.linear(il, "attn_q_b", &qa)?;
        let compressed = self.linear(il, "attn_kv_a_mqa", x)?;
        let rank = c.kv_rank;
        let normalized = self.rms(il, "attn_kv_a_norm", &compressed[..rank])?;
        let mut key = normalized.clone();
        let mut k_rope = compressed[rank..].to_vec();
        rope(&mut k_rope, pos, c);
        key.extend(k_rope);
        for h in q.chunks_exact_mut(c.head) {
            rope(&mut h[c.head - c.rotary..], pos, c);
        }
        let stride = rank + c.rotary;
        self.kv[il].k.truncate(pos * stride);
        self.kv[il].k.extend(key);
        self.kv[il].v.truncate(pos * rank);
        self.kv[il].v.extend(normalized);
        // Quantized head matrices have one independent slab for each attention head.
        let non_rotary: Vec<f32> = q
            .chunks_exact(c.head)
            .flat_map(|h| h[..c.head - c.rotary].iter().copied())
            .collect();
        let absorbed = self
            .weights
            .heads(&format!("blk.{il}.attn_k_b.weight"), &non_rotary)?;
        let kv = &self.kv[il];
        let values = if self.use_gpu_attention {
            let slot = &mut self.gpu_attention[il];
            if slot.is_none() {
                *slot = super::attention::CudaAttention::new(c.max_seq, 1, stride, rank, c.heads);
            }
            if let Some(attention) = slot {
                let queries: Vec<f32> = (0..c.heads)
                    .flat_map(|h| {
                        absorbed[h * rank..(h + 1) * rank]
                            .iter()
                            .chain(q[h * c.head + c.head - c.rotary..(h + 1) * c.head].iter())
                            .copied()
                    })
                    .collect();
                let mut values = vec![0.0; c.heads * rank];
                attention.run(
                    &queries,
                    &kv.k,
                    &kv.v,
                    pos,
                    0,
                    (c.head as f32).sqrt().recip(),
                    None,
                    &mut values,
                )?;
                Some(values)
            } else {
                None
            }
        } else {
            None
        };
        let values = if let Some(values) = values {
            values
        } else {
            let values: Vec<Vec<f32>> = (0..c.heads)
                .into_par_iter()
                .map(|h| {
                    let mut query = absorbed[h * rank..(h + 1) * rank].to_vec();
                    query.extend_from_slice(&q[h * c.head + c.head - c.rotary..(h + 1) * c.head]);
                    let mut scores: Vec<f32> = (0..=pos)
                        .map(|p| {
                            crate::ggml::simd::dot(&query, &kv.k[p * stride..(p + 1) * stride])
                                / (c.head as f32).sqrt()
                        })
                        .collect();
                    softmax(&mut scores, None);
                    let mut value = vec![0.0; rank];
                    for (p, s) in scores.into_iter().enumerate() {
                        for i in 0..rank {
                            value[i] += s * kv.k[p * stride + i];
                        }
                    }
                    value
                })
                .collect();
            values.into_iter().flatten().collect::<Vec<f32>>()
        };
        let attended = self
            .weights
            .heads(&format!("blk.{il}.attn_v_b.weight"), &values)?;
        self.linear(il, "attn_output", &attended)
    }
    fn forward(
        &mut self,
        token: u32,
        pos: usize,
        logits: bool,
        greedy: bool,
    ) -> Result<(Vec<f32>, Option<u32>)> {
        if let Some(cache) = &self.weights.expert_cache {
            cache
                .lock()
                .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?
                .begin_pass(pos);
        }
        let c = &self.cfg;
        if pos >= c.max_seq || token as usize >= c.vocab {
            return Err(BitNetError::Inference("token/context out of bounds".into()));
        }
        let tensor = self.weights.tensor("token_embd.weight")?;
        let mut x = vec![0.0; c.embd];
        crate::ggml::embedding_row_mmap(
            &self.weights.archive,
            tensor,
            token as usize,
            c.embd,
            c.vocab,
            &mut x,
        )?;
        if self
            .gpu_full
            .as_ref()
            .is_some_and(|full| full.is_segmented())
        {
            return self.resident_gpt_forward(&x, pos, logits, greedy);
        }
        if let Some(full) = &mut self.gpu_full {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            return full.run(&x, pos, logits, greedy);
        }
        if self.gpu_mla.is_some() {
            return self.resident_mla_forward(&x, pos, logits, greedy);
        }
        for il in 0..self.cfg.layers {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let h = self.rms(il, "attn_norm", &x)?;
            let attn = self.attention(il, pos, &h)?;
            add(&mut x, &attn)?;
            let h = self.rms(
                il,
                if self.cfg.family == Family::GptOss {
                    "post_attention_norm"
                } else {
                    "ffn_norm"
                },
                &x,
            )?;
            let ffn = if il < self.cfg.dense_layers {
                self.ffn(il, &h, false)?
            } else {
                self.moe(il, &h)?
            };
            add(&mut x, &ffn)?;
        }
        if !logits {
            return Ok((Vec::new(), None));
        }
        if let Some(head) = &mut self.gpu_head {
            return head.run(&x, greedy);
        }
        let x = norm(&x, self.weights.dense("output_norm.weight")?, self.cfg.eps)?;
        Ok((self.weights.matvec("output.weight", &x)?, None))
    }
    fn generate(
        &mut self,
        prompt: &str,
        limit: u32,
        sampling: SamplingOptions,
        mut events: Option<&mut (dyn FnMut(StreamEvent) -> Result<()> + Send)>,
    ) -> Result<(String, PhaseTimings)> {
        for kv in &mut self.kv {
            kv.k.clear();
            kv.v.clear();
        }
        for attention in self.gpu_attention.iter_mut().flatten() {
            attention.clear();
        }
        let enc = Instant::now();
        let ids = self.tokenizer.encode_ids(prompt, true)?;
        let encode_ms = enc.elapsed().as_millis() as u64;
        if ids.is_empty() {
            return Ok((String::new(), PhaseTimings::default()));
        }
        if ids.len().saturating_add(limit as usize) > self.cfg.max_seq {
            return Err(BitNetError::Inference(
                "prompt and generation exceed context capacity".into(),
            ));
        }
        let pf = Instant::now();
        if let Some(cache) = &self.weights.expert_cache {
            cache
                .lock()
                .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?
                .begin_sequence(ids.len());
        }
        let mut logits = Vec::new();
        let mut next_token = None;
        let gpu_greedy =
            (self.gpu_full.is_some() || self.gpu_mla.is_some() || self.gpu_head.is_some())
                && sampling.device_greedy_eligible();
        let reused = if let Some(full) = &mut self.gpu_full {
            full.restore_prefix(&ids)?
        } else if let Some(full) = &mut self.gpu_mla {
            full.restore_prefix(&ids)?
        } else {
            0
        };
        let block_capacity=self.gpu_full.as_ref().map_or(0,|f|f.prefill_capacity());
        if block_capacity>0 {
            let mut full=self.gpu_full.take().unwrap();
            let result=(|| {
                let tensor=self.weights.tensor("token_embd.weight")?;
                let mut pos=reused;
                while pos<ids.len() {
                    if inference_cancelled() {return Err(BitNetError::Inference("inference cancelled".into()));}
                    let count=block_capacity.min(ids.len()-pos);
                    let mut embeddings=vec![0.0;count*self.cfg.embd];
                    for (&token,row) in ids[pos..pos+count].iter().zip(embeddings.chunks_exact_mut(self.cfg.embd)) {
                        if token as usize>=self.cfg.vocab {return Err(BitNetError::Inference("block token out of bounds".into()));}
                        crate::ggml::embedding_row_mmap(&self.weights.archive,tensor,token as usize,self.cfg.embd,self.cfg.vocab,row)?;
                    }
                    (logits,next_token)=full.prefill(&embeddings,pos,count,pos+count==ids.len(),gpu_greedy)?;
                    pos+=count;
                }
                Ok(())
            })();
            self.gpu_full=Some(full);
            result?;
        }else {
            for (pos,&id) in ids.iter().enumerate().skip(reused) {
                (logits,next_token)=self.forward(id,pos,pos+1==ids.len(),gpu_greedy)?;
            }
        }
        if let Some(full) = &mut self.gpu_full {
            full.save_prefix(&ids)?;
        } else if let Some(full) = &mut self.gpu_mla {
            full.save_prefix(&ids)?;
        }
        let prefill_ms = pf.elapsed().as_millis() as u64;
        let mut rng = match sampling.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let mut generated = Vec::new();
        let stop = self.tokenizer.eos_token_ids();
        let dec = Instant::now();
        let mut previous = String::new();
        let mut emitted = String::new();
        for step in 0..limit {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let next = next_token
                .take()
                .unwrap_or_else(|| sample_token(&logits, &sampling, &generated, &mut rng));
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            let raw = self.tokenizer.decode_ids(&generated, false)?;
            let text = if self.cfg.family == Family::GptOss {
                super::output::gpt_oss_content(&raw).to_owned()
            } else {
                raw
            };
            if let Some(callback) = events.as_deref_mut() {
                emit_text_delta(&text, &mut emitted, false, callback)?;
            }
            previous = text;
            if step + 1 < limit {
                (logits, next_token) =
                    self.forward(next, ids.len() + step as usize, true, gpu_greedy)?;
            }
        }
        let phases = PhaseTimings {
            encode_ms,
            prefill_ms,
            decode_ms: dec.elapsed().as_millis() as u64,
            prompt_tokens: ids.len() as u32,
            completion_tokens: generated.len() as u32,
        };
        if let Some(callback) = events.as_deref_mut() {
            emit_text_delta(&previous, &mut emitted, true, callback)?;
        }
        if let Some(cache) = &self.weights.expert_cache {
            cache
                .lock()
                .map_err(|_| BitNetError::Inference("expert cache lock poisoned".into()))?
                .flush_trace();
        }
        Ok((previous, phases))
    }
}

pub(crate) struct NativeExecutor {
    kind: BackendKind,
    family: Family,
    id: String,
    runtime: Mutex<Runtime>,
}

impl NativeExecutor {
    pub fn load(
        archive: Arc<GgufArchive>,
        tokenizer: &Path,
        kind: BackendKind,
        family: Family,
    ) -> Result<Self> {
        let id = archive.suggested_openai_model_id();
        let runtime = Runtime::load(archive, tokenizer, kind, family)?;
        Ok(Self {
            kind,
            family,
            id,
            runtime: Mutex::new(runtime),
        })
    }
}
impl ModelExecutor for NativeExecutor {
    fn family(&self) -> &'static str {
        self.family.name()
    }
    fn backend(&self) -> BackendKind {
        self.kind
    }
    fn backend_accelerated(&self) -> bool {
        self.runtime
            .lock()
            .map(|r| r.weights.resident_bytes > 0)
            .unwrap_or(false)
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn openai_model_id(&self, _: Option<&GgufArchive>) -> Option<String> {
        Some(self.id.clone())
    }
    fn count_prompt_tokens(&self, prompt: &str) -> Result<u32> {
        let r = self
            .runtime
            .lock()
            .map_err(|_| BitNetError::Inference("runtime lock poisoned".into()))?;
        Ok(r.tokenizer.encode_ids(prompt, true)?.len() as u32)
    }
    fn offload_metadata(&self) -> Option<String> {
        self.runtime.lock().ok().map(|r| {
            let execution=if let Some(full)=&r.gpu_full {
                let mode = if full.is_segmented() {
                    "host-admitted per-layer segments; CPU routed FFN fallback available"
                } else {
                    "fixed expert banks; whole-token graph"
                };
                format!("resident quantized weights: {} MiB; resident GPT-OSS attention/router/head: true; fully resident GPT-OSS token pipeline: {}; {}; prefix snapshot ABI: {}; CUDA graphs: {}; split-KV: {}; context capacity: {}",(r.weights.resident_bytes+full.extra_weights_bytes)/(1024*1024),!full.is_segmented(),mode,full.supports_prefix(),full.graphs,full.split,r.cfg.max_seq)
            } else if let Some(full)=&r.gpu_mla {
                format!("resident quantized weights: {} MiB; resident MLA attention/router/head: true; compressed KV on device: true; host expert admission per layer; CPU routed FFN fallback available; CUDA graphs: {}; split-KV: {}; context capacity: {}",(r.weights.resident_bytes+full.extra_weights_bytes)/(1024*1024),full.graphs,full.split,r.cfg.max_seq)
            } else {
                format!("resident quantized weights: {} MiB; fully resident GPT-OSS token pipeline: false; resident output head: {}; resident routed expert layers: {}; attention GPU enabled: {}; remaining operations execute on CPU; context capacity: {}",r.weights.resident_bytes/(1024*1024),r.gpu_head.is_some(),r.gpu_moe.iter().filter(|m|m.is_some()).count(),r.use_gpu_attention,r.cfg.max_seq)
            };
            let gpt_block=r.gpu_full.as_ref().map_or(0,|f|f.prefill_capacity());
            let execution=format!("{execution}; GPT block prefill capacity: {gpt_block}");
            format!("{execution}; weight budget bytes: {}; native state reservation bytes: {}; fused resident expert contexts: {}/{}",r.weights.residency_budget_bytes,r.weights.state_reserve_bytes,r.gpu_moe.iter().flatten().filter(|m|m.is_fused()).count(),r.gpu_moe.iter().flatten().count())
        })
    }
    fn generate_with_timings(
        &self,
        prompt: &str,
        limit: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, PhaseTimings)> {
        self.runtime
            .lock()
            .map_err(|_| BitNetError::Inference("runtime lock poisoned".into()))?
            .generate(prompt, limit, sampling, None)
    }
    fn generate_streaming(
        &self,
        prompt: &str,
        limit: u32,
        sampling: SamplingOptions,
        callback: &mut (dyn FnMut(StreamEvent) -> Result<()> + Send),
    ) -> Result<()> {
        let (text, phases) = self
            .runtime
            .lock()
            .map_err(|_| BitNetError::Inference("runtime lock poisoned".into()))?
            .generate(prompt, limit, sampling, Some(callback))?;
        callback(StreamEvent::Done(crate::scheduler::InferenceOutput {
            text,
            stats: crate::scheduler::InferenceStats::from_phases(phases, false),
        }))
    }
}

#[cfg(test)]
mod tests {
    struct ScopedEnv(Vec<(String, Option<std::ffi::OsString>)>);
    impl ScopedEnv {
        fn new(keys: &[&str]) -> Self {
            Self(
                keys.iter()
                    .map(|&k| (k.to_owned(), std::env::var_os(k)))
                    .collect(),
            )
        }
    }
    impl Drop for ScopedEnv {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                if let Some(v) = v {
                    std::env::set_var(k, v);
                } else {
                    std::env::remove_var(k);
                }
            }
        }
    }
    #[test]
    fn opt_in_gpt_full_real_layer_diagnosis() {
        use super::*;
        if std::env::var("RBITNET_GPT_TRACE_TEST").as_deref() != Ok("1") {
            return;
        }
        let _env = ScopedEnv::new(&[
            "RBITNET_CUDA_GPT_FULL",
            "RBITNET_REQUIRE_GPT_FULL",
            "RBITNET_CUDA_SPLIT_KV",
        ]);
        let gguf = std::env::var("RBITNET_GPT_TEST_GGUF").unwrap();
        let tokenizer = std::env::var("RBITNET_GPT_TEST_TOKENIZER").unwrap();
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        std::env::set_var("RBITNET_CUDA_GPT_FULL", "0");
        std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "0");
        let mut r = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::GptOss,
        )
        .unwrap();
        let ids:Vec<_>=r.tokenizer.encode_ids(&"Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),true).unwrap().into_iter().take(273).collect();
        let mut embeddings = Vec::new();
        let mut expected = Vec::new();
        let mut raw_scores = Vec::new();
        let mut routes = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let c = &r.cfg;
            let mut x = vec![0.0; c.embd];
            crate::ggml::embedding_row_mmap(
                &r.weights.archive,
                r.weights.tensor("token_embd.weight").unwrap(),
                id as usize,
                c.embd,
                c.vocab,
                &mut x,
            )
            .unwrap();
            embeddings.push(x.clone());
            let mut hidden = Vec::new();
            let mut scores = Vec::new();
            let mut selected = Vec::new();
            for il in 0..r.cfg.layers {
                let h = r.rms(il, "attn_norm", &x).unwrap();
                let attn = r.attention(il, pos, &h).unwrap();
                add(&mut x, &attn).unwrap();
                let h = r.rms(il, "post_attention_norm", &x).unwrap();
                let raw = r.linear(il, "ffn_gate_inp", &h).unwrap();
                let bias = r.weights.dense(&format!("blk.{il}.exp_probs_b.bias")).ok();
                let mut ids: Vec<_> = (0..r.cfg.experts).collect();
                ids.sort_by(|&a, &b| {
                    (raw[b] + bias.map(|v| v[b]).unwrap_or(0.0))
                        .total_cmp(&(raw[a] + bias.map(|v| v[a]).unwrap_or(0.0)))
                        .then(a.cmp(&b))
                });
                ids.truncate(r.cfg.used);
                selected.push(ids);
                scores.push(raw);
                let ffn = r.moe(il, &h).unwrap();
                add(&mut x, &ffn).unwrap();
                hidden.push(x.clone());
            }
            expected.push(hidden);
            raw_scores.push(scores);
            routes.push(selected);
        }
        let embd = r.cfg.embd;
        let experts = r.cfg.experts;
        let used = r.cfg.used;
        drop(r);
        std::env::set_var("RBITNET_CUDA_GPT_FULL", "1");
        std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "1");
        std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
        let mut r = Runtime::load(
            archive,
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::GptOss,
        )
        .unwrap();
        let mut mismatches = 0;
        for (pos, x) in embeddings.iter().enumerate() {
            let trace = r.gpu_full.as_mut().unwrap().trace(x, pos, experts, used);
            for (il, row) in trace.chunks_exact(embd + experts + used).enumerate() {
                let ids: Vec<_> = row[embd + experts..].iter().map(|&v| v as usize).collect();
                let diff = row[..embd]
                    .iter()
                    .zip(&expected[pos][il])
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if ids != routes[pos][il] {
                    mismatches += 1;
                    eprintln!("pos={pos} layer={il} hidden_max_abs={diff:.6} actual_ids={ids:?} expected_ids={:?} actual_logits={:?} expected_logits={:?}",routes[pos][il],&row[embd..embd+experts],raw_scores[pos][il]);
                } else if [0, 127, 128, 254, 255, 256, 272].contains(&pos) {
                    eprintln!("pos={pos} layer={il} hidden_max_abs={diff:.6} routes_match");
                }
            }
        }
        eprintln!("GPT route disagreements={mismatches}");
    }

    #[test]
    fn opt_in_gpt_full_real_teacher_forcing_greedy_seed_penalty_and_reset() {
        use super::*;
        if std::env::var("RBITNET_GPT_FULL_TEST").as_deref() != Ok("1") {
            return;
        }
        let gguf = std::env::var("RBITNET_GPT_TEST_GGUF").unwrap();
        let tokenizer = std::env::var("RBITNET_GPT_TEST_TOKENIZER").unwrap();
        let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
        let corpus=["Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),"fn somme(values: &[i32]) -> i32 { values.iter().sum() } // vérifier les cas vides et les valeurs négatives\n".repeat(40)];
        let prompts=["<|start|>user<|message|>Quelle est la capitale de la France ? Réponds en un mot.<|end|><|start|>assistant<|channel|>final<|message|>","<|start|>user<|message|>Écris une fonction Python qui additionne deux nombres.<|end|><|start|>assistant<|channel|>final<|message|>"];
        let samples = [
            SamplingOptions::from_temperature(0.0),
            SamplingOptions {
                seed: Some(42),
                ..SamplingOptions::from_temperature(0.7)
            },
            SamplingOptions {
                frequency_penalty: 0.1,
                presence_penalty: 0.1,
                seed: Some(7),
                ..SamplingOptions::from_temperature(0.0)
            },
        ];
        let observed = [0, 1, 7, 15, 31, 63, 127, 128, 255, 256, 271, 272];
        // Keep all environment changes scoped even when an assertion unwinds.
        struct Env(Vec<(String, Option<std::ffi::OsString>)>);
        impl Drop for Env {
            fn drop(&mut self) {
                for (k, v) in &self.0 {
                    if let Some(v) = v {
                        std::env::set_var(k, v);
                    } else {
                        std::env::remove_var(k);
                    }
                }
            }
        }
        let _env = Env([
            "RBITNET_CUDA_GPT_FULL",
            "RBITNET_REQUIRE_GPT_FULL",
            "RBITNET_CUDA_GPT_FULL_GRAPH",
            "RBITNET_CUDA_SPLIT_KV",
            "RBITNET_MOE_CACHE_MB",
        ]
        .into_iter()
        .map(|k| (k.to_owned(), std::env::var_os(k)))
        .collect());
        std::env::set_var("RBITNET_MOE_CACHE_MB", "0");
        std::env::set_var("RBITNET_CUDA_GPT_FULL", "0");
        std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "0");
        let mut reference = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::GptOss,
        )
        .unwrap();
        eprintln!(
            "GPT reference resident weights {} MiB; fixed expert layers {}/{}",
            reference.weights.resident_bytes / (1024 * 1024),
            reference.gpu_moe.iter().flatten().count(),
            reference.cfg.layers
        );
        let inputs: Vec<Vec<u32>> = corpus
            .iter()
            .map(|p| {
                reference
                    .tokenizer
                    .encode_ids(p, true)
                    .unwrap()
                    .into_iter()
                    .take(273)
                    .collect()
            })
            .collect();
        assert!(inputs.iter().all(|v| v.len() == 273));
        let mut expected = Vec::new();
        for ids in &inputs {
            let mut outputs = Vec::new();
            for (pos, &id) in ids.iter().enumerate() {
                let (logits, _) = reference
                    .forward(id, pos, observed.contains(&pos), false)
                    .unwrap();
                if observed.contains(&pos) {
                    outputs.push(logits);
                }
            }
            expected.push(outputs);
        }
        let mut texts = Vec::new();
        for prompt in prompts {
            for sampling in &samples {
                texts.push(reference.generate(prompt, 32, *sampling, None).unwrap().0);
            }
        }
        drop(reference);
        let mut worst_kl = 0.0f64;
        let mut worst_nll = 0.0f64;
        for (graphs, split) in [("0", "0"), ("1", "0"), ("1", "1")] {
            std::env::set_var("RBITNET_CUDA_GPT_FULL", "1");
            std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "1");
            std::env::set_var("RBITNET_CUDA_GPT_FULL_GRAPH", graphs);
            std::env::set_var("RBITNET_CUDA_SPLIT_KV", split);
            let mut actual = Runtime::load(
                Arc::clone(&archive),
                Path::new(&tokenizer),
                BackendKind::Cuda,
                Family::GptOss,
            )
            .unwrap();
            assert!(actual
                .gpu_full
                .as_ref()
                .is_some_and(|full| !full.is_segmented()));
            for (case, ids) in inputs.iter().enumerate() {
                let mut n = 0;
                for (pos, &id) in ids.iter().enumerate() {
                    let (got, _) = actual
                        .forward(id, pos, observed.contains(&pos), false)
                        .unwrap();
                    if !observed.contains(&pos) {
                        continue;
                    }
                    let expected = &expected[case][n];
                    n += 1;
                    let argmax = |x: &[f32]| {
                        x.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap()
                            .0
                    };
                    assert_eq!(
                        argmax(&got),
                        argmax(expected),
                        "corpus {case} position {pos} graphs {graphs} split {split}"
                    );
                    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
                        assert!((a-b).abs()<=0.003*(1.0+b.abs()),"corpus={case} pos={pos} graphs={graphs} split={split} token={i}: {a} vs {b}");
                    }
                    let logprob = |x: &[f32]| {
                        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        let total =
                            x.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
                        x.iter().map(|&v| v as f64 - total).collect::<Vec<_>>()
                    };
                    let p = logprob(expected);
                    let q = logprob(&got);
                    let kl = p
                        .iter()
                        .zip(&q)
                        .map(|(&p, &q)| p.exp() * (p - q))
                        .sum::<f64>()
                        .max(0.0);
                    let id = ids.get(pos + 1).copied().unwrap_or(ids[pos]) as usize;
                    let nll = (p[id] - q[id]).abs();
                    worst_kl = worst_kl.max(kl);
                    worst_nll = worst_nll.max(nll);
                    assert!(
                        kl <= 1e-5 && nll <= 1e-3,
                        "KL={kl} NLL_delta={nll}, corpus {case}, pos {pos}"
                    );
                }
            }
            let mut n = 0;
            for prompt in prompts {
                for sampling in &samples {
                    let got = actual.generate(prompt, 32, *sampling, None).unwrap().0;
                    assert_eq!(got, texts[n], "graphs {graphs} split {split} sample {n}");
                    n += 1;
                }
            }
            eprintln!("GPT-OSS real teacher-forcing + six generations passed graphs={graphs}, split={split}");
            drop(actual);
        }
        eprintln!(
            "GPT real worst KL={worst_kl:.3e}, worst absolute target NLL delta={worst_nll:.3e}"
        );
    }

    #[test]
    fn mla_rotates_consecutive_pairs_and_gpt_oss_rotates_halves() {
        let mut cfg = super::Config {
            family: super::Family::Mla,
            embd: 4,
            vocab: 1,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head: 4,
            value: 4,
            rotary: 4,
            kv_rank: 2,
            max_seq: 16,
            eps: 1e-5,
            theta: 10000.0,
            yarn_factor: 1.0,
            yarn_orig: 16,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            experts: 1,
            used: 1,
            dense_layers: 0,
            groups: 1,
            groups_used: 1,
            sigmoid: true,
            weight_norm: true,
            weight_scale: 1.0,
            window: 0,
        };
        // Closed-form rotations at angles 1 and 0.01 radians, evaluated independently.
        let mut values = [1.0, 2.0, 3.0, 4.0];
        super::rope(&mut values, 1, &cfg);
        let (s, c) = 1.0_f64.sin_cos();
        let (s2, c2) = 0.01_f64.sin_cos();
        let expected = [
            c - 2.0 * s,
            s + 2.0 * c,
            3.0 * c2 - 4.0 * s2,
            3.0 * s2 + 4.0 * c2,
        ];
        for (&actual, expected) in values.iter().zip(expected) {
            assert!((actual as f64 - expected).abs() < 1e-6);
        }
        cfg.family = super::Family::GptOss;
        let mut values = [1.0, 2.0, 3.0, 4.0];
        super::rope(&mut values, 1, &cfg);
        let expected = [
            c - 3.0 * s,
            2.0 * c2 - 4.0 * s2,
            s + 3.0 * c,
            2.0 * s2 + 4.0 * c2,
        ];
        for (&actual, expected) in values.iter().zip(expected) {
            assert!((actual as f64 - expected).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[path = "mla_runtime_tests.rs"]
mod mla_runtime_tests;

#[cfg(test)]
#[path = "moe_policy_runtime_tests.rs"]
mod moe_policy_runtime_tests;

#[cfg(test)]
#[path="gpt_block_runtime_tests.rs"]
mod gpt_block_runtime_tests;
