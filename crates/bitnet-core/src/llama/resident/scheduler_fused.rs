//! Stateful CUDA decode waves for Sarathi + `RBITNET_FUSED_MULTI_SEQ` (#96).
//!
//! Keeps per-request KV owners and runs `rbitnet_cuda_llama_batch_step` when a wave
//! admits more than one ready sequence. Prefill for newly promoted sequences stays
//! serial (128-token partitions); decode rows share projection GEMMs on the GPU.

use super::transient_batch::NativeBatchWorkspace;
use super::{configured_page_limit, LlamaModel, Resident};
use crate::error::{BitNetError, Result};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::timings::{GenerationFinishReason, PhaseTimings};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SchedulerFusedOptions {
    pub maximum: usize,
    pub ordering: u32,
    pub pages: Option<u32>,
}

pub(crate) fn configured(
    backend: crate::backend::BackendKind,
) -> Result<Option<SchedulerFusedOptions>> {
    if backend != crate::backend::BackendKind::Cuda {
        return Ok(None);
    }
    if crate::llama::BatchOptions::configured(backend)?.is_some() {
        return Ok(None);
    }
    let enabled = crate::fused_batch::fused_multi_seq_enabled()
        || matches!(
            std::env::var("RBITNET_CUDA_FUSED_DECODE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
    if !enabled {
        return Ok(None);
    }
    if super::configured_kv_format()? != 0 {
        return Err(BitNetError::Inference(
            "CUDA fused scheduler decode requires F32 native KV".into(),
        ));
    }
    if std::env::var("RBITNET_CUDA_PREFILL").as_deref() != Ok("1") {
        return Err(BitNetError::Inference(
            "CUDA fused scheduler decode requires RBITNET_CUDA_PREFILL=1".into(),
        ));
    }
    if !matches!(
        std::env::var("RBITNET_CUDA_PREFILL_TOKENS").as_deref(),
        Err(std::env::VarError::NotPresent) | Ok("128")
    ) {
        return Err(BitNetError::Inference(
            "CUDA fused scheduler decode preserves 128-token prefill partitions".into(),
        ));
    }
    for key in [
        "RBITNET_CUDA_SPLIT_KV",
        "RBITNET_CUDA_PREFILL_TF32X3",
        "RBITNET_PREFIX_KV",
        "RBITNET_SPECULATIVE",
        "RBITNET_SPECULATIVE_PLD",
    ] {
        if matches!(
            std::env::var(key).as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        ) {
            return Err(BitNetError::Inference(format!(
                "{key} is not validated with CUDA fused scheduler decode"
            )));
        }
    }
    let parse = |key: &str, default: usize, min: usize, max: usize| -> Result<usize> {
        let value = std::env::var(key).ok().map_or(Ok(default), |s| {
            s.parse::<usize>()
                .map_err(|_| BitNetError::Inference(format!("{key} must be an integer")))
        })?;
        if !(min..=max).contains(&value) {
            return Err(BitNetError::Inference(format!(
                "{key} must be {min}..{max}"
            )));
        }
        Ok(value)
    };
    Ok(Some(SchedulerFusedOptions {
        maximum: parse("RBITNET_CUDA_FUSED_DECODE_SLOTS", 8, 1, 8)?,
        ordering: parse("RBITNET_CUDA_FUSED_DECODE_ORDERING", 0, 0, 1)? as u32,
        pages: configured_page_limit()?,
    }))
}

struct Sequence {
    resident: Resident,
    prompt_ids: Vec<u32>,
    generated: Vec<u32>,
    pending: Option<u32>,
    /// First sampled token from the terminal prefill chunk (not yet emitted).
    prefill_head: Option<u32>,
    next_position: usize,
    prefilled: usize,
    prefill_ms: u64,
    rng: StdRng,
    sampling: SamplingOptions,
    prompt_fingerprint: String,
}

pub(crate) struct SchedulerFusedLlama {
    model: Arc<LlamaModel>,
    tokenizer: Arc<LoadedPromptTokenizer>,
    workspace: NativeBatchWorkspace,
    page_limit: Option<u32>,
    sequences: BTreeMap<u64, Sequence>,
    maximum: usize,
    eos: Vec<u32>,
}

impl SchedulerFusedLlama {
    pub(crate) fn new(
        model: Arc<LlamaModel>,
        tokenizer: Arc<LoadedPromptTokenizer>,
        options: SchedulerFusedOptions,
    ) -> Result<Self> {
        let seed = Resident::new_with_pages(&model, options.pages, None)
            .ok_or_else(|| BitNetError::Inference("fused decode resident seed refused".into()))?;
        if seed.prefill.is_none() {
            return Err(BitNetError::Inference(
                "CUDA fused scheduler decode requires native block prefill".into(),
            ));
        }
        let workspace = NativeBatchWorkspace::new(
            Arc::clone(&model),
            &seed,
            options.maximum,
            options.ordering,
        )?;
        drop(seed);
        let eos = tokenizer.eos_token_ids();
        Ok(Self {
            model,
            tokenizer,
            workspace,
            page_limit: options.pages,
            sequences: BTreeMap::new(),
            maximum: options.maximum,
            eos,
        })
    }

    pub(crate) fn begin_batch(&mut self, active_ids: &[u64]) {
        self.sequences.retain(|id, _| active_ids.contains(id));
    }

    pub(crate) fn end_batch(&mut self) {
        self.sequences.clear();
    }

    pub(crate) fn native_stats(&self) -> Result<[u64; 3]> {
        self.workspace.stats()
    }

    fn ensure_sequence(&mut self, id: u64, prompt: &str, sampling: SamplingOptions) -> Result<()> {
        if self.sequences.contains_key(&id) {
            let seq = self.sequences.get(&id).expect("sequence");
            if seq.prompt_fingerprint != prompt {
                return Err(BitNetError::Inference(format!(
                    "fused decode prompt drift for request {id}"
                )));
            }
            return Ok(());
        }
        if self.sequences.len() >= self.maximum {
            return Err(BitNetError::Inference(
                "CUDA fused scheduler decode slot limit exceeded".into(),
            ));
        }
        let ids = self
            .tokenizer
            .encode_ids(prompt, crate::llama::llama_encode_add_special_tokens())?;
        let peer = self.sequences.values().next().map(|s| &s.resident);
        let resident = Resident::new_with_pages(&self.model, self.page_limit, peer)
            .ok_or_else(|| BitNetError::Inference("fused decode KV allocation refused".into()))?;
        let rng = match sampling.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        self.sequences.insert(
            id,
            Sequence {
                resident,
                prompt_ids: ids,
                generated: Vec::new(),
                pending: None,
                prefill_head: None,
                next_position: 0,
                prefilled: 0,
                prefill_ms: 0,
                rng,
                sampling,
                prompt_fingerprint: prompt.to_owned(),
            },
        );
        Ok(())
    }

    /// Run admitted prompt tokens as shared native waves. A wave contains one
    /// next token per active request, so its projections are shared even when
    /// prompt lengths and scheduler chunk sizes differ.
    pub(crate) fn prefill_wave(
        &mut self,
        items: &[(u64, String, usize, SamplingOptions)],
    ) -> Result<()> {
        for &(id, ref prompt, _, sampling) in items {
            self.ensure_sequence(id, prompt, sampling)?;
        }
        let mut remaining: BTreeMap<u64, usize> = items
            .iter()
            .map(|(id, _, count, _)| (*id, *count))
            .collect();

        while remaining.values().any(|count| *count > 0) {
            let mut advance_ids = Vec::new();
            let mut greedy_last_ids = Vec::new();
            let mut logits_last_ids = Vec::new();
            for (&id, count) in &remaining {
                if *count == 0 {
                    continue;
                }
                let seq = self.sequences.get(&id).ok_or_else(|| {
                    BitNetError::Inference("fused prefill sequence missing".into())
                })?;
                if seq.pending.is_some() || seq.prefill_head.is_some() {
                    return Err(BitNetError::Inference(
                        "fused prefill admitted after sequence became decode-ready".into(),
                    ));
                }
                if seq.prefilled >= seq.prompt_ids.len() {
                    return Err(BitNetError::Inference(
                        "fused prefill admission exceeds prompt length".into(),
                    ));
                }
                if seq.prefilled + 1 == seq.prompt_ids.len() {
                    if seq.sampling.device_greedy_eligible() {
                        greedy_last_ids.push(id);
                    } else {
                        logits_last_ids.push(id);
                    }
                } else {
                    advance_ids.push(id);
                }
            }

            if !advance_ids.is_empty() {
                self.prefill_advance_rows(&advance_ids)?;
            }
            if !greedy_last_ids.is_empty() {
                self.prefill_last_greedy_rows(&greedy_last_ids)?;
            }
            if !logits_last_ids.is_empty() {
                self.prefill_last_logits_rows(&logits_last_ids)?;
            }
            for id in advance_ids
                .into_iter()
                .chain(greedy_last_ids)
                .chain(logits_last_ids)
            {
                *remaining.get_mut(&id).expect("prefill admission") -= 1;
            }
        }
        Ok(())
    }

    fn prefill_advance_rows(&mut self, ids: &[u64]) -> Result<()> {
        let mut tokens = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        for id in ids {
            let seq = self.sequences.get(id).expect("prefill sequence");
            tokens.push(seq.prompt_ids[seq.prefilled]);
            positions.push(seq.prefilled as u32);
        }
        let started = Instant::now();
        let mut owners: Vec<*mut Resident> = ids
            .iter()
            .map(|id| {
                &mut self
                    .sequences
                    .get_mut(id)
                    .expect("prefill sequence")
                    .resident as *mut Resident
            })
            .collect();
        let mut refs: Vec<&mut Resident> = owners
            .iter_mut()
            .map(|owner| unsafe { &mut **owner })
            .collect();
        self.workspace.advance(&mut refs, &tokens, &positions)?;
        let elapsed = started.elapsed().as_millis() as u64;
        for id in ids {
            let seq = self.sequences.get_mut(id).expect("prefill sequence");
            seq.prefilled += 1;
            seq.prefill_ms = seq.prefill_ms.saturating_add(elapsed);
        }
        Ok(())
    }

    fn prefill_last_greedy_rows(&mut self, ids: &[u64]) -> Result<()> {
        let mut tokens = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        for id in ids {
            let seq = self.sequences.get(id).expect("prefill sequence");
            tokens.push(seq.prompt_ids[seq.prefilled]);
            positions.push(seq.prefilled as u32);
        }
        let started = Instant::now();
        let mut owners: Vec<*mut Resident> = ids
            .iter()
            .map(|id| {
                &mut self
                    .sequences
                    .get_mut(id)
                    .expect("prefill sequence")
                    .resident as *mut Resident
            })
            .collect();
        let mut refs: Vec<&mut Resident> = owners
            .iter_mut()
            .map(|owner| unsafe { &mut **owner })
            .collect();
        let sampled = self
            .workspace
            .greedy_ids(&mut refs, &tokens, &positions)?
            .to_vec();
        let elapsed = started.elapsed().as_millis() as u64;
        for (id, token) in ids.iter().zip(sampled) {
            let seq = self.sequences.get_mut(id).expect("prefill sequence");
            seq.prefilled += 1;
            seq.next_position = seq.prefilled;
            seq.prefill_head = Some(token);
            seq.prefill_ms = seq.prefill_ms.saturating_add(elapsed);
        }
        Ok(())
    }

    fn prefill_last_logits_rows(&mut self, ids: &[u64]) -> Result<()> {
        let mut tokens = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        for id in ids {
            let seq = self.sequences.get(id).expect("prefill sequence");
            tokens.push(seq.prompt_ids[seq.prefilled]);
            positions.push(seq.prefilled as u32);
        }
        let started = Instant::now();
        let mut owners: Vec<*mut Resident> = ids
            .iter()
            .map(|id| {
                &mut self
                    .sequences
                    .get_mut(id)
                    .expect("prefill sequence")
                    .resident as *mut Resident
            })
            .collect();
        let mut refs: Vec<&mut Resident> = owners
            .iter_mut()
            .map(|owner| unsafe { &mut **owner })
            .collect();
        let logits = self
            .workspace
            .full_logits(&mut refs, &tokens, &positions)?
            .to_vec();
        let elapsed = started.elapsed().as_millis() as u64;
        let vocab = self.model.cfg.n_vocab;
        for (row, id) in ids.iter().enumerate() {
            let seq = self.sequences.get_mut(id).expect("prefill sequence");
            let token = sample_token(
                &logits[row * vocab..(row + 1) * vocab],
                &seq.sampling,
                &seq.generated,
                &mut seq.rng,
            );
            seq.prefilled += 1;
            seq.next_position = seq.prefilled;
            seq.prefill_head = Some(token);
            seq.prefill_ms = seq.prefill_ms.saturating_add(elapsed);
        }
        Ok(())
    }

    fn prefill_until_ready(&mut self, id: u64) -> Result<()> {
        let seq = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| BitNetError::Inference("fused decode sequence missing".into()))?;
        if seq.pending.is_some() || seq.prefill_head.is_some() {
            return Ok(());
        }
        while seq.prefilled < seq.prompt_ids.len() {
            let count = (seq.prompt_ids.len() - seq.prefilled).min(128);
            let last = seq.prefilled + count == seq.prompt_ids.len();
            let greedy = seq.sampling.device_greedy_eligible();
            let started = Instant::now();
            let (logits, token) = self.workspace.prefill_chunk(
                &mut seq.resident,
                &seq.prompt_ids[seq.prefilled..seq.prefilled + count],
                seq.prefilled,
                last,
                greedy,
            )?;
            seq.prefill_ms = seq
                .prefill_ms
                .saturating_add(started.elapsed().as_millis() as u64);
            seq.prefilled += count;
            if !last {
                continue;
            }
            seq.next_position = seq.prefilled;
            let sampled = if greedy {
                token
            } else {
                sample_token(&logits, &seq.sampling, &[], &mut seq.rng)
            };
            seq.prefill_head = Some(sampled);
            return Ok(());
        }
        Ok(())
    }

    fn emit_prefill_head(&mut self, id: u64, max_tokens: u32) -> Result<(String, PhaseTimings)> {
        let seq = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| BitNetError::Inference("fused decode sequence missing".into()))?;
        let token = seq
            .prefill_head
            .take()
            .ok_or_else(|| BitNetError::Inference("fused decode prefill head missing".into()))?;
        seq.pending = Some(token);
        let text = if self.eos.contains(&token) {
            String::new()
        } else {
            seq.generated.push(token);
            self.tokenizer.decode_ids(&[token], true)?
        };
        seq.prompt_fingerprint.push_str(&text);
        let finish = if self.eos.contains(&token) {
            GenerationFinishReason::Stop
        } else if seq.generated.len() >= max_tokens as usize {
            GenerationFinishReason::Length
        } else {
            GenerationFinishReason::Length
        };
        let phases = PhaseTimings {
            encode_ms: 0,
            prefill_ms: seq.prefill_ms,
            decode_ms: 0,
            prompt_tokens: seq.prompt_ids.len() as u32,
            completion_tokens: u32::from(!self.eos.contains(&token)),
            finish_reason: finish,
        };
        if self.eos.contains(&token) || seq.generated.len() >= max_tokens as usize {
            self.sequences.remove(&id);
        }
        Ok((text, phases))
    }

    pub(crate) fn decode_wave(
        &mut self,
        items: &[(u64, String, u32, SamplingOptions)],
    ) -> Result<Vec<(String, PhaseTimings)>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        for &(id, ref prompt, _, sampling) in items {
            self.ensure_sequence(id, prompt, sampling)?;
        }
        for &(id, _, _, _) in items {
            self.prefill_until_ready(id)?;
        }

        let mut results: BTreeMap<u64, (String, PhaseTimings)> = BTreeMap::new();
        let mut batch_ids: Vec<u64> = Vec::new();
        for &(id, _, max_tokens, _) in items {
            if max_tokens == 0 {
                results.insert(
                    id,
                    (
                        String::new(),
                        PhaseTimings {
                            finish_reason: GenerationFinishReason::Length,
                            ..Default::default()
                        },
                    ),
                );
                continue;
            }
            if self
                .sequences
                .get(&id)
                .is_some_and(|s| s.prefill_head.is_some())
            {
                results.insert(id, self.emit_prefill_head(id, max_tokens)?);
            } else if self.sequences.get(&id).is_some_and(|s| s.pending.is_some()) {
                batch_ids.push(id);
            }
        }

        if !batch_ids.is_empty() {
            batch_ids.sort_unstable();
            let mut wave_tokens = Vec::with_capacity(batch_ids.len());
            let mut wave_positions = Vec::with_capacity(batch_ids.len());
            for id in &batch_ids {
                let seq = self.sequences.get(id).expect("batch sequence");
                wave_tokens.push(seq.pending.expect("pending token"));
                wave_positions.push(seq.next_position as u32);
            }

            let mut sampled: BTreeMap<u64, u32> = BTreeMap::new();
            if batch_ids.len() == 1 {
                let id = batch_ids[0];
                let seq = self.sequences.get_mut(&id).expect("sequence");
                let greedy = seq.sampling.device_greedy_eligible();
                let token = if greedy {
                    let mut owners = [&mut seq.resident];
                    self.workspace
                        .greedy_ids(&mut owners, &wave_tokens, &wave_positions)?[0]
                } else {
                    let mut owners = [&mut seq.resident];
                    let logits =
                        self.workspace
                            .full_logits(&mut owners, &wave_tokens, &wave_positions)?;
                    sample_token(
                        &logits[..self.model.cfg.n_vocab],
                        &seq.sampling,
                        &seq.generated,
                        &mut seq.rng,
                    )
                };
                sampled.insert(id, token);
            } else {
                let greedy = batch_ids.iter().all(|id| {
                    self.sequences
                        .get(id)
                        .expect("sequence")
                        .sampling
                        .device_greedy_eligible()
                });
                let tokens = if greedy {
                    let mut owners: Vec<*mut Resident> = batch_ids
                        .iter()
                        .map(|id| {
                            &mut self.sequences.get_mut(id).expect("sequence").resident
                                as *mut Resident
                        })
                        .collect();
                    let mut refs: Vec<&mut Resident> =
                        owners.iter_mut().map(|p| unsafe { &mut **p }).collect();
                    self.workspace
                        .greedy_ids(&mut refs, &wave_tokens, &wave_positions)?
                        .to_vec()
                } else {
                    let mut owners: Vec<*mut Resident> = batch_ids
                        .iter()
                        .map(|id| {
                            &mut self.sequences.get_mut(id).expect("sequence").resident
                                as *mut Resident
                        })
                        .collect();
                    let mut refs: Vec<&mut Resident> =
                        owners.iter_mut().map(|p| unsafe { &mut **p }).collect();
                    let logits =
                        self.workspace
                            .full_logits(&mut refs, &wave_tokens, &wave_positions)?;
                    let vocab = self.model.cfg.n_vocab;
                    batch_ids
                        .iter()
                        .enumerate()
                        .map(|(row, id)| {
                            let seq = self.sequences.get_mut(id).expect("sequence");
                            sample_token(
                                &logits[row * vocab..(row + 1) * vocab],
                                &seq.sampling,
                                &seq.generated,
                                &mut seq.rng,
                            )
                        })
                        .collect()
                };
                for (id, token) in batch_ids.iter().zip(tokens) {
                    sampled.insert(*id, token);
                }
            }

            for id in batch_ids {
                let max_tokens = items
                    .iter()
                    .find(|(i, _, _, _)| *i == id)
                    .map(|(_, _, m, _)| *m)
                    .unwrap_or(0);
                let prompt = items
                    .iter()
                    .find(|(i, _, _, _)| *i == id)
                    .map(|(_, p, _, _)| p.as_str())
                    .unwrap_or("");
                let token = sampled.remove(&id).ok_or_else(|| {
                    BitNetError::Inference("fused decode batch token missing".into())
                })?;
                let seq = self.sequences.get_mut(&id).expect("sequence");
                seq.next_position += 1;
                seq.pending = Some(token);
                let text = if self.eos.contains(&token) {
                    String::new()
                } else {
                    seq.generated.push(token);
                    self.tokenizer.decode_ids(&[token], true)?
                };
                if !self.eos.contains(&token) {
                    seq.prompt_fingerprint = format!("{prompt}{text}");
                }
                let mut phases = PhaseTimings {
                    encode_ms: 0,
                    prefill_ms: seq.prefill_ms,
                    decode_ms: 1,
                    prompt_tokens: seq.prompt_ids.len() as u32,
                    completion_tokens: u32::from(!self.eos.contains(&token)),
                    finish_reason: if self.eos.contains(&token) {
                        GenerationFinishReason::Stop
                    } else if seq.generated.len() >= max_tokens as usize {
                        GenerationFinishReason::Length
                    } else {
                        GenerationFinishReason::Stop
                    },
                };
                if phases.finish_reason == GenerationFinishReason::Stop
                    && !self.eos.contains(&token)
                    && seq.generated.len() < max_tokens as usize
                {
                    phases.finish_reason = GenerationFinishReason::Length;
                }
                if self.eos.contains(&token) || seq.generated.len() >= max_tokens as usize {
                    self.sequences.remove(&id);
                }
                results.insert(id, (text, phases));
            }
        }

        items
            .iter()
            .map(|(id, _, _, _)| {
                results.remove(id).ok_or_else(|| {
                    BitNetError::Inference(format!(
                        "fused decode wave missing result for request {id}"
                    ))
                })
            })
            .collect()
    }
}

// CUDA resident contexts are owned exclusively by this engine and accessed only
// through the executor mutex on the serving thread.
unsafe impl Send for SchedulerFusedLlama {}
unsafe impl Sync for SchedulerFusedLlama {}
