//! Private actual-model draft coupling. The target alone samples emitted IDs.
use super::*;

#[derive(Debug)]
pub(super) struct DraftRun {
    pub ids: Vec<u32>,
    pub eos: bool,
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub replayed: usize,
    pub prefill_ms: u64,
    pub decode_ms: u64,
}
impl Qwen35Runtime {
    pub(super) fn spec_check_draft(&self, draft: &Self) -> Result<()> {
        if self.cfg.n_expert != 0
            || draft.cfg.n_expert != 0
            || self.cfg.n_vocab != draft.cfg.n_vocab
            || self.gpu_full.is_none()
            || draft.gpu_full.is_none()
        {
            return Err(BitNetError::Inference("speculative pair requires two dense full-GPU Qwen models with identical vocabulary".into()));
        }
        match (self.tokenizer.as_ref(), draft.tokenizer.as_ref()) {
            (LoadedPromptTokenizer::Hf(a), LoadedPromptTokenizer::Hf(b))
                if a.to_string(false)
                    .map_err(|e| BitNetError::Inference(e.to_string()))?
                    == b.to_string(false)
                        .map_err(|e| BitNetError::Inference(e.to_string()))? => {}
            _ => {
                return Err(BitNetError::Inference(
                    "speculative pair tokenizer configurations differ".into(),
                ))
            }
        }
        if self.tokenizer.eos_token_ids() != draft.tokenizer.eos_token_ids() {
            return Err(BitNetError::Inference(
                "speculative pair EOS IDs differ".into(),
            ));
        }
        let keys = |a: &GgufArchive| {
            a.metadata
                .keys()
                .filter(|k| k.starts_with("tokenizer."))
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
        };
        if !self.archive.metadata.contains_key("tokenizer.ggml.tokens")
            || keys(&self.archive) != keys(&draft.archive)
            || keys(&self.archive).iter().any(|key| {
                format!("{:?}", self.archive.metadata[key])
                    != format!("{:?}", draft.archive.metadata[key])
            })
        {
            return Err(BitNetError::Inference(
                "speculative pair GGUF token IDs/metadata differ".into(),
            ));
        }
        Ok(())
    }
    fn spec_embeddings(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(ids.len() * self.cfg.n_embd);
        for &id in ids {
            if id as usize >= self.cfg.n_vocab {
                return Err(BitNetError::Inference(
                    "speculative token ID outside vocabulary".into(),
                ));
            }
            values.extend(token_embedding_row(
                &self.archive,
                &self.tok_embd,
                id as usize,
                self.cfg.n_embd,
                self.cfg.n_vocab,
            )?);
        }
        Ok(values)
    }
    fn spec_append(
        &self,
        id: u32,
        eos: &[u32],
        ids: &mut Vec<u32>,
        previous: &mut String,
        events: &mut Option<&mut (dyn FnMut(crate::stream::StreamEvent) -> Result<()> + Send)>,
    ) -> Result<bool> {
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        if eos.contains(&id) {
            return Ok(false);
        }
        ids.push(id);
        let text = self.tokenizer.decode_ids(ids, true)?;
        if let Some(callback) = events.as_deref_mut() {
            emit_text_delta(&text, previous, false, callback)?;
        }
        Ok(true)
    }
    // Caller validates the pair/configures native workspaces once at load time.
    // Ordinary prompt prefill arithmetic stays identical to the selected target
    // configuration; verification/replay uses ordered decode arithmetic only.
    pub(super) fn spec_generate_ids_with_draft(
        &mut self,
        draft: &mut Self,
        prompt_ids: &[u32],
        max_tokens: u32,
        depth: usize,
        sampling: SamplingOptions,
        mut events: Option<&mut (dyn FnMut(crate::stream::StreamEvent) -> Result<()> + Send)>,
    ) -> Result<DraftRun> {
        let structured = std::env::var("RBITNET_STRUCTURED_OUTPUT")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if sampling.structured_json
            || matches!(
                structured.as_str(),
                "json" | "tool" | "tool-call" | "tool_call"
            )
        {
            return Err(BitNetError::NotImplemented(
                "speculative Qwen structured-output grammar for subword tokenizers",
            ));
        }
        let coupled_proposals = match std::env::var("RBITNET_QWEN_SPEC_DRAFT_SAMPLING") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "greedy" => false,
            Ok(value) if value == "coupled" => true,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_QWEN_SPEC_DRAFT_SAMPLING must be greedy or coupled".into(),
                ))
            }
        };
        if depth == 0 || depth > 8 || prompt_ids.is_empty() {
            return Err(BitNetError::Inference(
                "speculative depth must be 1..8 with a nonempty prompt".into(),
            ));
        }
        crate::context_capacity::check_request(prompt_ids.len(), max_tokens, self.cfg.max_seq)?;
        crate::context_capacity::check_request(prompt_ids.len(), max_tokens, draft.cfg.max_seq)?;
        if inference_cancelled() {
            return Err(BitNetError::Inference("inference cancelled".into()));
        }
        let mut result = DraftRun {
            ids: Vec::new(),
            eos: false,
            rounds: 0,
            proposed: 0,
            accepted: 0,
            replayed: 0,
            prefill_ms: 0,
            decode_ms: 0,
        };
        if max_tokens == 0 {
            // Context/depth/cancellation checks above still apply; no draft or
            // target GPU work is needed when the requested output is empty.
            return Ok(result);
        }
        let greedy = sampling.device_greedy_eligible();
        let archive = Arc::clone(&self.archive);
        let draft_archive = Arc::clone(&draft.archive);
        let prefill = Instant::now();
        let mut logits = Vec::new();
        let mut next = None;
        let capacity = self
            .gpu_prefill_capacity()
            .min(self.prefill_chunk_tokens.max(1));
        for (chunk, part) in prompt_ids.chunks(capacity).enumerate() {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let position = chunk * capacity;
            let output = position + part.len() == prompt_ids.len();
            (logits, next) = if part.len() > 1 {
                self.forward_block(part, position, &archive, output, greedy)?
            } else {
                self.forward_inner(part[0], position, &archive, output, greedy)?
            };
        }
        let capacity = draft
            .gpu_prefill_capacity()
            .min(draft.prefill_chunk_tokens.max(1));
        for (chunk, part) in prompt_ids.chunks(capacity).enumerate() {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let position = chunk * capacity;
            if part.len() > 1 {
                draft.forward_block(part, position, &draft_archive, false, false)?;
            } else {
                draft.forward_inner(part[0], position, &draft_archive, false, false)?;
            }
        }
        result.prefill_ms = prefill.elapsed().as_millis() as u64;
        let decode = Instant::now();
        let mut rng = seeded_rng(sampling.seed);
        let mut previous = String::new();
        let mut position = prompt_ids.len();
        let eos = self.tokenizer.eos_token_ids();
        let mut first_finished = false;
        if max_tokens > 0 {
            let id = next
                .take()
                .unwrap_or_else(|| sample_token(&logits, &sampling, &result.ids, &mut rng));
            first_finished =
                !self.spec_append(id, &eos, &mut result.ids, &mut previous, &mut events)?;
            result.eos = first_finished;
        }
        while !first_finished && result.ids.len() < max_tokens as usize {
            if inference_cancelled() {
                return Err(BitNetError::Inference("inference cancelled".into()));
            }
            let target_nonce = self.gpu_full.as_mut().unwrap().spec_save()?;
            let draft_nonce = match draft.gpu_full.as_mut().unwrap().spec_save() {
                Ok(n) => n,
                Err(e) => {
                    let _ = self
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_finish(target_nonce, false);
                    return Err(e);
                }
            };
            let mut inputs = vec![*result.ids.last().unwrap()];
            let mut accepted = 0;
            let mut rejected = false;
            let mut finished = false;
            let round = (|| -> Result<()> {
                let count = depth.min(max_tokens as usize - result.ids.len());
                // A cloned RNG can propose the same stochastic path when the
                // draft distribution is close to the target. Only target draws
                // below advance the real sampler; rejected future proposals
                // never consume its entropy or change its output distribution.
                let use_greedy = greedy || !coupled_proposals;
                let mut proposal_rng = (!use_greedy).then(|| rng.clone());
                let mut proposal_history = if use_greedy {
                    Vec::new()
                } else {
                    result.ids.clone()
                };
                let draft_clock = Instant::now();
                for i in 0..count {
                    let (draft_logits, draft_next) = draft.forward_inner(
                        inputs[i],
                        position + i,
                        &draft_archive,
                        true,
                        use_greedy,
                    )?;
                    let id = if use_greedy {
                        draft_next.ok_or_else(|| {
                            BitNetError::Inference("draft device argmax unavailable".into())
                        })?
                    } else {
                        sample_token(
                            &draft_logits,
                            &sampling,
                            &proposal_history,
                            proposal_rng.as_mut().unwrap(),
                        )
                    };
                    if !use_greedy {
                        proposal_history.push(id);
                    }
                    inputs.push(id);
                }
                let draft_ns = draft_clock.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                result.rounds += 1;
                result.proposed += count;
                let verify_clock = Instant::now();
                let embeddings = self.spec_embeddings(&inputs)?;
                let (rows, ids) = self.gpu_full.as_mut().unwrap().spec_verify(
                    &embeddings,
                    position,
                    inputs.len(),
                    greedy,
                )?;
                let verify_ns = verify_clock.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                // Wall time includes the blocking Native verification and readback.
                // Checkpoint, replay, sampling and callbacks remain in decode_ms.
                crate::perf::record_gpu_verification(inputs.len(), 0);
                crate::perf::record_speculative_cost(draft_ns, verify_ns, false);
                for i in 0..=count {
                    if result.ids.len() == max_tokens as usize {
                        finished = true;
                        break;
                    }
                    let id = if greedy {
                        ids[i]
                    } else {
                        sample_token(
                            &rows[i * self.cfg.n_vocab..(i + 1) * self.cfg.n_vocab],
                            &sampling,
                            &result.ids,
                            &mut rng,
                        )
                    };
                    if i < count {
                        if id == inputs[i + 1] {
                            accepted += 1;
                            result.accepted += 1;
                        } else {
                            rejected = true;
                        }
                    }
                    if !self.spec_append(id, &eos, &mut result.ids, &mut previous, &mut events)? {
                        finished = true;
                        result.eos = true;
                        break;
                    }
                    if rejected {
                        break;
                    }
                }
                if result.ids.len() == max_tokens as usize {
                    finished = true;
                }
                Ok(())
            })();
            if let Err(e) = round {
                let _ = self
                    .gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(target_nonce, true);
                let _ = draft
                    .gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(draft_nonce, true);
                return Err(e);
            }
            if finished {
                self.gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(target_nonce, false)?;
                draft
                    .gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(draft_nonce, false)?;
                break;
            }
            if rejected {
                crate::perf::record_speculative_cost(0, 0, true);
                self.gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(target_nonce, true)?;
                draft
                    .gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(draft_nonce, true)?;
                let retained = &inputs[..accepted + 1];
                let target_input = self.spec_embeddings(retained)?;
                let draft_input = draft.spec_embeddings(retained)?;
                self.gpu_full.as_mut().unwrap().spec_replay(
                    &target_input,
                    position,
                    retained.len(),
                )?;
                draft.gpu_full.as_mut().unwrap().spec_replay(
                    &draft_input,
                    position,
                    retained.len(),
                )?;
                position += retained.len();
                result.replayed += retained.len();
            } else {
                // Draft proposal generation has consumed all inputs except its
                // final proposal; target verification consumed that token too.
                let count = inputs.len() - 1;
                draft.forward_inner(
                    inputs[count],
                    position + count,
                    &draft_archive,
                    false,
                    false,
                )?;
                self.gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(target_nonce, false)?;
                draft
                    .gpu_full
                    .as_mut()
                    .unwrap()
                    .spec_finish(draft_nonce, false)?;
                position += inputs.len();
            }
        }
        result.decode_ms = decode.elapsed().as_millis() as u64;
        let text = self.tokenizer.decode_ids(&result.ids, true)?;
        if let Some(callback) = events {
            emit_text_delta(&text, &mut previous, true, callback)?;
        }
        Ok(result)
    }
}
