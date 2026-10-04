//! Private request-local scheduling on the blocking Native multi-owner forward.
//! The caller drives ticks and transports events; no HTTP integration is implied.
use super::transient_batch::NativeBatchWorkspace;
use super::*;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::{sample_token, SamplingOptions};
use crate::stream::{emit_text_delta, StreamCallback, StreamEvent};
use crate::timings::{GenerationFinishReason, PhaseTimings};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

pub(super) struct WaveOutput {
    pub ids: Vec<u32>,
    pub text: String,
    pub phases: PhaseTimings,
    pub queued_ms: f64,
    pub ttft_wall_ms: Option<f64>,
    pub total_wall_ms: f64,
    pub inter_token_us: Vec<u64>,
}
#[derive(Default, Debug)]
pub(super) struct WaveTick {
    pub admitted: usize,
    pub decode_rows: usize,
    pub prefill_tokens: usize,
    pub retired: usize,
}
struct Pending {
    id: u64,
    ids: Vec<u32>,
    maximum: u32,
    sampling: SamplingOptions,
    cancel: Arc<AtomicBool>,
    callback: Option<StreamCallback>,
    arrival: Instant,
    encode_ms: u64,
}
struct Sequence {
    request: Pending,
    resident: Resident,
    prefilled: usize,
    next_position: usize,
    pending: Option<u32>,
    generated: Vec<u32>,
    text: String,
    emitted: String,
    rng: StdRng,
    queue_ms: f64,
    prefill_ms: u64,
    first_ready: Option<Instant>,
    first_output: Option<Instant>,
    previous_output: Option<Instant>,
    inter_token_us: Vec<u64>,
}
pub(super) struct ContinuousLlama {
    model: Arc<LlamaModel>,
    tokenizer: Arc<LoadedPromptTokenizer>,
    workspace: NativeBatchWorkspace,
    page_limit: Option<u32>,
    active: Vec<Option<Sequence>>,
    waiting: VecDeque<Pending>,
    completed: VecDeque<(u64, Result<WaveOutput>)>,
    maximum_queued: usize,
    iteration_budget: usize,
    prefill_cursor: usize,
    eos: Vec<u32>,
    skip_special: bool,
    failed: bool,
}
fn failure(message: &str) -> BitNetError {
    BitNetError::Inference(message.into())
}
impl ContinuousLlama {
    pub(super) fn new(
        model: Arc<LlamaModel>,
        tokenizer: Arc<LoadedPromptTokenizer>,
        maximum: usize,
        maximum_queued: usize,
        iteration_budget: usize,
        page_limit: Option<u32>,
        ordering: u32,
    ) -> Result<Self> {
        if !(1..=8).contains(&maximum)
            || !(1..=64).contains(&maximum_queued)
            || iteration_budget < 128 + maximum
            || iteration_budget > 4096
        {
            return Err(failure("continuous capacity/prefill budget refused"));
        }
        if !matches!(
            std::env::var("RBITNET_CUDA_KV_FORMAT").as_deref(),
            Err(std::env::VarError::NotPresent) | Ok("f32")
        ) {
            return Err(failure("continuous scheduling currently requires F32 KV"));
        }
        let seed = Resident::new_with_pages(&model, page_limit, None)
            .ok_or_else(|| failure("continuous resident seed refused"))?;
        let workspace = NativeBatchWorkspace::new(Arc::clone(&model), &seed, maximum, ordering)?;
        if seed.prefill.is_none() {
            return Err(failure(
                "continuous scheduling requires Native block prefill",
            ));
        }
        // Native retains only the key/configuration and its own scratch; the
        // immutable model remains owned here. No idle KV context is retained.
        drop(seed);
        let skip_special = !matches!(
            std::env::var("RBITNET_LLAMA_DECODE_SKIP_SPECIAL").as_deref(),
            Ok("0") | Ok("false") | Ok("no")
        );
        let eos = tokenizer.eos_token_ids();
        Ok(Self {
            model,
            tokenizer,
            workspace,
            page_limit,
            active: (0..maximum).map(|_| None).collect(),
            waiting: VecDeque::new(),
            completed: VecDeque::new(),
            maximum_queued,
            iteration_budget,
            prefill_cursor: 0,
            eos,
            skip_special,
            failed: false,
        })
    }
    pub(super) fn submit(
        &mut self,
        id: u64,
        prompt: &str,
        maximum: u32,
        sampling: SamplingOptions,
        cancel: Arc<AtomicBool>,
        callback: Option<StreamCallback>,
    ) -> Result<()> {
        if self.failed {
            return Err(failure(
                "continuous workspace failed; a fresh engine is required",
            ));
        }
        sampling.validate_structured_output()?;
        if self.waiting.len() + self.completed.len() >= self.maximum_queued {
            return Err(failure("continuous pending/result queue full"));
        }
        if self.waiting.iter().any(|r| r.id == id)
            || self.active.iter().flatten().any(|s| s.request.id == id)
            || self.completed.iter().any(|(old, _)| *old == id)
        {
            return Err(failure("continuous duplicate request id"));
        }
        let arrival = Instant::now();
        let ids = self
            .tokenizer
            .encode_ids(prompt, crate::llama::llama_encode_add_special_tokens())?;
        crate::context_capacity::check_request(ids.len(), maximum, self.model.cfg.max_seq)?;
        if ids.is_empty() {
            return Err(failure("continuous empty prompt encoding refused"));
        }
        let encode_ms = arrival.elapsed().as_millis() as u64;
        self.waiting.push_back(Pending {
            id,
            ids,
            maximum,
            sampling,
            cancel,
            callback,
            arrival,
            encode_ms,
        });
        Ok(())
    }
    pub(super) fn cancel(&mut self, id: u64) -> bool {
        if let Some(request) = self.waiting.iter().find(|r| r.id == id) {
            request.cancel.store(true, Ordering::Release);
            return true;
        }
        if let Some(sequence) = self.active.iter().flatten().find(|s| s.request.id == id) {
            sequence.request.cancel.store(true, Ordering::Release);
            return true;
        }
        false
    }
    pub(super) fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.active.iter().all(Option::is_none)
    }
    pub(super) fn take_completed(&mut self) -> Vec<(u64, Result<WaveOutput>)> {
        self.completed.drain(..).collect()
    }
    pub(super) fn shutdown(&mut self) {
        self.failed = true;
        for index in 0..self.active.len() {
            self.fail_sequence(index, failure("continuous engine stopped"));
        }
        while let Some(request) = self.waiting.pop_front() {
            self.completed
                .push_back((request.id, Err(failure("continuous engine stopped"))));
        }
    }
    pub(super) fn stats(&self) -> Result<[u64; 3]> {
        self.workspace.stats()
    }
    fn phases(s: &Sequence, reason: GenerationFinishReason) -> PhaseTimings {
        PhaseTimings {
            encode_ms: s.request.encode_ms,
            prefill_ms: s.prefill_ms,
            decode_ms: s.first_ready.map_or(0, |t| t.elapsed().as_millis() as u64),
            prompt_tokens: s.request.ids.len() as u32,
            completion_tokens: s.generated.len() as u32,
            finish_reason: reason,
        }
    }
    fn finish(&mut self, index: usize, reason: GenerationFinishReason) -> Result<()> {
        let mut s = self.active[index]
            .take()
            .ok_or_else(|| failure("continuous owner already retired"))?;
        let phases = Self::phases(&s, reason);
        let callback_result = if let Some(callback) = s.request.callback.as_deref_mut() {
            emit_text_delta(&s.text, &mut s.emitted, true, callback).and_then(|_| {
                callback(StreamEvent::Done(crate::scheduler::InferenceOutput {
                    text: s.text.clone(),
                    stats: crate::scheduler::InferenceStats::from_phases(phases, false),
                }))
            })
        } else {
            Ok(())
        };
        let result = callback_result.map(|_| WaveOutput {
            ids: s.generated,
            text: s.text,
            phases,
            queued_ms: s.queue_ms,
            ttft_wall_ms: s
                .first_output
                .map(|t| t.duration_since(s.request.arrival).as_secs_f64() * 1000.),
            total_wall_ms: s.request.arrival.elapsed().as_secs_f64() * 1000.,
            inter_token_us: s.inter_token_us,
        });
        self.completed.push_back((s.request.id, result));
        Ok(())
    }
    fn fail_sequence(&mut self, index: usize, error: BitNetError) {
        if let Some(s) = self.active[index].take() {
            self.completed.push_back((s.request.id, Err(error)));
        }
    }
    fn publish(&mut self, index: usize, token: u32) -> Result<()> {
        let s = self.active[index]
            .as_mut()
            .ok_or_else(|| failure("continuous sequence missing after wave"))?;
        if s.request.cancel.load(Ordering::Acquire) {
            self.fail_sequence(index, failure("request cancelled"));
            return Ok(());
        }
        if self.eos.contains(&token) {
            return self.finish(index, GenerationFinishReason::Stop);
        }
        s.generated.push(token);
        s.pending = Some(token);
        let now = Instant::now();
        if let Some(previous) = s.previous_output {
            s.inter_token_us
                .push(now.duration_since(previous).as_micros() as u64);
        }
        s.first_output.get_or_insert(now);
        s.previous_output = Some(now);
        s.text = self.tokenizer.decode_ids(&s.generated, self.skip_special)?;
        if let Some(callback) = s.request.callback.as_deref_mut() {
            if let Err(error) = emit_text_delta(&s.text, &mut s.emitted, false, callback) {
                self.fail_sequence(index, error);
                return Ok(());
            }
        }
        if s.request.cancel.load(Ordering::Acquire) {
            self.fail_sequence(index, failure("request cancelled"));
            return Ok(());
        }
        if s.generated.len() >= s.request.maximum as usize {
            return self.finish(index, GenerationFinishReason::Length);
        }
        Ok(())
    }
    fn admit(&mut self, tick: &mut WaveTick) {
        // Remove cancelled requests even when all GPU slots are occupied.
        let count = self.waiting.len();
        for _ in 0..count {
            let r = self
                .waiting
                .pop_front()
                .expect("bounded pending queue length");
            if r.cancel.load(Ordering::Acquire) {
                self.completed
                    .push_back((r.id, Err(failure("request cancelled before admission"))));
                tick.retired += 1;
            } else {
                self.waiting.push_back(r);
            }
        }
        for index in 0..self.active.len() {
            if self.active[index].is_some() {
                continue;
            }
            let Some(mut request) = self.waiting.pop_front() else {
                break;
            };
            if request.maximum == 0 {
                let phases = PhaseTimings {
                    encode_ms: request.encode_ms,
                    prompt_tokens: request.ids.len() as u32,
                    finish_reason: GenerationFinishReason::Length,
                    ..Default::default()
                };
                let result = if let Some(callback) = request.callback.as_deref_mut() {
                    callback(StreamEvent::Done(crate::scheduler::InferenceOutput {
                        text: String::new(),
                        stats: crate::scheduler::InferenceStats::from_phases(phases, false),
                    }))
                } else {
                    Ok(())
                };
                self.completed.push_back((
                    request.id,
                    result.map(|_| WaveOutput {
                        ids: Vec::new(),
                        text: String::new(),
                        phases,
                        queued_ms: request.arrival.elapsed().as_secs_f64() * 1000.,
                        ttft_wall_ms: None,
                        total_wall_ms: request.arrival.elapsed().as_secs_f64() * 1000.,
                        inter_token_us: Vec::new(),
                    }),
                ));
                tick.retired += 1;
                continue;
            }
            let peer = if self.page_limit.is_some() {
                self.active.iter().flatten().next().map(|s| &s.resident)
            } else {
                None
            };
            let Some(resident) = Resident::new_with_pages(&self.model, self.page_limit, peer)
            else {
                self.completed.push_back((
                    request.id,
                    Err(failure("request Native KV/context allocation refused")),
                ));
                tick.retired += 1;
                continue;
            };
            let rng = match request.sampling.seed {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            };
            let queue_ms = (request.arrival.elapsed().as_secs_f64() * 1000.
                - request.encode_ms as f64)
                .max(0.);
            self.active[index] = Some(Sequence {
                request,
                resident,
                prefilled: 0,
                next_position: 0,
                pending: None,
                generated: Vec::new(),
                text: String::new(),
                emitted: String::new(),
                rng,
                queue_ms,
                prefill_ms: 0,
                first_ready: None,
                first_output: None,
                previous_output: None,
                inter_token_us: Vec::new(),
            });
            tick.admitted += 1;
        }
    }
    pub(super) fn tick(&mut self) -> Result<WaveTick> {
        if self.failed {
            return Err(failure(
                "continuous workspace failed; a fresh engine is required",
            ));
        }
        match self.tick_inner() {
            Ok(tick) => Ok(tick),
            Err(error) => {
                self.failed = true;
                let message = format!("continuous engine failed: {error}");
                for index in 0..self.active.len() {
                    self.fail_sequence(index, failure(&message));
                }
                while let Some(request) = self.waiting.pop_front() {
                    self.completed
                        .push_back((request.id, Err(failure(&message))));
                }
                Err(error)
            }
        }
    }
    fn tick_inner(&mut self) -> Result<WaveTick> {
        let mut tick = WaveTick::default();
        let previous_completed = self.completed.len();
        for index in 0..self.active.len() {
            if self.active[index]
                .as_ref()
                .is_some_and(|s| s.request.cancel.load(Ordering::Acquire))
            {
                self.fail_sequence(index, failure("request cancelled"));
                tick.retired += 1;
            }
        }
        self.admit(&mut tick);
        let indices: Vec<_> = self
            .active
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().filter(|s| s.pending.is_some()).map(|_| i))
            .collect();
        let greedy = indices.iter().all(|&i| {
            self.active[i]
                .as_ref()
                .unwrap()
                .request
                .sampling
                .device_greedy_eligible()
        });
        if !indices.is_empty() {
            let ids: Vec<_> = indices
                .iter()
                .map(|&i| self.active[i].as_ref().unwrap().pending.unwrap())
                .collect();
            let positions: Vec<_> = indices
                .iter()
                .map(|&i| self.active[i].as_ref().unwrap().next_position as u32)
                .collect();
            let mut owners: Vec<_> = self
                .active
                .iter_mut()
                .enumerate()
                .filter_map(|(i, s)| {
                    indices
                        .contains(&i)
                        .then(|| &mut s.as_mut().unwrap().resident)
                })
                .collect();
            let tokens = if greedy {
                self.workspace
                    .greedy_ids(&mut owners, &ids, &positions)?
                    .to_vec()
            } else {
                let logits = self.workspace.full_logits(&mut owners, &ids, &positions)?;
                drop(owners);
                indices
                    .iter()
                    .enumerate()
                    .map(|(row, &index)| {
                        let s = self.active[index].as_mut().unwrap();
                        sample_token(
                            &logits
                                [row * self.model.cfg.n_vocab..(row + 1) * self.model.cfg.n_vocab],
                            &s.request.sampling,
                            &s.generated,
                            &mut s.rng,
                        )
                    })
                    .collect()
            };
            tick.decode_rows = indices.len();
            for (&index, token) in indices.iter().zip(tokens) {
                self.active[index].as_mut().unwrap().next_position += 1;
                self.publish(index, token)?;
            }
        }
        // Decode was executed first. Prefill consumes real GPU work, in the
        // same 128-token partitions as the independent resident baseline.
        let mut remaining = self.iteration_budget - tick.decode_rows;
        for offset in 0..self.active.len() {
            let index = (self.prefill_cursor + offset) % self.active.len();
            let Some(s) = self.active[index].as_mut() else {
                continue;
            };
            if s.pending.is_some() {
                continue;
            }
            let count = (s.request.ids.len() - s.prefilled).min(128);
            if count > remaining {
                continue;
            }
            if s.request.cancel.load(Ordering::Acquire) {
                self.fail_sequence(index, failure("request cancelled"));
                continue;
            }
            let last = s.prefilled + count == s.request.ids.len();
            let greedy = s.request.sampling.device_greedy_eligible();
            let started = Instant::now();
            let (logits, token) = self.workspace.prefill_chunk(
                &mut s.resident,
                &s.request.ids[s.prefilled..s.prefilled + count],
                s.prefilled,
                last,
                greedy,
            )?;
            s.prefill_ms = s
                .prefill_ms
                .saturating_add(started.elapsed().as_millis() as u64);
            s.prefilled += count;
            remaining -= count;
            tick.prefill_tokens += count;
            if !last {
                continue;
            }
            s.next_position = s.prefilled;
            s.first_ready = Some(Instant::now());
            if let Some(callback) = s.request.callback.as_deref_mut() {
                let stats = crate::scheduler::InferenceStats::from_phases(
                    PhaseTimings {
                        encode_ms: s.request.encode_ms,
                        prefill_ms: s.prefill_ms,
                        prompt_tokens: s.request.ids.len() as u32,
                        ..Default::default()
                    },
                    false,
                );
                if let Err(error) = callback(StreamEvent::FirstToken { stats }) {
                    self.fail_sequence(index, error);
                    continue;
                }
            }
            let token = if greedy {
                token
            } else {
                sample_token(&logits, &s.request.sampling, &[], &mut s.rng)
            };
            self.publish(index, token)?;
        }
        self.prefill_cursor = (self.prefill_cursor + 1) % self.active.len();
        tick.retired = self.completed.len() - previous_completed;
        if tick.decode_rows + tick.prefill_tokens > 0 {
            crate::perf::record_scheduler_stall_free_iter(self.iteration_budget);
        }
        Ok(tick)
    }
}
