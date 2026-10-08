//! One GPU worker, bounded command/event transport, request-local cancellation.
use super::continuous::{ContinuousLlama, WaveOutput};
use super::LlamaModel;
use crate::error::{BitNetError, Result};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use crate::scheduler::{InferenceOutput, InferenceStats};
use crate::stream::StreamEvent;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Arc,
};
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchOptions {
    pub slots: usize,
    pub queued: usize,
    pub token_budget: usize,
    pub pages: Option<u32>,
    pub ordering: u32,
    /// Keep newly admitted prompt work within the current decode-first budget.
    /// Disabled by default while this serving policy is validated.
    pub adaptive_admission: bool,
}

#[cfg(test)]
mod integration_options_tests {
    use super::*;

    #[test]
    fn combined_options_refuse_encoded_kv_and_context_tiers_before_native_start() {
        let keys = [
            "RBITNET_CUDA_CONTINUOUS",
            "RBITNET_CUDA_KV_FORMAT",
            "RBITNET_CONTEXT_TIERS",
            "RBITNET_CUDA_PREFILL",
            "RBITNET_CUDA_PREFILL_TOKENS",
            "RBITNET_CUDA_KV_PAGE_LIMIT",
            "RBITNET_CUDA_CONTINUOUS_SLOTS",
            "RBITNET_CUDA_CONTINUOUS_QUEUE",
            "RBITNET_CUDA_CONTINUOUS_TOKEN_BUDGET",
            "RBITNET_CUDA_CONTINUOUS_ORDERING",
            "RBITNET_CUDA_CONTINUOUS_ADMISSION",
            "RBITNET_CUDA_LIVE_SSE_MUX",
            "RBITNET_PREFIX_KV",
            "RBITNET_SPECULATIVE",
            "RBITNET_SPECULATIVE_PLD",
            "RBITNET_CUDA_SPLIT_KV",
            "RBITNET_CUDA_PREFILL_TF32X3",
            "RBITNET_CONTINUOUS_BATCHING",
            "RBITNET_FUSED_MULTI_SEQ",
            "RBITNET_MTP_K",
        ];
        struct Environment(Vec<(&'static str, Option<std::ffi::OsString>)>);
        impl Drop for Environment {
            fn drop(&mut self) {
                for (key, value) in &self.0 {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
        let _restore = Environment(
            keys.iter()
                .map(|&key| (key, std::env::var_os(key)))
                .collect(),
        );
        for key in keys {
            std::env::remove_var(key);
        }
        std::env::set_var("RBITNET_CUDA_CONTINUOUS", "1");
        std::env::set_var("RBITNET_CUDA_PREFILL", "1");
        let backend = crate::backend::BackendKind::Cuda;
        assert!(BatchOptions::configured(backend).unwrap().is_some());
        for format in ["f16", "q8"] {
            std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
            let error = BatchOptions::configured(backend).unwrap_err().to_string();
            assert!(error.contains("requires F32 native KV"), "{error}");
        }
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
        for tiers in ["1", "true"] {
            std::env::set_var("RBITNET_CONTEXT_TIERS", tiers);
            let error = BatchOptions::configured(backend).unwrap_err().to_string();
            assert!(error.contains("context tiers are not validated"), "{error}");
        }
        std::env::set_var("RBITNET_CONTEXT_TIERS", "0");
        assert!(BatchOptions::configured(backend).unwrap().is_some());

        std::env::set_var("RBITNET_CUDA_CONTINUOUS_ADMISSION", "adaptive");
        assert!(
            BatchOptions::configured(backend)
                .unwrap()
                .expect("continuous options")
                .adaptive_admission
        );
        std::env::set_var("RBITNET_CUDA_CONTINUOUS_ADMISSION", "invalid");
        let error = BatchOptions::configured(backend).unwrap_err().to_string();
        assert!(
            error.contains("ADMISSION must be fifo or adaptive"),
            "{error}"
        );
        std::env::set_var("RBITNET_CUDA_CONTINUOUS_ADMISSION", "fifo");

        std::env::set_var("RBITNET_CUDA_LIVE_SSE_MUX", "1");
        std::env::set_var("RBITNET_CONTINUOUS_BATCHING", "1");
        std::env::set_var("RBITNET_FUSED_MULTI_SEQ", "1");
        assert!(BatchOptions::configured(backend).unwrap().is_some());

        std::env::set_var("RBITNET_CUDA_LIVE_SSE_MUX", "0");
        let error = BatchOptions::configured(backend).unwrap_err().to_string();
        assert!(error.contains("RBITNET_CONTINUOUS_BATCHING"), "{error}");
    }
}
impl BatchOptions {
    pub(crate) fn configured(backend: crate::backend::BackendKind) -> Result<Option<Self>> {
        let enabled = match std::env::var("RBITNET_CUDA_CONTINUOUS") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(s) if s == "0" => false,
            Ok(s) if s == "1" => true,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_CUDA_CONTINUOUS must be 0 or 1".into(),
                ))
            }
        };
        if !enabled {
            return Ok(None);
        }
        let live_sse_mux = match std::env::var("RBITNET_CUDA_LIVE_SSE_MUX") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_CUDA_LIVE_SSE_MUX must be 0 or 1".into(),
                ))
            }
        };
        if backend != crate::backend::BackendKind::Cuda {
            return Err(BitNetError::Inference(
                "RBITNET_CUDA_CONTINUOUS requires the CUDA Llama backend".into(),
            ));
        }
        if super::configured_kv_format()? != 0 {
            return Err(BitNetError::Inference(
                "RBITNET_CUDA_CONTINUOUS requires F32 native KV storage".into(),
            ));
        }
        if crate::context_native::enabled() {
            return Err(BitNetError::Inference(
                "context tiers are not validated with RBITNET_CUDA_CONTINUOUS".into(),
            ));
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
        let slots = parse("RBITNET_CUDA_CONTINUOUS_SLOTS", 4, 1, 8)?;
        let queued = parse("RBITNET_CUDA_CONTINUOUS_QUEUE", 32, 1, 64)?;
        let token_budget = parse(
            "RBITNET_CUDA_CONTINUOUS_TOKEN_BUDGET",
            256,
            128 + slots,
            4096,
        )?;
        let ordering = parse("RBITNET_CUDA_CONTINUOUS_ORDERING", 0, 0, 1)? as u32;
        let adaptive_admission = match std::env::var("RBITNET_CUDA_CONTINUOUS_ADMISSION") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "fifo" => false,
            Ok(value) if value == "adaptive" => true,
            _ => {
                return Err(BitNetError::Inference(
                    "RBITNET_CUDA_CONTINUOUS_ADMISSION must be fifo or adaptive".into(),
                ))
            }
        };
        for key in [
            "RBITNET_PREFIX_KV",
            "RBITNET_SPECULATIVE",
            "RBITNET_SPECULATIVE_PLD",
            "RBITNET_CUDA_SPLIT_KV",
            "RBITNET_CUDA_PREFILL_TF32X3",
        ] {
            if matches!(
                std::env::var(key).as_deref(),
                Ok("1") | Ok("true") | Ok("yes")
            ) {
                return Err(BitNetError::Inference(format!(
                    "{key} is not validated with RBITNET_CUDA_CONTINUOUS"
                )));
            }
        }
        if !live_sse_mux {
            for key in ["RBITNET_CONTINUOUS_BATCHING", "RBITNET_FUSED_MULTI_SEQ"] {
                if matches!(
                    std::env::var(key).as_deref(),
                    Ok("1") | Ok("true") | Ok("yes")
                ) {
                    return Err(BitNetError::Inference(format!(
                        "{key} is not validated with RBITNET_CUDA_CONTINUOUS"
                    )));
                }
            }
        }
        if std::env::var("RBITNET_MTP_K")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .is_some_and(|n| n > 1)
        {
            return Err(BitNetError::Inference(
                "RBITNET_MTP_K bursts are not validated with RBITNET_CUDA_CONTINUOUS".into(),
            ));
        }
        if std::env::var("RBITNET_CUDA_PREFILL").as_deref() != Ok("1") {
            return Err(BitNetError::Inference(
                "RBITNET_CUDA_CONTINUOUS requires RBITNET_CUDA_PREFILL=1".into(),
            ));
        }
        if !matches!(
            std::env::var("RBITNET_CUDA_PREFILL_TOKENS").as_deref(),
            Err(std::env::VarError::NotPresent) | Ok("128")
        ) {
            return Err(BitNetError::Inference(
                "RBITNET_CUDA_CONTINUOUS currently preserves 128-token prefill partitions".into(),
            ));
        }
        Ok(Some(Self {
            slots,
            queued,
            token_budget,
            pages: super::configured_page_limit()?,
            ordering,
            adaptive_admission,
        }))
    }
}
enum Reply {
    Event(StreamEvent),
    Complete(Result<WaveOutput>),
}
enum Command {
    Submit {
        id: u64,
        prompt: String,
        maximum: u32,
        sampling: SamplingOptions,
        cancel: Arc<AtomicBool>,
        reply: mpsc::SyncSender<Reply>,
    },
    Query(mpsc::SyncSender<Result<[u64; 3]>>),
    Shutdown,
}
pub(crate) struct BatchController {
    sender: mpsc::SyncSender<Command>,
    stop: Arc<AtomicBool>,
    next: AtomicU64,
    worker: Option<JoinHandle<()>>,
    capacity: usize,
    tokenizer: Arc<LoadedPromptTokenizer>,
}
impl Drop for BatchController {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.sender.try_send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn error(message: &str) -> BitNetError {
    BitNetError::Inference(message.into())
}
impl BatchController {
    pub(crate) fn start(
        model: Arc<LlamaModel>,
        tokenizer: Arc<LoadedPromptTokenizer>,
        options: BatchOptions,
    ) -> Result<Self> {
        let capacity = model.cfg.max_seq;
        let worker_tokenizer = Arc::clone(&tokenizer);
        let (sender, receiver) = mpsc::sync_channel::<Command>(options.queued);
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<()>>(1);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = std::thread::Builder::new()
            .name("rbitnet-cuda-batch".into())
            .spawn(move || {
                let mut driver = match ContinuousLlama::new(
                    model,
                    worker_tokenizer,
                    options.slots,
                    options.queued,
                    options.token_budget,
                    options.pages,
                    options.ordering,
                    options.adaptive_admission,
                ) {
                    Ok(driver) => driver,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                let mut replies: BTreeMap<u64, mpsc::SyncSender<Reply>> = BTreeMap::new();
                loop {
                    let mut shutdown = worker_stop.load(Ordering::Acquire);
                    let first = if driver.is_idle() {
                        receiver.recv().ok()
                    } else {
                        receiver.try_recv().ok()
                    };
                    let mut commands = Vec::with_capacity(options.queued);
                    if let Some(command) = first {
                        commands.push(command);
                    }
                    while commands.len() < options.queued {
                        match receiver.try_recv() {
                            Ok(command) => commands.push(command),
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                shutdown = true;
                                break;
                            }
                        }
                    }
                    for command in commands {
                        match command {
                            Command::Shutdown => shutdown = true,
                            Command::Query(tx) => {
                                let _ = tx.try_send(driver.stats());
                            }
                            Command::Submit {
                                id,
                                prompt,
                                maximum,
                                sampling,
                                cancel,
                                reply,
                            } => {
                                if shutdown {
                                    let _ = reply.try_send(Reply::Complete(Err(error(
                                        "continuous engine stopped",
                                    ))));
                                    continue;
                                }
                                let event_tx = reply.clone();
                                let event_cancel = Arc::clone(&cancel);
                                let callback = Box::new(move |event| {
                                    event_tx.try_send(Reply::Event(event)).map_err(|_| {
                                        event_cancel.store(true, Ordering::Release);
                                        error("owned request event channel closed or full")
                                    })
                                });
                                match driver.submit(
                                    id,
                                    &prompt,
                                    maximum,
                                    sampling,
                                    cancel,
                                    Some(callback),
                                ) {
                                    Ok(()) => {
                                        replies.insert(id, reply);
                                    }
                                    Err(e) => {
                                        let _ = reply.try_send(Reply::Complete(Err(e)));
                                    }
                                }
                            }
                        }
                    }
                    if shutdown {
                        driver.shutdown();
                    } else if !driver.is_idle() {
                        // Native failures seal the engine and retire every owner;
                        // request callbacks/disconnects retire only their own ID.
                        if driver.tick().is_err() {
                            shutdown = true;
                        }
                    }
                    for (id, result) in driver.take_completed() {
                        if let Some(tx) = replies.remove(&id) {
                            let _ = tx.try_send(Reply::Complete(result));
                        }
                    }
                    if shutdown {
                        worker_stop.store(true, Ordering::Release);
                        break;
                    }
                }
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender,
                stop,
                next: AtomicU64::new(0),
                worker: Some(worker),
                capacity,
                tokenizer,
            }),
            Ok(Err(e)) => {
                let _ = worker.join();
                Err(e)
            }
            Err(_) => {
                let _ = worker.join();
                Err(error("continuous worker failed before initialization"))
            }
        }
    }
    pub(crate) fn native_stats(&self) -> Result<[u64; 3]> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.sender
            .try_send(Command::Query(tx))
            .map_err(|_| error("continuous query queue closed or full"))?;
        rx.recv()
            .map_err(|_| error("continuous query worker closed"))?
    }
    fn transact(
        &self,
        prompt: &str,
        maximum: u32,
        sampling: SamplingOptions,
        mut callback: Option<&mut (dyn FnMut(StreamEvent) -> Result<()> + Send)>,
    ) -> Result<WaveOutput> {
        if self.stop.load(Ordering::Acquire) {
            return Err(error("continuous engine stopped"));
        }
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
            return Err(error(
                "continuous Llama structured-output grammar is not validated",
            ));
        }
        let started = Instant::now();
        let ids = self
            .tokenizer
            .encode_ids(prompt, crate::llama::llama_encode_add_special_tokens())?;
        crate::context_capacity::check_request(ids.len(), maximum, self.capacity)?;
        let id = self
            .next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| error("continuous request IDs exhausted"))?;
        let cancel = Arc::new(AtomicBool::new(false));
        struct Cancel(Arc<AtomicBool>);
        impl Drop for Cancel {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _guard = Cancel(Arc::clone(&cancel));
        let (tx, rx) = mpsc::sync_channel(16);
        self.sender
            .try_send(Command::Submit {
                id,
                prompt: prompt.to_owned(),
                maximum,
                sampling,
                cancel,
                reply: tx,
            })
            .map_err(|_| error("continuous request queue closed or full"))?;
        let mut first_ready_ms = None;
        let mut terminal_seen = false;
        loop {
            match rx
                .recv()
                .map_err(|_| error("owned continuous request channel closed"))?
            {
                Reply::Event(mut event) => {
                    match &mut event {
                        StreamEvent::FirstToken { stats } => {
                            let ms = started.elapsed().as_millis() as u64;
                            first_ready_ms = Some(ms);
                            stats.ttft_ms = ms;
                        }
                        StreamEvent::Done(output) => {
                            output.stats.ttft_ms = first_ready_ms
                                .unwrap_or_else(|| started.elapsed().as_millis() as u64);
                            output.stats.total_wall_ms = started.elapsed().as_millis() as u64;
                            terminal_seen = true;
                            continue;
                        }
                        StreamEvent::Delta { .. } => {}
                    }
                    if let Some(cb) = callback.as_deref_mut() {
                        cb(event)?;
                    }
                }
                Reply::Complete(result) => {
                    let mut output = result?;
                    if !terminal_seen {
                        return Err(error("continuous completion lacks its terminal event"));
                    }
                    output.total_wall_ms = started.elapsed().as_secs_f64() * 1000.;
                    if let Some(ms) = first_ready_ms {
                        output.ttft_wall_ms = Some(ms as f64);
                    }
                    if let Some(cb) = callback.as_deref_mut() {
                        cb(StreamEvent::Done(Self::inference_output(&output)))?;
                    }
                    return Ok(output);
                }
            }
        }
    }
    pub(crate) fn generate_output(
        &self,
        prompt: &str,
        maximum: u32,
        sampling: SamplingOptions,
    ) -> Result<InferenceOutput> {
        let output = self.transact(prompt, maximum, sampling, None)?;
        Ok(Self::inference_output(&output))
    }
    fn inference_output(output: &WaveOutput) -> InferenceOutput {
        let mut stats = InferenceStats::from_phases(output.phases, false);
        stats.ttft_ms = output.ttft_wall_ms.unwrap_or(output.total_wall_ms).round() as u64;
        stats.total_wall_ms = output.total_wall_ms.round() as u64;
        if !output.inter_token_us.is_empty() {
            stats.itl_us =
                output.inter_token_us.iter().sum::<u64>() / output.inter_token_us.len() as u64;
            stats.tpot_us = stats.itl_us;
        }
        InferenceOutput {
            text: output.text.clone(),
            stats,
        }
    }
    pub(crate) fn generate_with_timings(
        &self,
        prompt: &str,
        maximum: u32,
        sampling: SamplingOptions,
    ) -> Result<(String, crate::timings::PhaseTimings)> {
        let output = self.transact(prompt, maximum, sampling, None)?;
        Ok((output.text, output.phases))
    }
    pub(crate) fn generate_streaming(
        &self,
        prompt: &str,
        maximum: u32,
        sampling: SamplingOptions,
        callback: &mut (dyn FnMut(StreamEvent) -> Result<()> + Send),
    ) -> Result<()> {
        self.transact(prompt, maximum, sampling, Some(callback))
            .map(|_| ())
    }
}
