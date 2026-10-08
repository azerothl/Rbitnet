//! Small HTTP rendezvous for Sarathi batches.
//!
//! The core scheduler can execute an [`InferenceBatch`] as one CUDA fused decode
//! wave, but independent HTTP handlers otherwise submit singleton batches. This
//! collector holds compatible requests for a short window, then invokes the
//! core batch entry point once.

use bitnet_core::inference::Engine;
use bitnet_core::scheduler::{InferenceOutput, InferenceRequest};
use bitnet_core::stream::StreamEvent;
use bitnet_core::{BitNetError, Result};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const COALESCE_WINDOW: Duration = Duration::from_millis(2);
const MAX_FUSED_ROWS: usize = 8;

struct Pending {
    engine: Arc<Engine>,
    request: InferenceRequest,
    reply: Reply,
}

enum Reply {
    Complete(mpsc::SyncSender<Result<InferenceOutput>>),
    Stream(mpsc::SyncSender<std::result::Result<StreamEvent, String>>),
}

/// Process-wide per-server collection point for compatible HTTP requests.
///
/// Requests are only coalesced with the same loaded `Engine`; registry-backed
/// requests for another model retain their own batch and KV ownership.
#[derive(Default)]
pub(crate) struct ContinuousBatcher {
    pending: Mutex<Vec<Pending>>,
    dispatch_scheduled: AtomicBool,
    #[cfg(test)]
    dispatches: AtomicUsize,
}

impl ContinuousBatcher {
    pub(crate) fn complete(
        self: &Arc<Self>,
        engine: Arc<Engine>,
        request: InferenceRequest,
    ) -> Result<InferenceOutput> {
        let (reply, receive) = mpsc::sync_channel(1);
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| BitNetError::Inference("continuous batch queue poisoned".into()))?;
            pending.push(Pending {
                engine,
                request,
                reply: Reply::Complete(reply),
            });
        }

        if self
            .dispatch_scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let batcher = Arc::clone(self);
            std::thread::Builder::new()
                .name("rbitnet-http-sarathi".into())
                .spawn(move || {
                    std::thread::sleep(COALESCE_WINDOW);
                    batcher.dispatch();
                })
                .map_err(|e| {
                    BitNetError::Inference(format!("start continuous batch worker: {e}"))
                })?;
        }

        receive
            .recv()
            .map_err(|_| BitNetError::Inference("continuous batch worker stopped".into()))?
    }

    /// Execute an SSE request in the same Sarathi batch as compatible JSON
    /// requests. A batch result becomes its content delta and terminal event.
    pub(crate) fn complete_streaming(
        self: &Arc<Self>,
        engine: Arc<Engine>,
        request: InferenceRequest,
        on_event: &mut (dyn FnMut(StreamEvent) -> Result<()> + Send),
    ) -> Result<()> {
        let (reply, receive) = mpsc::sync_channel(4);
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| BitNetError::Inference("continuous batch queue poisoned".into()))?;
            pending.push(Pending {
                engine,
                request,
                reply: Reply::Stream(reply),
            });
        }

        if self
            .dispatch_scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let batcher = Arc::clone(self);
            std::thread::Builder::new()
                .name("rbitnet-http-sarathi".into())
                .spawn(move || {
                    std::thread::sleep(COALESCE_WINDOW);
                    batcher.dispatch();
                })
                .map_err(|e| {
                    BitNetError::Inference(format!("start continuous batch worker: {e}"))
                })?;
        }

        loop {
            match receive
                .recv()
                .map_err(|_| BitNetError::Inference("continuous batch worker stopped".into()))?
            {
                Ok(StreamEvent::Done(output)) => {
                    on_event(StreamEvent::Done(output))?;
                    return Ok(());
                }
                Ok(event) => on_event(event)?,
                Err(message) => return Err(BitNetError::Inference(message)),
            }
        }
    }

    fn dispatch(&self) {
        #[cfg(test)]
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        let pending = {
            let mut queued = match self.pending.lock() {
                Ok(queued) => queued,
                Err(_) => return,
            };
            self.dispatch_scheduled.store(false, Ordering::Release);
            std::mem::take(&mut *queued)
        };

        let mut groups: Vec<(Arc<Engine>, Vec<Pending>)> = Vec::new();
        for entry in pending {
            if let Some((_, members)) = groups
                .iter_mut()
                .find(|(engine, _)| Arc::ptr_eq(engine, &entry.engine))
            {
                members.push(entry);
            } else {
                groups.push((Arc::clone(&entry.engine), vec![entry]));
            }
        }

        for (engine, mut members) in groups {
            while !members.is_empty() {
                let take = members.len().min(fused_row_capacity());
                let members: Vec<_> = members.drain(..take).collect();
                self.complete_group(&engine, members);
            }
        }
    }

    fn complete_group(&self, engine: &Arc<Engine>, members: Vec<Pending>) {
        let requests: Vec<_> = members.iter().map(|entry| entry.request.clone()).collect();
        match engine.complete_batch_detailed(&requests) {
            Ok(outputs) if outputs.len() == members.len() => {
                for (entry, output) in members.into_iter().zip(outputs) {
                    Self::reply_output(entry.reply, output);
                }
            }
            Ok(outputs) => {
                let message = format!(
                    "continuous batch response count mismatch: {} outputs for {} requests",
                    outputs.len(),
                    members.len()
                );
                for entry in members {
                    Self::reply_error(entry.reply, message.clone());
                }
            }
            Err(error) => {
                let message = error.to_string();
                for entry in members {
                    Self::reply_error(entry.reply, message.clone());
                }
            }
        }
    }

    fn reply_output(reply: Reply, output: InferenceOutput) {
        match reply {
            Reply::Complete(tx) => {
                let _ = tx.send(Ok(output));
            }
            Reply::Stream(tx) => {
                if !output.text.is_empty() {
                    let _ = tx.send(Ok(StreamEvent::Delta {
                        text: output.text.clone(),
                    }));
                }
                let _ = tx.send(Ok(StreamEvent::Done(output)));
            }
        }
    }

    fn reply_error(reply: Reply, message: String) {
        match reply {
            Reply::Complete(tx) => {
                let _ = tx.send(Err(BitNetError::Inference(message)));
            }
            Reply::Stream(tx) => {
                let _ = tx.send(Err(message));
            }
        }
    }

    #[cfg(test)]
    fn dispatch_count(&self) -> usize {
        self.dispatches.load(Ordering::Relaxed)
    }
}

fn fused_row_capacity() -> usize {
    std::env::var("RBITNET_CUDA_FUSED_DECODE_SLOTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (1..=MAX_FUSED_ROWS).contains(&value))
        .unwrap_or(MAX_FUSED_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::inference::stub_engine;
    use bitnet_core::sampling::SamplingOptions;
    use std::sync::mpsc;

    #[test]
    fn streaming_member_receives_its_delta_and_terminal_event() {
        let batcher = Arc::new(ContinuousBatcher::default());
        let request = InferenceRequest {
            prompt: "stream this response".into(),
            max_tokens: 4,
            sampling: SamplingOptions::default(),
        };
        let mut events = Vec::new();

        batcher
            .complete_streaming(Arc::new(stub_engine()), request, &mut |event| {
                events.push(event);
                Ok(())
            })
            .unwrap();

        assert!(matches!(events.first(), Some(StreamEvent::Delta { text }) if !text.is_empty()));
        assert!(matches!(events.last(), Some(StreamEvent::Done(_))));
    }

    #[test]
    fn streaming_and_json_members_share_one_dispatch() {
        let batcher = Arc::new(ContinuousBatcher::default());
        let engine = Arc::new(stub_engine());
        let (json_tx, json_rx) = mpsc::sync_channel(1);
        let (stream_tx, stream_rx) = mpsc::sync_channel(2);
        let request = |prompt: &str| InferenceRequest {
            prompt: prompt.into(),
            max_tokens: 4,
            sampling: SamplingOptions::default(),
        };
        batcher.pending.lock().unwrap().extend([
            Pending {
                engine: Arc::clone(&engine),
                request: request("json member"),
                reply: Reply::Complete(json_tx),
            },
            Pending {
                engine,
                request: request("stream member"),
                reply: Reply::Stream(stream_tx),
            },
        ]);

        batcher.dispatch();

        assert!(!json_rx.recv().unwrap().unwrap().text.is_empty());
        assert!(matches!(
            stream_rx.recv().unwrap().unwrap(),
            StreamEvent::Delta { .. }
        ));
        assert!(matches!(
            stream_rx.recv().unwrap().unwrap(),
            StreamEvent::Done(_)
        ));
        assert_eq!(batcher.dispatch_count(), 1);
    }
}
