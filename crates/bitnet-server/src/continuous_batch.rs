//! Small HTTP rendezvous for Sarathi batches.
//!
//! The core scheduler can execute an [`InferenceBatch`] as one CUDA fused decode
//! wave, but independent HTTP handlers otherwise submit singleton batches. This
//! collector holds compatible requests for a short window, then invokes the
//! core batch entry point once.

use bitnet_core::inference::Engine;
use bitnet_core::scheduler::{InferenceOutput, InferenceRequest};
use bitnet_core::{BitNetError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const COALESCE_WINDOW: Duration = Duration::from_millis(2);
const MAX_FUSED_ROWS: usize = 8;

struct Pending {
    engine: Arc<Engine>,
    request: InferenceRequest,
    reply: mpsc::SyncSender<Result<InferenceOutput>>,
}

/// Process-wide per-server collection point for non-streaming HTTP requests.
///
/// Requests are only coalesced with the same loaded `Engine`; registry-backed
/// requests for another model retain their own batch and KV ownership.
#[derive(Default)]
pub(crate) struct ContinuousBatcher {
    pending: Mutex<Vec<Pending>>,
    dispatch_scheduled: AtomicBool,
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
                reply,
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

    fn dispatch(&self) {
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
                    let _ = entry.reply.send(Ok(output));
                }
            }
            Ok(outputs) => {
                let message = format!(
                    "continuous batch response count mismatch: {} outputs for {} requests",
                    outputs.len(),
                    members.len()
                );
                for entry in members {
                    let _ = entry
                        .reply
                        .send(Err(BitNetError::Inference(message.clone())));
                }
            }
            Err(error) => {
                let message = error.to_string();
                for entry in members {
                    let _ = entry
                        .reply
                        .send(Err(BitNetError::Inference(message.clone())));
                }
            }
        }
    }
}

fn fused_row_capacity() -> usize {
    std::env::var("RBITNET_CUDA_FUSED_DECODE_SLOTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (1..=MAX_FUSED_ROWS).contains(&value))
        .unwrap_or(MAX_FUSED_ROWS)
}
