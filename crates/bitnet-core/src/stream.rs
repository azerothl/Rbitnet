//! Token streaming events for OpenAI-compatible SSE.

use crate::error::{BitNetError, Result};
use crate::scheduler::{InferenceOutput, InferenceStats};

/// Incremental output from a streaming generation pass.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A decoded text delta (UTF-8 safe suffix since the previous event).
    Delta { text: String },
    /// First token available (after encode + prefill); carries partial stats.
    FirstToken { stats: InferenceStats },
    /// Generation finished with full text and final stats.
    Done(InferenceOutput),
}

pub type StreamCallback = Box<dyn FnMut(StreamEvent) -> Result<()> + Send>;

/// Adapter for `std::sync::mpsc::Sender` used by the HTTP server.
pub fn channel_callback(
    tx: std::sync::mpsc::Sender<StreamEvent>,
) -> Box<dyn FnMut(StreamEvent) -> Result<()> + Send> {
    Box::new(move |ev| {
        tx.send(ev)
            .map_err(|e| BitNetError::Inference(format!("stream channel closed: {e}")))?;
        Ok(())
    })
}
