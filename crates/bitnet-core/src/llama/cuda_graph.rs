//! CUDA graph capture scaffolding for decode loops (rvllm-style D2D token feedback).
//!
//! When `RBITNET_CUDA_GRAPH=1` and the backend is CUDA, the runtime records decode
//! steps into a graph once shapes stabilize. Until full device capture lands, this
//! module tracks eager vs graphed mode and exports metrics for A/B comparison.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeGraphMode {
    #[default]
    Eager,
    Graphed,
}

#[derive(Debug)]
pub struct CudaDecodeGraph {
    pub mode: DecodeGraphMode,
    eager_steps: AtomicU64,
    graphed_steps: AtomicU64,
}

impl CudaDecodeGraph {
    pub fn from_env() -> Self {
        let enabled = matches!(
            std::env::var("RBITNET_CUDA_GRAPH").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        );
        Self {
            mode: if enabled {
                DecodeGraphMode::Graphed
            } else {
                DecodeGraphMode::Eager
            },
            eager_steps: AtomicU64::new(0),
            graphed_steps: AtomicU64::new(0),
        }
    }

    pub fn record_decode_step(&self) {
        match self.mode {
            DecodeGraphMode::Eager => {
                self.eager_steps.fetch_add(1, Ordering::Relaxed);
            }
            DecodeGraphMode::Graphed => {
                self.graphed_steps.fetch_add(1, Ordering::Relaxed);
                crate::perf::record_cuda_graph_replay();
            }
        }
    }

    pub fn eager_steps(&self) -> u64 {
        self.eager_steps.load(Ordering::Relaxed)
    }

    pub fn graphed_steps(&self) -> u64 {
        self.graphed_steps.load(Ordering::Relaxed)
    }
}
