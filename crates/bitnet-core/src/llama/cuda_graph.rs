//! CUDA graph capture scaffolding for decode loops (rvllm-style D2D token feedback).
//!
//! `RBITNET_CUDA_GRAPH=1` enables legacy scheduling diagnostics only.
//! Actual capture/replay lives in `resident.rs` and the optional native CUDA DLL.
//! This module does not contribute to the native replay performance counter.

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
    capture_rounds: AtomicU64,
    stable_shape_steps: AtomicU64,
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
            capture_rounds: AtomicU64::new(0),
            stable_shape_steps: AtomicU64::new(0),
        }
    }

    /// Called each decode step; after enough stable-shape steps, counts as a capture round (device replay pending).
    pub fn record_decode_step(&self, stable_shape: bool) {
        if stable_shape {
            self.stable_shape_steps.fetch_add(1, Ordering::Relaxed);
        }
        match self.mode {
            DecodeGraphMode::Eager => {
                self.eager_steps.fetch_add(1, Ordering::Relaxed);
            }
            DecodeGraphMode::Graphed => {
                if stable_shape && self.stable_shape_steps.load(Ordering::Relaxed) == 1 {
                    self.capture_rounds.fetch_add(1, Ordering::Relaxed);
                }
                self.graphed_steps.fetch_add(1, Ordering::Relaxed);
                // This legacy scheduling hook does not launch a device graph.
                // Only successful native graph launches increment the shared replay metric.
            }
        }
    }

    pub fn capture_rounds(&self) -> u64 {
        self.capture_rounds.load(Ordering::Relaxed)
    }

    pub fn eager_steps(&self) -> u64 {
        self.eager_steps.load(Ordering::Relaxed)
    }

    pub fn graphed_steps(&self) -> u64 {
        self.graphed_steps.load(Ordering::Relaxed)
    }
}
