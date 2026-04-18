//! **Rbitnet** — pure Rust BitNet inference core (work in progress).
//!
//! Modules:
//! - [`gguf`] — minimal GGUF header validation
//! - [`kernels`] — reference ternary linear ops
//! - [`inference`] — [`Engine`] façade and stub mode for HTTP testing

pub mod error;
pub mod gguf;
pub mod inference;
pub mod kernels;

pub use error::{BitNetError, Result};
pub use inference::Engine;
