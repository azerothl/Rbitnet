//! Multimodal projector (`mmproj`) sidecar discovery and metadata.
//!
//! Phase-1 scaffolding for [#143](https://github.com/azerothl/Rbitnet/issues/143):
//! open / inspect an mmproj GGUF. Does **not** encode images or inject patches —
//! HTTP image requests still return 501 until a native encoder lands.

mod info;
mod resolve;

pub use info::MmprojInfo;
pub use resolve::{resolve_mmproj_path, ResolveMmprojOpts};
