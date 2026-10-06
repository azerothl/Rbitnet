//! Multimodal projector (`mmproj`) — sidecar discovery, metadata, CPU vision encode,
//! and prompt fusion for Llama prefill.
//!
//! Phase 1 for [#143](https://github.com/azerothl/Rbitnet/issues/143):
//! - resolve / inspect an mmproj GGUF
//! - load CLIP ViT + LLaVA MLP projector and encode JPEG/PNG/data-URL bytes to
//!   patch embeddings `[n_patches × n_embd_llm]`
//! - expand `<image>` placeholders into patch slots for the Llama CPU prefill path

mod config;
mod encoder;
mod fuse;
mod image_io;
mod info;
mod resolve;

pub use config::{MmprojConfig, VitFfnOp, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD};
pub use encoder::MmprojEncoder;
pub use fuse::{expand_prompt_with_patches, PrefillItem, IMAGE_PLACEHOLDER};
pub use image_io::{decode_and_normalize, decode_bytes_or_data_url, NormalizedImage};
pub use info::MmprojInfo;
pub use resolve::{resolve_mmproj_path, ResolveMmprojOpts};
