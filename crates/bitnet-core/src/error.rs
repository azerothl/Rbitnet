//! Error types for the inference stack.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BitNetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid GGUF file: {0}")]
    InvalidGguf(String),

    #[error("inference not implemented: {0}")]
    NotImplemented(&'static str),

    #[error("model not loaded; set RBITNET_MODEL to a .gguf path or use stub mode")]
    ModelNotLoaded,

    #[error("inference error: {0}")]
    Inference(String),

    #[error("tokenizer required: set RBITNET_TOKENIZER or place tokenizer.json next to the GGUF")]
    TokenizerMissing,

    #[error("unsupported GGML tensor type {0} for dequantization")]
    UnsupportedGgmlType(u32),
}

pub type Result<T> = std::result::Result<T, BitNetError>;
