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

    #[error("tokenizer required: set RBITNET_TOKENIZER to tokenizer.json (or tokenizer.model if supported), or place tokenizer.json / tokenizer.model next to the GGUF")]
    TokenizerMissing,

    #[error("unsupported GGML tensor type {0} for dequantization")]
    UnsupportedGgmlType(u32),
}

pub type Result<T> = std::result::Result<T, BitNetError>;

impl BitNetError {
    /// HTTP status for chat/completions style APIs: client misconfiguration vs server bugs.
    #[must_use]
    pub fn http_status_for_chat_completion(&self) -> u16 {
        match self {
            Self::TokenizerMissing => 400,
            Self::InvalidGguf(_) => 400,
            Self::NotImplemented(_) => 501,
            Self::ModelNotLoaded => 503,
            Self::Inference(msg) => {
                let m = msg.to_ascii_lowercase();
                if m.contains("tokenizer")
                    || m.contains("rbitnet_")
                    || m.contains("not found")
                    || m.contains("missing")
                    || m.contains("mmap")
                    || m.contains("path")
                    || m.contains("out of range")
                    || m.contains("token id")
                    || m.contains("sequence position")
                    || m.contains("max_seq")
                {
                    400
                } else {
                    500
                }
            }
            Self::UnsupportedGgmlType(_) => 400,
            Self::Io(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    400
                } else {
                    500
                }
            }
        }
    }

    /// Short FR + EN hint for JSON error bodies (troubleshooting).
    #[must_use]
    pub fn user_troubleshooting_hint(&self) -> Option<&'static str> {
        match self {
            Self::TokenizerMissing => Some(
                "FR: placez tokenizer.json à côté du .gguf ou définissez RBITNET_TOKENIZER. EN: add tokenizer.json beside the GGUF or set RBITNET_TOKENIZER. See docs/USAGE.md",
            ),
            Self::InvalidGguf(_) => Some(
                "FR: vérifiez le chemin du fichier .gguf. EN: verify the .gguf path. See docs/LIMITATIONS.md",
            ),
            Self::ModelNotLoaded => Some(
                "FR: définissez RBITNET_MODEL, ou RBITNET_STUB=1 pour un serveur sans poids. EN: set RBITNET_MODEL or RBITNET_STUB=1. See docs/USAGE.md",
            ),
            Self::Inference(msg) if msg.to_ascii_lowercase().contains("mmap") => Some(
                "FR: disque plein, fichier verrouillé ou accès refusé. EN: disk full, file lock, or permissions. See docs/USAGE.md",
            ),
            _ => None,
        }
    }
}
