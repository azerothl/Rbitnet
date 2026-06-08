//! Hard errors when a roadmap-tagged GGUF cannot run on the Llama runtime.

use crate::error::BitNetError;

/// Explains that MoE-only / non-Llama topology is rejected at load time (no silent stub).
pub(crate) fn roadmap_architecture_not_supported(architecture_key: &str) -> BitNetError {
    BitNetError::Inference(format!(
        "GGUF architecture `{architecture_key}` is only runnable when tensors match the built-in Llama loader \
(token_embd, blk.N.* in Llama layout). Pure MoE / MLA exports are not implemented in Rbitnet yet — \
use a Llama-compatible GGUF export, or set `RBITNET_ARCHITECTURE=llama` when the file is mis-tagged and actually Llama-shaped.",
    ))
}
