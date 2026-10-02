//! Hard errors when a roadmap-tagged GGUF cannot run on the Llama runtime.

use crate::error::BitNetError;

/// Explains that MoE-only / non-Llama topology is rejected at load time (no silent stub).
///
/// Spike clarity for [#25](https://github.com/azerothl/Rbitnet/issues/25): refuse early with a
/// support-table pointer; do not pretend DeepSeek MoE / MLA or Mixtral-class graphs run yet.
pub(crate) fn roadmap_architecture_not_supported(architecture_key: &str) -> BitNetError {
    BitNetError::Inference(format!(
        "GGUF architecture `{architecture_key}` is refused: tensors are not Llama-shaped (need token_embd + blk.N.* in the built-in Llama layout). DeepSeek-style MLA attention and non-Mixtral MoE graphs are not implemented in Rbitnet yet (see docs/ARCHITECTURE_GGUF_MATRIX.md and docs/LIMITATIONS.md — issue #25). Supported today: Llama/Mistral-shaped GGUF, dense `qwen3` (CPU), Mixtral MoE (`mixtral`, CPU top-k experts), and experimental `qwen35moe` only with RBITNET_BACKEND=cuda|hybrid. If this file is mis-tagged but actually Llama-shaped, set RBITNET_ARCHITECTURE=llama; otherwise use a Llama-compatible dense export (not a full DeepSeek MoE GGUF)."
    ))
}

#[cfg(test)]
mod tests {
    use super::roadmap_architecture_not_supported;

    #[test]
    fn refuse_message_names_arch_and_moe_mla_gap() {
        let err = roadmap_architecture_not_supported("deepseek2");
        let msg = err.to_string();
        assert!(
            msg.contains("deepseek2"),
            "must echo the architecture key: {msg}"
        );
        assert!(
            msg.contains("MoE") && msg.contains("MLA"),
            "must name MoE/MLA gap: {msg}"
        );
        assert!(
            msg.contains("ARCHITECTURE_GGUF_MATRIX") || msg.contains("LIMITATIONS"),
            "must point at support docs: {msg}"
        );
        assert!(
            msg.contains("qwen3"),
            "must mention dense qwen3 as supported path: {msg}"
        );
        assert!(
            msg.contains("mixtral"),
            "must mention Mixtral MoE as supported path: {msg}"
        );
        assert!(
            !msg.contains("not implemented in Rbitnet yet — use a Llama-compatible"),
            "old short refuse wording should be replaced"
        );
    }

    #[test]
    fn refuse_message_mentions_override_for_mistagged_llama() {
        let msg = roadmap_architecture_not_supported("glm4moe").to_string();
        assert!(msg.contains("RBITNET_ARCHITECTURE=llama"));
        assert!(msg.contains("refused") || msg.contains("not Llama-shaped"));
    }
}
