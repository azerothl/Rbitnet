//! Metadata helpers for `deepseek2` GGUF (best-effort `llama.*` hyperparameters).

use crate::gguf::GgufArchive;

#[derive(Debug, Clone)]
pub struct DeepSeek2GgufMeta {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
}

impl DeepSeek2GgufMeta {
    pub fn probe(archive: &GgufArchive) -> Self {
        let h = archive.llama_hyper_params();
        Self {
            n_embd: h.embedding_length.unwrap_or(0) as usize,
            n_layer: h.block_count.unwrap_or(0) as usize,
            n_vocab: h.vocab_size.unwrap_or(0) as usize,
        }
    }
}
