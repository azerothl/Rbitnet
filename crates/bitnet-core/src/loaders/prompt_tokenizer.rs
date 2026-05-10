//! Prompt tokenizer: Hugging Face `tokenizer.json` or SentencePiece `tokenizer.model`.

use std::path::Path;

use sentencepiece::SentencePieceProcessor;
use tokenizers::Tokenizer;

use crate::error::{BitNetError, Result};

/// Supports the two common layouts shipped next to GGUF files.
pub(crate) enum LoadedPromptTokenizer {
    Hf(Tokenizer),
    Sp(SentencePieceProcessor),
}

impl LoadedPromptTokenizer {
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        let lower = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if lower.ends_with(".model") {
            let sp = SentencePieceProcessor::open(path).map_err(|e| {
                BitNetError::Inference(format!("sentencepiece load (tokenizer.model): {e}"))
            })?;
            return Ok(Self::Sp(sp));
        }
        let tokenizer = Tokenizer::from_file(path).map_err(|e| {
            BitNetError::Inference(format!("Hugging Face tokenizer load (tokenizer.json): {e}"))
        })?;
        Ok(Self::Hf(tokenizer))
    }

    pub(crate) fn encode_ids(&self, prompt: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match self {
            Self::Hf(t) => {
                let enc = t
                    .encode(prompt, add_special_tokens)
                    .map_err(|e| BitNetError::Inference(format!("encode: {e}")))?;
                Ok(enc.get_ids().to_vec())
            }
            Self::Sp(sp) => {
                let pieces = sp
                    .encode(prompt)
                    .map_err(|e| BitNetError::Inference(format!("encode: {e}")))?;
                Ok(pieces.into_iter().map(|p| p.id).collect())
            }
        }
    }

    pub(crate) fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Hf(t) => t
                .decode(ids, skip_special_tokens)
                .map_err(|e| BitNetError::Inference(format!("decode: {e}"))),
            Self::Sp(sp) => sp
                .decode_piece_ids(ids)
                .map_err(|e| BitNetError::Inference(format!("decode: {e}"))),
        }
    }

    /// Best-effort EOS id for Llama/Mistral/Qwen-style chat checkpoints.
    pub(crate) fn eos_token_id(&self) -> Option<u32> {
        const CANDS: &[&str] = &["</s>", "<|endoftext|>", "<|im_end|>", "<|end|>"];
        match self {
            Self::Hf(t) => CANDS.iter().find_map(|s| t.token_to_id(s)),
            Self::Sp(sp) => CANDS
                .iter()
                .find_map(|s| match sp.piece_to_id(s) {
                    Ok(Some(id)) => Some(id),
                    Ok(None) | Err(_) => None,
                })
                .or_else(|| sp.eos_id()),
        }
    }
}
