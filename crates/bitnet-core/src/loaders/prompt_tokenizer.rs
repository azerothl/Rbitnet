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
                // Llama 3 chat templates already include BOS. The tokenizer's
                // post-processor would prepend a second one when special tokens are enabled.
                let add_special_tokens = add_special_tokens
                    && !(prompt.starts_with("<|begin_of_text|>")
                        && t.token_to_id("<|begin_of_text|>").is_some());
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
        self.eos_token_ids().into_iter().next()
    }

    /// A chat turn and the entire sequence may have different stop tokens (Llama 3).
    pub(crate) fn eos_token_ids(&self) -> Vec<u32> {
        const CANDS: &[&str] = &[
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ];
        let mut ids: Vec<u32> = match self {
            Self::Hf(t) => CANDS.iter().filter_map(|s| t.token_to_id(s)).collect(),
            Self::Sp(sp) => CANDS
                .iter()
                .filter_map(|s| match sp.piece_to_id(s) {
                    Ok(Some(id)) => Some(id),
                    Ok(None) | Err(_) => None,
                })
                .chain(sp.eos_id())
                .collect(),
        };
        ids.dedup();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        models::wordlevel::WordLevel, processors::template::TemplateProcessing, AddedToken,
    };

    fn llama3_tokenizer() -> LoadedPromptTokenizer {
        let vocab = [
            ("[UNK]", 0),
            ("Hello", 1),
            ("<|begin_of_text|>", 2),
            ("<|eot_id|>", 3),
            ("<|end_of_text|>", 4),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|begin_of_text|>", true),
            AddedToken::from("<|eot_id|>", true),
            AddedToken::from("<|end_of_text|>", true),
        ]);
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<|begin_of_text|> $A")
                .unwrap()
                .special_tokens(vec![("<|begin_of_text|>", 2)])
                .build()
                .unwrap(),
        ));
        LoadedPromptTokenizer::Hf(tokenizer)
    }

    #[test]
    fn llama3_formatted_prompt_has_exactly_one_bos() {
        let tokenizer = llama3_tokenizer();
        assert_eq!(
            tokenizer
                .encode_ids("<|begin_of_text|>Hello", true)
                .unwrap(),
            [2, 1]
        );
        assert_eq!(tokenizer.encode_ids("Hello", true).unwrap(), [2, 1]);
        assert_eq!(tokenizer.encode_ids("Hello", false).unwrap(), [1]);
    }

    #[test]
    fn llama3_stops_on_both_turn_and_sequence_end() {
        let tokenizer = llama3_tokenizer();
        assert_eq!(tokenizer.eos_token_ids(), [3, 4]);
        assert_eq!(tokenizer.decode_ids(&[1, 3], true).unwrap(), "Hello");
    }
}
