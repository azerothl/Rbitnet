//! Prompt tokenizer: Hugging Face `tokenizer.json` or SentencePiece `tokenizer.model`.

use std::path::Path;

mod sentencepiece_codec;
use sentencepiece_codec::SentencePieceCodec;
use tokenizers::Tokenizer;

use crate::error::{BitNetError, Result};

/// Supports the two common layouts shipped next to GGUF files.
pub(crate) enum LoadedPromptTokenizer {
    Hf(Tokenizer),
    Sp(SentencePieceCodec),
}

impl LoadedPromptTokenizer {
    #[cfg(test)]
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        Self::load(path, None)
    }
    pub(crate) fn from_path_for_gguf(
        path: &Path,
        archive: &crate::gguf::GgufArchive,
    ) -> Result<Self> {
        Self::load(path, Some(archive))
    }
    fn load(path: &Path, archive: Option<&crate::gguf::GgufArchive>) -> Result<Self> {
        let lower = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if lower.ends_with(".model") {
            let sp = SentencePieceCodec::load(path, archive)?;
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
                let mut ids = enc.get_ids().to_vec();
                if add_special_tokens && prompt.starts_with("<s>") {
                    if let Some(bos) = t.token_to_id("<s>") {
                        if ids.first() == Some(&bos) {
                            let plain = t.encode(prompt, false).map_err(|e| {
                                BitNetError::Inference(format!("encode explicit BOS: {e}"))
                            })?;
                            if ids.len() > plain.get_ids().len()
                                && ids[1..].starts_with(plain.get_ids())
                            {
                                ids.remove(0);
                            }
                        }
                    }
                }
                Ok(ids)
            }
            Self::Sp(sp) => sp.encode(prompt, add_special_tokens),
        }
    }

    pub(crate) fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Hf(t) => t
                .decode(ids, skip_special_tokens)
                .map_err(|e| BitNetError::Inference(format!("decode: {e}"))),
            Self::Sp(sp) => sp.decode(ids, skip_special_tokens),
        }
    }

    /// Best-effort EOS id for Llama/Mistral/Qwen-style chat checkpoints.
    pub(crate) fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_ids().into_iter().next()
    }

    /// A chat turn and the entire sequence may have different stop tokens (Llama 3).
    pub(crate) fn eos_token_ids(&self) -> Vec<u32> {
        // Harmony's <|end|> ends an analysis message, not the assistant turn.
        if let Self::Hf(t) = self {
            if t.token_to_id("<|channel|>").is_some()
                && t.token_to_id("<|message|>").is_some()
                && t.token_to_id("<|start|>").is_some()
            {
                if let Some(fim_suffix) = t
                    .token_to_id("<|fim_suffix|>")
                    .or_else(|| t.token_to_id("<|return|>"))
                {
                    return [Some(fim_suffix), t.token_to_id("<|endoftext|>")]
                        .into_iter()
                        .flatten()
                        .collect();
                }
            }
        }
        const CANDS: &[&str] = &[
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|user|>",
            "<|observation|>",
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
    /// Optional real tokenizer check; fixtures contain complete IDs from llama.cpp.
    #[test]
    fn optional_all_prompt_ids_match_reference() {
        let (Ok(tokenizer), Ok(fixtures)) = (
            std::env::var("RBITNET_PROMPT_TOKENIZER"),
            std::env::var("RBITNET_PROMPT_FIXTURES"),
        ) else {
            return;
        };
        let tokenizer =
            super::LoadedPromptTokenizer::from_path(std::path::Path::new(&tokenizer)).unwrap();
        let fixtures: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(fixtures).unwrap()).unwrap();
        assert!(!fixtures.is_empty());
        for fixture in fixtures {
            let actual = tokenizer
                .encode_ids(fixture["prompt"].as_str().unwrap(), true)
                .unwrap();
            let expected: Vec<u32> = serde_json::from_value(fixture["token_ids"].clone()).unwrap();
            assert_eq!(actual, expected, "{}: prompt IDs differ", fixture["id"]);
        }
    }
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

    #[test]
    fn harmony_continues_after_analysis_message_end() {
        let vocab = [
            ("[UNK]", 0),
            ("<|end|>", 1),
            ("<|return|>", 2),
            ("<|endoftext|>", 3),
            ("<|channel|>", 4),
            ("<|message|>", 5),
            ("<|start|>", 6),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let tok = LoadedPromptTokenizer::Hf(Tokenizer::new(model));
        assert_eq!(tok.eos_token_ids(), [2, 3]);
    }
}
