//! Expand a text prompt containing an image placeholder into a mixed
//! token / patch-embedding prefill sequence.

use crate::error::{BitNetError, Result};
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;

/// Placeholder string inserted by the HTTP layer for each image part.
pub const IMAGE_PLACEHOLDER: &str = "<image>";

#[derive(Debug, Clone)]
pub enum PrefillItem {
    Token(u32),
    /// Row index into a contiguous `[n_patches × n_embd]` patch buffer.
    Patch(usize),
}

/// Split `prompt` on [`IMAGE_PLACEHOLDER`], tokenize text segments, and insert
/// `n_patches` patch slots where each placeholder occurs.
pub(crate) fn expand_prompt_with_patches(
    tokenizer: &LoadedPromptTokenizer,
    prompt: &str,
    n_patches: usize,
    add_special: bool,
) -> Result<Vec<PrefillItem>> {
    if n_patches == 0 {
        return Err(BitNetError::Inference(
            "vision expand requires n_patches > 0".into(),
        ));
    }
    let parts: Vec<&str> = prompt.split(IMAGE_PLACEHOLDER).collect();
    if parts.len() == 1 {
        return Err(BitNetError::Inference(format!(
            "vision prompt missing `{IMAGE_PLACEHOLDER}` placeholder"
        )));
    }
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        if !part.is_empty() {
            // Only the first segment may receive BOS/special tokens.
            let add = add_special && i == 0;
            let ids = tokenizer.encode_ids(part, add)?;
            out.extend(ids.into_iter().map(PrefillItem::Token));
        }
        if i + 1 < parts.len() {
            for p in 0..n_patches {
                out.push(PrefillItem::Patch(p));
            }
        }
    }
    if out.is_empty() {
        return Err(BitNetError::Inference(
            "vision expand produced empty prefill sequence".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_constant_is_llava_style() {
        assert_eq!(IMAGE_PLACEHOLDER, "<image>");
    }
}
