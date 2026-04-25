//! Tokenizer + generation loop.

use std::path::Path;

use rand::Rng;
use tokenizers::Tokenizer;

use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;

use super::model::{KvCache, LlamaModel};

/// Loads [`LlamaModel`] from GGUF and a Hugging Face tokenizer file (`tokenizer.json`, or `tokenizer.model` when loadable).
pub struct LlamaRuntime {
    model: LlamaModel,
    tokenizer: Tokenizer,
    kv: KvCache,
}

fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
    let lower = tokenizer_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
        if lower.ends_with(".model") {
            BitNetError::Inference(format!(
                "tokenizer load (tokenizer.model): {e}. If this is a raw SentencePiece protobuf, export tokenizer.json from Hugging Face (Save Pretrained) or use a repo that ships tokenizer.json."
            ))
        } else {
            BitNetError::Inference(format!("tokenizer load: {e}"))
        }
    })?;
    Ok(tokenizer)
}

impl LlamaRuntime {
    pub fn load(archive: &GgufArchive, tokenizer_path: &Path) -> Result<Self> {
        let model = LlamaModel::from_gguf(archive)?;
        let tokenizer = load_tokenizer(tokenizer_path)?;
        let kv = KvCache::new(&model.cfg);
        Ok(Self {
            model,
            tokenizer,
            kv,
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        self.kv.clear();
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| BitNetError::Inference(format!("encode: {e}")))?;
        let prompt_ids: Vec<u32> = enc.get_ids().iter().copied().collect();
        if prompt_ids.is_empty() {
            return Ok(String::new());
        }

        let mut logits = Vec::new();
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = self.model.forward(&mut self.kv, tid, pos)?;
        }

        let eos_id = self
            .tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"));

        let mut gen = Vec::new();
        let mut rng = rand::thread_rng();
        let mut pos = prompt_ids.len();

        for _ in 0..max_tokens {
            let next_id = sample_token(&logits, temperature, &mut rng);
            if Some(next_id) == eos_id {
                break;
            }
            gen.push(next_id);
            logits = self.model.forward(&mut self.kv, next_id, pos)?;
            pos += 1;
        }

        self.tokenizer
            .decode(&gen, true)
            .map_err(|e| BitNetError::Inference(format!("decode: {e}")))
    }
}

fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
                let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
                a.total_cmp(&b)
            })
            .unwrap()
            .0 as u32;
    }
    let scaled: Vec<f32> = logits.iter().map(|z| z / temperature).collect();
    let m = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|z| (z - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let r = rng.gen::<f32>() * s;
    let mut c = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        c += e;
        if c >= r {
            return i as u32;
        }
    }
    (exps.len().saturating_sub(1)) as u32
}
