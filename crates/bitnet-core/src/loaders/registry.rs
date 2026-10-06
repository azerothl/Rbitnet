//! Dispatch `general.architecture` (and env overrides) to a concrete [`ModelExecutor`] builder.

use std::path::Path;
use std::sync::Arc;

use crate::backend::BackendKind;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::model::ModelExecutor;

use super::arch_key::{resolve_architecture_key, resolve_architecture_key_for_load};
use super::bitnet;
use super::llama;
use super::mixtral;
use super::qwen3;
use super::qwen35;

use crate::deepseek2;
use crate::glm4_moe;
use crate::gpt_oss;

fn unsupported_non_llama_gguf_architecture(key: &str) -> Option<&'static str> {
    match key {
        "spark2_5" | "spark2-5" | "spark25" | "spark_2_5" => Some(
            "GGUF `general.architecture` spark2_5 (Spark-X2.5) is not supported. \
Rbitnet does not load it as Llama. Tracking: https://github.com/azerothl/Rbitnet/issues/142 \
and docs/ARCHITECTURE_GGUF_MATRIX.md.",
        ),
        "qwen2" | "qwen2vl" | "qwen2_moe" | "gemma" | "gemma2" | "gemma3" | "phi3" | "phi4"
        | "bloom" | "gpt2" | "t5" | "rwkv" => Some(
            "GGUF `general.architecture` is not supported by Rbitnet's Llama-compatible loader. \
For Qwen3 MoE checkpoints use `qwen35moe` with `RBITNET_BACKEND=cuda`. \
If this file is actually Llama/Mistral-shaped (mis-tagged), set `RBITNET_ARCHITECTURE=llama`.",
        ),
        _ => None,
    }
}
