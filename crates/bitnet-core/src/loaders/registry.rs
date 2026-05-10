//! Dispatch `general.architecture` (and env overrides) to a concrete [`ModelExecutor`] builder.

use std::path::Path;
use std::sync::Arc;

use crate::backend::BackendKind;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::model::ModelExecutor;

use super::arch_key::{resolve_architecture_key, resolve_architecture_key_for_load};
use super::llama;
use super::qwen35;

use crate::deepseek2;
use crate::glm4_moe;
use crate::gpt_oss;

const BITNET_NOT_IMPL: &str = "BitNet GGUF inference is not yet implemented. \
Use RBITNET_TOY=1 for a toy model, or provide a Llama-architecture GGUF. \
Set RBITNET_MODEL_FAMILY=llama to override auto-detected architecture from the file.";

fn unsupported_non_llama_gguf_architecture(key: &str) -> Option<&'static str> {
    match key {
        "qwen2" | "qwen2vl" | "qwen2_moe" | "gemma" | "gemma2" | "gemma3" | "phi3" | "phi4"
        | "bloom" | "gpt2" | "t5" | "rwkv" => Some(
            "GGUF `general.architecture` is not supported by Rbitnet's Llama-compatible loader. \
For Qwen3 MoE checkpoints use `qwen35moe` with `RBITNET_BACKEND=cuda`. \
If this file is actually Llama/Mistral-shaped (mis-tagged), set `RBITNET_ARCHITECTURE=llama`.",
        ),
        _ => None,
    }
}

/// Build a [`ModelExecutor`] for a memory-mapped GGUF (uses `RBITNET_*` env for tokenizer / architecture overrides).
pub fn dispatch_gguf_executor(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
) -> Result<Box<dyn ModelExecutor>> {
    dispatch_gguf_executor_inner(backend_kind, gguf, model_path, false, None, None)
}

/// Same as [`dispatch_gguf_executor`] but does not use tokenizer/architecture from environment:
/// optional registry overrides plus GGUF-local tokenizer discovery only.
pub fn dispatch_gguf_executor_for_load(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
    tokenizer_override: Option<&Path>,
    architecture_override: Option<&str>,
) -> Result<Box<dyn ModelExecutor>> {
    dispatch_gguf_executor_inner(
        backend_kind,
        gguf,
        model_path,
        true,
        tokenizer_override,
        architecture_override,
    )
}

fn dispatch_gguf_executor_inner(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
    isolated_from_env: bool,
    tokenizer_override: Option<&Path>,
    architecture_override: Option<&str>,
) -> Result<Box<dyn ModelExecutor>> {
    let key = if isolated_from_env {
        resolve_architecture_key_for_load(&gguf, architecture_override)
    } else {
        resolve_architecture_key(&gguf)
    };

    if key == "bitnet" {
        return Err(BitNetError::Inference(BITNET_NOT_IMPL.into()));
    }

    if key == "qwen35moe" {
        if backend_kind != BackendKind::Cuda {
            return Err(BitNetError::Inference(
                "GGUF `qwen35moe`: native CUDA path requires `RBITNET_BACKEND=cuda` (Phase 1; CPU Llama-compatible loader is unsupported for this topology)."
                    .into(),
            ));
        }
        return qwen35::build_qwen35_moe_executor(
            backend_kind,
            gguf,
            model_path,
            isolated_from_env,
            tokenizer_override,
        );
    }

    if key == "deepseek2" {
        if backend_kind != BackendKind::Cuda {
            return Err(BitNetError::Inference(format!(
                "GGUF `{key}`: CUDA backend required (`RBITNET_BACKEND=cuda`). See docs/ARCHITECTURE_GGUF_MATRIX.md."
            )));
        }
        return deepseek2::build_deepseek2_executor(
            backend_kind,
            gguf,
            model_path,
            isolated_from_env,
            tokenizer_override,
        );
    }

    if key == "gptoss" {
        if backend_kind != BackendKind::Cuda {
            return Err(BitNetError::Inference(format!(
                "GGUF `{key}`: CUDA backend required (`RBITNET_BACKEND=cuda`). See docs/ARCHITECTURE_GGUF_MATRIX.md."
            )));
        }
        return gpt_oss::build_gptoss_executor(
            backend_kind,
            gguf,
            model_path,
            isolated_from_env,
            tokenizer_override,
        );
    }

    if matches!(key.as_str(), "glm4moe" | "glm4_moe") {
        if backend_kind != BackendKind::Cuda {
            return Err(BitNetError::Inference(format!(
                "GGUF `{key}`: CUDA backend required (`RBITNET_BACKEND=cuda`). See docs/ARCHITECTURE_GGUF_MATRIX.md."
            )));
        }
        return glm4_moe::build_glm4_moe_executor(
            key.as_str(),
            backend_kind,
            gguf,
            model_path,
            isolated_from_env,
            tokenizer_override,
        );
    }

    if let Some(msg) = unsupported_non_llama_gguf_architecture(&key) {
        return Err(BitNetError::Inference(msg.into()));
    }

    llama::build_llama_executor(
        backend_kind,
        gguf,
        model_path,
        isolated_from_env,
        tokenizer_override,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::backend::BackendKind;
    use crate::error::BitNetError;
    use crate::loaders::test_lock::env_test_lock;

    fn write_minimal_gguf_with_arch(path: &Path, arch: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(path)?;
        fn u32_le(w: &mut File, x: u32) -> std::io::Result<()> {
            w.write_all(&x.to_le_bytes())
        }
        fn u64_le(w: &mut File, x: u64) -> std::io::Result<()> {
            w.write_all(&x.to_le_bytes())
        }
        f.write_all(b"GGUF")?;
        u32_le(&mut f, 3u32)?;
        u64_le(&mut f, 0u64)?;
        u64_le(&mut f, 1u64)?;
        let key_meta = "general.architecture";
        u64_le(&mut f, key_meta.len() as u64)?;
        f.write_all(key_meta.as_bytes())?;
        u32_le(&mut f, 8u32)?;
        u64_le(&mut f, arch.len() as u64)?;
        f.write_all(arch.as_bytes())?;
        let pos = f.metadata()?.len() as usize;
        let align = 32usize;
        let pad = (align - (pos % align)) % align;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }

    #[test]
    fn dispatch_requires_cuda_for_qwen35moe_native() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.gguf");
        write_minimal_gguf_with_arch(&p, "qwen35moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let err = match dispatch_gguf_executor(BackendKind::Cpu, Arc::clone(&g), &p) {
            Ok(_) => panic!("expected Cpu dispatch error for qwen35moe"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.to_ascii_lowercase().contains("cuda"), "msg={msg}");
    }

    #[test]
    fn dispatch_qwen35moe_cuda_requires_tokenizer() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        write_minimal_gguf_with_arch(&p, "qwen35moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        match dispatch_gguf_executor(BackendKind::Cuda, g, &p) {
            Err(BitNetError::TokenizerMissing) => {}
            Err(e) => panic!("expected TokenizerMissing from qwen35moe builder, got {e}"),
            Ok(_) => panic!("expected Err without tokenizer beside GGUF"),
        }
    }

    #[test]
    fn dispatch_glm4moe_requires_cuda_on_cpu() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("glm.gguf");
        write_minimal_gguf_with_arch(&p, "glm4moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let err = match dispatch_gguf_executor(BackendKind::Cpu, g, &p) {
            Ok(_) => panic!("expected glm4moe Cpu dispatch to fail"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.to_ascii_lowercase().contains("cuda"), "msg={msg}");
    }

    #[test]
    fn dispatch_gptoss_requires_cuda_on_cpu() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("go.gguf");
        write_minimal_gguf_with_arch(&p, "gptoss").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let err = match dispatch_gguf_executor(BackendKind::Cpu, g, &p) {
            Ok(_) => panic!("expected gptoss Cpu dispatch to fail"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.to_ascii_lowercase().contains("cuda"), "msg={msg}");
    }

    #[test]
    fn dispatch_deepseek2_requires_cuda_on_cpu() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ds.gguf");
        write_minimal_gguf_with_arch(&p, "deepseek2").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let err = match dispatch_gguf_executor(BackendKind::Cpu, g, &p) {
            Ok(_) => panic!("expected deepseek2 Cpu dispatch to fail"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.to_ascii_lowercase().contains("cuda"), "msg={msg}");
    }

    #[test]
    fn dispatch_glm4moe_cuda_requires_tokenizer() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("glm.gguf");
        write_minimal_gguf_with_arch(&p, "glm4moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        match dispatch_gguf_executor(BackendKind::Cuda, g, &p) {
            Err(BitNetError::TokenizerMissing) => {}
            Err(e) => panic!("expected TokenizerMissing for glm4moe dispatch, got {e}"),
            Ok(_) => panic!("expected Err without tokenizer beside GGUF"),
        }
    }

    #[test]
    fn dispatch_llama_override_skips_block_but_needs_tokenizer() {
        let _g = env_test_lock();
        std::env::remove_var("RBITNET_TOKENIZER");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.gguf");
        write_minimal_gguf_with_arch(&p, "qwen35moe").unwrap();
        std::env::set_var("RBITNET_ARCHITECTURE", "llama");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let r = dispatch_gguf_executor(BackendKind::Cpu, g, &p);
        std::env::remove_var("RBITNET_ARCHITECTURE");
        match r {
            Err(BitNetError::TokenizerMissing) => {}
            Err(e) => panic!("expected TokenizerMissing after routing to Llama loader, got {e}"),
            Ok(_) => panic!("expected TokenizerMissing after routing to Llama loader, got Ok"),
        }
    }
}
