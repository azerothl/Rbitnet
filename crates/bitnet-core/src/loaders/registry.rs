//! Dispatch `general.architecture` (and env overrides) to a concrete [`ModelExecutor`] builder.

use std::path::Path;
use std::sync::Arc;

use crate::backend::BackendKind;
use crate::error::{BitNetError, Result};
use crate::gguf::GgufArchive;
use crate::model::ModelExecutor;

use super::arch_key::resolve_architecture_key;
use super::llama;

const BITNET_NOT_IMPL: &str = "BitNet GGUF inference is not yet implemented. \
Use RBITNET_TOY=1 for a toy model, or provide a Llama-architecture GGUF. \
Set RBITNET_MODEL_FAMILY=llama to override auto-detected architecture from the file.";

fn unsupported_arch_message(arch: &str) -> String {
    format!(
        "GGUF architecture `{arch}` is not implemented in Rbitnet yet. \
Use a Llama-family checkpoint and tokenizer, or contribute a loader under `crates/bitnet-core/src/loaders/`. \
Inspect `general.architecture` via `cargo run -p bitnet-core --example inspect_gguf -- your.gguf`. \
To force attempting the Llama loader anyway, set RBITNET_ARCHITECTURE=llama."
    )
}

/// Loaders for architectures that have a dedicated inference path (not the generic Llama GGUF stack).
fn is_known_non_llama_executor(arch: &str) -> bool {
    matches!(arch, "qwen35moe")
}

/// Build a [`ModelExecutor`] for a memory-mapped GGUF.
pub fn dispatch_gguf_executor(
    backend_kind: BackendKind,
    gguf: Arc<GgufArchive>,
    model_path: &Path,
) -> Result<Box<dyn ModelExecutor>> {
    let key = resolve_architecture_key(&gguf);

    if key == "bitnet" {
        return Err(BitNetError::Inference(BITNET_NOT_IMPL.into()));
    }

    if is_known_non_llama_executor(&key) {
        return Err(BitNetError::Inference(unsupported_arch_message(&key)));
    }

    llama::build_llama_executor(backend_kind, gguf, model_path)
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
    fn dispatch_rejects_qwen35moe_early() {
        let _g = env_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.gguf");
        write_minimal_gguf_with_arch(&p, "qwen35moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let g = Arc::new(GgufArchive::mmap_path(&p).unwrap());
        let err = match dispatch_gguf_executor(BackendKind::Cpu, Arc::clone(&g), &p) {
            Ok(_) => panic!("expected error for qwen35moe"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("qwen35moe") || msg.contains("`qwen35moe`"),
            "msg={msg}"
        );
    }

    #[test]
    fn dispatch_llama_override_skips_block_but_needs_tokenizer() {
        let _g = env_test_lock();
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
