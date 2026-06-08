//! Resolve the architecture key used by [`super::dispatch_gguf_executor`].

use crate::gguf::GgufArchive;

/// Normalize a user-supplied slug (ASCII lowercase, trimmed).
pub fn normalize_architecture_slug(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

/// Normalized slug for [`resolve_architecture_key`] when `RBITNET_MODEL_FAMILY` selects a
/// **Llama-GGUF–compatible** runtime path (same tensor layout as `llama.cpp` “Llama” loader).
///
/// Covers Mistral/Mixtral/DeepSeek-coded Llama exports, chat templates (*Yi*, *Zephyr*, …),
/// and vendor labels sometimes used without matching `general.architecture` (*openai*, *z.ai*, …).
pub fn family_override_token(family_normalized: &str) -> Option<&'static str> {
    match family_normalized {
        "llama" | "mistral" | "mixtral" | "codellama" | "deepseek" | "deepseek2"
        | "deepseekcoder" | "yi" | "vicuna" | "wizardlm" | "orca" | "starling" | "zephyr"
        | "openchat" | "neural-chat" | "solar" | "stablelm" | "phi" | "phi2" | "openai"
        | "gpt-oss" | "zai" | "glm" | "chatglm" => Some("llama"),
        _ => None,
    }
}

/// SSOT priority:
/// 1. `RBITNET_ARCHITECTURE` (explicit override).
/// 2. `RBITNET_MODEL_FAMILY` — see [`family_override_token`] (Llama-lineage aliases vs `bitnet`).
/// 3. **`auto`** (default family): detect `general.architecture == bitnet`; else GGUF slug; else **`llama`**
///    when `general.architecture` is missing (legacy GGUF compat).
///
/// Atlas-style loaders dispatch on this normalized key.
pub fn resolve_architecture_key(gguf: &GgufArchive) -> String {
    if let Ok(v) = std::env::var("RBITNET_ARCHITECTURE") {
        let t = normalize_architecture_slug(&v);
        if !t.is_empty() {
            return t;
        }
    }

    if let Ok(f) = std::env::var("RBITNET_MODEL_FAMILY") {
        let fam = normalize_architecture_slug(&f);
        if fam.is_empty() || fam == "auto" {
            // fall through
        } else if fam == "bitnet" {
            return "bitnet".to_string();
        } else if let Some(slug) = family_override_token(&fam) {
            return slug.to_string();
        }
        // unrecognized family string: treat like auto — use GGUF + fallbacks below
    }

    if gguf
        .architecture()
        .is_some_and(|a| a.eq_ignore_ascii_case("bitnet"))
    {
        return "bitnet".to_string();
    }

    gguf.normalized_architecture()
        .unwrap_or_else(|| "llama".to_string())
}

/// Resolve architecture for a GGUF loaded in isolation from `RBITNET_ARCHITECTURE` / `RBITNET_MODEL_FAMILY`
/// so concurrent multi-model loads do not race on environment variables.
///
/// Priority:
/// 1. Non-empty `explicit_override` (registry `architecture` field).
/// 2. Same detection as [`resolve_architecture_key`] when env vars are unset: BitNet flag from GGUF,
///    then `general.architecture` slug, else `llama`.
pub fn resolve_architecture_key_for_load(
    gguf: &GgufArchive,
    explicit_override: Option<&str>,
) -> String {
    if let Some(raw) = explicit_override {
        let t = normalize_architecture_slug(raw);
        if !t.is_empty() {
            return t;
        }
    }

    if gguf
        .architecture()
        .is_some_and(|a| a.eq_ignore_ascii_case("bitnet"))
    {
        return "bitnet".to_string();
    }

    gguf.normalized_architecture()
        .unwrap_or_else(|| "llama".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use crate::loaders::test_lock::env_test_lock;

    #[test]
    fn normalize_slug_trims_and_lowercases() {
        assert_eq!(normalize_architecture_slug("  Llama  "), "llama");
    }

    #[test]
    fn resolve_respects_rbitnet_architecture_override() {
        let _g = env_test_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.gguf");
        write_minimal_gguf_with_arch(Path::new(&path), "llama").unwrap();
        std::env::set_var("RBITNET_ARCHITECTURE", "qwen35moe");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        let gguf = GgufArchive::mmap_path(Path::new(&path)).unwrap();
        let key = resolve_architecture_key(&gguf);
        std::env::remove_var("RBITNET_ARCHITECTURE");
        assert_eq!(key, "qwen35moe");
    }

    #[test]
    fn resolve_model_family_llama_override_overrides_slugs() {
        let _g = env_test_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.gguf");
        write_minimal_gguf_with_arch(Path::new(&path), "qwen35moe").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::set_var("RBITNET_MODEL_FAMILY", "llama");
        let gguf = GgufArchive::mmap_path(Path::new(&path)).unwrap();
        let key = resolve_architecture_key(&gguf);
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        assert_eq!(key, "llama");
    }

    #[test]
    fn resolve_reads_general_architecture_when_auto() {
        let _g = env_test_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t.gguf");
        write_minimal_gguf_with_arch(Path::new(&path), "Mistral").unwrap();
        std::env::remove_var("RBITNET_ARCHITECTURE");
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        std::env::set_var("RBITNET_MODEL_FAMILY", "auto");
        let gguf = GgufArchive::mmap_path(Path::new(&path)).unwrap();
        assert_eq!(gguf.architecture(), Some("Mistral"));
        let key = resolve_architecture_key(&gguf);
        std::env::remove_var("RBITNET_MODEL_FAMILY");
        assert_eq!(key, "mistral");
    }

    /// Minimal GGUF v3: 0 tensors, 1 KV `general.architecture` string, default alignment 32.
    fn write_minimal_gguf_with_arch(path: &Path, arch: &str) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        fn u32_le(w: &mut File, x: u32) -> std::io::Result<()> {
            w.write_all(&x.to_le_bytes())
        }
        fn u64_le(w: &mut File, x: u64) -> std::io::Result<()> {
            w.write_all(&x.to_le_bytes())
        }
        f.write_all(b"GGUF")?;
        u32_le(&mut f, 3u32)?;
        u64_le(&mut f, 0u64)?; // tensor_count
        u64_le(&mut f, 1u64)?; // kv_count
        let key = "general.architecture";
        u64_le(&mut f, key.len() as u64)?;
        f.write_all(key.as_bytes())?;
        u32_le(&mut f, 8u32)?; // STRING
        u64_le(&mut f, arch.len() as u64)?;
        f.write_all(arch.as_bytes())?;
        let pos = f.metadata()?.len() as usize;
        let align = 32usize;
        let pad = (align - (pos % align)) % align;
        f.write_all(&vec![0u8; pad])?;
        Ok(())
    }
}
