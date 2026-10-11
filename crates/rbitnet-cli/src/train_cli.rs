//! Delegate to optional Python recipes under `training/` (LoRA/SFT).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Run `training/<recipe>` with `python`, forwarding `passthrough` args.
pub fn run_train(repo_root: &Path, recipe: &Path, passthrough: &[String]) -> Result<(), String> {
    let root = fs::canonicalize(repo_root).map_err(|e| {
        format!(
            "--repo-root {} cannot be resolved: {e}",
            repo_root.display()
        )
    })?;
    // Reject absolute paths and parent-directory components to prevent path traversal.
    if recipe.is_absolute() {
        return Err(format!(
            "--recipe must be a relative path under training/ (got: {})",
            recipe.display()
        ));
    }
    if recipe
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(format!(
            "--recipe must not contain '..' path components: {}",
            recipe.display()
        ));
    }

    let script = root.join("training").join(recipe);
    if !script.is_file() {
        return Err(format!(
            "recipe not found: {} (point --repo-root at the Rbitnet repository root; expected training/ tree)",
            script.display()
        ));
    }

    // Canonicalize the resolved path and verify it stays inside training/.
    let canonical_script = fs::canonicalize(&script)
        .map_err(|e| format!("cannot resolve recipe path {}: {e}", script.display()))?;
    let canonical_training = fs::canonicalize(root.join("training"))
        .map_err(|e| format!("cannot resolve training/ directory: {e}"))?;
    if !canonical_script.starts_with(&canonical_training) {
        return Err(format!(
            "--recipe resolves outside the training/ directory: {}",
            recipe.display()
        ));
    }

    let py = resolve_python()?;
    let mut cmd = Command::new(&py);
    cmd.arg(&canonical_script);
    for a in passthrough {
        cmd.arg(a);
    }
    cmd.current_dir(&root);

    let status = cmd
        .status()
        .map_err(|e| format!("failed to run {}: {e}", py.display()))?;
    if !status.success() {
        return Err(format!(
            "Python exited with status {status} (set RBITNET_PYTHON to override the interpreter)"
        ));
    }
    Ok(())
}

fn resolve_python() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("RBITNET_PYTHON") {
        return Ok(PathBuf::from(p));
    }
    for exe in ["python", "python3", "py"] {
        let mut c = Command::new(exe);
        c.arg("--version");
        if c.output().map(|o| o.status.success()).unwrap_or(false) {
            return Ok(PathBuf::from(exe));
        }
    }
    Err(
        "Python 3.10+ not found in PATH. Install Python or set RBITNET_PYTHON to the interpreter."
            .into(),
    )
}

/// Compact checklist for HF checkpoint → GGUF → Rbitnet.
pub fn export_gguf_hint(checkpoint: Option<&Path>) -> String {
    let mut out = String::from(
        "Rbitnet consumes supported GGUF architectures. Export Llama-family Safetensors checkpoints with the llama.cpp tools for your revision.\n\
Upstream: https://github.com/ggml-org/llama.cpp\n\
Walkthrough: docs/UNSLOTH_TO_RBITNET.md\n\
Compatibility: docs/TRAINING_AND_COMPATIBILITY.md\n",
    );
    if let Some(p) = checkpoint {
        if !p.exists() {
            out.push_str(&format!(
                "warning: --checkpoint path does not exist yet: {}\n",
                p.display()
            ));
        }
        out.push_str(&format!(
            "\nCheckpoint directory: {}\n\n\
Typical flow (verify against your llama.cpp checkout):\n\
  1. python convert_hf_to_gguf.py \"{}\" --outfile model-f16.gguf\n\
  2. Optional: run the llama.cpp quantize tool for Q4_K_M or similar.\n\
  3. Copy tokenizer.json beside the .gguf or set RBITNET_TOKENIZER.\n\
  4. Set RBITNET_MODEL to the absolute .gguf path.\n\
  5. rbitnet serve\n",
            p.display(),
            p.display()
        ));
    }
    out
}

