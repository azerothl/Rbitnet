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
    let script = root.join("training").join(recipe);
    if !script.is_file() {
        return Err(format!(
            "recipe not found: {} (point --repo-root at the Rbitnet repository root; expected training/ tree)",
            script.display()
        ));
    }

    let py = resolve_python()?;
    let mut cmd = Command::new(&py);
    cmd.arg(&script);
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

/// Print a compact checklist for HF checkpoint → GGUF → Rbitnet.
pub fn print_export_gguf_hint(checkpoint: Option<&Path>) {
    eprintln!(
        "Rbitnet loads Llama-shaped GGUF files only. Convert Hugging Face Safetensors checkpoints with"
    );
    eprintln!(
        "the **llama.cpp** tools for your revision (script names change over time). Upstream:"
    );
    eprintln!("  https://github.com/ggerganov/llama.cpp");
    eprintln!();
    eprintln!("Full guide: docs/TRAINING_AND_COMPATIBILITY.md");
    if let Some(p) = checkpoint {
        if !p.exists() {
            eprintln!(
                "warning: --checkpoint path does not exist yet: {}",
                p.display()
            );
        }
        println!();
        println!("Checkpoint directory: {}", p.display());
        println!();
        println!("Typical flow (verify against your llama.cpp checkout):");
        println!(
            "  1. python convert_hf_to_gguf.py {} --outfile model-f16.gguf",
            p.display()
        );
        println!("  2. Optional: run the llama.cpp quantize tool for Q4_K_M or similar.");
        println!("  3. Copy tokenizer.json beside the .gguf or set RBITNET_TOKENIZER.");
        println!("  4. export RBITNET_MODEL=/absolute/path/to/model.gguf");
        println!("  5. rbitnet serve   # or rbitnet-server");
    } else {
        println!();
        println!("Pass --checkpoint DIR for a concrete convert_hf_to_gguf.py example line.");
    }
}
