//! CLI helper: print GGUF summary (metadata, Llama hyperparameters, first tensors).
//!
//! Usage:
//! ```text
//! cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
//! ```
//!
//! Use this to verify a BitNet-produced GGUF (e.g. after converting
//! `1bitLLM/bitnet_b1_58-large` with Microsoft BitNet tooling) before enabling `RBITNET_MODEL`.

use std::env;
use std::path::Path;

use bitnet_core::GgufArchive;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: inspect_gguf <file.gguf>")?;
    let path = Path::new(&path);
    let arch = GgufArchive::mmap_path(path)?;
    println!("{}", arch.summary_line());
    println!(
        "tensor_data_len={} bytes",
        arch.tensor_data().len()
    );

    let hp = arch.llama_hyper_params();
    println!(
        "LlamaHyperParams: context_length={:?} embedding_length={:?} block_count={:?} \
         head_count={:?} head_count_kv={:?} vocab_size={:?}",
        hp.context_length,
        hp.embedding_length,
        hp.block_count,
        hp.head_count,
        hp.head_count_kv,
        hp.vocab_size
    );

    let show = 20.min(arch.tensors.len());
    println!("First {show} tensors:");
    for t in arch.tensors.iter().take(show) {
        println!(
            "  {:50} dims={:?} ggml_type={}",
            t.name, t.dimensions, t.ggml_type
        );
    }
    if arch.tensors.len() > show {
        println!("  ... and {} more", arch.tensors.len() - show);
    }

    Ok(())
}
