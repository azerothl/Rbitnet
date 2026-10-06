//! GPU residency probe for Llama GGUFs (local-only evidence, not CI).
//!
//! Loads one GGUF with the requested weights backend, counts resident
//! quantized matrices, runs a short greedy decode on a fixed prompt, and
//! prints the shared `CudaRuntime` counter delta
//! (`device_resident_quant_gemv_calls`) alongside timing.
//!
//! Usage:
//! ```text
//! RBITNET_CUDA_QUANT_LIB=<dll> cargo run -p bitnet-core --release \
//!   --example gpu_residency_probe -- <GGUF> <TOKENIZER_JSON> <cpu|cuda> [MAX_NEW_TOKENS]
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bitnet_core::backend::{make_backend, BackendKind};
use bitnet_core::gguf::GgufArchive;
use bitnet_core::llama::{KvStorage, LlamaModel, MatrixWeights};
use bitnet_core::scratch::ScratchArena;

fn parse_backend(s: &str) -> BackendKind {
    match s {
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        other => panic!("expected cpu|cuda, got {other}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 || args.len() > 4 {
        return Err("GGUF TOKENIZER_JSON cpu|cuda [MAX_NEW_TOKENS]".into());
    }
    let backend_kind = parse_backend(&args[2]);
    let max_new: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16);

    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&args[0]))?);
    let tokenizer =
        tokenizers::Tokenizer::from_file(&args[1]).map_err(|e| format!("tokenizer: {e}"))?;

    let prompt = "What is the capital of France? Answer in one word.";
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();

    let load_started = Instant::now();
    let model = LlamaModel::from_gguf_arc_for_backend(Arc::clone(&archive), backend_kind)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let mut total_cuda_quant = 0usize;
    let mut resident_cuda_quant = 0usize;
    let mut total_cuda_dense = 0usize;
    let mut resident_cuda_dense = 0usize;
    let mut total_quant_mmap = 0usize;
    let mut total_dense = 0usize;
    let mut residency_snapshot_before = None;
    let mut residency_snapshot_after_load = None;
    let mut visit = |weights: &MatrixWeights| {
        match weights {
            MatrixWeights::CudaQuant { device, .. } => {
                total_cuda_quant += 1;
                if device.is_device_resident() {
                    resident_cuda_quant += 1;
                }
                if residency_snapshot_before.is_none() {
                    residency_snapshot_before = device.runtime_metrics();
                }
                residency_snapshot_after_load = device.runtime_metrics();
            }
            MatrixWeights::CudaDense { .. } => {
                total_cuda_dense += 1;
                resident_cuda_dense += 1;
            }
            MatrixWeights::Quant { .. } => total_quant_mmap += 1,
            MatrixWeights::Dense(_) => total_dense += 1,
        }
    };
    visit(&model.token_embd);
    for layer in &model.layers {
        visit(&layer.wq);
        visit(&layer.wk);
        visit(&layer.wv);
        visit(&layer.wo);
        visit(&layer.ffn_gate);
        visit(&layer.ffn_up);
        visit(&layer.ffn_down);
    }
    visit(&model.output);

    let backend = make_backend(backend_kind);
    let mut kv = KvStorage::new_dense(&model.cfg);
    let mut scratch = ScratchArena::default();

    let prefill_started = Instant::now();
    let mut logits = Vec::new();
    for (pos, id) in prompt_ids.iter().enumerate() {
        logits = model.forward_with_backend_and_scratch(
            &mut kv,
            *id,
            pos,
            backend.as_ref(),
            &mut scratch,
        )?;
    }
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;

    let decode_started = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    for step in 0..max_new {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        if next == 128009 || next == 128001 {
            break;
        }
        generated.push(next);
        logits = model.forward_with_backend_and_scratch(
            &mut kv,
            next,
            prompt_ids.len() + step,
            backend.as_ref(),
            &mut scratch,
        )?;
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = if decode_ms > 0.0 {
        generated.len() as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| format!("decode: {e}"))?;

    // Re-read the shared runtime counter through any resident matrix.
    let mut residency_snapshot_after = None;
    let mut scan = |weights: &MatrixWeights| {
        if residency_snapshot_after.is_none() {
            if let MatrixWeights::CudaQuant { device, .. } = weights {
                residency_snapshot_after = device.runtime_metrics();
            }
        }
    };
    scan(&model.token_embd);
    for layer in &model.layers {
        scan(&layer.wq);
        scan(&layer.wk);
        scan(&layer.wv);
        scan(&layer.wo);
        scan(&layer.ffn_gate);
        scan(&layer.ffn_up);
        scan(&layer.ffn_down);
    }
    scan(&model.output);

    let (before_quant, before_gemv) = residency_snapshot_before
        .map(|m| {
            (
                m.device_resident_quant_gemv_calls,
                m.gemv_calls,
            )
        })
        .unwrap_or((0, 0));
    let (after_quant, after_gemv) = residency_snapshot_after
        .map(|m| {
            (
                m.device_resident_quant_gemv_calls,
                m.gemv_calls,
            )
        })
        .unwrap_or((0, 0));

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend": backend_kind.as_str(),
            "prompt": prompt,
            "prompt_tokens": prompt_ids.len(),
            "generated_ids": generated,
            "generated": generated.len(),
            "text": text,
            "load_ms": load_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tps": decode_tps,
            "matrices": {
                "cuda_quant_total": total_cuda_quant,
                "cuda_quant_resident": resident_cuda_quant,
                "cuda_dense_total": total_cuda_dense,
                "cuda_dense_resident": resident_cuda_dense,
                "mmap_quant": total_quant_mmap,
                "dense_f32": total_dense,
            },
            "device_resident_quant_gemv_calls_before": before_quant,
            "device_resident_quant_gemv_calls_after": after_quant,
            "device_resident_quant_gemv_calls_delta": after_quant.saturating_sub(before_quant),
            "gemv_calls_before": before_gemv,
            "gemv_calls_after": after_gemv,
            "residency_after_load": residency_snapshot_after_load
                .map(|m| m.device_resident_quant_gemv_calls)
                .unwrap_or(0),
        }))
        .unwrap()
    );
    Ok(())
}
