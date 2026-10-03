//! Profile the benchmark's real forward path; the CPU-attention variant is a diagnostic ablation.
//! cargo run -p bitnet-core --release --features profile-llama --example profile_llama --
//!   MODEL.gguf llama32-1b-prompts.json TOKENIZER.json cpu cpu 2 16

use bitnet_core::{
    backend::{make_backend, BackendKind},
    gguf::GgufArchive,
    llama::{profile, KvStorage, LlamaModel},
    perf,
    scratch::ScratchArena,
};
use serde_json::{json, Value};
use std::{path::Path, sync::Arc, time::Instant};

fn kind(s: &str) -> BackendKind {
    match s {
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        _ => panic!("cpu or cuda required"),
    }
}

fn delta(a: &perf::PerfSnapshot, b: &perf::PerfSnapshot) -> Value {
    json!({"gpu_cublas_gemv_calls": b.gpu_gemv_calls - a.gpu_gemv_calls,
        "gpu_upload_bytes": b.gpu_upload_bytes - a.gpu_upload_bytes,
        "gpu_download_bytes": b.gpu_download_bytes - a.gpu_download_bytes,
        "cpu_quant_calls": b.quant_matvec_calls - a.quant_matvec_calls,
        "cpu_quant_ns": b.quant_matvec_ns - a.quant_matvec_ns})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "MODEL PROMPTS TOKENIZER WEIGHTS_BACKEND ATTENTION_BACKEND REPEATS TOKENS".into(),
        );
    }
    let rows: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let prompt = rows
        .iter()
        .find(|p| p["id"] == "throughput-1")
        .ok_or("missing throughput-1")?;
    let ids: Vec<u32> = serde_json::from_value(prompt["token_ids"].clone())?;
    let tokenizer = tokenizers::Tokenizer::from_file(&args[2]).map_err(|e| e.to_string())?;
    let encoded = tokenizer
        .encode(prompt["prompt"].as_str().unwrap(), true)
        .map_err(|e| e.to_string())?;
    assert_eq!(encoded.get_ids(), ids, "benchmark tokenization drift");
    let repeats: usize = args[5].parse()?;
    let tokens: usize = args[6].parse()?;
    if repeats == 0 || tokens == 0 || ids.is_empty() {
        return Err("positive counts required".into());
    }
    let started = Instant::now();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&args[0]))?);
    let model = LlamaModel::from_gguf_arc_for_backend(archive, kind(&args[3]))?;
    let backend = make_backend(kind(&args[4]));
    if args[4] == "cuda" && !backend.is_native_accelerated() {
        return Err("CUDA unavailable".into());
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut kv = KvStorage::new_dense(&model.cfg);
    let mut scratch = ScratchArena::default();
    // Initialize pools and CUDA libraries; excluded from measured spans.
    for (pos, id) in ids.iter().take(3).enumerate() {
        model.forward_with_backend_and_scratch(
            &mut kv,
            *id,
            pos,
            backend.as_ref(),
            &mut scratch,
        )?;
    }
    profile::take();
    let mut measurements = Vec::new();
    for repetition in 0..repeats {
        kv.clear();
        let before = perf::snapshot();
        let started = Instant::now();
        let mut logits = Vec::new();
        for (pos, id) in ids.iter().enumerate() {
            logits = model.forward_with_backend_and_scratch(
                &mut kv,
                *id,
                pos,
                backend.as_ref(),
                &mut scratch,
            )?;
        }
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
        let prefill_stages = profile::take();
        let middle = perf::snapshot();
        let started = Instant::now();
        let mut generated = Vec::new();
        for step in 0..tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            if next == 128009 || next == 128001 {
                break;
            }
            generated.push(next);
            // Match the current runtime, including its final unused forward.
            logits = model.forward_with_backend_and_scratch(
                &mut kv,
                next,
                ids.len() + step,
                backend.as_ref(),
                &mut scratch,
            )?;
        }
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let decode_stages = profile::take();
        measurements.push(json!({"repetition": repetition, "prompt_tokens": ids.len(),
            "completion_tokens": generated.len(), "prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "decode_tps": generated.len() as f64 / (decode_ms / 1000.0),
            "prefill_stages": prefill_stages, "decode_stages": decode_stages,
            "prefill_counters": delta(&before, &middle), "decode_counters": delta(&middle, &perf::snapshot()),
            "text": tokenizer.decode(&generated, true).map_err(|e| e.to_string())?, "token_ids": generated}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"format": "rbitnet-llama-profile-v1",
        "weights_backend": args[3], "attention_backend": args[4],
        "output_offload": std::env::var("RBITNET_HYBRID_OUTPUT").unwrap_or_else(|_| "0".into()),
        "weight_mode": std::env::var("RBITNET_LLAMA_WEIGHT_MODE").unwrap_or_else(|_| "auto".into()),
        "cuda_dense_ablation": std::env::var("RBITNET_PROFILE_CUDA_DENSE").unwrap_or_else(|_| "0".into()),
        "load_ms": load_ms, "max_seq": model.cfg.max_seq,
        "measurements": measurements}))?
    );
    Ok(())
}
