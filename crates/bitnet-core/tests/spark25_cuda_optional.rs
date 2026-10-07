//! Optional Spark-X2.5 CUDA matvec smoke on real Q4_K GGUF.
//!
//! ```powershell
//! $env:RBITNET_SPARK25_SMOKE='1'
//! $env:RBITNET_TEST_GGUF='D:/Rbitnet-benchmark-models/spark25-1.7b/Spark-X2.5-1.7B-Q4_K_M.gguf'
//! $env:RBITNET_TOKENIZER='D:/Rbitnet-benchmark-models/spark25-1.7b'
//! cargo test -p bitnet-core spark25_cuda_matvec_smoke -- --nocapture
//! ```

use std::path::Path;
use std::sync::Arc;

use bitnet_core::backend::BackendKind;
use bitnet_core::gguf::GgufArchive;
use bitnet_core::loaders::dispatch_gguf_executor_for_load;
use bitnet_core::model::ModelExecutor as _;
use bitnet_core::sampling::SamplingOptions;

#[test]
fn spark25_cuda_matvec_smoke() {
    if std::env::var("RBITNET_SPARK25_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_TEST_GGUF").expect("RBITNET_TEST_GGUF");
    let tokenizer = std::env::var("RBITNET_TOKENIZER").expect("RBITNET_TOKENIZER");
    let model_path = Path::new(&gguf);
    let tok_dir = Path::new(&tokenizer);
    let archive = Arc::new(GgufArchive::mmap_path(model_path).unwrap());
    let exec = dispatch_gguf_executor_for_load(
        BackendKind::Cuda,
        archive,
        model_path,
        Some(tok_dir),
        None,
    )
    .unwrap();
    assert!(
        exec.backend_accelerated(),
        "expected device-resident quant matvec (build cuda_quant, RBITNET_BACKEND=cuda)"
    );
    let before = bitnet_core::backend::CudaRuntime::try_load()
        .map(|rt| rt.metrics_snapshot().device_resident_quant_gemv_calls)
        .unwrap_or(0);
    let sampling = SamplingOptions::from_temperature(0.0);
    let (text, phases) = exec
        .generate_with_timings("Hello", 8, sampling)
        .expect("generate");
    let after = bitnet_core::backend::CudaRuntime::try_load()
        .map(|rt| rt.metrics_snapshot().device_resident_quant_gemv_calls)
        .unwrap_or(0);
    assert!(
        after > before,
        "device_resident_quant_gemv_calls did not increase ({before} -> {after})"
    );
    assert!(phases.completion_tokens > 0, "expected completion tokens");
    let preview: String = text.chars().take(80).collect();
    eprintln!(
        "spark25 cuda smoke ok: {} tokens, gemv {} -> {}, text preview: {:?}",
        phases.completion_tokens, before, after, preview
    );
}
