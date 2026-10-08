//! CUDA fused scheduler decode: shared batch stats vs serial owners.
use super::scheduler_fused::{configured, SchedulerFusedLlama, SchedulerFusedOptions};
use super::*;
use crate::loaders::prompt_tokenizer::LoadedPromptTokenizer;
use crate::sampling::SamplingOptions;
use std::sync::Arc;

fn load() -> (Arc<LlamaModel>, Arc<LoadedPromptTokenizer>) {
    let archive = Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let tokenizer = Arc::new(
        LoadedPromptTokenizer::from_path_for_gguf(
            std::path::Path::new(&std::env::var("RBITNET_TOKENIZER").unwrap()),
            &archive,
        )
        .unwrap(),
    );
    let model = Arc::new(
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap(),
    );
    (model, tokenizer)
}

#[test]
fn fused_scheduler_batch_shared_projections_exceed_serial() {
    if std::env::var("RBITNET_LLAMA_FUSED_SCHEDULER_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_KV_FORMAT", "f32");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "0");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::set_var("RBITNET_FUSED_MULTI_SEQ", "1");
    let options = configured(crate::backend::BackendKind::Cuda)
        .unwrap()
        .expect("fused scheduler options");
    let (model, tokenizer) = load();
    let mut engine = SchedulerFusedLlama::new(model, tokenizer, options).unwrap();
    let prompt = "The capital of France is";
    let sampling = SamplingOptions {
        temperature: 0.,
        ..Default::default()
    };
    engine.begin_batch(&[1, 2]);
    let wave1 = engine
        .decode_wave(&[
            (1, prompt.to_owned(), 4, sampling),
            (2, prompt.to_owned(), 4, sampling),
        ])
        .unwrap();
    assert_eq!(wave1.len(), 2);
    let stats = engine.native_stats().unwrap();
    assert!(stats[0] >= 1, "batch waves: {stats:?}");
    assert!(stats[1] >= 2, "batch rows: {stats:?}");
    assert!(
        stats[2] > stats[0],
        "shared projection count should exceed wave count: {stats:?}"
    );
    let _ = engine.decode_wave(&[
        (1, format!("{prompt}{}", wave1[0].0), 4, sampling),
        (2, format!("{prompt}{}", wave1[1].0), 4, sampling),
    ]);
    let stats2 = engine.native_stats().unwrap();
    assert!(
        stats2[2] > stats2[0],
        "multi-row decode should keep shared GEMMs: {stats2:?}"
    );
    engine.end_batch();
}
