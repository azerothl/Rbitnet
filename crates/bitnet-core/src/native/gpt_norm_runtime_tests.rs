//! Optional real GPT-OSS block/fusion comparison to its serial resident pipeline.
use super::*;
#[test]
fn actual_gpt_ordered_fastpaths_cross_library_logits_seeded_outputs_prefixes_and_cancellation() {
    if std::env::var("RBITNET_GPT_BLOCK_RUNTIME_TEST").as_deref() != Ok("1") {
        return;
    }
    let path = std::env::var("RBITNET_GPT_BLOCK_GGUF").unwrap();
    let tok = std::env::var("RBITNET_GPT_BLOCK_TOKENIZER").unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&path)).unwrap());
    for (k, v) in [
        ("RBITNET_MAX_SEQ", "2048"),
        ("RBITNET_MOE_CACHE_MB", "0"),
        ("RBITNET_MOE_EXECUTION", "cache"),
        ("RBITNET_CUDA_GPT_FULL", "1"),
        ("RBITNET_REQUIRE_GPT_FULL", "1"),
        ("RBITNET_CUDA_GPT_SEGMENTED", "0"),
        ("RBITNET_CUDA_GPT_FULL_GRAPH", "1"),
        ("RBITNET_CUDA_SPLIT_KV", "1"),
        ("RBITNET_CUDA_GPT_PREFILL", "0"),
        ("RBITNET_REQUIRE_GPT_PREFILL", "0"),
        ("RBITNET_CUDA_MOE_FUSED", "0"),
        ("RBITNET_REQUIRE_FUSED_MOE", "0"),
        ("RBITNET_PREFIX_KV", "0"),
    ] {
        std::env::set_var(k, v);
    }
    let mut reference = Runtime::load(
        Arc::clone(&archive),
        Path::new(&tok),
        BackendKind::Cuda,
        Family::GptOss,
    )
    .unwrap();
    assert!(!reference.gpu_full.as_ref().unwrap().is_segmented());
    let corpus=["Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),
 "fn somme(values: &[i32]) -> i32 { values.iter().sum() } // vérifier les cas vides et les valeurs négatives\n".repeat(40)];
    let inputs: Vec<Vec<u32>> = corpus
        .iter()
        .map(|s| {
            reference
                .tokenizer
                .encode_ids(s, true)
                .unwrap()
                .into_iter()
                .take(160)
                .collect()
        })
        .collect();
    let observed = [0, 1, 7, 31, 63, 127, 145, 146, 147, 148, 149, 159];
    let mut expected = Vec::new();
    for ids in &inputs {
        assert_eq!(ids.len(), 160);
        let mut outputs = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let (out, _) = reference
                .forward(id, pos, observed.contains(&pos), false)
                .unwrap();
            if observed.contains(&pos) {
                outputs.push(out);
            }
        }
        expected.push(outputs);
    }
    let prompts=["<|start|>user<|message|>Quelle est la capitale de la France ? Réponds en un mot.<|end|><|start|>assistant<|channel|>final<|message|>",
 "<|start|>user<|message|>Écris une fonction Python qui additionne deux nombres.<|end|><|start|>assistant<|channel|>final<|message|>"];
    let samples = [
        SamplingOptions::from_temperature(0.0),
        SamplingOptions {
            seed: Some(42),
            ..SamplingOptions::from_temperature(0.7)
        },
        SamplingOptions {
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            seed: Some(7),
            ..SamplingOptions::from_temperature(0.0)
        },
    ];
    let mut texts = Vec::new();
    for p in prompts {
        for s in samples {
            texts.push(reference.generate(p, 32, s, None).unwrap().0);
        }
    }
    let folder =
        std::path::PathBuf::from(std::env::var("RBITNET_GPT_FASTPATH_OUTPUT_DIR").unwrap());
    std::fs::create_dir_all(&folder).unwrap();
    for (corpus, outputs) in expected.iter().enumerate() {
        for (observation, row) in outputs.iter().enumerate() {
            assert!(row.iter().all(|x| x.is_finite()));
            let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            std::fs::write(
                folder.join(format!(
                    "corpus-{corpus}-position-{}.f32",
                    observed[observation]
                )),
                bytes,
            )
            .unwrap();
        }
    }
    std::fs::write(
        folder.join("texts.json"),
        serde_json::to_vec(&texts).unwrap(),
    )
    .unwrap();
    drop(reference);
    let mut worst_kl = 0.0f64;
    let mut worst_nll = 0.0f64;
    for (count, tile, fused) in [(16, 0, 0), (32, 0, 0), (32, 1, 0), (16, 0, 1), (0, 0, 1)] {
        std::env::set_var(
            "RBITNET_CUDA_GPT_PREFILL",
            if count > 0 { "1" } else { "0" },
        );
        std::env::set_var(
            "RBITNET_REQUIRE_GPT_PREFILL",
            if count > 0 { "1" } else { "0" },
        );
        std::env::set_var("RBITNET_CUDA_GPT_PREFILL_TOKENS", count.to_string());
        std::env::set_var("RBITNET_CUDA_GPT_PREFILL_TILE", tile.to_string());
        std::env::set_var("RBITNET_CUDA_MOE_FUSED", fused.to_string());
        std::env::set_var("RBITNET_REQUIRE_FUSED_MOE", fused.to_string());
        std::env::set_var("RBITNET_PREFIX_KV", "1");
        let mut actual = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tok),
            BackendKind::Cuda,
            Family::GptOss,
        )
        .unwrap();
        assert!(!actual.gpu_full.as_ref().unwrap().is_segmented());
        assert_eq!(actual.gpu_full.as_ref().unwrap().prefill_capacity(), count);
        assert_eq!(
            actual
                .gpu_moe
                .iter()
                .flatten()
                .filter(|m| m.is_fused())
                .count(),
            if fused == 1 { actual.cfg.layers } else { 0 }
        );
        for (corpus, ids) in inputs.iter().enumerate() {
            let mut full = actual.gpu_full.take().unwrap();
            let tensor = actual.weights.tensor("token_embd.weight").unwrap();
            let mut pos = 0;
            let mut observation = 0;
            while pos < ids.len() {
                let next_observed = observed[observation];
                let n = if count > 0 {
                    count.min(next_observed + 1 - pos)
                } else {
                    1
                };
                let mut embeddings = vec![0.0; n * actual.cfg.embd];
                for (&id, row) in ids[pos..pos + n]
                    .iter()
                    .zip(embeddings.chunks_exact_mut(actual.cfg.embd))
                {
                    crate::ggml::embedding_row_mmap(
                        &actual.weights.archive,
                        tensor,
                        id as usize,
                        actual.cfg.embd,
                        actual.cfg.vocab,
                        row,
                    )
                    .unwrap();
                }
                let emit = pos + n - 1 == next_observed;
                let (out, _) = if count > 0 {
                    full.prefill(&embeddings, pos, n, emit, false).unwrap()
                } else {
                    full.run(&embeddings, pos, emit, false).unwrap()
                };
                pos += n;
                if !emit {
                    continue;
                }
                let want = &expected[corpus][observation];
                assert_eq!(out.len(), want.len());
                assert!(out.iter().all(|x| x.is_finite()));
                assert!(
                    out.iter()
                        .zip(want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "complete ordered block/serial vectors must agree bit for bit"
                );
                let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
                std::fs::write(folder.join(format!("actual-count-{count}-tile-{tile}-fused-{fused}-corpus-{corpus}-position-{}.f32",pos-1)),bytes).unwrap();
                observation += 1;
                let argmax = |xs: &[f32]| {
                    xs.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0
                };
                assert_eq!(
                    argmax(&out),
                    argmax(want),
                    "count={count} fused={fused} pos={}",
                    pos - 1
                );
                let logp = |xs: &[f32]| {
                    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let z = max + xs.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln();
                    xs.iter().map(|&x| x as f64 - z).collect::<Vec<_>>()
                };
                let p = logp(want);
                let q = logp(&out);
                let kl = p
                    .iter()
                    .zip(&q)
                    .map(|(&p, &q)| p.exp() * (p - q))
                    .sum::<f64>()
                    .abs();
                let target = ids.get(pos).copied().unwrap_or(ids[pos - 1]) as usize;
                let nll = (p[target] - q[target]).abs();
                assert!(
                    kl < 1e-5 && nll < 1e-3,
                    "count={count} fused={fused} pos={} KL={kl} NLL={nll}",
                    pos - 1
                );
                worst_kl = worst_kl.max(kl);
                worst_nll = worst_nll.max(nll);
            }
            actual.gpu_full = Some(full);
        }
        let mut i = 0;
        for p in prompts {
            for s in samples {
                for _ in 0..2 {
                    assert_eq!(
                        actual.generate(p, 32, s, None).unwrap().0,
                        texts[i],
                        "count={count} tile={tile} fused={fused} sample={i}"
                    );
                }
                i += 1;
            }
        }
        let mut closed = |_event: StreamEvent| -> Result<()> {
            Err(BitNetError::Inference("fixture client disconnected".into()))
        };
        assert!(actual
            .generate(prompts[1], 128, samples[0], Some(&mut closed))
            .is_err());
        if count > 0 {
            crate::cancel::clear_inference_cancel();
            let before = crate::perf::snapshot().gpu_prefill_blocks;
            let interrupt = std::thread::spawn(move || {
                let deadline = Instant::now() + std::time::Duration::from_secs(10);
                while crate::perf::snapshot().gpu_prefill_blocks <= before {
                    assert!(Instant::now() < deadline, "block prefill never started");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                crate::cancel::request_inference_cancel();
            });
            let cancelled = actual.generate(&corpus[0], 32, samples[0], None);
            interrupt.join().unwrap();
            crate::cancel::clear_inference_cancel();
            assert!(cancelled.unwrap_err().to_string().contains("cancelled"));
        }
        assert_eq!(
            actual.generate(prompts[0], 32, samples[0], None).unwrap().0,
            texts[0]
        );
        let metrics = actual.weights.moe_metrics.as_ref().unwrap();
        let fused_calls: u64 = metrics
            .layers
            .iter()
            .map(|l| l.fused_gpu_ffns.load(std::sync::atomic::Ordering::Relaxed))
            .sum();
        assert_eq!(fused_calls > 0, fused == 1);
        println!("Real GPT block count={count} tile={tile} fused={fused} completed fused FFNs={fused_calls}; logits/seed/penalty/prefix/cancellation exact");
    }
    println!(
        "GPT_ORDERED_FASTPATH_ACTUAL_DONE worst KL={worst_kl:.3e}, absolute target NLL delta={worst_nll:.3e}"
    );
}
