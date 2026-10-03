//! Actual GPT sequences, including the token147 router near-tie and immutable prefixes.
#[test]
fn opt_in_gpt_segmented_real_teacher_forcing_greedy_seed_penalty_reset_and_prefix() {
    use super::*;
    if std::env::var("RBITNET_GPT_SEGMENTED_TEST").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_GPT_TEST_GGUF").unwrap();
    let tokenizer = std::env::var("RBITNET_GPT_TEST_TOKENIZER").unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    let corpus=["Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),"fn somme(values: &[i32]) -> i32 { values.iter().sum() } // vérifier les cas vides et les valeurs négatives\n".repeat(40)];
    let prompts=["<|start|>user<|message|>Quelle est la capitale de la France ? Réponds en un mot.<|end|><|start|>assistant<|channel|>final<|message|>","<|start|>user<|message|>Écris une fonction Python qui additionne deux nombres.<|end|><|start|>assistant<|channel|>final<|message|>"];
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
    let observed = [
        0, 1, 7, 15, 31, 63, 127, 128, 145, 146, 147, 148, 149, 255, 256, 271, 272,
    ];
    // Keep all environment changes scoped even when an assertion unwinds.
    struct Env(Vec<(String, Option<std::ffi::OsString>)>);
    impl Drop for Env {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                if let Some(v) = v {
                    std::env::set_var(k, v);
                } else {
                    std::env::remove_var(k);
                }
            }
        }
    }
    let _env = Env([
        "RBITNET_CUDA_GPT_FULL",
        "RBITNET_REQUIRE_GPT_FULL",
        "RBITNET_CUDA_GPT_FULL_GRAPH",
        "RBITNET_CUDA_SPLIT_KV",
        "RBITNET_MOE_CACHE_MB",
        "RBITNET_CUDA_GPT_SEGMENTED",
        "RBITNET_PREFIX_KV",
        "RBITNET_MAX_SEQ",
        "RBITNET_MOE_EXECUTION",
    ]
    .into_iter()
    .map(|k| (k.to_owned(), std::env::var_os(k)))
    .collect());
    let cache = std::env::var("RBITNET_GPT_TEST_CACHE").unwrap_or_else(|_| "8192".into());
    assert!(matches!(cache.as_str(), "0" | "16" | "8192"));
    std::env::set_var("RBITNET_MOE_CACHE_MB", &cache);
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_MAX_SEQ", "2048");
    std::env::set_var("RBITNET_CUDA_GPT_SEGMENTED", "0");
    std::env::set_var("RBITNET_MOE_EXECUTION", "cache");
    std::env::set_var("RBITNET_CUDA_GPT_FULL", "0");
    std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "0");
    let mut reference = Runtime::load(
        Arc::clone(&archive),
        Path::new(&tokenizer),
        BackendKind::Cuda,
        Family::GptOss,
    )
    .unwrap();
    eprintln!(
        "GPT reference resident weights {} MiB; fixed expert layers {}/{}",
        reference.weights.resident_bytes / (1024 * 1024),
        reference.gpu_moe.iter().flatten().count(),
        reference.cfg.layers
    );
    let inputs: Vec<Vec<u32>> = corpus
        .iter()
        .map(|p| {
            reference
                .tokenizer
                .encode_ids(p, true)
                .unwrap()
                .into_iter()
                .take(273)
                .collect()
        })
        .collect();
    assert!(inputs.iter().all(|v| v.len() == 273));
    let mut expected = Vec::new();
    for ids in &inputs {
        let mut outputs = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let (logits, _) = reference
                .forward(id, pos, observed.contains(&pos), false)
                .unwrap();
            if observed.contains(&pos) {
                outputs.push(logits);
            }
        }
        expected.push(outputs);
    }
    let mut texts = Vec::new();
    for prompt in prompts {
        for sampling in &samples {
            texts.push(reference.generate(prompt, 32, *sampling, None).unwrap().0);
        }
    }
    drop(reference);
    let mut worst_kl = 0.0f64;
    let mut worst_nll = 0.0f64;
    for (graphs, split, prefix) in [
        ("0", "0", "0"),
        ("1", "0", "0"),
        ("1", "1", "0"),
        ("1", "1", "1"),
    ] {
        std::env::set_var("RBITNET_CUDA_GPT_SEGMENTED", "1");
        std::env::set_var("RBITNET_PREFIX_KV", prefix);
        std::env::set_var("RBITNET_CUDA_GPT_FULL", "1");
        std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "1");
        std::env::set_var("RBITNET_CUDA_GPT_FULL_GRAPH", graphs);
        std::env::set_var("RBITNET_CUDA_SPLIT_KV", split);
        let mut actual = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::GptOss,
        )
        .unwrap();
        assert!(actual
            .gpu_full
            .as_ref()
            .is_some_and(|full| full.is_segmented()));
        assert!(actual.gpu_full.as_ref().unwrap().supports_prefix());
        let counters_before = crate::perf::snapshot();
        for (case, ids) in inputs.iter().enumerate() {
            let mut n = 0;
            for (pos, &id) in ids.iter().enumerate() {
                let (got, _) = actual
                    .forward(id, pos, observed.contains(&pos), false)
                    .unwrap();
                if !observed.contains(&pos) {
                    continue;
                }
                let expected = &expected[case][n];
                n += 1;
                let argmax = |x: &[f32]| {
                    x.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                assert_eq!(
                    argmax(&got),
                    argmax(expected),
                    "corpus {case} position {pos} graphs {graphs} split {split}"
                );
                for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
                    assert!((a-b).abs()<=0.003*(1.0+b.abs()),"corpus={case} pos={pos} graphs={graphs} split={split} token={i}: {a} vs {b}");
                }
                let logprob = |x: &[f32]| {
                    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let total = x.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
                    x.iter().map(|&v| v as f64 - total).collect::<Vec<_>>()
                };
                let p = logprob(expected);
                let q = logprob(&got);
                let kl = p
                    .iter()
                    .zip(&q)
                    .map(|(&p, &q)| p.exp() * (p - q))
                    .sum::<f64>()
                    .max(0.0);
                let id = ids.get(pos + 1).copied().unwrap_or(ids[pos]) as usize;
                let nll = (p[id] - q[id]).abs();
                worst_kl = worst_kl.max(kl);
                worst_nll = worst_nll.max(nll);
                assert!(
                    kl <= 1e-5 && nll <= 1e-3,
                    "KL={kl} NLL_delta={nll}, corpus {case}, pos {pos}"
                );
            }
        }
        let mut n = 0;
        for prompt in prompts {
            for sampling in &samples {
                let got = actual.generate(prompt, 32, *sampling, None).unwrap().0;
                assert_eq!(got, texts[n], "graphs {graphs} split {split} sample {n}");
                n += 1;
            }
        }
        if prefix == "1" {
            let before = crate::perf::snapshot();
            let mut n = 0;
            for prompt in prompts {
                for sampling in &samples {
                    let got = actual.generate(prompt, 32, *sampling, None).unwrap().0;
                    assert_eq!(got, texts[n], "warm prefix sample {n}");
                    n += 1;
                }
            }
            let after = crate::perf::snapshot();
            assert!(after.prefix_cache_hits > before.prefix_cache_hits);
        }
        let counters_after = crate::perf::snapshot();
        assert!(counters_after.gpu_gpt_full_tokens > counters_before.gpu_gpt_full_tokens);
        if cache == "16" {
            assert!(
                counters_after.native_moe_fallback_layers
                    > counters_before.native_moe_fallback_layers
            );
            assert_eq!(
                counters_after.native_moe_resident_layers,
                counters_before.native_moe_resident_layers
            );
        } else {
            assert!(
                counters_after.native_moe_resident_layers
                    > counters_before.native_moe_resident_layers
            );
        }
        if cache == "0" && std::env::var("RBITNET_CUDA_DEVICE_BUDGET_MB").as_deref() == Ok("6144") {
            assert!(
                counters_after.native_moe_fallback_layers
                    > counters_before.native_moe_fallback_layers
            );
        }
        eprintln!("GPT-OSS segmented real teacher-forcing + six generations passed cache={cache}, graphs={graphs}, split={split}, prefix={prefix}");
        drop(actual);
    }
    eprintln!("GPT real worst KL={worst_kl:.3e}, worst absolute target NLL delta={worst_nll:.3e}");
}
