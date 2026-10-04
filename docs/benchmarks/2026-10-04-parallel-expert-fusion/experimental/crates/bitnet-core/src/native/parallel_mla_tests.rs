//! Opt-in actual GLM-4.7-Flash sequences, separate from synthetic numerical oracles.
#[test]
fn actual_mla_parallel_fusion_teacher_forcing_greedy_seed_penalty_and_reset() {
    use super::*;
    if std::env::var("RBITNET_MLA_FULL_TEST").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_MLA_TEST_GGUF").unwrap();
    let tokenizer = std::env::var("RBITNET_MLA_TEST_TOKENIZER").unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    let corpus=["Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),"fn somme(values: &[i32]) -> i32 { values.iter().sum() } // vérifier les cas vides et les valeurs négatives\n".repeat(40)];
    let prompts=["[gMASK]<sop><|user|>\nQuelle est la capitale de la France ? Réponds en un mot.<|assistant|>\n","[gMASK]<sop><|user|>\nÉcris une fonction Python qui additionne deux nombres.<|assistant|>\n"];
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
    let observed = [0, 1, 7, 15, 31, 63, 127, 128, 255, 256, 271, 272];
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
        "RBITNET_CUDA_MLA_FULL",
        "RBITNET_REQUIRE_MLA_FULL",
        "RBITNET_CUDA_MLA_FULL_GRAPH",
        "RBITNET_CUDA_SPLIT_KV",
        "RBITNET_MOE_CACHE_MB",
        "RBITNET_PREFIX_KV",
    ]
    .into_iter()
    .map(|k| (k.to_owned(), std::env::var_os(k)))
    .collect());
    std::env::set_var("RBITNET_MOE_CACHE_MB", "8192");
    std::env::set_var("RBITNET_CUDA_MOE_FUSED", "0");
    std::env::set_var("RBITNET_REQUIRE_FUSED_MOE", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_MLA_FULL", "0");
    std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "0");
    let mut reference = Runtime::load(
        Arc::clone(&archive),
        Path::new(&tokenizer),
        BackendKind::Cuda,
        Family::Mla,
    )
    .unwrap();
    eprintln!(
        "MLA reference resident weights {} MiB; resident routed layers {}/{}",
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
        std::env::set_var("RBITNET_PREFIX_KV", prefix);
        std::env::set_var("RBITNET_CUDA_MOE_FUSED", "2");
        std::env::set_var("RBITNET_REQUIRE_FUSED_MOE", "1");
        std::env::set_var("RBITNET_CUDA_MLA_FULL", "1");
        std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "1");
        std::env::set_var("RBITNET_CUDA_MLA_FULL_GRAPH", graphs);
        std::env::set_var("RBITNET_CUDA_SPLIT_KV", split);
        let mut actual = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::Mla,
        )
        .unwrap();
        assert!(actual.gpu_mla.is_some());
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
                if prefix == "1" {
                    let warm = actual.generate(prompt, 32, *sampling, None).unwrap().0;
                    assert_eq!(
                        warm, texts[n],
                        "restored prefix graphs {graphs} split {split} sample {n}"
                    );
                }
                n += 1;
            }
        }
        eprintln!(
            "MLA real teacher-forcing + six generations passed graphs={graphs}, split={split}, prefix={prefix}"
        );
        drop(actual);
    }
    eprintln!("MLA real worst KL={worst_kl:.3e}, worst absolute target NLL delta={worst_nll:.3e}");
}

#[test]
fn opt_in_real_mla_cpu_routed_fallback_preserves_logits_sampling_and_prefix() {
    use super::*;
    if std::env::var("RBITNET_MLA_FALLBACK_TEST").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_MLA_TEST_GGUF").unwrap();
    let tokenizer = std::env::var("RBITNET_MLA_TEST_TOKENIZER").unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    struct Env(Vec<(&'static str, Option<std::ffi::OsString>)>);
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
        "RBITNET_CUDA_MLA_FULL",
        "RBITNET_REQUIRE_MLA_FULL",
        "RBITNET_CUDA_MLA_FULL_GRAPH",
        "RBITNET_CUDA_SPLIT_KV",
        "RBITNET_MOE_CACHE_MB",
        "RBITNET_PREFIX_KV",
        "RBITNET_MAX_SEQ",
    ]
    .into_iter()
    .map(|k| (k, std::env::var_os(k)))
    .collect());
    std::env::set_var("RBITNET_MOE_CACHE_MB", "16");
    std::env::set_var("RBITNET_MAX_SEQ", "128");
    std::env::set_var("RBITNET_CUDA_MLA_FULL", "0");
    std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "0");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    let mut reference = Runtime::load(
        Arc::clone(&archive),
        Path::new(&tokenizer),
        BackendKind::Cuda,
        Family::Mla,
    )
    .unwrap();
    let corpus =
        "Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. "
            .repeat(10);
    let ids: Vec<_> = reference
        .tokenizer
        .encode_ids(&corpus, true)
        .unwrap()
        .into_iter()
        .take(64)
        .collect();
    assert_eq!(ids.len(), 64);
    let observed = [0, 1, 7, 15, 31, 63];
    let mut expected = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let (logits, _) = reference
            .forward(id, pos, observed.contains(&pos), false)
            .unwrap();
        if observed.contains(&pos) {
            expected.push(logits);
        }
    }
    let prompt="[gMASK]<sop><|user|>\nQuelle est la capitale de la France ? Réponds en un mot.<|assistant|>\n";
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
    let texts: Vec<_> = samples
        .iter()
        .map(|&s| reference.generate(prompt, 8, s, None).unwrap().0)
        .collect();
    drop(reference);
    std::env::set_var("RBITNET_CUDA_MLA_FULL", "1");
    std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "1");
    std::env::set_var("RBITNET_CUDA_MLA_FULL_GRAPH", "1");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "1");
    std::env::set_var("RBITNET_PREFIX_KV", "1");
    let mut actual = Runtime::load(
        archive,
        Path::new(&tokenizer),
        BackendKind::Cuda,
        Family::Mla,
    )
    .unwrap();
    assert!(actual.gpu_mla.is_some());
    let before = crate::perf::snapshot();
    let mut n = 0;
    let mut max_kl = 0.0f64;
    let mut max_nll = 0.0f64;
    for (pos, &id) in ids.iter().enumerate() {
        let (got, _) = actual
            .forward(id, pos, observed.contains(&pos), false)
            .unwrap();
        if !observed.contains(&pos) {
            continue;
        }
        let expected = &expected[n];
        n += 1;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(
            argmax(&got),
            argmax(expected),
            "CPU routed fallback position {pos}"
        );
        for (&a, &b) in got.iter().zip(expected) {
            assert!(
                (a - b).abs() <= 0.003 * (1.0 + b.abs()),
                "fallback pos {pos}: {a} vs {b}"
            );
        }
        let logprob = |v: &[f32]| {
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z = v.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln() + max;
            v.iter().map(|&x| x as f64 - z).collect::<Vec<_>>()
        };
        let p = logprob(expected);
        let q = logprob(&got);
        let kl = p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| p.exp() * (p - q))
            .sum::<f64>()
            .max(0.0);
        let next = ids.get(pos + 1).copied().unwrap_or(id) as usize;
        let nll = (p[next] - q[next]).abs();
        assert!(
            kl <= 1e-5 && nll <= 1e-3,
            "fallback pos {pos}: KL {kl}, NLL {nll}"
        );
        max_kl = max_kl.max(kl);
        max_nll = max_nll.max(nll);
    }
    for (i, &s) in samples.iter().enumerate() {
        assert_eq!(actual.generate(prompt, 8, s, None).unwrap().0, texts[i]);
        assert_eq!(actual.generate(prompt, 8, s, None).unwrap().0, texts[i]);
    }
    let after = crate::perf::snapshot();
    assert!(after.native_moe_fallback_layers > before.native_moe_fallback_layers);
    assert_eq!(
        after.native_moe_resident_layers,
        before.native_moe_resident_layers
    );
    assert!(after.gpu_mla_full_tokens > before.gpu_mla_full_tokens);
    assert!(after.prefix_cache_hits > before.prefix_cache_hits);
    eprintln!("GLM actual CPU routed fallback: six logit positions, three generations plus three restored prefixes; KL {max_kl:.3e}, NLL {max_nll:.3e}");
}
