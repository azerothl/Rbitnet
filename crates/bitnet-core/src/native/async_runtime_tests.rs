//! Opt-in actual GGUF policy parity; never uses a stub or a reference engine.
use super::*;
#[test]
fn real_moe_async_cache_preserves_logits_generations_prefix_and_model_lifetime() {
    if std::env::var("RBITNET_MOE_POLICY_TEST").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("RBITNET_MOE_POLICY_GGUF").unwrap();
    let tokenizer = std::env::var("RBITNET_MOE_POLICY_TOKENIZER").unwrap();
    let family = match std::env::var("RBITNET_MOE_POLICY_FAMILY").as_deref() {
        Ok("gptoss") => Family::GptOss,
        Ok("mla") => Family::Mla,
        _ => panic!("policy fixture family required"),
    };
    let family_name = family.name();
    let cache = std::env::var("RBITNET_MOE_POLICY_CACHE").unwrap_or_else(|_| "8192".into());
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    std::env::set_var("RBITNET_MAX_SEQ", "2048");
    std::env::set_var("RBITNET_MOE_CACHE_MB", &cache);
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_CUDA_GPT_FULL", "0");
    std::env::set_var("RBITNET_CUDA_MLA_FULL", "0");
    std::env::set_var("RBITNET_REQUIRE_GPT_FULL", "0");
    std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "0");
    std::env::set_var("RBITNET_MOE_EXECUTION", "cache");
    let arena_requested = crate::backend::expert_arena::from_env().unwrap();
    std::env::set_var("RBITNET_MOE_ARENA", "0");
    std::env::set_var("RBITNET_MOE_ASYNC", "0");
    let mut reference = Runtime::load(
        Arc::clone(&archive),
        Path::new(&tokenizer),
        BackendKind::Cuda,
        family,
    )
    .unwrap();
    let corpus=["Paris est la capitale de la France. Un robot visite une bibliothèque et lit des livres. ".repeat(40),
        "fn somme(values: &[i32]) -> i32 { values.iter().sum() } // vérifier les cas vides et les valeurs négatives\n".repeat(40)];
    let inputs: Vec<Vec<u32>> = corpus
        .iter()
        .map(|p| {
            reference
                .tokenizer
                .encode_ids(p, true)
                .unwrap()
                .into_iter()
                .take(160)
                .collect()
        })
        .collect();
    assert!(inputs.iter().all(|ids| ids.len() == 160));
    let observed = [0, 1, 7, 31, 63, 127, 145, 146, 147, 148, 149, 159];
    let mut expected = Vec::new();
    for ids in &inputs {
        let mut outputs = Vec::new();
        for (pos, &token) in ids.iter().enumerate() {
            let (out, _) = reference
                .forward(token, pos, observed.contains(&pos), false)
                .unwrap();
            if observed.contains(&pos) {
                outputs.push(out);
            }
        }
        expected.push(outputs);
    }
    let prompts = if family == Family::GptOss {
        ["<|start|>user<|message|>Quelle est la capitale de la France ? Réponds en un mot.<|end|><|start|>assistant<|channel|>final<|message|>",
        "<|start|>user<|message|>Écris une fonction Python qui additionne deux nombres.<|end|><|start|>assistant<|channel|>final<|message|>"]
    } else {
        [
            "Quelle est la capitale de la France ? Réponds en un mot.",
            "Écris une fonction Python qui additionne deux nombres.",
        ]
    };
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
    for prompt in prompts {
        for sample in samples {
            texts.push(reference.generate(prompt, 32, sample, None).unwrap().0);
        }
    }
    drop(reference);
    let mut worst_kl = 0.0f64;
    let mut worst_nll = 0.0f64;
    for policy in ["off", "demand", "previous-pass"] {
        std::env::set_var(
            "RBITNET_MOE_ARENA",
            if arena_requested && policy != "off" {
                "1"
            } else {
                "0"
            },
        );
        std::env::set_var("RBITNET_MOE_EXECUTION", "cache");
        std::env::set_var("RBITNET_MOE_ASYNC", if policy == "off" { "0" } else { "1" });
        std::env::set_var(
            "RBITNET_MOE_PREFETCH",
            if policy == "previous-pass" {
                "previous-pass"
            } else {
                "off"
            },
        );
        std::env::set_var("RBITNET_MOE_PINNED_SLOTS", "2");
        std::env::set_var(
            "RBITNET_CUDA_GPT_FULL",
            if family == Family::GptOss { "1" } else { "0" },
        );
        std::env::set_var(
            "RBITNET_CUDA_MLA_FULL",
            if family == Family::Mla { "1" } else { "0" },
        );
        std::env::set_var(
            "RBITNET_REQUIRE_GPT_FULL",
            if family == Family::GptOss { "1" } else { "0" },
        );
        std::env::set_var(
            "RBITNET_REQUIRE_MLA_FULL",
            if family == Family::Mla { "1" } else { "0" },
        );
        std::env::set_var("RBITNET_CUDA_SPLIT_KV", "1");
        std::env::set_var("RBITNET_CUDA_GPT_FULL_GRAPH", "1");
        std::env::set_var("RBITNET_CUDA_MLA_FULL_GRAPH", "1");
        std::env::set_var("RBITNET_PREFIX_KV", "1");
        let mut actual = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            family,
        )
        .unwrap();
        let weak = Arc::downgrade(actual.weights.moe_metrics.as_ref().unwrap());
        let metric_id = actual.weights.moe_metrics.as_ref().unwrap().id;
        assert!(actual.weights.expert_cache.is_some());
        for (corpus, ids) in inputs.iter().enumerate() {
            let mut observation = 0;
            for (pos, &token) in ids.iter().enumerate() {
                let (got, _) = actual
                    .forward(token, pos, observed.contains(&pos), false)
                    .unwrap();
                if !observed.contains(&pos) {
                    continue;
                }
                let want = &expected[corpus][observation];
                observation += 1;
                let argmax = |xs: &[f32]| {
                    xs.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0
                };
                assert_eq!(
                    argmax(&got),
                    argmax(want),
                    "family={family_name} policy={policy} pos={pos}"
                );
                let logp = |xs: &[f32]| {
                    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let z = xs.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln() + max;
                    xs.iter().map(|&x| x as f64 - z).collect::<Vec<_>>()
                };
                let p = logp(want);
                let q = logp(&got);
                let kl = p
                    .iter()
                    .zip(&q)
                    .map(|(&p, &q)| p.exp() * (p - q))
                    .sum::<f64>()
                    .abs();
                let target = ids.get(pos + 1).copied().unwrap_or(token) as usize;
                let nll = (p[target] - q[target]).abs();
                assert!(
                    kl < 1e-5 && nll < 1e-3,
                    "family={family_name} policy={policy} pos={pos} KL={kl} NLL={nll}"
                );
                worst_kl = worst_kl.max(kl);
                worst_nll = worst_nll.max(nll);
            }
        }
        let mut n = 0;
        for prompt in prompts {
            for sample in samples {
                for repeat in 0..2 {
                    assert_eq!(
                        actual.generate(prompt, 32, sample, None).unwrap().0,
                        texts[n],
                        "family={family_name} policy={policy} sample={n} repeat={repeat}"
                    );
                }
                n += 1;
            }
        }
        let metrics = actual.weights.moe_metrics.as_ref().unwrap();
        use std::sync::atomic::Ordering;
        let cpu: u64 = metrics
            .layers
            .iter()
            .map(|l| l.fallback_ffns.load(Ordering::Relaxed))
            .sum();
        let gpu: u64 = metrics
            .layers
            .iter()
            .map(|l| l.gpu_ffns.load(Ordering::Relaxed))
            .sum();
        assert!(
            gpu > 0,
            "cache modes must exercise completed GPU expert FFNs"
        );
        assert_eq!(
            metrics.async_enabled.load(Ordering::Relaxed),
            u64::from(policy != "off")
        );
        assert_eq!(metrics.async_failed.load(Ordering::Relaxed), 0);
        if policy != "off" {
            assert_eq!(metrics.pinned_slots.load(Ordering::Relaxed), 2);
            assert!(
                metrics.async_pool_bytes.load(Ordering::Relaxed)
                    <= cache.parse::<u64>().unwrap() * 1048576
            );
            assert!(
                metrics
                    .layers
                    .iter()
                    .map(|l| l.copy_dma_ns.load(Ordering::Relaxed))
                    .sum::<u64>()
                    > 0
            );
        }
        let predicted: u64 = metrics
            .layers
            .iter()
            .map(|l| l.prefetch_requested.load(Ordering::Relaxed))
            .sum();
        if policy != "previous-pass" {
            assert_eq!(predicted, 0);
        }
        if policy == "previous-pass"
            && std::env::var("RBITNET_MOE_REQUIRE_PREFETCH").as_deref() == Ok("1")
        {
            assert!(
                predicted > 0,
                "the explicitly capacity-limited case must execute actual prefetch copies"
            );
        }
        eprintln!("MoE async actual prefetch copies family={family_name} policy={policy} cache={cache}: {predicted}");
        eprintln!("MoE async actual family={family_name} policy={policy} cache={cache}: CPU FFNs {cpu}, GPU FFNs {gpu}, weights {} MiB",actual.weights.resident_bytes/1048576);
        drop(actual);
        assert!(weak.upgrade().is_none());
        assert!(!crate::native::moe_metrics::prometheus_text()
            .contains(&format!("model_id=\"{metric_id}\"")));
    }
    eprintln!(
        "MoE async real worst KL={worst_kl:.3e}, worst absolute target NLL delta={worst_nll:.3e}"
    );
}
