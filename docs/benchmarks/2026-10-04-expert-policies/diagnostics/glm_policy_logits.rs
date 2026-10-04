//! Private frozen-DLL diagnostic: identical token history, all logits and argmax.
use super::*;

fn top(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(b.cmp(&a)));
    ids.into_iter().take(8).map(|i| (i, logits[i])).collect()
}

fn vectors(
    rt: &mut Runtime,
    prompt: &str,
    want: Option<&[Vec<f32>]>,
    ids: &mut Vec<u32>,
    label: &str,
) -> Vec<Vec<f32>> {
    let tokens = rt.tokenizer.encode_ids(prompt, true).unwrap();
    assert_eq!(
        tokens.len(),
        127,
        "exact failed HTTP prompt must be reconstructed"
    );
    rt.weights
        .expert_cache
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .begin_sequence(tokens.len());
    let mut next = None;
    for (position, &token) in tokens.iter().enumerate() {
        (_, next) = rt
            .forward(token, position, position + 1 == tokens.len(), true)
            .unwrap();
    }
    let mut result = Vec::new();
    for step in 0..128 {
        let device_id = next.take().unwrap();
        let (logits, _) = rt.gpu_mla.as_mut().unwrap().end(true, false).unwrap();
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "{label} nonfinite step={step}"
        );
        let best = top(&logits);
        assert_eq!(
            device_id as usize, best[0].0,
            "{label} device/host argmax step={step}, top={best:?}"
        );
        if let Some(want) = want {
            if let Some(first) = logits
                .iter()
                .zip(&want[step])
                .position(|(a, b)| a.to_bits() != b.to_bits())
            {
                let worst = logits
                    .iter()
                    .zip(&want[step])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "GLM_POLICY_LOGIT_MISMATCH {}",
                    serde_json::json!({"label":label,"step":step,"first_logit":first,"got_bits":logits[first].to_bits(),"want_bits":want[step][first].to_bits(),"max_abs":worst,"got_top":best,"want_top":top(&want[step])})
                );
                panic!(
                    "identical weights, token history and arithmetic must preserve full logit bits"
                );
            }
            assert_eq!(device_id, ids[step]);
        } else {
            ids.push(device_id);
        }
        result.push(logits);
        if step + 1 < 128 {
            // Use the reference history even when diagnosing a candidate.
            (_, next) = rt
                .forward(ids[step], tokens.len() + step, true, true)
                .unwrap();
        }
    }
    eprintln!("GLM_POLICY_LOGITS_EXACT {label} positions=128");
    result
}

#[test]
fn optional_glm_policy_full_logits_repeated_history_exact() {
    if std::env::var("RBITNET_GLM_POLICY_LOGITS_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "2048");
    std::env::set_var("RBITNET_MOE_CACHE_MB", "8192");
    std::env::set_var("RBITNET_PREFIX_KV", "0");
    std::env::set_var("RBITNET_MOE_ASYNC", "0");
    std::env::set_var("RBITNET_MOE_PREFETCH", "off");
    std::env::set_var("RBITNET_MOE_EXECUTION", "cache");
    std::env::set_var("RBITNET_CUDA_MLA_FULL", "1");
    std::env::set_var("RBITNET_REQUIRE_MLA_FULL", "1");
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", "1");
    std::env::set_var("RBITNET_CUDA_MLA_FULL_GRAPH", "1");
    let gguf = std::env::var("RBITNET_MLA_TEST_GGUF").unwrap();
    let tokenizer = std::env::var("RBITNET_MLA_TEST_TOKENIZER").unwrap();
    let archive = Arc::new(GgufArchive::mmap_path(Path::new(&gguf)).unwrap());
    let mut system =
        "Tu es un assistant précis. Voici des notes communes à cette conversation.\n".to_owned();
    system.push_str(
        &(0..4)
            .map(|i| {
                format!("Note {i}: Les villes ont des bibliothèques, des jardins et des musées.")
            })
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let render = |question: &str| {
        format!("[gMASK]<sop><|system|>{system}<|user|>{question}<|assistant|></think>")
    };
    let story = render("Écris un récit de 150 mots sur un robot qui explore une bibliothèque.");
    let code=render("Écris une fonction Python qui calcule une moyenne en ignorant les valeurs None et explique le cas d’une liste vide.");
    let capital="[gMASK]<sop><|user|>Quelle est la capitale de la France ? Réponds en un mot.<|assistant|></think>";
    let mut expected: Option<Vec<Vec<f32>>> = None;
    let mut ids = Vec::new();
    for policy in ["lru", "lfu", "least-stale"] {
        std::env::set_var("RBITNET_MOE_CACHE_POLICY", policy);
        let mut rt = Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
            Family::Mla,
        )
        .unwrap();
        assert!(rt.gpu_mla.is_some());
        for cycle in 0..4 {
            let got = vectors(
                &mut rt,
                &story,
                expected.as_deref(),
                &mut ids,
                &format!("{policy}-cycle{cycle}"),
            );
            if expected.is_none() {
                let capture = std::env::var("RBITNET_GLM_FAILED_CAPTURE").unwrap();
                let frozen: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(capture).unwrap()).unwrap();
                let text = rt.tokenizer.decode_ids(&ids, false).unwrap();
                assert_eq!(
                    text,
                    frozen["rows"][0]["response"]["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap(),
                    "exact failed HTTP story must be reproduced"
                );
                expected = Some(got);
            }
            rt.generate(&code, 128, SamplingOptions::from_temperature(0.0), None)
                .unwrap();
            rt.generate(capital, 128, SamplingOptions::from_temperature(0.0), None)
                .unwrap();
        }
    }
    eprintln!("GLM_POLICY_FULL_LOGITS_DONE policies=3 cycles=4 positions=1536");
}
