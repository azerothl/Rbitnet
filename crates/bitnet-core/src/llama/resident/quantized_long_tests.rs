//! Optional actual GPU tests against a predeclared bounded quality envelope.
use super::*;
use std::sync::Arc;

fn load() -> (
    LlamaModel,
    tokenizers::Tokenizer,
    Arc<crate::backend::CudaRuntime>,
) {
    let capacity = std::env::var("RBITNET_KV_TEST_CONTEXT").unwrap_or_else(|_| "2048".into());
    assert!(matches!(capacity.as_str(), "512" | "2048" | "8192"));
    std::env::set_var("RBITNET_MAX_SEQ", capacity);
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let rt = crate::backend::CudaRuntime::try_load().expect("actual CUDA required");
    let archive = Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let model =
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::env::var("RBITNET_TOKENIZER").unwrap()).unwrap();
    (model, tokenizer, rt)
}
fn fill(r: &mut Resident, m: &LlamaModel, ids: &[u32]) {
    for (chunk, part) in ids.chunks(16).enumerate() {
        r.prefill(m, part, chunk * 16, false).unwrap();
    }
}
fn log_probs(logits: &[f32]) -> Vec<f64> {
    assert!(logits.iter().all(|x| x.is_finite()));
    let maximum = logits
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
    let sum = logits
        .iter()
        .map(|&x| (x as f64 - maximum).exp())
        .sum::<f64>()
        .ln();
    logits.iter().map(|&x| x as f64 - maximum - sum).collect()
}
fn exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()),
        "{label}"
    );
}
#[test]
fn optional_quantized_long_llama_forced_quality_and_physical_bytes() {
    if std::env::var("RBITNET_CUDA_KV_TEST").as_deref() != Ok("1") {
        return;
    }
    let (model, tokenizer, rt) = load();
    let limits = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(std::env::var("RBITNET_KV_QUALITY_PLAN").unwrap()).unwrap(),
    )
    .unwrap();
    let mut all = Vec::new();
    for graph in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graph);
        for format in ["f16", "q8"] {
            let mut kl = Vec::new();
            let mut nll = Vec::new();
            let mut disagreements = 0;
            for content in ["Le robot cherche un livre sur les étoiles puis visite le jardin calme. ","fn checked_sum(values: &[i32]) -> Option<i32> { values.iter().try_fold(0, |a, &b| a.checked_add(b)) } "] {
                let encoded=tokenizer.encode(format!("<|begin_of_text|>{}",content.repeat(900)),false).unwrap();let ids=encoded.get_ids();assert!(ids.len()>model.cfg.max_seq);
                for prefix in [33usize,255,model.cfg.max_seq-19] {
                    std::env::set_var("RBITNET_CUDA_KV_FORMAT","f32");
                    let before=rt.managed_memory_stats().unwrap().categories[1];
                    let mut reference=Resident::new_with_pages(&model,None,None).unwrap();
                    let f32_bytes=rt.managed_memory_stats().unwrap().categories[1]-before;
                    assert_eq!(f32_bytes,(model.cfg.n_layer*model.cfg.n_kv*model.cfg.head_dim*model.cfg.max_seq*8)as u64);
                    std::env::set_var("RBITNET_CUDA_KV_FORMAT",format);
                    let before=rt.managed_memory_stats().unwrap().categories[1];
                    let mut quantized=Resident::new_with_pages(&model,None,None).unwrap();
                    let expected=quantized.kv_bytes_per_token*model.cfg.max_seq;
                    assert_eq!(rt.managed_memory_stats().unwrap().categories[1]-before,expected as u64,"true physical encoded K/V including Q8 scales");
                    assert!(expected<f32_bytes as usize);
                    fill(&mut reference,&model,&ids[..prefix]);fill(&mut quantized,&model,&ids[..prefix]);
                    for position in prefix..prefix+16 {
                        let a=reference.forward(&model,ids[position],position,true).unwrap();
                        let b=quantized.forward(&model,ids[position],position,true).unwrap();
                        let p=log_probs(&a);let q=log_probs(&b);
                        kl.push(p.iter().zip(&q).map(|(p,q)|p.exp()*(p-q)).sum::<f64>().max(0.));
                        nll.push((p[ids[position+1]as usize]-q[ids[position+1]as usize]).abs());
                        let argmax=|x:&[f32]|x.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
                        disagreements+=usize::from(argmax(&a)!=argmax(&b));
                    }
                }
            }
            let mean_kl = kl.iter().sum::<f64>() / kl.len() as f64;
            let worst_kl = kl.iter().copied().fold(0., f64::max);
            let mean_nll = nll.iter().sum::<f64>() / nll.len() as f64;
            let worst_nll = nll.iter().copied().fold(0., f64::max);
            let l = &limits["numerical_limits"][format];
            let row = serde_json::json!({"context_capacity":model.cfg.max_seq,"observed_prefixes":[33usize,255,model.cfg.max_seq-19],"format":format,"graph":graph,"split":std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap(),"positions":kl.len(),"mean_kl":mean_kl,"worst_kl":worst_kl,"mean_abs_target_nll_delta":mean_nll,"worst_abs_target_nll_delta":worst_nll,"argmax_disagreements":disagreements,"kl_samples":kl,"target_nll_samples":nll});
            println!("KV_QUALITY {row}");
            all.push(row);
            assert!(
                mean_kl <= l["observed_mean_target_kl_max"].as_f64().unwrap()
                    && worst_kl <= l["observed_worst_target_kl_max"].as_f64().unwrap()
            );
            assert!(
                mean_nll <= l["mean_abs_target_nll_delta_max"].as_f64().unwrap()
                    && worst_nll <= l["worst_abs_target_nll_delta_max"].as_f64().unwrap()
            );
        }
    }
    println!("ACTUAL_KV_QUALITY bounded forced Llama F16/Q8 logits, target NLL, disagreements and physical bytes passed: {} modes",all.len());
}
#[test]
fn optional_quantized_snapshots_truncate_and_invalid_values_refuse_then_recover() {
    if std::env::var("RBITNET_CUDA_KV_TEST").as_deref() != Ok("1") {
        return;
    }
    let (model, tokenizer, _) = load();
    let encoded = tokenizer
        .encode(
            "Le robot lit un livre dans un jardin calme. ".repeat(30),
            false,
        )
        .unwrap();
    let ids = encoded.get_ids();
    for format in ["f16", "q8"] {
        std::env::set_var("RBITNET_CUDA_KV_FORMAT", format);
        for paged in [false, true] {
            let mut r = Resident::new_with_pages(&model, paged.then_some(8), None).unwrap();
            fill(&mut r, &model, &ids[..33]);
            let api = r.snapshots.as_ref().unwrap();
            let snapshot = SavedPrefix {
                context: unsafe { (api.create)(r.context as *mut c_void, 33) } as usize,
                destroy: api.destroy,
            };
            assert_ne!(snapshot.context, 0);
            let a = r.forward(&model, ids[33], 33, true).unwrap();
            r.forward(&model, ids[34], 34, true).unwrap();
            assert_eq!(
                unsafe { (r.truncate.unwrap())(r.context as *mut c_void, 33) },
                0
            );
            exact(
                &a,
                &r.forward(&model, ids[33], 33, true).unwrap(),
                "truncation replay exact within format",
            );
            let restore = r.snapshots.as_ref().unwrap().restore;
            assert_eq!(
                unsafe {
                    restore(
                        r.context as *mut c_void,
                        snapshot.context as *const c_void,
                        33,
                    )
                },
                0
            );
            let invalid = vec![f32::NAN; model.cfg.n_embd];
            let mut logits = vec![0.; r.vocab];
            assert_ne!(
                unsafe {
                    (r.step)(
                        r.context as *mut c_void,
                        invalid.as_ptr(),
                        33,
                        1,
                        logits.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                },
                0,
                "invalid encoded vectors must refuse host output"
            );
            assert!(
                r.forward(&model, ids[33], 33, true).is_err(),
                "poisoned state cannot silently continue"
            );
            assert_eq!(
                unsafe {
                    restore(
                        r.context as *mut c_void,
                        snapshot.context as *const c_void,
                        33,
                    )
                },
                0
            );
            exact(
                &a,
                &r.forward(&model, ids[33], 33, true).unwrap(),
                "snapshot restores clean encoded payload and Q8 scales",
            );
            fill(&mut r, &model, &ids[..33]);
            exact(
                &a,
                &r.forward(&model, ids[33], 33, true).unwrap(),
                "reset replay within format",
            );
        }
    }
    println!("ACTUAL_KV_LIFETIME dense/page F16/Q8 snapshot, truncation, nonfinite refusal/poison/reset/restore passed");
}

/// Full teacher forcing over 1024 next-token targets in each of two fixed
/// synthetic prose/code corpora. This is distinct from sparse long-tail probes.
#[test]
fn optional_quantized_full_1024_targets_nll_and_perplexity() {
    if std::env::var("RBITNET_CUDA_KV_TEST").as_deref() != Ok("1") {
        return;
    }
    let (model, tokenizer, _) = load();
    assert!(model.cfg.max_seq >= 2048);
    std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", "1");
    let limits = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(std::env::var("RBITNET_KV_QUALITY_PLAN").unwrap()).unwrap(),
    )
    .unwrap();
    for (corpus,content) in ["Le robot cherche un livre sur les étoiles puis visite le jardin calme. ","fn checked_sum(values: &[i32]) -> Option<i32> { values.iter().try_fold(0, |a, &b| a.checked_add(b)) } "].iter().enumerate() {
        let encoded=tokenizer.encode(format!("<|begin_of_text|>{}",content.repeat(120)),false).unwrap();
        let ids=&encoded.get_ids()[..1025];let mut reference=Vec::new();
        for format in ["f32","f16","q8"] {
            std::env::set_var("RBITNET_CUDA_KV_FORMAT",format);
            let mut r=Resident::new_with_pages(&model,None,None).unwrap();
            let mut nll=Vec::new();
            for position in 0..1024 {
                let logits=r.forward(&model,ids[position],position,true).unwrap();
                nll.push(-log_probs(&logits)[ids[position+1]as usize]);
            }
            if format=="f32" {reference=nll.clone();}
            let delta:Vec<_>=nll.iter().zip(&reference).map(|(a,b)|(a-b).abs()).collect();
            let mean=nll.iter().sum::<f64>()/1024.;let ref_mean=reference.iter().sum::<f64>()/1024.;
            let mean_delta=delta.iter().sum::<f64>()/1024.;let worst=delta.iter().copied().fold(0.,f64::max);
            let row=serde_json::json!({"corpus":corpus,"synthetic_corpus":true,"targets":1024,"context_capacity":model.cfg.max_seq,"format":format,"graph":"1","split":std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap(),
                "mean_target_nll":mean,"perplexity":mean.exp(),"f32_mean_target_nll":ref_mean,"f32_perplexity":ref_mean.exp(),"perplexity_ratio":(mean-ref_mean).exp(),
                "mean_abs_target_nll_delta":mean_delta,"worst_abs_target_nll_delta":worst,"target_nll_samples":nll,"abs_target_nll_samples":delta});
            println!("KV_FULL_NLL {row}");
            if format!="f32" {
                assert!(mean_delta<=limits["numerical_limits"][format]["mean_abs_target_nll_delta_max"].as_f64().unwrap());
                assert!(worst<=limits["numerical_limits"][format]["worst_abs_target_nll_delta_max"].as_f64().unwrap());
            }
        }
    }
    println!("ACTUAL_KV_FULL_NLL two fixed synthetic corpora x 1024 imposed next-token targets x F32/F16/Q8 passed; this is not a natural-language evaluation suite");
}
