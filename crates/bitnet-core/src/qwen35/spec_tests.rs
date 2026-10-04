//! Actual-model exact all-position logits, greedy IDs and one-shot GDN rollback.
use super::*;

fn exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    assert!(
        a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()),
        "{label}"
    );
}
fn embeddings(rt: &Qwen35Runtime, ids: &[u32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(ids.len() * rt.cfg.n_embd);
    for &id in ids {
        values.extend(
            token_embedding_row(
                &rt.archive,
                &rt.tok_embd,
                id as usize,
                rt.cfg.n_embd,
                rt.cfg.n_vocab,
            )
            .unwrap(),
        );
    }
    values
}
fn fill(rt: &mut Qwen35Runtime, ids: &[u32]) {
    let archive = Arc::clone(&rt.archive);
    for (pos, &id) in ids.iter().enumerate() {
        rt.forward_one(id, pos, &archive, false).unwrap();
    }
}

#[test]
fn optional_actual_qwen_all_position_verify_and_gdn_rollback_exact() {
    if std::env::var("RBITNET_QWEN_SPEC_TEST").as_deref() != Ok("1") {
        return;
    }
    let archive = Arc::new(
        GgufArchive::mmap_path(Path::new(&std::env::var("RBITNET_QWEN_TEST_GGUF").unwrap()))
            .unwrap(),
    );
    let tokenizer = std::env::var("RBITNET_QWEN_TEST_TOKENIZER").unwrap();
    let device = crate::backend::CudaRuntime::try_load().unwrap();
    let mut observations = 0;
    for graphs in ["0", "1"] {
        std::env::set_var("RBITNET_CUDA_QWEN_FULL_GRAPH", graphs);
        std::env::set_var("RBITNET_CUDA_QWEN_PREFILL", "1");
        let mut reference = Qwen35Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
        )
        .unwrap();
        let before = device.managed_memory_stats().unwrap().categories[5];
        let reference_bytes = reference
            .gpu_full
            .as_mut()
            .unwrap()
            .spec_configure()
            .unwrap();
        assert_eq!(
            device.managed_memory_stats().unwrap().categories[5] - before,
            reference_bytes as u64
        );
        let mut verify = Qwen35Runtime::load(
            Arc::clone(&archive),
            Path::new(&tokenizer),
            BackendKind::Cuda,
        )
        .unwrap();
        let before = device.managed_memory_stats().unwrap().categories[5];
        let verify_bytes = verify.gpu_full.as_mut().unwrap().spec_configure().unwrap();
        assert_eq!(verify_bytes, reference_bytes);
        assert_eq!(
            device.managed_memory_stats().unwrap().categories[5] - before,
            verify_bytes as u64
        );
        assert!(
            verify.gpu_full.as_mut().unwrap().spec_configure().is_err(),
            "single preallocated workspace only"
        );
        assert!(
            verify.gpu_full.as_mut().unwrap().spec_save().is_err(),
            "empty uninitialized state cannot be saved"
        );
        for sentence in [
            "Paris est la capitale de la France. Les oiseaux visitent un jardin. ",
            "fn sum(values: &[i32]) -> i32 { values.iter().sum() }\n",
        ] {
            let ids = reference
                .tokenizer
                .encode_ids(&sentence.repeat(35), true)
                .unwrap();
            assert!(ids.len() > 270);
            for prefix in [1usize, 33, 129, 257] {
                for count in [1usize, 2, 4, 8, 9] {
                    fill(&mut reference, &ids[..prefix]);
                    fill(&mut verify, &ids[..prefix]);
                    exact(
                        &reference.gpu_full.as_ref().unwrap().spec_state(),
                        &verify.gpu_full.as_ref().unwrap().spec_state(),
                        "prefix GDN/conv",
                    );
                    let checkpoint = verify.gpu_full.as_mut().unwrap().spec_save().unwrap();
                    let foreign = reference.gpu_full.as_mut().unwrap().spec_save().unwrap();
                    assert_ne!(checkpoint, foreign);
                    assert!(
                        verify
                            .gpu_full
                            .as_mut()
                            .unwrap()
                            .spec_finish(foreign, true)
                            .is_err(),
                        "other owner nonce refused without state mutation"
                    );
                    reference
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_finish(foreign, false)
                        .unwrap();
                    let input = embeddings(&verify, &ids[prefix..prefix + count]);
                    let (actual, _) = verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_verify(&input, prefix, count, false)
                        .unwrap();
                    let mut expected = Vec::new();
                    for (i, &id) in ids[prefix..prefix + count].iter().enumerate() {
                        expected.extend(
                            reference
                                .forward_one(id, prefix + i, &archive, true)
                                .unwrap(),
                        );
                    }
                    exact(&actual, &expected, "all target position logits");
                    observations += count;
                    exact(
                        &reference.gpu_full.as_ref().unwrap().spec_state(),
                        &verify.gpu_full.as_ref().unwrap().spec_state(),
                        "verified final GDN/conv",
                    );
                    // Divergent speculative tail is rolled back before replaying a
                    // different accepted prefix; stale future attention KV is excluded.
                    verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_finish(checkpoint, true)
                        .unwrap();
                    assert!(
                        verify
                            .gpu_full
                            .as_mut()
                            .unwrap()
                            .spec_finish(checkpoint, true)
                            .is_err(),
                        "one-shot restore"
                    );
                    fill(&mut reference, &ids[..prefix]);
                    for i in 0..5 {
                        let id = ids[prefix + 17 + i];
                        let a = reference
                            .forward_one(id, prefix + i, &archive, true)
                            .unwrap();
                        let b = verify.forward_one(id, prefix + i, &archive, true).unwrap();
                        exact(&a, &b, "divergent successor after rollback");
                    }
                    exact(
                        &reference.gpu_full.as_ref().unwrap().spec_state(),
                        &verify.gpu_full.as_ref().unwrap().spec_state(),
                        "replayed divergent GDN/conv",
                    );
                    fill(&mut reference, &ids[..prefix]);
                    fill(&mut verify, &ids[..prefix]);
                    let nonce = verify.gpu_full.as_mut().unwrap().spec_save().unwrap();
                    let (_, actual_ids) = verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_verify(&input, prefix, count, true)
                        .unwrap();
                    for (i, &actual) in actual_ids.iter().enumerate() {
                        let e = reference
                            .forward_inner(ids[prefix + i], prefix + i, &archive, true, true)
                            .unwrap()
                            .1
                            .unwrap();
                        assert_eq!(actual, e, "position argmax");
                    }
                    verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_finish(nonce, true)
                        .unwrap();
                    verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_replay(&input, prefix, count)
                        .unwrap();
                    exact(
                        &reference.gpu_full.as_ref().unwrap().spec_state(),
                        &verify.gpu_full.as_ref().unwrap().spec_state(),
                        "headless ordered replay GDN/conv",
                    );
                    let nonce = verify.gpu_full.as_mut().unwrap().spec_save().unwrap();
                    verify
                        .gpu_full
                        .as_mut()
                        .unwrap()
                        .spec_finish(nonce, false)
                        .unwrap();
                    assert!(
                        verify
                            .gpu_full
                            .as_mut()
                            .unwrap()
                            .spec_finish(nonce, true)
                            .is_err(),
                        "discarded checkpoint refused"
                    );
                    let nonce = verify.gpu_full.as_mut().unwrap().spec_save().unwrap();
                    verify.forward_one(ids[0], 0, &archive, false).unwrap();
                    assert!(
                        verify
                            .gpu_full
                            .as_mut()
                            .unwrap()
                            .spec_finish(nonce, true)
                            .is_err(),
                        "new request reset invalidates old checkpoint"
                    );
                }
            }
        }
        let bad = vec![f32::NAN; verify.cfg.n_embd];
        assert!(verify
            .gpu_full
            .as_mut()
            .unwrap()
            .spec_verify(&bad, 1, 1, false)
            .is_err());
        assert!(verify
            .gpu_full
            .as_mut()
            .unwrap()
            .spec_verify(&[], 1, 0, false)
            .is_err());
        assert!(verify
            .gpu_full
            .as_mut()
            .unwrap()
            .spec_verify(&vec![0.0; verify.cfg.n_embd * 10], 1, 10, false)
            .is_err());
        eprintln!("QWEN_SPEC_VERIFY graphs={graphs} observations={observations} bounded workspace={verify_bytes} bytes; full logits, IDs, GDN/conv rollback/nonce/reset exact");
    }
}
