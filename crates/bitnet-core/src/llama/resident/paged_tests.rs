//! Optional real-GGUF CUDA page arithmetic, sharing, ownership and lifetime fixtures.
use super::*;
use std::sync::Arc;

fn saved(runtime: &Resident, length: usize) -> SavedPrefix {
    let api = runtime.snapshots.as_ref().unwrap();
    let context = unsafe { (api.create)(runtime.context as *mut c_void, length as u32) } as usize;
    assert_ne!(context, 0);
    SavedPrefix {
        context,
        destroy: api.destroy,
    }
}
fn restore(runtime: &mut Resident, snapshot: &SavedPrefix, length: usize) -> i32 {
    unsafe {
        (runtime.snapshots.as_ref().unwrap().restore)(
            runtime.context as *mut c_void,
            snapshot.context as *const c_void,
            length as u32,
        )
    }
}
fn exact(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(a.to_bits(), b.to_bits(), "{label} logit {index}: {a}/{b}");
    }
}

#[test]
fn optional_native_paged_shared_prefix_cow_graph_block_split_and_reclaim() {
    if std::env::var("RBITNET_CUDA_PAGES_TEST").as_deref() != Ok("1") {
        return;
    }
    std::env::set_var("RBITNET_MAX_SEQ", "256");
    std::env::set_var("RBITNET_CUDA_PREFILL", "1");
    std::env::set_var("RBITNET_CUDA_PREFILL_TF32X3", "0");
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let rt = crate::backend::CudaRuntime::try_load().expect("CUDA required");
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
    let prompt = "<|begin_of_text|>Le robot explore la bibliothèque et découvre un jardin calme. "
        .repeat(20);
    let encoded = tokenizer.encode(prompt, false).unwrap();
    let ids = encoded.get_ids();
    assert!(ids.len() > 100);
    let prefix = &ids[..33]; // share a full page and one partially written page
    let split = std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap_or_else(|_| "0".into());
    for graphs in ["0", "1"] {
        for split in [split.as_str()] {
            std::env::set_var("RBITNET_CUDA_RESIDENT_GRAPH", graphs);
            std::env::set_var("RBITNET_CUDA_SPLIT_KV", split);
            let mut dense = Resident::new_with_pages(&model, None, None).unwrap();
            for count in [1usize, 4, 8] {
                let before = rt.managed_memory_stats().unwrap();
                let mut root = Resident::new_with_pages(&model, Some(24), None).unwrap();
                assert_eq!(
                    root.split_layers,
                    if split == "1" {
                        model.cfg.n_layer as u64
                    } else {
                        0
                    },
                    "actual paged attention variant, including Windows DLL environment separation"
                );
                assert_eq!(
                    root.split_layers, dense.split_layers,
                    "dense/page oracle has matching actual arithmetic"
                );
                assert_eq!(root.page_stats().unwrap().unwrap().allocated_pages, 0);
                let a = root.prefill(&model, prefix, 0, false).unwrap().0;
                let b = dense.prefill(&model, prefix, 0, false).unwrap().0;
                exact(&a, &b, "initial prefix");
                let snapshot = saved(&root, 33);
                let mut peers = Vec::new();
                for _ in 1..count {
                    peers.push(Resident::new_with_pages(&model, Some(24), Some(&root)).unwrap());
                }
                for peer in &mut peers {
                    assert_eq!(restore(peer, &snapshot, 33), 0);
                }
                let initial = root.page_stats().unwrap().unwrap();
                assert_eq!(initial.allocated_pages, 2);
                assert_eq!(initial.referenced_pages, 2);
                let table_bytes =
                    (model.cfg.max_seq.div_ceil(32) * 2 * std::mem::size_of::<usize>() * count)
                        as u64;
                assert_eq!(
                    rt.managed_memory_stats().unwrap().categories[1] - before.categories[1],
                    2 * initial.bytes_per_page + table_bytes,
                    "physical pages are charged once across contexts/snapshots"
                );
                for (branch, runtime) in std::iter::once(&mut root)
                    .chain(peers.iter_mut())
                    .enumerate()
                {
                    assert_eq!(restore(runtime, &snapshot, 33), 0);
                    dense.prefill(&model, prefix, 0, false).unwrap();
                    let inputs: Vec<u32> =
                        (0..37).map(|i| ids[33 + (branch * 7 + i) % 60]).collect();
                    // Check the block verifier itself on the same dense/native arithmetic.
                    match (
                        runtime.verify(&model, &inputs[..7], 33, false).unwrap(),
                        dense.verify(&model, &inputs[..7], 33, false).unwrap(),
                    ) {
                        (Verified::Logits(a, n), Verified::Logits(b, m)) => {
                            assert_eq!(n, m);
                            exact(&a, &b, "paged verifier");
                        }
                        _ => panic!("full verifier logits required"),
                    }
                    runtime.truncate(33).unwrap();
                    dense.truncate(33).unwrap();
                    for (i, &token) in inputs.iter().enumerate() {
                        let a = runtime.forward(&model, token, 33 + i, true).unwrap();
                        let b = dense.forward(&model, token, 33 + i, true).unwrap();
                        exact(&a, &b, "divergent branch");
                    }
                    assert_eq!(runtime.page_stats().unwrap().unwrap().active_pages, 3);
                }
                let final_stats = root.page_stats().unwrap().unwrap();
                assert_eq!(final_stats.allocated_pages, 2 + 2 * count as u64);
                assert!(final_stats.cow_pages >= count as u64);
                assert_eq!(final_stats.refusals, 0);
                // One sibling rolls back into a shared full page. Its COW must leave
                // both the original prefix snapshot and other continuations intact.
                let runtime = peers.last_mut().unwrap_or(&mut root);
                assert_eq!(restore(runtime, &snapshot, 31), 0);
                dense.prefill(&model, &ids[..31], 0, false).unwrap();
                let a = runtime.forward(&model, ids[31], 31, true).unwrap();
                let b = dense.forward(&model, ids[31], 31, true).unwrap();
                exact(&a, &b, "short partial rollback");
                assert_eq!(restore(runtime, &snapshot, 33), 0);
                dense.prefill(&model, prefix, 0, false).unwrap();
                let a = runtime.forward(&model, ids[33], 33, true).unwrap();
                let b = dense.forward(&model, ids[33], 33, true).unwrap();
                exact(&a, &b, "immutable shared snapshot");
                for runtime in std::iter::once(&mut root).chain(peers.iter_mut()) {
                    runtime.truncate(0).unwrap();
                }
                root.trim_pages().unwrap();
                assert_eq!(root.page_stats().unwrap().unwrap().allocated_pages, 2);
                drop(snapshot);
                root.trim_pages().unwrap();
                assert_eq!(root.page_stats().unwrap().unwrap().allocated_pages, 0);
                assert_eq!(
                    rt.managed_memory_stats().unwrap().categories[1] - before.categories[1],
                    table_bytes
                );
                drop(peers);
                drop(root);
                assert_eq!(
                    rt.managed_memory_stats().unwrap().categories[1],
                    before.categories[1]
                );
                println!("PAGED_NATIVE graphs={graphs} split={split} sequences={count} pages={} bytes/page={} COW={} exact=true",final_stats.allocated_pages,final_stats.bytes_per_page,final_stats.cow_pages);
            }
            // Two full pages fit; COW of the partial shared page must fail safely.
            let mut bounded = Resident::new_with_pages(&model, Some(2), None).unwrap();
            bounded.prefill(&model, prefix, 0, false).unwrap();
            let snapshot = saved(&bounded, 33);
            assert!(bounded.forward(&model, ids[33], 33, true).is_err());
            let stats = bounded.page_stats().unwrap().unwrap();
            assert_eq!(stats.allocated_pages, 2);
            assert_eq!(stats.tokens, 33);
            assert!(stats.refusals > 0);
            assert_eq!(restore(&mut bounded, &snapshot, 33), 0);
            drop(snapshot);
            dense.prefill(&model, prefix, 0, false).unwrap();
            exact(
                &bounded.forward(&model, ids[33], 33, true).unwrap(),
                &dense.forward(&model, ids[33], 33, true).unwrap(),
                "retry after snapshot release",
            );
            // Independent pools, even for the same weights, cannot import pages.
            let foreign = saved(&bounded, 34);
            let mut unrelated = Resident::new_with_pages(&model, Some(2), None).unwrap();
            assert_ne!(restore(&mut unrelated, &foreign, 34), 0);
            assert!(
                Resident::new_with_pages(&model, Some(3), Some(&bounded)).is_none(),
                "peer limits must agree"
            );
            assert!(Resident::new_with_pages(&model, Some(0), None).is_none());
            drop(foreign);
            drop(unrelated);
            drop(bounded);
        }
    }
}

#[test]
fn optional_native_paged_owner_variants_and_snapshot_outlives_root() {
    if std::env::var("RBITNET_CUDA_PAGES_TEST").as_deref() != Ok("1") {
        return;
    }
    for (k, v) in [
        ("RBITNET_MAX_SEQ", "256"),
        ("RBITNET_CUDA_PREFILL", "1"),
        ("RBITNET_CUDA_PREFILL_TF32X3", "0"),
        ("RBITNET_CUDA_RESIDENT_GRAPH", "1"),
    ] {
        std::env::set_var(k, v);
    }
    let split = std::env::var("RBITNET_CUDA_SPLIT_KV").unwrap_or_else(|_| "0".into());
    std::env::remove_var("RBITNET_CUDA_KV_PAGE_LIMIT");
    let rt = crate::backend::CudaRuntime::try_load().unwrap();
    let archive = Arc::new(
        crate::gguf::GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let model = LlamaModel::from_gguf_arc_for_backend(
        Arc::clone(&archive),
        crate::backend::BackendKind::Cuda,
    )
    .unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::env::var("RBITNET_TOKENIZER").unwrap()).unwrap();
    let encoded = tokenizer
        .encode(
            "Le robot lit un livre dans un jardin calme. ".repeat(20),
            false,
        )
        .unwrap();
    let ids = encoded.get_ids();
    assert!(ids.len() > 34);
    let prefix = &ids[..33];
    let before = rt.managed_memory_stats().unwrap().categories[1];
    let mut dense = Resident::new_with_pages(&model, None, None).unwrap();
    dense.prefill(&model, prefix, 0, false).unwrap();
    let dense_snapshot = saved(&dense, 33);
    let mut dense_other = Resident::new_with_pages(&model, None, None).unwrap();
    assert_ne!(
        restore(&mut dense_other, &dense_snapshot, 33),
        0,
        "dense checkpoints have a context generation owner"
    );
    dense_other.prefill(&model, prefix, 0, false).unwrap();
    exact(
        &dense.forward(&model, ids[33], 33, true).unwrap(),
        &dense_other.forward(&model, ids[33], 33, true).unwrap(),
        "dense recovery after foreign checkpoint refusal",
    );
    drop(dense_other);
    drop(dense_snapshot);
    let mut root = Resident::new_with_pages(&model, Some(2), None).unwrap();
    root.prefill(&model, prefix, 0, false).unwrap();
    let snapshot = saved(&root, 33);
    let initial = root.page_stats().unwrap().unwrap();
    std::env::set_var(
        "RBITNET_CUDA_SPLIT_KV",
        if split == "1" { "0" } else { "1" },
    );
    assert!(
        Resident::new_with_pages(&model, Some(2), Some(&root)).is_none(),
        "a shared pool must use the same attention reduction variant"
    );
    std::env::set_var("RBITNET_CUDA_SPLIT_KV", &split);
    let mut peer = Resident::new_with_pages(&model, Some(2), Some(&root)).unwrap();
    assert_eq!(peer.split_layers, dense.split_layers);
    let lib = crate::ggml::load_cuda_quant_library().unwrap();
    type Configure = unsafe extern "C" fn(*mut c_void, u32) -> i32;
    let configure = unsafe {
        *lib.get::<Configure>(b"rbitnet_cuda_llama_configure_tensor_prefill\0")
            .unwrap()
    };
    assert_ne!(
        unsafe { configure(peer.context as *mut c_void, 1) },
        0,
        "sharing a populated pool cannot change TF32 arithmetic"
    );
    assert_ne!(
        unsafe { configure(root.context as *mut c_void, 0) },
        0,
        "populated contexts cannot be reconfigured"
    );
    assert_eq!(restore(&mut peer, &snapshot, 33), 0);
    // Identical GGUF bytes held by another device weight owner still cannot
    // inject state into this pool, whose model descriptors are frozen.
    let foreign_model =
        LlamaModel::from_gguf_arc_for_backend(archive, crate::backend::BackendKind::Cuda).unwrap();
    assert!(Resident::new_with_pages(&foreign_model, Some(2), Some(&root)).is_none());
    drop(foreign_model);
    drop(root);
    drop(snapshot);
    dense.prefill(&model, prefix, 0, false).unwrap();
    exact(
        &peer.forward(&model, ids[33], 33, true).unwrap(),
        &dense.forward(&model, ids[33], 33, true).unwrap(),
        "peer survives root and snapshot release",
    );
    assert_eq!(peer.page_stats().unwrap().unwrap().allocated_pages, 2);
    let surviving = saved(&peer, 34);
    drop(peer);
    drop(dense);
    assert_eq!(
        rt.managed_memory_stats().unwrap().categories[1] - before,
        2 * initial.bytes_per_page,
        "snapshot alone retains exactly its physical pages"
    );
    drop(surviving);
    assert_eq!(rt.managed_memory_stats().unwrap().categories[1], before);
    println!("PAGED_OWNER dense foreign/attention variant/TF32 variant/weight owner refusals, peer and snapshot lifetime, one-charge reclamation exact");
}
