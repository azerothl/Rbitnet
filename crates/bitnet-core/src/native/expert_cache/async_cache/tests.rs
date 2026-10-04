use super::*;

#[test]
fn optional_async_pool_uses_one_budget_protects_ready_and_pending_and_drains_unload() {
    if std::env::var("RBITNET_CUDA_ASYNC_TEST").as_deref() != Ok("1") {
        return;
    }
    let archive = Arc::new(
        GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let rt = CudaRuntime::try_load().expect("CUDA required");
    let before = rt.managed_memory_stats().unwrap();
    let bytes: [usize; 3] = ["gate", "up", "down"].map(|p| {
        let t = archive
            .tensor_by_name(&format!("blk.0.ffn_{p}_exps.weight"))
            .unwrap();
        crate::ggml::ggml_row_size(t.ggml_type, t.dimensions[0]).unwrap() * t.dimensions[1] as usize
    });
    let group_bytes = bytes.iter().sum::<usize>();
    for (budget, slots) in [(group_bytes - 1, 2), (2 * group_bytes, 3)] {
        let mut invalid = ExpertCache::new(Arc::clone(&archive), Arc::clone(&rt), budget);
        assert!(invalid.enable_async(slots, false).is_err());
        drop(invalid);
        assert_eq!(
            rt.managed_memory_stats().unwrap().categories[4],
            before.categories[4]
        );
    }
    {
        // LFU would prefer a just-prefetched selected entry as victim unless
        // pending hits are completed and pinned before the first demand miss.
        let mut adversarial =
            ExpertCache::new(Arc::clone(&archive), Arc::clone(&rt), 2 * group_bytes);
        adversarial.enable_async(2, false).unwrap();
        adversarial.policy = Policy::Lfu;
        let old = adversarial.acquire(0, 0).unwrap().unwrap();
        let expected: Vec<_> = old.matrices.iter().map(|m| m.device_address()).collect();
        drop(old);
        drop(adversarial.acquire(0, 1).unwrap().unwrap());
        for _ in 0..3 {
            drop(adversarial.acquire(0, 1).unwrap().unwrap());
        }
        let mut state = adversarial.async_state.take().unwrap();
        assert!(state.stage(&mut adversarial, (0, 2), true).unwrap());
        adversarial.async_state = Some(state);
        let groups = adversarial.acquire_selected(0, &[3, 2]).unwrap().unwrap();
        assert_eq!(
            groups[1]
                .matrices
                .iter()
                .map(|m| m.device_address())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(adversarial.entries.contains_key(&(0, 2)));
        assert!(!adversarial.entries.contains_key(&(0, 1)));
        drop(groups);
        drop(adversarial);
        assert_eq!(
            rt.managed_memory_stats().unwrap().categories[4],
            before.categories[4]
        );
    }
    let model = super::super::super::moe_metrics::Model::new(
        "async-pool-fixture".into(),
        "gpt-oss".into(),
        2,
    );
    let mut cache = ExpertCache::new(Arc::clone(&archive), Arc::clone(&rt), 2 * group_bytes);
    cache.enable_async(2, true).unwrap();
    cache.set_metrics(Arc::clone(&model));
    assert_eq!(
        model.async_pool_bytes.load(Ordering::Relaxed),
        2 * group_bytes as u64
    );
    assert_eq!(
        model.pinned_bytes.load(Ordering::Relaxed),
        2 * group_bytes as u64
    );
    assert_eq!(model.pinned_slots.load(Ordering::Relaxed), 2);
    assert_eq!(model.async_enabled.load(Ordering::Relaxed), 1);
    assert_eq!(model.async_failed.load(Ordering::Relaxed), 0);
    assert_eq!(cache.bytes, 2 * group_bytes);
    assert_eq!(
        rt.managed_memory_stats().unwrap().categories[4],
        before.categories[4] + 2 * group_bytes as u64
    );
    cache.begin_sequence(2);
    cache.begin_pass(0);
    let first = cache.acquire(0, 0).unwrap().unwrap();
    let addresses: Vec<_> = first.matrices.iter().map(|m| m.device_address()).collect();
    let mut state = cache.async_state.take().unwrap();
    assert!(state.stage(&mut cache, (0, 2), true).unwrap());
    assert!(
        !cache.entries.contains_key(&(0, 2)),
        "even a finished upload stays unpublished until completed"
    );
    assert_eq!(state.pending.len(), 1);
    cache.async_state = Some(state);
    // Counter reattachment must not double a pending gauge or the physical pool.
    cache.set_metrics(Arc::clone(&model));
    assert_eq!(model.layers[0].pending_experts.load(Ordering::Relaxed), 1);
    assert_eq!(
        model.layers[0].pending_bytes.load(Ordering::Relaxed),
        group_bytes as u64
    );
    assert_eq!(
        model.async_pool_bytes.load(Ordering::Relaxed),
        2 * group_bytes as u64
    );
    // Materialize the selected pending hit before protecting the full selection.
    // The first allocation is externally leased throughout both acquisitions.
    let selected = cache.acquire_selected(0, &[2, 0]).unwrap().unwrap();
    assert!(Arc::ptr_eq(&selected[1], &first));
    assert_eq!(
        selected[1]
            .matrices
            .iter()
            .map(|m| m.device_address())
            .collect::<Vec<_>>(),
        addresses
    );
    assert!(cache.acquire(0, 1).unwrap().is_none());
    assert_eq!(cache.bytes, 2 * group_bytes);
    let reused: Vec<_> = selected[0]
        .matrices
        .iter()
        .map(|m| m.device_address())
        .collect();
    drop(selected);
    let mut state = cache.async_state.take().unwrap();
    assert!(state.stage(&mut cache, (0, 1), true).unwrap());
    assert!(!cache.entries.contains_key(&(0, 1)));
    cache.async_state = Some(state);
    let replacement = cache.acquire(0, 1).unwrap().unwrap();
    assert_eq!(
        replacement
            .matrices
            .iter()
            .map(|m| m.device_address())
            .collect::<Vec<_>>(),
        reused
    );
    for (projection, matrix) in ["gate", "up", "down"]
        .into_iter()
        .zip(&replacement.matrices)
    {
        let t = archive
            .tensor_by_name(&format!("blk.0.ffn_{projection}_exps.weight"))
            .unwrap();
        let size = matrix.bytes();
        assert_eq!(
            matrix.host_payload(),
            &archive.tensor_payload(t).unwrap()[size..2 * size]
        );
    }
    assert_eq!(model.layers[0].prefetch_used.load(Ordering::Relaxed), 2);
    assert_eq!(model.layers[0].pending_experts.load(Ordering::Relaxed), 0);
    drop(replacement);
    // Leave an unused transfer pending during unload. The matrix/pinned/event
    // owners must survive until that DMA finishes, including early cancellation.
    let mut state = cache.async_state.take().unwrap();
    assert!(state.stage(&mut cache, (1, 3), true).unwrap());
    cache.async_state = Some(state);
    cache.trace_route(1, &[4], group_bytes);
    assert_eq!(model.layers[1].prefetch_unused.load(Ordering::Relaxed), 1);
    assert_eq!(
        model.layers[1]
            .prefetch_wasted_bytes
            .load(Ordering::Relaxed),
        group_bytes as u64
    );
    drop(cache);
    assert_eq!(model.async_pool_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(model.pinned_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(model.pinned_slots.load(Ordering::Relaxed), 0);
    assert_eq!(model.async_enabled.load(Ordering::Relaxed), 0);
    assert_eq!(model.layers[0].ready_experts.load(Ordering::Relaxed), 0);
    assert_eq!(model.layers[1].pending_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(
        rt.managed_memory_stats().unwrap().categories[4],
        before.categories[4] + group_bytes as u64,
        "only the external active lease survives model cache unload"
    );
    drop(first);
    assert_eq!(
        rt.managed_memory_stats().unwrap().categories[4],
        before.categories[4]
    );
}
