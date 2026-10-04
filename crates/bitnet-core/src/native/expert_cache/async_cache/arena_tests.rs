use super::*;

#[test]
fn optional_actual_expert_arena_cache_refill_poison_and_external_lease_preserve_budget() {
    if std::env::var("RBITNET_EXPERT_ARENA_TEST").as_deref() != Ok("1") {
        return;
    }
    assert!(crate::backend::expert_arena::from_env().unwrap());
    let archive = Arc::new(
        GgufArchive::mmap_path(std::path::Path::new(
            &std::env::var("RBITNET_TEST_GGUF").unwrap(),
        ))
        .unwrap(),
    );
    let rt = CudaRuntime::try_load().expect("real CUDA required");
    let before = rt.managed_memory_stats().unwrap();
    let projections = ["gate", "up", "down"];
    let spans = projections.map(|projection| {
        archive
            .tensors
            .iter()
            .filter(|t| {
                t.name.starts_with("blk.")
                    && t.name.ends_with(&format!("ffn_{projection}_exps.weight"))
            })
            .map(|t| {
                crate::ggml::ggml_row_size(t.ggml_type, t.dimensions[0]).unwrap()
                    * t.dimensions[1] as usize
            })
            .max()
            .unwrap()
    });
    let group_bytes = spans.iter().map(|span| (span + 255) & !255).sum::<usize>();
    let mut cache = ExpertCache::new(Arc::clone(&archive), Arc::clone(&rt), 2 * group_bytes);
    let routed_layers = archive
        .tensors
        .iter()
        .filter(|t| t.name.ends_with("ffn_gate_exps.weight"))
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let first_layer = routed_layers[0];
    let layers = routed_layers.iter().max().unwrap() + 1;
    let metrics = super::super::super::moe_metrics::Model::new(
        "arena-lifetime-fixture".into(),
        "moe".into(),
        layers,
    );
    cache.set_metrics(Arc::clone(&metrics));
    cache.enable_async(2, false).unwrap();
    let live = rt.managed_memory_stats().unwrap();
    assert_eq!(live.allocations - before.allocations, 1);
    assert_eq!(
        live.categories[4] - before.categories[4],
        (2 * group_bytes) as u64
    );
    assert_eq!(cache.bytes, 2 * group_bytes);
    let first = cache.acquire(first_layer, 0).unwrap().unwrap();
    let addresses = first
        .matrices
        .iter()
        .map(|m| m.device_address())
        .collect::<Vec<_>>();
    drop(cache.acquire(first_layer, 1).unwrap().unwrap());
    assert!(cache.acquire(first_layer, 2).unwrap().is_some());
    assert_eq!(
        first
            .matrices
            .iter()
            .map(|m| m.device_address())
            .collect::<Vec<_>>(),
        addresses
    );
    // Exercise every layer's original Q4/Q6/MXFP4 metadata through the same
    // unleased slot while the other group's views remain externally leased.
    for &layer in &routed_layers {
        let before_refill = rt.managed_memory_stats().unwrap().allocations;
        let group = cache.acquire(layer, 1).unwrap().unwrap();
        assert_eq!(
            rt.managed_memory_stats().unwrap().allocations,
            before_refill,
            "cache refills must never allocate device memory"
        );
        for (projection, matrix) in projections.into_iter().zip(&group.matrices) {
            let tensor = archive
                .tensor_by_name(&format!("blk.{layer}.ffn_{projection}_exps.weight"))
                .unwrap();
            let bytes = matrix.bytes();
            assert_eq!(matrix.ggml_type(), tensor.ggml_type);
            assert_eq!(
                matrix.host_payload(),
                &archive.tensor_payload(tensor).unwrap()[bytes..2 * bytes]
            );
            let x = (0..matrix.in_cols())
                .map(|i| ((i * 13 % 71) as f32 - 35.) / 256.)
                .collect::<Vec<_>>();
            let expected = crate::ggml::QuantMatvecKernel::cpu_parallel()
                .matvec_payload(
                    tensor.ggml_type,
                    matrix.host_payload(),
                    &x,
                    matrix.in_cols(),
                    matrix.out_rows(),
                )
                .unwrap();
            // The low-level quant launcher has thread-local scratch. A scoped
            // worker releases that scratch before checking arena-only lifetime.
            let actual = std::thread::scope(|scope| {
                scope.spawn(|| matrix.matvec(&x).unwrap()).join().unwrap()
            });
            for (a, b) in actual.into_iter().zip(expected) {
                assert!(
                    (a - b).abs() <= 3e-4 * (1. + b.abs()),
                    "arena quant {a}/{b}"
                );
            }
        }
        drop(group);
    }
    let mut state = cache.async_state.take().unwrap();
    assert!(state.stage(&mut cache, (first_layer, 3), true).unwrap());
    state.poison(&mut cache); // pending copies drain before their views disappear
    assert_eq!(
        cache.bytes,
        2 * group_bytes,
        "poison must retain the whole physical arena charge"
    );
    assert_eq!(
        metrics.async_pool_bytes.load(Ordering::Relaxed),
        (2 * group_bytes) as u64
    );
    assert_eq!(metrics.async_failed.load(Ordering::Relaxed), 1);
    assert!(metrics
        .layers
        .iter()
        .all(|l| l.pending_bytes.load(Ordering::Relaxed) == 0));
    cache.async_state = Some(state);
    let refused = match cache.acquire(first_layer, 3) {
        Err(error) => error,
        Ok(_) => panic!("a poisoned asynchronous cache must refuse further acquisition"),
    };
    assert!(refused
        .to_string()
        .contains("async cache failed; reload the model before retrying"));
    drop(cache);
    assert_eq!(
        rt.managed_memory_stats().unwrap().categories[4],
        live.categories[4],
        "external views keep the complete physical allocation alive after cache unload"
    );
    for matrix in &first.matrices {
        let actual = std::thread::scope(|scope| {
            scope
                .spawn(|| matrix.matvec(&vec![0.; matrix.in_cols()]).unwrap())
                .join()
                .unwrap()
        });
        assert!(actual.iter().all(|x| *x == 0.));
    }
    drop(first);
    let after = rt.managed_memory_stats().unwrap();
    assert_eq!(after.live, before.live);
    assert_eq!(after.categories, before.categories);
    println!("EXPERT_ARENA_CACHE_DONE layers={} physical_allocations=1 refill_allocations=0 poison_and_last_lease=true",routed_layers.len());
}
