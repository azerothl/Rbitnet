//! Prefix cache metrics: radix hit + `record_prefix_hit` increments counters.

use bitnet_core::perf::{record_prefix_hit, snapshot};
use bitnet_core::prefix_kv::{PrefixKvBlockCache, PrefixKvKey, PrefixKvScope};
use bitnet_core::prefix_kv_exec::{DenseKvSnapshot, PrefixKvExecutionCache};

#[test]
fn radix_longest_prefix_increments_hits() {
    let scope = PrefixKvScope {
        model_id: "test-model".into(),
        tokenizer_id: "tok".into(),
        template_id: "raw".into(),
        kv_format: "dense".into(),
        rope_id: "default".into(),
    };
    let mut cache = PrefixKvBlockCache::default();
    let key = PrefixKvKey::with_invalidation(
        &scope.model_id,
        &scope.tokenizer_id,
        &scope.template_id,
        &scope.kv_format,
        &scope.rope_id,
        &[1, 2, 3],
    );
    cache.insert_tokens(key, &[1, 2, 3], vec![10, 11]);
    let before = snapshot().prefix_cache_hits;
    let hit = cache
        .longest_token_prefix(&scope, &[1, 2, 3, 9])
        .expect("hit");
    assert!(hit.matched_tokens >= 3);
    // Runtime records after a successful KV restore; mirror that here.
    record_prefix_hit(hit.bytes_saved);
    let after = snapshot().prefix_cache_hits;
    assert!(after > before, "prefix_cache_hits should increase on radix hit");
}

#[test]
fn dense_prefix_kv_execution_cache_store_lookup() {
    std::env::set_var("RBITNET_PREFIX_KV", "1");
    std::env::set_var("RBITNET_PREFIX_KV_MAX_ENTRIES", "4");
    let mut cache = PrefixKvExecutionCache::from_env();
    assert!(cache.enabled());
    let key = PrefixKvKey::with_invalidation("m", "t", "raw", "dense", "default", &[7, 8]);
    let snap = DenseKvSnapshot {
        k: vec![vec![1.0, 2.0]],
        v: vec![vec![3.0, 4.0]],
        token_count: 2,
    };
    cache.store(key.clone(), snap.clone());
    let got = cache.lookup(&key).expect("lookup after store");
    assert_eq!(got.token_count, 2);
    assert_eq!(got.k, snap.k);
    assert_eq!(got.v, snap.v);
    std::env::remove_var("RBITNET_PREFIX_KV");
    std::env::remove_var("RBITNET_PREFIX_KV_MAX_ENTRIES");
}
