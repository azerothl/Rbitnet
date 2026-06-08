//! Prefix cache metrics: radix longest-prefix hit increments `prefix_cache_hits`.

use bitnet_core::perf::snapshot;
use bitnet_core::prefix_kv::{PrefixKvBlockCache, PrefixKvKey, PrefixKvScope};

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
    let after = snapshot().prefix_cache_hits;
    assert!(after > before, "prefix_cache_hits should increase on radix hit");
}
