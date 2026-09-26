//! Prefix KV block cache scaffolding (RadixAttention-style [2312.07104]).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Snapshot of dense per-layer KV rows after prefill.
#[derive(Debug, Clone)]
pub struct DenseKvSnapshot {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub token_count: usize,
}

/// Paged KV block table snapshot (per-layer logical → physical mapping).
#[derive(Debug, Clone)]
pub struct PagedKvSnapshot {
    pub block_phys: Vec<Vec<usize>>,
    pub token_count: usize,
}

/// Payload restored on a radix hit (dense rows or paged block table).
#[derive(Debug, Clone)]
pub enum PrefixKvSnap {
    Dense(DenseKvSnapshot),
    Paged(PagedKvSnapshot),
}

#[derive(Debug, Default)]
pub struct PrefixKvBlockCache {
    inner: HashMap<PrefixKvKey, Arc<[usize]>>,
    radix: RadixPrefixCache,
}

impl PrefixKvBlockCache {
    pub fn from_env() -> Self {
        Self {
            inner: HashMap::new(),
            radix: RadixPrefixCache::from_env(),
        }
    }

    pub fn insert(&mut self, key: PrefixKvKey, blocks: Vec<usize>) {
        self.inner.insert(key, blocks.into());
    }

    pub fn get(&self, key: &PrefixKvKey) -> Option<&Arc<[usize]>> {
        self.inner.get(key)
    }

    pub fn insert_blocks(&mut self, prefix_hash: u64, blocks: Vec<usize>) {
        self.insert(PrefixKvKey::legacy(prefix_hash), blocks);
    }

    pub fn get_blocks(&self, prefix_hash: u64) -> Option<&Arc<[usize]>> {
        self.get(&PrefixKvKey::legacy(prefix_hash))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn insert_tokens(&mut self, key: PrefixKvKey, token_ids: &[u32], blocks: Vec<usize>) {
        self.radix.insert(key, token_ids, blocks, None);
    }

    /// Insert radix path + optional dense/paged KV snapshot at the leaf (agent prefix reuse).
    pub fn insert_tokens_with_snap(
        &mut self,
        key: PrefixKvKey,
        token_ids: &[u32],
        blocks: Vec<usize>,
        snap: Option<PrefixKvSnap>,
    ) {
        self.radix.insert(key, token_ids, blocks, snap);
    }

    pub fn longest_token_prefix(
        &mut self,
        scope: &PrefixKvScope,
        token_ids: &[u32],
    ) -> Option<PrefixKvMatch> {
        self.radix.longest_prefix(scope, token_ids)
    }

    pub fn radix_entries(&self) -> usize {
        self.radix.entry_count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKvKey {
    pub model_id: String,
    pub tokenizer_id: String,
    pub template_id: String,
    pub kv_format: String,
    pub rope_id: String,
    pub prefix_hash: u64,
}

impl PrefixKvKey {
    pub fn new(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        template_id: impl Into<String>,
        token_ids: &[u32],
    ) -> Self {
        Self::with_invalidation(
            model_id,
            tokenizer_id,
            template_id,
            "f32",
            "default",
            token_ids,
        )
    }

    pub fn with_invalidation(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        template_id: impl Into<String>,
        kv_format: impl Into<String>,
        rope_id: impl Into<String>,
        token_ids: &[u32],
    ) -> Self {
        Self {
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            template_id: template_id.into(),
            kv_format: kv_format.into(),
            rope_id: rope_id.into(),
            prefix_hash: hash_prefix_tokens(token_ids),
        }
    }

    fn legacy(prefix_hash: u64) -> Self {
        Self {
            model_id: "legacy".into(),
            tokenizer_id: "legacy".into(),
            template_id: "legacy".into(),
            kv_format: "f32".into(),
            rope_id: "default".into(),
            prefix_hash,
        }
    }

    pub fn scope(&self) -> PrefixKvScope {
        PrefixKvScope {
            model_id: self.model_id.clone(),
            tokenizer_id: self.tokenizer_id.clone(),
            template_id: self.template_id.clone(),
            kv_format: self.kv_format.clone(),
            rope_id: self.rope_id.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKvScope {
    pub model_id: String,
    pub tokenizer_id: String,
    pub template_id: String,
    pub kv_format: String,
    pub rope_id: String,
}

#[derive(Debug, Clone)]
pub struct PrefixKvMatch {
    pub matched_tokens: usize,
    pub blocks: Arc<[usize]>,
    pub bytes_saved: usize,
    pub snap: Option<PrefixKvSnap>,
}

#[derive(Debug)]
pub struct RadixPrefixCache {
    root: RadixNode,
    max_leaves: usize,
    /// LRU of (scope, token path) leaf keys for eviction.
    lru: Vec<(PrefixKvScope, Vec<u32>)>,
    clock: u64,
}

impl Default for RadixPrefixCache {
    fn default() -> Self {
        Self {
            root: RadixNode::default(),
            max_leaves: 256,
            lru: Vec::new(),
            clock: 0,
        }
    }
}

#[derive(Debug, Default)]
struct RadixNode {
    scope_blocks: HashMap<PrefixKvScope, Arc<[usize]>>,
    scope_snaps: HashMap<PrefixKvScope, PrefixKvSnap>,
    children: HashMap<u32, RadixNode>,
    last_used: u64,
}

impl RadixPrefixCache {
    pub fn from_env() -> Self {
        let max_leaves = std::env::var("RBITNET_PREFIX_KV_RADIX_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256);
        Self {
            root: RadixNode::default(),
            max_leaves,
            lru: Vec::new(),
            clock: 0,
        }
    }

    pub fn entry_count(&self) -> usize {
        self.lru.len()
    }

    pub fn insert(
        &mut self,
        key: PrefixKvKey,
        token_ids: &[u32],
        blocks: Vec<usize>,
        snap: Option<PrefixKvSnap>,
    ) {
        let scope = key.scope();
        let mut node = &mut self.root;
        for &tok in token_ids {
            node = node.children.entry(tok).or_default();
        }
        self.clock = self.clock.saturating_add(1);
        node.last_used = self.clock;
        node.scope_blocks.insert(scope.clone(), blocks.into());
        if let Some(s) = snap {
            node.scope_snaps.insert(scope.clone(), s);
        }

        let leaf = (scope, token_ids.to_vec());
        if let Some(pos) = self.lru.iter().position(|e| e == &leaf) {
            self.lru.remove(pos);
        }
        self.lru.push(leaf);
        while self.lru.len() > self.max_leaves {
            if let Some((evict_scope, path)) = self.lru.first().cloned() {
                self.lru.remove(0);
                Self::evict_leaf(&mut self.root, &evict_scope, &path);
            } else {
                break;
            }
        }
    }

    fn evict_leaf(root: &mut RadixNode, scope: &PrefixKvScope, path: &[u32]) {
        if path.is_empty() {
            root.scope_blocks.remove(scope);
            root.scope_snaps.remove(scope);
            return;
        }
        fn walk(node: &mut RadixNode, scope: &PrefixKvScope, path: &[u32]) -> bool {
            if path.is_empty() {
                node.scope_blocks.remove(scope);
                node.scope_snaps.remove(scope);
                return node.children.is_empty()
                    && node.scope_blocks.is_empty()
                    && node.scope_snaps.is_empty();
            }
            let tok = path[0];
            let mut child_empty = false;
            if let Some(child) = node.children.get_mut(&tok) {
                child_empty = walk(child, scope, &path[1..]);
            }
            if child_empty {
                node.children.remove(&tok);
            }
            node.children.is_empty() && node.scope_blocks.is_empty() && node.scope_snaps.is_empty()
        }
        walk(root, scope, path);
    }

    pub fn longest_prefix(
        &mut self,
        scope: &PrefixKvScope,
        token_ids: &[u32],
    ) -> Option<PrefixKvMatch> {
        let mut node = &self.root;
        let mut best: Option<(usize, Arc<[usize]>, Option<PrefixKvSnap>)> = None;
        for (idx, &tok) in token_ids.iter().enumerate() {
            let Some(next) = node.children.get(&tok) else {
                break;
            };
            node = next;
            if let Some(blocks) = node.scope_blocks.get(scope) {
                best = Some((
                    idx + 1,
                    Arc::clone(blocks),
                    node.scope_snaps.get(scope).cloned(),
                ));
            }
        }
        let (matched_tokens, blocks, snap) = best?;
        if matched_tokens == 0 {
            return None;
        }
        // Touch LRU for this leaf path.
        let leaf_path = token_ids[..matched_tokens].to_vec();
        let leaf = (scope.clone(), leaf_path);
        if let Some(pos) = self.lru.iter().position(|e| e == &leaf) {
            let e = self.lru.remove(pos);
            self.lru.push(e);
        }
        self.clock = self.clock.saturating_add(1);
        let bytes_saved = matched_tokens.saturating_mul(64);
        Some(PrefixKvMatch {
            matched_tokens,
            blocks,
            bytes_saved,
            snap,
        })
    }
}

pub fn hash_prefix_tokens(ids: &[u32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_hash_stable_per_process() {
        let x = hash_prefix_tokens(&[10, 20, 30]);
        assert_eq!(x, hash_prefix_tokens(&[10, 20, 30]));
    }

    #[test]
    fn cache_roundtrip() {
        let mut c = PrefixKvBlockCache::default();
        c.insert_blocks(42, vec![0, 1, 2]);
        assert_eq!(c.get_blocks(42).unwrap().as_ref(), [0, 1, 2]);
    }

    #[test]
    fn key_separates_model_template() {
        let a = PrefixKvKey::new("m1", "tok", "raw", &[1, 2, 3]);
        let b = PrefixKvKey::new("m1", "tok", "chatml", &[1, 2, 3]);
        assert_ne!(a, b);
    }

    #[test]
    fn radix_returns_longest_scoped_prefix() {
        let mut c = PrefixKvBlockCache::default();
        let key = PrefixKvKey::with_invalidation("m1", "tok", "raw", "q8", "rope", &[1, 2]);
        let scope = key.scope();
        c.insert_tokens(key, &[1, 2], vec![9, 10]);
        let hit = c.longest_token_prefix(&scope, &[1, 2, 3]).unwrap();
        assert_eq!(hit.matched_tokens, 2);
        assert_eq!(hit.blocks.as_ref(), [9, 10]);
    }

    #[test]
    fn radix_lru_evicts_oldest_leaf() {
        let mut radix = RadixPrefixCache {
            max_leaves: 2,
            ..RadixPrefixCache::default()
        };
        let scope = PrefixKvScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            template_id: "x".into(),
            kv_format: "f32".into(),
            rope_id: "default".into(),
        };
        let k1 = PrefixKvKey {
            prefix_hash: 1,
            ..PrefixKvKey::with_invalidation("m", "t", "x", "f32", "default", &[1])
        };
        let k2 = PrefixKvKey {
            prefix_hash: 2,
            ..PrefixKvKey::with_invalidation("m", "t", "x", "f32", "default", &[2])
        };
        let k3 = PrefixKvKey {
            prefix_hash: 3,
            ..PrefixKvKey::with_invalidation("m", "t", "x", "f32", "default", &[3])
        };
        radix.insert(k1, &[1], vec![1], None);
        radix.insert(k2, &[2], vec![2], None);
        assert_eq!(radix.entry_count(), 2);
        radix.insert(k3, &[3], vec![3], None);
        assert_eq!(radix.entry_count(), 2);
        assert!(radix.longest_prefix(&scope, &[1]).is_none());
        assert!(radix.longest_prefix(&scope, &[2]).is_some());
        assert!(radix.longest_prefix(&scope, &[3]).is_some());
    }

    #[test]
    fn agent_style_prefix_hit_rate_after_warmup() {
        let mut cache = PrefixKvBlockCache::from_env();
        let scope = PrefixKvScope {
            model_id: "tiny".into(),
            tokenizer_id: "tok".into(),
            template_id: "chat".into(),
            kv_format: "f32".into(),
            rope_id: "default".into(),
        };
        // Shared system + tools prefix (agent pattern).
        let system: Vec<u32> = (1..=40).collect();
        let mut hits = 0u32;
        let mut total = 0u32;
        for i in 0..50u32 {
            let mut prompt = system.clone();
            prompt.push(1000 + i); // unique user turn
            if i == 0 {
                let key = PrefixKvKey::with_invalidation(
                    &scope.model_id,
                    &scope.tokenizer_id,
                    &scope.template_id,
                    &scope.kv_format,
                    &scope.rope_id,
                    &system,
                );
                cache.insert_tokens_with_snap(
                    key,
                    &system,
                    (0..system.len()).collect(),
                    Some(PrefixKvSnap::Dense(DenseKvSnapshot {
                        k: vec![vec![1.0]; 1],
                        v: vec![vec![1.0]; 1],
                        token_count: system.len(),
                    })),
                );
                total += 1;
                continue; // warm-up miss
            }
            total += 1;
            if let Some(m) = cache.longest_token_prefix(&scope, &prompt) {
                if m.matched_tokens >= system.len() && m.snap.is_some() {
                    hits += 1;
                }
            }
        }
        let rate = hits as f64 / (total - 1) as f64;
        assert!(
            rate >= 0.70,
            "agent prefix_hit rate {rate:.2} (hits={hits}/{}) below 70%",
            total - 1
        );
    }
}
