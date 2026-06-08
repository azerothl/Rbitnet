//! Prefix KV block cache scaffolding.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct PrefixKvBlockCache {
    inner: HashMap<PrefixKvKey, Arc<[usize]>>,
    radix: RadixPrefixCache,
}

impl PrefixKvBlockCache {
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
        self.radix.insert(key, token_ids, blocks);
    }

    pub fn longest_token_prefix(
        &self,
        scope: &PrefixKvScope,
        token_ids: &[u32],
    ) -> Option<PrefixKvMatch> {
        let hit = self.radix.longest_prefix(scope, token_ids);
        if let Some(ref m) = hit {
            crate::perf::record_prefix_cache_hit(m.bytes_saved);
        } else {
            crate::perf::record_prefix_cache_miss();
        }
        hit
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixKvMatch {
    pub matched_tokens: usize,
    pub blocks: Arc<[usize]>,
    pub bytes_saved: usize,
}

#[derive(Debug, Default)]
pub struct RadixPrefixCache {
    root: RadixNode,
}

#[derive(Debug, Default)]
struct RadixNode {
    scope_blocks: HashMap<PrefixKvScope, Arc<[usize]>>,
    children: HashMap<u32, RadixNode>,
}

impl RadixPrefixCache {
    pub fn insert(&mut self, key: PrefixKvKey, token_ids: &[u32], blocks: Vec<usize>) {
        let scope = key.scope();
        let mut node = &mut self.root;
        for &tok in token_ids {
            node = node.children.entry(tok).or_default();
        }
        node.scope_blocks.insert(scope, blocks.into());
    }

    pub fn longest_prefix(
        &self,
        scope: &PrefixKvScope,
        token_ids: &[u32],
    ) -> Option<PrefixKvMatch> {
        let mut node = &self.root;
        let mut best: Option<PrefixKvMatch> = node.scope_blocks.get(scope).map(|blocks| {
            let bytes_saved = blocks.len().saturating_mul(std::mem::size_of::<usize>());
            PrefixKvMatch {
                matched_tokens: 0,
                blocks: Arc::clone(blocks),
                bytes_saved,
            }
        });
        for (idx, &tok) in token_ids.iter().enumerate() {
            let Some(next) = node.children.get(&tok) else {
                break;
            };
            node = next;
            if let Some(blocks) = node.scope_blocks.get(scope) {
                let bytes_saved = blocks.len().saturating_mul(std::mem::size_of::<usize>());
                best = Some(PrefixKvMatch {
                    matched_tokens: idx + 1,
                    blocks: Arc::clone(blocks),
                    bytes_saved,
                });
            }
        }
        best
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
}
