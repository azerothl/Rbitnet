//! Prefix KV block cache scaffolding (Inference stack v2 phase C.1).
//!
//! Maps a hash of tokenizer ids to physical page indices once block-structured KV is shared across requests.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct PrefixKvBlockCache {
    inner: HashMap<PrefixKvKey, Arc<[usize]>>,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixKvKey {
    pub model_id: String,
    pub tokenizer_id: String,
    pub template_id: String,
    pub prefix_hash: u64,
}

impl PrefixKvKey {
    pub fn new(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        template_id: impl Into<String>,
        token_ids: &[u32],
    ) -> Self {
        Self {
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            template_id: template_id.into(),
            prefix_hash: hash_prefix_tokens(token_ids),
        }
    }

    fn legacy(prefix_hash: u64) -> Self {
        Self {
            model_id: "legacy".into(),
            tokenizer_id: "legacy".into(),
            template_id: "legacy".into(),
            prefix_hash,
        }
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
}
