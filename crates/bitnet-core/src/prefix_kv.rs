//! Prefix KV block cache scaffolding (Inference stack v2 phase C.1).
//!
//! Maps a hash of tokenizer ids to physical page indices once block-structured KV is shared across requests.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct PrefixKvBlockCache {
    inner: HashMap<u64, Arc<[usize]>>,
}

impl PrefixKvBlockCache {
    pub fn insert_blocks(&mut self, prefix_hash: u64, blocks: Vec<usize>) {
        self.inner.insert(prefix_hash, blocks.into());
    }

    pub fn get_blocks(&self, prefix_hash: u64) -> Option<&Arc<[usize]>> {
        self.inner.get(&prefix_hash)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
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
}
