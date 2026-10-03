//! Runtime-local, byte-bounded prefix snapshots. An entry never crosses model runtimes.
use std::collections::BTreeMap;

struct Entry<T> {
    tokens: Vec<u32>,
    snapshot: T,
    bytes: usize,
    used: u64,
}

/// Token index plus LRU eviction. Snapshot ownership also owns the device allocations.
pub(crate) struct PrefixStore<T> {
    entries: BTreeMap<u64, Entry<T>>,
    budget: usize,
    max_entries: usize,
    bytes: usize,
    clock: u64,
}

pub(crate) fn enabled() -> bool {
    matches!(
        std::env::var("RBITNET_PREFIX_KV").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

pub(crate) fn minimum_tokens() -> usize {
    std::env::var("RBITNET_PREFIX_KV_MIN_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1)
}

impl<T> PrefixStore<T> {
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries.values().any(|entry| entry.tokens == tokens)
    }
    pub fn from_env() -> Self {
        let mb = std::env::var("RBITNET_CUDA_PREFIX_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256);
        let count = std::env::var("RBITNET_CUDA_PREFIX_ENTRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        Self::new(mb.saturating_mul(1024 * 1024), count)
    }

    pub fn new(budget: usize, max_entries: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget,
            max_entries,
            bytes: 0,
            clock: 0,
        }
    }

    /// Returns the longest reusable prefix. Recurrent snapshots require a full checkpoint;
    /// ordinary attention snapshots may be truncated to any common prefix.
    pub fn lookup(
        &mut self,
        tokens: &[u32],
        truncatable: bool,
        minimum: usize,
    ) -> Option<(&T, usize)> {
        let best = self
            .entries
            .iter()
            .filter_map(|(&key, entry)| {
                let matched = entry
                    .tokens
                    .iter()
                    .zip(tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                (matched >= minimum && (truncatable || matched == entry.tokens.len()))
                    .then_some((key, matched))
            })
            .max_by_key(|&(key, matched)| (matched, self.entries[&key].used));
        let (key, matched) = best?;
        self.clock = self.clock.saturating_add(1);
        let entry = self.entries.get_mut(&key)?;
        entry.used = self.clock;
        Some((&entry.snapshot, matched))
    }

    /// Reserve before allocating, so replacing an entry does not temporarily exceed budget.
    pub fn reserve(&mut self, tokens: &[u32], bytes: usize) -> bool {
        if bytes > self.budget || bytes == 0 || self.max_entries == 0 {
            return false;
        }
        let duplicate = self
            .entries
            .iter()
            .find(|(_, e)| e.tokens == tokens)
            .map(|(&k, _)| k);
        if let Some(key) = duplicate {
            self.remove(key);
        }
        while self.bytes.saturating_add(bytes) > self.budget
            || self.entries.len() >= self.max_entries
        {
            let Some(key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.used)
                .map(|(&k, _)| k)
            else {
                break;
            };
            self.remove(key);
        }
        true
    }

    pub fn insert(&mut self, tokens: Vec<u32>, snapshot: T, bytes: usize) {
        if !self.reserve(&tokens, bytes) {
            return;
        }
        self.clock = self.clock.saturating_add(1);
        self.bytes += bytes;
        self.entries.insert(
            self.clock,
            Entry {
                tokens,
                snapshot,
                bytes,
                used: self.clock,
            },
        );
    }

    fn remove(&mut self, key: u64) {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -= entry.bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Owned(Arc<AtomicUsize>);
    impl Drop for Owned {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn recurrent_checkpoints_cannot_be_truncated_but_attention_can() {
        let mut cache = PrefixStore::new(128, 4);
        cache.insert(vec![1, 2, 3, 4], 42, 32);
        assert_eq!(cache.lookup(&[1, 2, 9], false, 1), None);
        assert_eq!(cache.lookup(&[1, 2, 9], true, 1), Some((&42, 2)));
        assert_eq!(cache.lookup(&[1, 2, 3, 4, 5], false, 1), Some((&42, 4)));
        assert_eq!(cache.lookup(&[9], true, 1), None);
    }

    #[test]
    fn byte_budget_lru_and_replacement_release_owned_snapshots() {
        let released = Arc::new(AtomicUsize::new(0));
        let mut cache = PrefixStore::new(64, 2);
        for id in [1, 2] {
            cache.insert(vec![id], Owned(released.clone()), 32);
        }
        cache.lookup(&[1], true, 1).unwrap();
        assert!(cache.reserve(&[3], 32));
        assert_eq!(released.load(Ordering::SeqCst), 1);
        assert!(cache.lookup(&[2], true, 1).is_none());
        cache.insert(vec![3], Owned(released.clone()), 32);
        assert!(cache.reserve(&[1], 32));
        assert_eq!(released.load(Ordering::SeqCst), 2);
        assert!(!cache.reserve(&[4], 65));
        assert_eq!(cache.bytes, 32);
        drop(cache);
        assert_eq!(released.load(Ordering::SeqCst), 3);
    }
}
