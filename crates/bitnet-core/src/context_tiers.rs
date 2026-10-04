//! Opt-in host/disk prefix checkpoints. Native transfers are owned by the caller.
//! Budgets are per model runtime; no SSD access occurs inside token decoding.
use crate::portable_envelope::{self, Compatibility, HostCheckpoint};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[derive(Clone, Debug)]
pub(crate) struct Policy {
    pub directory: PathBuf,
    pub ram_bytes: usize,
    pub disk_bytes: usize,
    pub entries: usize,
    pub retention: Duration,
}
impl Policy {
    pub fn from_env() -> io::Result<Option<Self>> {
        if !matches!(
            std::env::var("RBITNET_CONTEXT_TIERS").as_deref(),
            Ok("1" | "true")
        ) {
            return Ok(None);
        }
        fn number(name: &str, default: usize) -> io::Result<usize> {
            match std::env::var(name) {
                Ok(value) => value
                    .trim()
                    .parse()
                    .map_err(|_| invalid("invalid context budget")),
                Err(_) => Ok(default),
            }
        }
        let directory = std::env::var_os("RBITNET_CONTEXT_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| invalid("RBITNET_CONTEXT_DIR is required for context tiers"))?;
        let mib = |name, default| {
            number(name, default)?
                .checked_mul(1024 * 1024)
                .ok_or_else(|| invalid("context budget overflow"))
        };
        let policy = Self {
            directory,
            ram_bytes: mib("RBITNET_CONTEXT_RAM_MB", 256)?,
            disk_bytes: mib("RBITNET_CONTEXT_DISK_MB", 2048)?,
            entries: number("RBITNET_CONTEXT_ENTRIES", 64)?,
            retention: Duration::from_secs(number("RBITNET_CONTEXT_TTL_SECS", 1800)? as u64),
        };
        if policy.ram_bytes == 0
            || policy.entries == 0
            || policy.entries > 4096
            || policy.retention.is_zero()
        {
            return Err(invalid(
                "context RAM, entry count and TTL must be positive and bounded",
            ));
        }
        Ok(Some(policy))
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}
fn address(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn interrupted_write_name(name: &str) -> bool {
    let Some(body) = name
        .strip_prefix('.')
        .and_then(|value| value.strip_suffix(".tmp"))
    else {
        return false;
    };
    let mut parts = body.split('-');
    matches!(parts.next(), Some(value) if address(value))
        && matches!(parts.next(), Some(value) if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) && value.parse::<u32>().is_ok())
        && matches!(parts.next(), Some(value) if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) && value.parse::<u64>().is_ok())
        && parts.next().is_none()
}

/// A borrow owns its payload even if a cache entry is evicted. Its bytes stay charged
/// until the last lease is dropped, preventing hidden over-budget active snapshots.
pub(crate) struct Lease {
    pub checkpoint: HostCheckpoint,
    charged: Arc<AtomicUsize>,
    bytes: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.charged.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct Entry {
    tokens: Vec<u32>,
    payload_bytes: usize,
    file: Option<PathBuf>,
    file_bytes: usize,
    host: Option<Arc<Lease>>,
    created: SystemTime,
    used: u64,
}
#[derive(Default, Clone, Debug, serde::Serialize)]
pub(crate) struct Stats {
    pub ram_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub write_failures: u64,
    pub read_failures: u64,
    pub captures: u64,
    pub capture_refusals: u64,
    pub evictions: u64,
    pub read_ns: u64,
    pub write_ns: u64,
}
pub(crate) struct Store {
    key: Compatibility,
    policy: Policy,
    directory: PathBuf,
    // Exclusive OS lock isolates cooperative runtimes and is released by process exit.
    _directory_lock: File,
    entries: BTreeMap<String, Entry>,
    charged: Arc<AtomicUsize>,
    disk_used: usize,
    clock: u64,
    pub stats: Stats,
}
impl Store {
    pub fn open(key: Compatibility, policy: Policy) -> io::Result<Self> {
        let identity = hex(&Sha256::digest(serde_json::to_vec(&key)?));
        let directory = policy.directory.join("rbitnet-state-v1").join(identity);
        std::fs::create_dir_all(&directory)?;
        if std::fs::symlink_metadata(&directory)?
            .file_type()
            .is_symlink()
        {
            return Err(invalid("checkpoint namespace must not be a symlink"));
        }
        let lock_path = directory.join(".owner.lock");
        if lock_path.exists()
            && std::fs::symlink_metadata(&lock_path)?
                .file_type()
                .is_symlink()
        {
            return Err(invalid("checkpoint lock must not be a symlink"));
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        lock.try_lock()
            .map_err(|error| io::Error::new(io::ErrorKind::WouldBlock, error.to_string()))?;
        let mut store = Self {
            key,
            policy,
            directory,
            _directory_lock: lock,
            entries: BTreeMap::new(),
            charged: Arc::new(AtomicUsize::new(0)),
            disk_used: 0,
            clock: 0,
            stats: Stats::default(),
        };
        // Header inspection is bounded indexing only. Integrity and all identities
        // are checked again by read_checkpoint before any payload reaches CUDA.
        let scan_limit = store.policy.entries.saturating_mul(4) + 32;
        let scanned = std::fs::read_dir(&store.directory)?
            .take(scan_limit + 1)
            .collect::<io::Result<Vec<_>>>()?;
        if scanned.len() > scan_limit {
            return Err(invalid(
                "checkpoint namespace exceeds bounded scan; replay required",
            ));
        }
        for item in scanned {
            let path = item.path();
            // An exclusive namespace lock proves no cooperative writer is alive.
            // Reclaim only this format's generated temporary names after a crash.
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(interrupted_write_name)
            {
                let metadata = std::fs::symlink_metadata(&path)?;
                if !metadata.is_file() || metadata.file_type().is_symlink() {
                    return Err(invalid("derived checkpoint temporary is not regular"));
                }
                std::fs::remove_file(&path)?;
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            if path.extension().and_then(|value| value.to_str()) != Some("state") || !address(stem)
            {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(invalid("derived checkpoint object is not regular"));
            }
            let info = match portable_envelope::inspect_checkpoint(
                &path,
                &store.key,
                store.policy.ram_bytes,
            ) {
                Ok(info) => info,
                Err(_) => {
                    // Invalid, incompatible or oversized derived objects must not
                    // accumulate outside the disk quota after a process restart.
                    std::fs::remove_file(&path)?;
                    continue;
                }
            };
            let created = metadata.modified()?;
            let Ok(file_bytes) = usize::try_from(metadata.len()) else {
                continue;
            };
            store.disk_used = store
                .disk_used
                .checked_add(file_bytes)
                .ok_or_else(|| invalid("disk accounting overflow"))?;
            store.entries.insert(
                stem.into(),
                Entry {
                    tokens: info.tokens,
                    payload_bytes: info.payload_bytes,
                    file: Some(path.clone()),
                    file_bytes,
                    host: None,
                    created,
                    used: 0,
                },
            );
        }
        store.expire()?;
        store.trim_disk(0, None)?;
        store.trim_entries()?;
        Ok(store)
    }
    pub fn ram_used(&self) -> usize {
        self.charged.load(Ordering::Acquire)
    }
    pub fn disk_used(&self) -> usize {
        self.disk_used
    }
    fn remove(&mut self, id: &str) -> io::Result<()> {
        // Only generated content-addressed files inside the exclusively locked namespace.
        if let Some(entry) = self.entries.get(id) {
            if let Some(file) = &entry.file {
                if file.parent() != Some(self.directory.as_path()) || !address(id) {
                    return Err(invalid("unowned checkpoint path"));
                }
                match std::fs::remove_file(file) {
                    Ok(()) => (),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => (),
                    Err(error) => return Err(error),
                }
            }
        }
        if let Some(entry) = self.entries.remove(id) {
            self.disk_used = self.disk_used.saturating_sub(entry.file_bytes);
            self.stats.evictions += 1;
        }
        Ok(())
    }
    fn expire(&mut self) -> io::Result<()> {
        let now = SystemTime::now();
        let ids: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                now.duration_since(entry.created).unwrap_or_default() >= self.policy.retention
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            self.remove(&id)?;
        }
        Ok(())
    }
    fn trim_entries(&mut self) -> io::Result<()> {
        while self.entries.len() > self.policy.entries {
            let id = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(id, _)| id.clone())
                .unwrap();
            self.remove(&id)?;
        }
        Ok(())
    }
    fn trim_disk(&mut self, incoming: usize, protect: Option<&str>) -> io::Result<()> {
        if incoming > self.policy.disk_bytes {
            return Err(invalid("checkpoint exceeds disk budget"));
        }
        while self.disk_used.saturating_add(incoming) > self.policy.disk_bytes {
            let id = self
                .entries
                .iter()
                .filter(|(id, entry)| entry.file.is_some() && Some(id.as_str()) != protect)
                .min_by_key(|(_, entry)| entry.used)
                .map(|(id, _)| id.clone())
                .ok_or_else(|| invalid("disk budget cannot be reclaimed"))?;
            self.remove(&id)?;
        }
        Ok(())
    }
    fn reserve_ram(&mut self, bytes: usize) -> bool {
        if bytes == 0 || bytes > self.policy.ram_bytes {
            return false;
        }
        while self.ram_used().saturating_add(bytes) > self.policy.ram_bytes {
            let id = self
                .entries
                .iter()
                .filter(|(_, entry)| {
                    entry
                        .host
                        .as_ref()
                        .is_some_and(|lease| Arc::strong_count(lease) == 1)
                })
                .min_by_key(|(_, entry)| entry.used)
                .map(|(id, _)| id.clone());
            let Some(id) = id else { return false };
            self.entries.get_mut(&id).unwrap().host = None;
            if self.entries[&id].file.is_none() {
                // With no cold copy the prefix is no longer cached. Retaining
                // its token index would falsely suppress the next capture.
                self.entries.remove(&id);
                self.stats.evictions += 1;
            }
        }
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        true
    }
    /// Capture at a validated prefill boundary. Allocation and pending active leases
    /// share the host budget; failed transfers never produce usable entries.
    pub fn capture(
        &mut self,
        tokens: &[u32],
        bytes: usize,
        fill: impl FnOnce(&mut [f32]) -> io::Result<()>,
    ) -> io::Result<bool> {
        self.expire()?;
        if tokens.is_empty() || tokens.len() > 8192 || bytes % 4 != 0 {
            return Err(invalid("invalid capture geometry"));
        }
        if self.entries.values().any(|entry| entry.tokens == tokens) {
            return Ok(true);
        }
        if !self.reserve_ram(bytes) {
            self.stats.capture_refusals += 1;
            return Ok(false);
        }
        // Own the charge before allocation so every error path releases it.
        let mut lease = Lease {
            checkpoint: HostCheckpoint {
                tokens: tokens.to_vec(),
                values: Vec::new(),
            },
            charged: Arc::clone(&self.charged),
            bytes,
        };
        lease
            .checkpoint
            .values
            .try_reserve_exact(bytes / 4)
            .map_err(|_| invalid("host snapshot allocation refused"))?;
        lease.checkpoint.values.resize(bytes / 4, 0.0);
        fill(&mut lease.checkpoint.values)?;
        if lease
            .checkpoint
            .values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("non-finite capture"));
        }
        let id = hex(&Sha256::digest(serde_json::to_vec(tokens)?));
        let mut file = None;
        let mut file_bytes = 0;
        if self.policy.disk_bytes > 0 {
            // Reserve the complete sealed object and its temporary write, not just tensor bytes.
            let bound = portable_envelope::sealed_size(&self.key, tokens, bytes)?;
            let started = std::time::Instant::now();
            match self.trim_disk(bound, None).and_then(|_| {
                portable_envelope::write_checkpoint(
                    &self.directory,
                    &self.key,
                    tokens,
                    &lease.checkpoint.values,
                    self.policy.ram_bytes,
                )
            }) {
                Ok(path) => {
                    file_bytes = usize::try_from(std::fs::metadata(&path)?.len())
                        .map_err(|_| invalid("sealed size overflow"))?;
                    file = Some(path);
                    self.disk_used += file_bytes;
                    self.stats.writes += 1;
                }
                Err(error) => {
                    self.stats.write_failures += 1;
                    tracing::warn!(%error,"context persistence failed; host checkpoint/replay remains available");
                }
            }
            self.stats.write_ns = self
                .stats
                .write_ns
                .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        self.clock = self.clock.saturating_add(1);
        self.entries.insert(
            id,
            Entry {
                tokens: tokens.to_vec(),
                payload_bytes: bytes,
                file,
                file_bytes,
                host: Some(Arc::new(lease)),
                created: SystemTime::now(),
                used: self.clock,
            },
        );
        self.stats.captures += 1;
        self.trim_entries()?;
        Ok(true)
    }
    pub fn lookup(
        &mut self,
        tokens: &[u32],
        expected: impl Fn(usize) -> usize,
    ) -> io::Result<Option<Arc<Lease>>> {
        self.expire()?;
        let id = self
            .entries
            .iter()
            .filter(|(_, entry)| tokens.starts_with(&entry.tokens))
            .max_by_key(|(_, entry)| (entry.tokens.len(), entry.used))
            .map(|(id, _)| id.clone());
        let Some(id) = id else {
            self.stats.misses += 1;
            return Ok(None);
        };
        let bytes = self.entries[&id].payload_bytes;
        let length = self.entries[&id].tokens.len();
        if bytes != expected(length) {
            self.stats.read_failures += 1;
            self.remove(&id)?;
            return Ok(None);
        }
        self.clock = self.clock.saturating_add(1);
        self.entries.get_mut(&id).unwrap().used = self.clock;
        if let Some(host) = self.entries[&id].host.as_ref() {
            self.stats.ram_hits += 1;
            return Ok(Some(Arc::clone(host)));
        }
        let Some(file) = self.entries[&id].file.clone() else {
            self.stats.misses += 1;
            return Ok(None);
        };
        if !self.reserve_ram(bytes) {
            self.stats.misses += 1;
            return Ok(None);
        }
        let mut lease = Lease {
            checkpoint: HostCheckpoint {
                tokens: Vec::new(),
                values: Vec::new(),
            },
            charged: Arc::clone(&self.charged),
            bytes,
        };
        let started = std::time::Instant::now();
        let read =
            portable_envelope::read_checkpoint(&file, &self.key, self.policy.ram_bytes, bytes);
        self.stats.read_ns = self
            .stats
            .read_ns
            .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        match read {
            Ok(checkpoint) => {
                if checkpoint.tokens != self.entries[&id].tokens {
                    self.stats.read_failures += 1;
                    self.remove(&id)?;
                    return Ok(None);
                }
                lease.checkpoint = checkpoint;
                let lease = Arc::new(lease);
                self.entries.get_mut(&id).unwrap().host = Some(Arc::clone(&lease));
                self.stats.disk_hits += 1;
                Ok(Some(lease))
            }
            Err(error) => {
                self.stats.read_failures += 1;
                self.remove(&id)?;
                tracing::warn!(%error,"invalid persisted context; prompt will be replayed");
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key() -> Compatibility {
        Compatibility {
            model_sha256: [1; 32],
            tokenizer_sha256: [2; 32],
            native_library_sha256: [3; 32],
            execution_config_sha256: [4; 32],
            layout: "fixture-f32-v1".into(),
        }
    }
    fn policy(directory: &Path, ram_bytes: usize) -> Policy {
        Policy {
            directory: directory.into(),
            ram_bytes,
            disk_bytes: 65536,
            entries: 8,
            retention: Duration::from_secs(1800),
        }
    }
    fn capture(store: &mut Store, tokens: &[u32], value: f32) -> bool {
        store
            .capture(tokens, 64, |values| {
                values.fill(value);
                Ok(())
            })
            .unwrap()
    }
    #[test]
    fn restart_reclaims_interrupted_writes_and_malformed_objects_without_removing_notes() {
        let directory = tempfile::tempdir().unwrap();
        let configuration = policy(directory.path(), 128);
        let mut store = Store::open(key(), configuration.clone()).unwrap();
        assert!(capture(&mut store, &[1, 2], 0.5));
        let namespace = store.directory.clone();
        let temporary = namespace.join(format!(".{}-1234-9.tmp", "a".repeat(64)));
        let malformed = namespace.join(format!("{}.state", "b".repeat(64)));
        let note = namespace.join("user-note.tmp");
        let unrelated = namespace.join(format!(".{}-1234-not-a-counter.tmp", "c".repeat(64)));
        for path in [&temporary, &malformed, &note, &unrelated] {
            std::fs::write(path, b"interrupted or user content").unwrap();
        }
        drop(store);
        let mut reopened = Store::open(key(), configuration).unwrap();
        assert!(!temporary.exists());
        assert!(!malformed.exists());
        assert_eq!(
            std::fs::read(&note).unwrap(),
            b"interrupted or user content"
        );
        assert!(unrelated.exists());
        assert!(reopened.lookup(&[1, 2, 3], |_| 64).unwrap().is_some());
        let physical_owned_bytes: usize = std::fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("state"))
            .map(|path| std::fs::metadata(path).unwrap().len() as usize)
            .sum();
        assert_eq!(reopened.disk_used(), physical_owned_bytes);
        assert!(reopened.disk_used() <= reopened.policy.disk_bytes);
    }
    #[test]
    fn held_payload_remains_charged_and_cannot_be_evicted_to_hide_memory() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = Store::open(key(), policy(directory.path(), 64)).unwrap();
        assert!(capture(&mut store, &[1, 2], 0.5));
        let held = store.lookup(&[1, 2, 3], |_| 64).unwrap().unwrap();
        assert!(!capture(&mut store, &[7, 8], 1.0));
        assert_eq!(store.ram_used(), 64);
        assert_eq!(held.checkpoint.values, vec![0.5; 16]);
        drop(held);
        assert!(capture(&mut store, &[7, 8], 1.0));
        assert_eq!(store.ram_used(), 64);
        let restored = store.lookup(&[1, 2, 3], |_| 64).unwrap().unwrap();
        assert_eq!(restored.checkpoint.values, vec![0.5; 16]);
        assert_eq!(store.stats.disk_hits, 1);
    }
    #[test]
    fn failed_capture_releases_pending_charge_and_never_creates_a_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = Store::open(key(), policy(directory.path(), 128)).unwrap();
        assert!(store
            .capture(&[1], 64, |values| {
                values.fill(9.0);
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "transfer interrupted",
                ))
            })
            .is_err());
        assert_eq!(store.ram_used(), 0);
        assert_eq!(store.disk_used(), 0);
        assert!(store.lookup(&[1, 2], |_| 64).unwrap().is_none());
        assert!(store
            .capture(&[1], 64, |values| {
                values.fill(f32::NAN);
                Ok(())
            })
            .is_err());
        assert_eq!(store.ram_used(), 0);
    }
    #[test]
    fn host_only_eviction_allows_a_fresh_capture_of_the_same_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let mut configuration = policy(directory.path(), 64);
        configuration.disk_bytes = 0;
        let mut store = Store::open(key(), configuration).unwrap();
        assert!(capture(&mut store, &[1, 2], 0.5));
        assert!(capture(&mut store, &[3, 4], 0.75));
        assert!(store.lookup(&[1, 2, 5], |_| 64).unwrap().is_none());
        assert_eq!(store.entries.len(), 1);
        assert!(capture(&mut store, &[1, 2], 1.25));
        let recaptured = store.lookup(&[1, 2, 5], |_| 64).unwrap().unwrap();
        assert_eq!(recaptured.checkpoint.values, vec![1.25; 16]);
        assert_eq!(store.stats.captures, 3);
        assert_eq!(store.stats.writes, 0);
        assert_eq!(store.ram_used(), 64);
        assert_eq!(store.disk_used(), 0);
    }
    #[test]
    fn restart_releases_lock_and_restores_only_exact_identity_and_full_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let configuration = policy(directory.path(), 128);
        {
            let mut store = Store::open(key(), configuration.clone()).unwrap();
            assert!(capture(&mut store, &[1, 2, 3], 0.25));
            assert!(Store::open(key(), configuration.clone()).is_err());
        }
        let mut reopened = Store::open(key(), configuration.clone()).unwrap();
        assert!(reopened.lookup(&[1, 2, 9], |_| 64).unwrap().is_none());
        let saved = reopened.lookup(&[1, 2, 3, 4], |_| 64).unwrap().unwrap();
        assert_eq!(saved.checkpoint.tokens, [1, 2, 3]);
        assert_eq!(saved.checkpoint.values, vec![0.25; 16]);
        assert_eq!(reopened.stats.disk_hits, 1);
        for field in 0..4 {
            let mut other = key();
            match field {
                0 => other.model_sha256[0] ^= 1,
                1 => other.tokenizer_sha256[0] ^= 1,
                2 => other.native_library_sha256[0] ^= 1,
                _ => other.execution_config_sha256[0] ^= 1,
            };
            let mut isolated = Store::open(other, configuration.clone()).unwrap();
            assert!(isolated.lookup(&[1, 2, 3, 4], |_| 64).unwrap().is_none());
        }
    }
    #[test]
    fn corrupted_or_wrong_size_payload_is_refused_before_use_and_can_be_recaptured() {
        let directory = tempfile::tempdir().unwrap();
        let configuration = policy(directory.path(), 128);
        let file = {
            let mut store = Store::open(key(), configuration.clone()).unwrap();
            assert!(capture(&mut store, &[1, 2], 0.5));
            store.entries.values().next().unwrap().file.clone().unwrap()
        };
        let mut raw = std::fs::read(&file).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 1;
        std::fs::write(&file, raw).unwrap();
        let mut reopened = Store::open(key(), configuration).unwrap();
        assert!(reopened.lookup(&[1, 2, 3], |_| 64).unwrap().is_none());
        assert_eq!(reopened.ram_used(), 0);
        assert_eq!(reopened.stats.read_failures, 1);
        assert!(capture(&mut reopened, &[1, 2], 0.75));
        assert!(reopened.lookup(&[1, 2, 3], |_| 128).unwrap().is_none());
        assert_eq!(reopened.ram_used(), 0);
    }
    #[test]
    fn disk_quota_failure_keeps_host_state_and_disabled_disk_does_not_persist() {
        let directory = tempfile::tempdir().unwrap();
        let mut configuration = policy(directory.path(), 128);
        configuration.disk_bytes = 32;
        let mut store = Store::open(key(), configuration.clone()).unwrap();
        assert!(capture(&mut store, &[1], 0.5));
        assert_eq!(store.stats.write_failures, 1);
        assert_eq!(store.disk_used(), 0);
        assert!(store.lookup(&[1, 2], |_| 64).unwrap().is_some());
        drop(store);
        configuration.disk_bytes = 0;
        let mut host_only = Store::open(key(), configuration.clone()).unwrap();
        assert!(capture(&mut host_only, &[2], 0.5));
        assert_eq!(host_only.stats.writes, 0);
        drop(host_only);
        let mut restarted = Store::open(key(), configuration).unwrap();
        assert!(restarted.lookup(&[2, 3], |_| 64).unwrap().is_none());
    }
    #[test]
    fn retention_and_entry_limits_reclaim_only_derived_cache_objects() {
        let directory = tempfile::tempdir().unwrap();
        let mut configuration = policy(directory.path(), 128);
        configuration.entries = 1;
        let mut store = Store::open(key(), configuration).unwrap();
        assert!(capture(&mut store, &[1], 0.5));
        assert!(capture(&mut store, &[2], 0.75));
        assert_eq!(store.entries.len(), 1);
        assert!(store.disk_used() <= store.policy.disk_bytes);
        let note = store.directory.join("user-note.txt");
        std::fs::write(&note, "keep").unwrap();
        for entry in store.entries.values_mut() {
            entry.created = SystemTime::UNIX_EPOCH;
        }
        store.expire().unwrap();
        assert_eq!(store.ram_used(), 0);
        assert_eq!(store.disk_used(), 0);
        assert_eq!(std::fs::read_to_string(note).unwrap(), "keep");
    }
}
