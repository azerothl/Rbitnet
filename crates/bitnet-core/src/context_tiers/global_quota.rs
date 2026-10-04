//! Cooperative, fail-closed disk admission across compatibility namespaces.
use super::{address, interrupted_write_name, invalid, linked, owned_directory};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(super) struct Guard {
    root: PathBuf,
    limit: usize,
    _lock: File,
}

fn lock_file(path: &Path) -> io::Result<File> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if linked(&metadata) || !metadata.is_file() => {
            return Err(invalid("quota ownership file must not redirect"));
        }
        Ok(_) => (),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error),
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0x1 | 0x2).custom_flags(0x00200000);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if linked(&metadata) || !metadata.is_file() {
        return Err(invalid("opened quota ownership file must not redirect"));
    }
    file.try_lock()
        .map_err(|error| io::Error::new(io::ErrorKind::WouldBlock, error.to_string()))?;
    Ok(file)
}

impl Guard {
    pub fn open(root: &Path, limit: usize) -> io::Result<Self> {
        let mut lock = lock_file(&root.join(".disk-quota.lock"))?;
        // Persist one explicit cap for the root. Mixed configurations must not
        // silently enlarge an already registered global quota.
        let bytes = lock.metadata()?.len();
        if bytes == 0 {
            lock.write_all(&(limit as u64).to_le_bytes())?;
            lock.sync_all()?;
        } else {
            if bytes != 8 {
                return Err(invalid("invalid global quota registration"));
            }
            let mut cap = [0; 8];
            lock.seek(SeekFrom::Start(0))?;
            lock.read_exact(&mut cap)?;
            if u64::from_le_bytes(cap) != limit as u64 {
                return Err(invalid(
                    "global disk quota differs from registered root cap",
                ));
            }
        }
        Ok(Self {
            root: root.into(),
            limit,
            _lock: lock,
        })
    }

    /// Returns false only when admission needs reclamation in the caller's own
    /// namespace. Other active stores are counted but never reclaimed.
    pub fn reserve(&self, current: &Path, incoming: usize) -> io::Result<bool> {
        if incoming > self.limit {
            return Err(invalid("checkpoint exceeds global disk quota"));
        }
        let mut total = 0usize;
        let mut candidates = Vec::new();
        // These directory and owner handles protect every candidate until its
        // deletion. Nonblocking locks avoid cycles between two active stores.
        let mut leases = Vec::new();
        for item in std::fs::read_dir(&self.root)? {
            let item = item?;
            let path = item.path();
            if !item.file_name().to_str().is_some_and(address) {
                continue;
            }
            let metadata = std::fs::symlink_metadata(&path)?;
            if linked(&metadata) || !metadata.is_dir() {
                return Err(invalid("managed quota namespace must not redirect"));
            }
            let reclaim = if path == current {
                false
            } else {
                let directory = owned_directory(&path)?;
                match lock_file(&path.join(".owner.lock")) {
                    Ok(owner) => {
                        leases.push((directory, owner));
                        true
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => false,
                    Err(error) => return Err(error),
                }
            };
            for entry in std::fs::read_dir(&path)? {
                let object = entry?.path();
                let name = object.file_name().and_then(|name| name.to_str());
                let derived = name.is_some_and(interrupted_write_name)
                    || (object.extension().and_then(|v| v.to_str()) == Some("state")
                        && object
                            .file_stem()
                            .and_then(|v| v.to_str())
                            .is_some_and(address));
                if !derived {
                    continue;
                }
                let metadata = std::fs::symlink_metadata(&object)?;
                if linked(&metadata) || !metadata.is_file() {
                    return Err(invalid("managed quota object must be regular"));
                }
                let size =
                    usize::try_from(metadata.len()).map_err(|_| invalid("quota size overflow"))?;
                total = total
                    .checked_add(size)
                    .ok_or_else(|| invalid("global disk accounting overflow"))?;
                if reclaim {
                    candidates.push((
                        metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                        object,
                        size,
                    ));
                }
            }
        }
        candidates.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        for (_, object, size) in candidates {
            if total
                .checked_add(incoming)
                .is_some_and(|sum| sum <= self.limit)
            {
                return Ok(true);
            }
            std::fs::remove_file(&object)?;
            total -= size;
        }
        Ok(total
            .checked_add(incoming)
            .is_some_and(|sum| sum <= self.limit))
    }
}
