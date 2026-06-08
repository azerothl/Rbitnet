//! Serialize tests that mutate process environment variables.

use std::sync::{Mutex, OnceLock};

pub fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static M: OnceLock<Mutex<()>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env test lock poisoned")
}
