//! Actual GGUF admission checks for unsupported async-cache configurations.
use super::*;

#[test]
fn optional_async_configuration_refuses_silent_fallback_and_releases() {
    if std::env::var("RBITNET_ASYNC_CONFIG_TEST").as_deref() != Ok("1") {
        return;
    }
    let archive = Arc::new(
        GgufArchive::mmap_path(Path::new(&std::env::var("RBITNET_TEST_GGUF").unwrap())).unwrap(),
    );
    let rt = crate::backend::CudaRuntime::try_load().expect("actual CUDA runtime required");
    let before = rt.managed_memory_stats().unwrap();
    let defaults = [
        ("RBITNET_MOE_ASYNC", "1"),
        ("RBITNET_MOE_PINNED_SLOTS", "2"),
        ("RBITNET_MOE_PREFETCH", "off"),
        ("RBITNET_MOE_CACHE_MB", "512"),
        ("RBITNET_MOE_EXECUTION", "cache"),
        ("RBITNET_CUDA_MOE", "1"),
    ];
    struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            for (key, value) in &self.0 {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
    let _restore = Restore(
        defaults
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect(),
    );
    for (kind, key, value, message) in [
        (
            BackendKind::Cpu,
            "RBITNET_MOE_ASYNC",
            "1",
            "requires CUDA/hybrid",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_CACHE_MB",
            "0",
            "positive expert budget",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_CUDA_MOE",
            "0",
            "native dynamic API",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_EXECUTION",
            "cpu",
            "cache execution policy",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_EXECUTION",
            "adaptive",
            "cache execution policy",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_PREFETCH",
            "unknown",
            "must be off or previous-pass",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_PINNED_SLOTS",
            "3",
            "one or two pinned slots",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_PINNED_SLOTS",
            "invalid",
            "invalid pinned slot count",
        ),
        (
            BackendKind::Cuda,
            "RBITNET_MOE_ASYNC",
            "yes",
            "must be 0 or 1",
        ),
    ] {
        for (key, value) in defaults {
            std::env::set_var(key, value);
        }
        std::env::set_var(key, value);
        let error = Weights::new(Arc::clone(&archive), kind)
            .err()
            .expect("requested async mode must refuse unsupported configuration");
        assert!(
            error.to_string().contains(message),
            "{key}={value}: {error}"
        );
        let after = rt.managed_memory_stats().unwrap();
        assert_eq!(after.live, before.live);
        assert_eq!(after.categories, before.categories);
        println!("ASYNC_CONFIG_REFUSAL {key}={value}: {error}; physical categories unchanged");
    }
    for (key, value) in defaults {
        std::env::set_var(key, value);
    }
    std::env::set_var("RBITNET_MOE_ASYNC", "0");
    let cpu = Weights::new(archive, BackendKind::Cpu).unwrap();
    assert!(cpu.expert_cache.is_none());
    drop(cpu);
    assert_eq!(rt.managed_memory_stats().unwrap().live, before.live);
    println!(
        "ASYNC_CONFIG_REFUSAL nine actual admission checks passed; disabled CPU path retained"
    );
}
