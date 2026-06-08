//! `rbitnet tune` — apply battery / latency / throughput presets (mistral.rs-style).

use std::io::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneProfile {
    Battery,
    Latency,
    Throughput,
}

impl TuneProfile {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "battery" | "power" | "eco" => Some(Self::Battery),
            "latency" | "low-latency" | "fast" => Some(Self::Latency),
            "throughput" | "batch" | "server" => Some(Self::Throughput),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Battery => "battery",
            Self::Latency => "latency",
            Self::Throughput => "throughput",
        }
    }
}

/// Env vars set by each profile (name → value).
pub fn profile_env(profile: TuneProfile) -> &'static [(&'static str, &'static str)] {
    match profile {
        TuneProfile::Battery => &[
            ("RBITNET_MAX_CONCURRENT", "1"),
            ("RBITNET_PREFIX_KV", "0"),
            ("RBITNET_CONTINUOUS_BATCHING", "0"),
            ("RBITNET_KV_POOL", "0"),
            ("RBITNET_CUDA_GRAPH", "0"),
            ("RBITNET_PREFILL_CHUNK_TOKENS", "256"),
            ("RBITNET_BACKEND", "cpu"),
        ],
        TuneProfile::Latency => &[
            ("RBITNET_MAX_CONCURRENT", "2"),
            ("RBITNET_PREFIX_KV", "1"),
            ("RBITNET_CONTINUOUS_BATCHING", "0"),
            ("RBITNET_KV_POOL", "0"),
            ("RBITNET_CUDA_GRAPH", "1"),
            ("RBITNET_PREFILL_CHUNK_TOKENS", "512"),
            ("RBITNET_BACKEND", "cuda"),
        ],
        TuneProfile::Throughput => &[
            ("RBITNET_MAX_CONCURRENT", "8"),
            ("RBITNET_PREFIX_KV", "1"),
            ("RBITNET_CONTINUOUS_BATCHING", "1"),
            ("RBITNET_KV_POOL", "1"),
            ("RBITNET_CUDA_GRAPH", "1"),
            ("RBITNET_PREFILL_CHUNK_TOKENS", "1024"),
            ("RBITNET_BACKEND", "cuda"),
        ],
    }
}

pub fn apply_profile(profile: TuneProfile, export_shell: bool) {
    for (k, v) in profile_env(profile) {
        if export_shell {
            #[cfg(windows)]
            {
                let _ = writeln!(std::io::stdout(), "set {k}={v}");
            }
            #[cfg(not(windows))]
            {
                let _ = writeln!(std::io::stdout(), "export {k}={v}");
            }
        } else {
            std::env::set_var(k, v);
        }
    }
    eprintln!(
        "Applied tune profile '{}' ({} vars). Start serve with these env vars in the same shell.",
        profile.as_str(),
        profile_env(profile).len()
    );
}
