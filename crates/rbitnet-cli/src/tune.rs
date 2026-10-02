//! `rbitnet tune` — apply battery / latency / throughput presets (mistral.rs-style).

use std::io::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneProfile {
    Battery,
    /// Low-latency interactive chat: prefix KV on, continuous batching off, CPU-safe defaults.
    Interactive,
    Latency,
    Throughput,
    /// Microsoft BitNet / Akasha BitNetProvider CPU recipe (prefix KV measured path).
    BitnetCpu,
}

impl TuneProfile {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "battery" | "power" | "eco" => Some(Self::Battery),
            "interactive" | "chat" => Some(Self::Interactive),
            "latency" | "low-latency" | "fast" => Some(Self::Latency),
            "throughput" | "batch" | "server" => Some(Self::Throughput),
            "bitnet-cpu" | "bitnet_cpu" | "bitnet" => Some(Self::BitnetCpu),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Battery => "battery",
            Self::Interactive => "interactive",
            Self::Latency => "latency",
            Self::Throughput => "throughput",
            Self::BitnetCpu => "bitnet-cpu",
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
        // Measured serving path for ≥2 sessions: prefix KV on; CB stays off until fused matmul.
        TuneProfile::Interactive => &[
            ("RBITNET_MAX_CONCURRENT", "2"),
            ("RBITNET_PREFIX_KV", "1"),
            ("RBITNET_CONTINUOUS_BATCHING", "0"),
            ("RBITNET_KV_POOL", "0"),
            ("RBITNET_CUDA_GRAPH", "0"),
            ("RBITNET_PREFILL_CHUNK_TOKENS", "512"),
            ("RBITNET_BACKEND", "cpu"),
            ("RBITNET_SESSIONS", "1"),
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
        // Continuous batching + Sarathi stall-free schedule; not fused GPU.
        // KV Q8 saves paged RSS (~4× vs F32 pages); keep off for max numerical safety.
        TuneProfile::Throughput => &[
            ("RBITNET_MAX_CONCURRENT", "8"),
            ("RBITNET_PREFIX_KV", "1"),
            ("RBITNET_CONTINUOUS_BATCHING", "1"),
            ("RBITNET_KV_POOL", "1"),
            ("RBITNET_LLAMA_PAGED_KV", "1"),
            ("RBITNET_KV_QUANT", "q8"),
            ("RBITNET_CUDA_GRAPH", "0"),
            ("RBITNET_PREFILL_CHUNK_TOKENS", "256"),
            ("RBITNET_ITERATION_TOKEN_BUDGET", "512"),
            ("RBITNET_BACKEND", "cpu"),
            ("RBITNET_SESSIONS", "1"),
        ],
        TuneProfile::BitnetCpu => &[
            ("RBITNET_ARCHITECTURE", "bitnet"),
            ("RBITNET_BACKEND", "cpu"),
            ("RBITNET_PREFIX_KV", "1"),
            ("RBITNET_CONTINUOUS_BATCHING", "0"),
            ("RBITNET_LLAMA_PAGED_KV", "1"),
            ("RBITNET_KV_QUANT", "q8"),
            ("RBITNET_MAX_CONCURRENT", "2"),
            ("RBITNET_CUDA_GRAPH", "0"),
            ("RBITNET_BIND", "127.0.0.1:8080"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn env_map(profile: TuneProfile) -> std::collections::BTreeMap<&'static str, &'static str> {
        profile_env(profile).iter().copied().collect()
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(TuneProfile::parse("interactive"), Some(TuneProfile::Interactive));
        assert_eq!(TuneProfile::parse("chat"), Some(TuneProfile::Interactive));
        assert_eq!(TuneProfile::parse("throughput"), Some(TuneProfile::Throughput));
        assert_eq!(TuneProfile::parse("batch"), Some(TuneProfile::Throughput));
        assert_eq!(TuneProfile::parse("bitnet-cpu"), Some(TuneProfile::BitnetCpu));
        assert_eq!(TuneProfile::parse("battery"), Some(TuneProfile::Battery));
        assert!(TuneProfile::parse("nope").is_none());
    }

    #[test]
    fn interactive_exports_low_latency_flags() {
        let m = env_map(TuneProfile::Interactive);
        assert_eq!(m.get("RBITNET_PREFIX_KV"), Some(&"1"));
        assert_eq!(m.get("RBITNET_CONTINUOUS_BATCHING"), Some(&"0"));
        assert_eq!(m.get("RBITNET_BACKEND"), Some(&"cpu"));
        assert_eq!(m.get("RBITNET_MAX_CONCURRENT"), Some(&"2"));
    }

    #[test]
    fn throughput_exports_batch_flags() {
        let m = env_map(TuneProfile::Throughput);
        assert_eq!(m.get("RBITNET_CONTINUOUS_BATCHING"), Some(&"1"));
        assert_eq!(m.get("RBITNET_KV_POOL"), Some(&"1"));
        assert_eq!(m.get("RBITNET_KV_QUANT"), Some(&"q8"));
        assert_eq!(m.get("RBITNET_PREFIX_KV"), Some(&"1"));
    }

    #[test]
    fn bitnet_cpu_exports_provider_flags() {
        let m = env_map(TuneProfile::BitnetCpu);
        assert_eq!(m.get("RBITNET_ARCHITECTURE"), Some(&"bitnet"));
        assert_eq!(m.get("RBITNET_BACKEND"), Some(&"cpu"));
        assert_eq!(m.get("RBITNET_PREFIX_KV"), Some(&"1"));
        assert_eq!(m.get("RBITNET_CUDA_GRAPH"), Some(&"0"));
    }

    #[test]
    fn battery_disables_heavy_paths() {
        let m = env_map(TuneProfile::Battery);
        assert_eq!(m.get("RBITNET_MAX_CONCURRENT"), Some(&"1"));
        assert_eq!(m.get("RBITNET_PREFIX_KV"), Some(&"0"));
        assert_eq!(m.get("RBITNET_CONTINUOUS_BATCHING"), Some(&"0"));
    }
}
