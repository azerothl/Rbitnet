//! Model profile and residency policy scaffolding.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub gguf: PathBuf,
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub max_vram_mb: Option<u64>,
    #[serde(default)]
    pub max_ram_mb: Option<u64>,
    #[serde(default)]
    pub hybrid_layers: Option<String>,
    #[serde(default)]
    pub hybrid_policy: Option<String>,
    #[serde(default)]
    pub warmup: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelResidency {
    pub id: String,
    pub loaded: bool,
    pub warm: bool,
    pub last_used_ms_ago: Option<u64>,
    pub estimated_vram_mb: Option<u64>,
    pub estimated_ram_mb: Option<u64>,
    pub fallback_reason: Option<String>,
    pub offload_decisions: Vec<OffloadDecision>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct OffloadDecision {
    pub tensor: String,
    pub device: String,
    pub reason: String,
}

#[derive(Debug, Default)]
pub struct ModelManager {
    profiles: HashMap<String, ModelProfile>,
    loaded: HashMap<String, LoadedModelState>,
    max_loaded: usize,
}

#[derive(Debug)]
struct LoadedModelState {
    warm: bool,
    last_used: Instant,
    estimated_vram_mb: Option<u64>,
    estimated_ram_mb: Option<u64>,
    fallback_reason: Option<String>,
    offload_decisions: Vec<OffloadDecision>,
}

impl ModelManager {
    pub fn new(max_loaded: usize) -> Self {
        Self {
            profiles: HashMap::new(),
            loaded: HashMap::new(),
            max_loaded: max_loaded.max(1),
        }
    }

    pub fn upsert_profile(&mut self, profile: ModelProfile) {
        self.profiles.insert(profile.id.clone(), profile);
    }

    pub fn profile(&self, id: &str) -> Option<&ModelProfile> {
        self.profiles.get(id)
    }

    pub fn mark_loaded(
        &mut self,
        id: impl Into<String>,
        warm: bool,
        estimated_vram_mb: Option<u64>,
        estimated_ram_mb: Option<u64>,
        fallback_reason: Option<String>,
    ) -> Vec<String> {
        let id = id.into();
        self.loaded.insert(
            id,
            LoadedModelState {
                warm,
                last_used: Instant::now(),
                estimated_vram_mb,
                estimated_ram_mb,
                fallback_reason,
                offload_decisions: Vec::new(),
            },
        );
        self.evict_lru()
    }

    pub fn set_offload_decisions(&mut self, id: &str, decisions: Vec<OffloadDecision>) {
        if let Some(state) = self.loaded.get_mut(id) {
            state.offload_decisions = decisions;
            state.last_used = Instant::now();
        }
    }

    pub fn touch(&mut self, id: &str) {
        if let Some(state) = self.loaded.get_mut(id) {
            state.last_used = Instant::now();
        }
    }

    pub fn residency(&self) -> Vec<ModelResidency> {
        let now = Instant::now();
        let mut out = Vec::new();
        for id in self.profiles.keys() {
            if let Some(state) = self.loaded.get(id) {
                out.push(ModelResidency {
                    id: id.clone(),
                    loaded: true,
                    warm: state.warm,
                    last_used_ms_ago: Some(now.duration_since(state.last_used).as_millis() as u64),
                    estimated_vram_mb: state.estimated_vram_mb,
                    estimated_ram_mb: state.estimated_ram_mb,
                    fallback_reason: state.fallback_reason.clone(),
                    offload_decisions: state.offload_decisions.clone(),
                });
            } else {
                out.push(ModelResidency {
                    id: id.clone(),
                    loaded: false,
                    warm: false,
                    last_used_ms_ago: None,
                    estimated_vram_mb: None,
                    estimated_ram_mb: None,
                    fallback_reason: None,
                    offload_decisions: Vec::new(),
                });
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    fn evict_lru(&mut self) -> Vec<String> {
        let mut evicted = Vec::new();
        while self.loaded.len() > self.max_loaded {
            let Some((victim, _)) = self
                .loaded
                .iter()
                .min_by_key(|(_, state)| state.last_used)
                .map(|(id, state)| (id.clone(), state.last_used))
            else {
                break;
            };
            self.loaded.remove(&victim);
            evicted.push(victim);
        }
        evicted
    }
}
