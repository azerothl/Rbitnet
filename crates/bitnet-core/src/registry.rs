//! Kernel registry skeleton with model-family namespaces.

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct KernelRegistry {
    entries: HashMap<String, String>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, namespace: &str, op: &str, kernel_id: &str) {
        let key = format!("{namespace}/{op}");
        self.entries.insert(key, kernel_id.to_string());
    }

    pub fn resolve(&self, namespace: &str, op: &str) -> Option<&str> {
        self.entries
            .get(&format!("{namespace}/{op}"))
            .map(|s| s.as_str())
    }

    pub fn bootstrap_default() -> Self {
        let mut registry = Self::new();
        registry.register("llama", "matvec", "cpu_ref_matvec");
        registry.register("llama", "attention_decode", "cpu_ref_attention_decode");
        registry.register("bitnet", "matvec", "cpu_ref_bitnet_matvec");
        registry.register("bitnet", "attention_decode", "cpu_ref_attention_decode");
        registry
    }
}
