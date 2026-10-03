//! Draft for #83/#86: scoped statistics without retaining a loaded model.
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Default)]
pub(crate) struct Layer {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub upload_bytes: AtomicU64,
    pub upload_ns: AtomicU64,
    pub lock_wait_ns: AtomicU64,
    pub capacity_refusals: AtomicU64,
    pub ready_experts: AtomicU64,
    pub ready_bytes: AtomicU64,
    pub gpu_ffns: AtomicU64,
    pub fused_gpu_ffns: AtomicU64,
    pub fallback_ffns: AtomicU64,
    pub gpu_ffn_ns: AtomicU64,
    pub fallback_ffn_ns: AtomicU64,
    pub timed_gpu_ffns: AtomicU64,
    pub timed_fallback_ffns: AtomicU64,
    pub cpu_decisions: AtomicU64,
    pub gpu_decisions: AtomicU64,
}
pub(crate) struct Model {
    pub id: u64,
    pub name: String,
    pub architecture: String,
    pub layers: Vec<Layer>,
}
static NEXT: AtomicU64 = AtomicU64::new(1);
fn registry() -> &'static Mutex<Vec<Weak<Model>>> {
    static MODELS: OnceLock<Mutex<Vec<Weak<Model>>>> = OnceLock::new();
    MODELS.get_or_init(|| Mutex::new(Vec::new()))
}
impl Model {
    pub(crate) fn new(name: String, architecture: String, layers: usize) -> Arc<Self> {
        let model = Arc::new(Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            name,
            architecture,
            layers: (0..layers).map(|_| Layer::default()).collect(),
        });
        let mut models = registry().lock().unwrap_or_else(|e| e.into_inner());
        models.retain(|m| m.strong_count() > 0);
        models.push(Arc::downgrade(&model));
        model
    }
    pub(crate) fn layer(&self, layer: usize) -> Option<&Layer> {
        self.layers.get(layer)
    }
    pub(crate) fn fused_ffn(&self,layer:usize) {
        if let Some(l)=self.layer(layer){l.fused_gpu_ffns.fetch_add(1,Ordering::Relaxed);}
    }
    pub(crate) fn ffn(&self, layer: usize, gpu: bool, ns: Option<u64>) {
        let Some(l) = self.layer(layer) else { return };
        let (count, time, timed) = if gpu {
            (&l.gpu_ffns, &l.gpu_ffn_ns, &l.timed_gpu_ffns)
        } else {
            (&l.fallback_ffns, &l.fallback_ffn_ns, &l.timed_fallback_ffns)
        };
        count.fetch_add(1, Ordering::Relaxed);
        if let Some(ns) = ns {
            time.fetch_add(ns, Ordering::Relaxed);
            timed.fetch_add(1, Ordering::Relaxed);
        }
    }
}
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
pub(crate) fn prometheus_text() -> String {
    // Temporary Arc owners protect only these counters, never weights, streams
    // or cache leases. Unloaded runtimes disappear on the next scrape.
    let models = {
        let mut entries = registry().lock().unwrap_or_else(|e| e.into_inner());
        let models: Vec<_> = entries.iter().filter_map(Weak::upgrade).collect();
        entries.retain(|m| m.strong_count() > 0);
        models
    };
    let mut text = String::new();
    let names = [
        ("cache_hits_total", "counter"),
        ("cache_misses_total", "counter"),
        ("cache_evictions_total", "counter"),
        ("cache_upload_bytes_total", "counter"),
        ("cache_upload_ns_total", "counter"),
        ("cache_lock_wait_ns_total", "counter"),
        ("cache_capacity_refusals_total", "counter"),
        ("cache_ready_experts", "gauge"),
        ("cache_ready_bytes", "gauge"),
        ("gpu_ffns_total", "counter"),
        ("fused_gpu_ffns_total", "counter"),
        ("fallback_ffns_total", "counter"),
        ("gpu_ffn_ns_total", "counter"),
        ("fallback_ffn_ns_total", "counter"),
        ("timed_gpu_ffns_total", "counter"),
        ("timed_fallback_ffns_total", "counter"),
        ("cpu_decisions_total", "counter"),
        ("gpu_decisions_total", "counter"),
    ];
    for (name, kind) in names {
        writeln!(text, "# TYPE rbitnet_moe_layer_{name} {kind}").unwrap();
    }
    for model in models {
        for (index, l) in model.layers.iter().enumerate() {
            let counters = [
                &l.hits,
                &l.misses,
                &l.evictions,
                &l.upload_bytes,
                &l.upload_ns,
                &l.lock_wait_ns,
                &l.capacity_refusals,
                &l.ready_experts,
                &l.ready_bytes,
                &l.gpu_ffns,
                &l.fused_gpu_ffns,
                &l.fallback_ffns,
                &l.gpu_ffn_ns,
                &l.fallback_ffn_ns,
                &l.timed_gpu_ffns,
                &l.timed_fallback_ffns,
                &l.cpu_decisions,
                &l.gpu_decisions,
            ];
            for ((name, _), value) in names.iter().zip(counters) {
                writeln!(text,"rbitnet_moe_layer_{name}{{model_id=\"{}\",model=\"{}\",architecture=\"{}\",layer=\"{index}\"}} {}",
                    model.id,label(&model.name),label(&model.architecture),value.load(Ordering::Relaxed)).unwrap();
            }
        }
    }
    text
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unload_prunes_scoped_counters_without_retaining_model_weights() {
        let first = Model::new("metrics-unload-fixture".into(), "gpt-oss".into(), 2);
        let second = Model::new("metrics-survivor-fixture".into(), "deepseek2".into(), 1);
        let weak = Arc::downgrade(&first);
        first.ffn(1, true, None);
        second.ffn(0, false, Some(17));
        let text = prometheus_text();
        assert!(text.contains("metrics-unload-fixture"));
        assert!(text.contains("metrics-survivor-fixture"));
        assert_eq!(first.layers[1].timed_gpu_ffns.load(Ordering::Relaxed), 0);
        assert_eq!(second.layers[0].fallback_ffn_ns.load(Ordering::Relaxed), 17);
        drop(first);
        assert!(weak.upgrade().is_none());
        let text = prometheus_text();
        assert!(!text.contains("metrics-unload-fixture"));
        assert!(text.contains("metrics-survivor-fixture"));
    }
    #[test]
    fn labels_escape_quotes_backslashes_and_line_breaks() {
        assert_eq!(label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
