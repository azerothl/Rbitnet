//! Simple Prometheus-style counters (no external exporter dependency).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Escape a Prometheus label value per the text exposition format spec:
/// backslash → `\\`, double-quote → `\"`, newline → `\n`.
fn escape_label_value(v: &str) -> String {
    v.replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

/// Request and inference counters for `GET /metrics`.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub chat_requests_total: AtomicU64,
    pub chat_errors_total: AtomicU64,
    pub inference_timeouts_total: AtomicU64,
    pub inference_ms_total: AtomicU64,
    pub inference_calls_total: AtomicU64,
    pub unauthorized_total: AtomicU64,
    pub inference_by_backend_family: Mutex<Vec<((String, String), u64)>>,
    pub inference_ttft_ms_total: AtomicU64,
    pub inference_encode_ms_total: AtomicU64,
    pub inference_prefill_ms_total: AtomicU64,
    pub inference_decode_ms_total: AtomicU64,
    pub inference_itl_us_total: AtomicU64,
    pub inference_tpot_us_total: AtomicU64,
    pub speculative_requests_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    pub native_accelerated_calls_total: AtomicU64,
    pub model_unloads_total: AtomicU64,
}

impl ServerMetrics {
    pub fn record_backend_family_call(&self, backend: &str, family: &str) {
        let mut rows = match self.inference_by_backend_family.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some((_, count)) = rows
            .iter_mut()
            .find(|((b, f), _)| b == backend && f == family)
        {
            *count += 1;
            return;
        }
        rows.push(((backend.to_string(), family.to_string()), 1));
    }

    pub fn prometheus_text(&self) -> String {
        let cr = self.chat_requests_total.load(Ordering::Relaxed);
        let ce = self.chat_errors_total.load(Ordering::Relaxed);
        let ct = self.inference_timeouts_total.load(Ordering::Relaxed);
        let it = self.inference_ms_total.load(Ordering::Relaxed);
        let ic = self.inference_calls_total.load(Ordering::Relaxed);
        let ua = self.unauthorized_total.load(Ordering::Relaxed);
        let ttft_sum = self.inference_ttft_ms_total.load(Ordering::Relaxed);
        let encode_sum = self.inference_encode_ms_total.load(Ordering::Relaxed);
        let prefill_sum = self.inference_prefill_ms_total.load(Ordering::Relaxed);
        let decode_sum = self.inference_decode_ms_total.load(Ordering::Relaxed);
        let itl_sum = self.inference_itl_us_total.load(Ordering::Relaxed);
        let tpot_sum = self.inference_tpot_us_total.load(Ordering::Relaxed);
        let spec_total = self.speculative_requests_total.load(Ordering::Relaxed);
        let completion_tokens = self.completion_tokens_total.load(Ordering::Relaxed);
        let native_calls = self.native_accelerated_calls_total.load(Ordering::Relaxed);
        let unloads = self.model_unloads_total.load(Ordering::Relaxed);

        let mut s = String::new();
        writeln!(
            s,
            "# HELP rbitnet_chat_requests_total Chat completion requests"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_chat_requests_total counter").unwrap();
        writeln!(s, "rbitnet_chat_requests_total {cr}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_chat_errors_total Chat completion handler errors (4xx/5xx from handler)"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_chat_errors_total counter").unwrap();
        writeln!(s, "rbitnet_chat_errors_total {ce}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_inference_timeouts_total Inference wall-clock timeouts"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_timeouts_total counter").unwrap();
        writeln!(s, "rbitnet_inference_timeouts_total {ct}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_inference_ms_sum Sum of inference wall times in milliseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_ms_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_ms_sum {it}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_inference_calls_total Completed inference calls (excludes timeouts)"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_calls_total counter").unwrap();
        writeln!(s, "rbitnet_inference_calls_total {ic}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_unauthorized_total Rejected requests (missing/invalid API key)"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_unauthorized_total counter").unwrap();
        writeln!(s, "rbitnet_unauthorized_total {ua}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_model_unloads_total Admin or idle-triggered model unload events"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_model_unloads_total counter").unwrap();
        writeln!(s, "rbitnet_model_unloads_total {unloads}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_inference_ttft_ms_sum Sum of estimated time-to-first-token in milliseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_ttft_ms_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_ttft_ms_sum {ttft_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_inference_encode_ms_sum Sum of tokenizer encode wall times in milliseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_encode_ms_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_encode_ms_sum {encode_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_inference_prefill_ms_sum Sum of model prefill phase wall times in milliseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_prefill_ms_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_prefill_ms_sum {prefill_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_inference_decode_ms_sum Sum of model decode phase wall times in milliseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_decode_ms_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_decode_ms_sum {decode_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_inference_itl_us_sum Sum of average inter-token latency (microseconds per generated token) per completed inference"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_itl_us_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_itl_us_sum {itl_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_inference_tpot_us_sum Sum of estimated time-per-output-token in microseconds"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_inference_tpot_us_sum counter").unwrap();
        writeln!(s, "rbitnet_inference_tpot_us_sum {tpot_sum}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_speculative_requests_total Requests handled with speculative path"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_speculative_requests_total counter").unwrap();
        writeln!(s, "rbitnet_speculative_requests_total {spec_total}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_completion_tokens_total Total completion tokens (generated; from tokenizer counts when the runtime reports them)"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_completion_tokens_total counter").unwrap();
        writeln!(s, "rbitnet_completion_tokens_total {completion_tokens}").unwrap();
        writeln!(
            s,
            "# HELP rbitnet_native_accelerated_calls_total Inference calls executed on native accelerated backend paths"
        )
        .unwrap();
        writeln!(s, "# TYPE rbitnet_native_accelerated_calls_total counter").unwrap();
        writeln!(s, "rbitnet_native_accelerated_calls_total {native_calls}").unwrap();

        writeln!(
            s,
            "# HELP rbitnet_inference_calls_by_backend_family_total Completed inference calls by backend/model family"
        )
        .unwrap();
        writeln!(
            s,
            "# TYPE rbitnet_inference_calls_by_backend_family_total counter"
        )
        .unwrap();
        let rows = match self.inference_by_backend_family.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for ((backend, family), count) in rows.iter() {
            writeln!(
                s,
                "rbitnet_inference_calls_by_backend_family_total{{backend=\"{}\",family=\"{}\"}} {}",
                escape_label_value(backend),
                escape_label_value(family),
                count
            )
            .unwrap();
        }

        s
    }
}
