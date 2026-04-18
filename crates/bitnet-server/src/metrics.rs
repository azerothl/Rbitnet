//! Simple Prometheus-style counters (no external exporter dependency).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// Request and inference counters for `GET /metrics`.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub chat_requests_total: AtomicU64,
    pub chat_errors_total: AtomicU64,
    pub inference_timeouts_total: AtomicU64,
    pub inference_ms_total: AtomicU64,
    pub inference_calls_total: AtomicU64,
    pub unauthorized_total: AtomicU64,
}

impl ServerMetrics {
    pub fn prometheus_text(&self) -> String {
        let cr = self.chat_requests_total.load(Ordering::Relaxed);
        let ce = self.chat_errors_total.load(Ordering::Relaxed);
        let ct = self.inference_timeouts_total.load(Ordering::Relaxed);
        let it = self.inference_ms_total.load(Ordering::Relaxed);
        let ic = self.inference_calls_total.load(Ordering::Relaxed);
        let ua = self.unauthorized_total.load(Ordering::Relaxed);

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

        s
    }
}
