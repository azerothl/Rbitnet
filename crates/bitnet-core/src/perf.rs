//! Lightweight engine-wide performance counters.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PerfSnapshot {
    pub quant_matvec_calls: u64,
    pub quant_matvec_rows: u64,
    pub quant_matvec_cols: u64,
    pub quant_matvec_ns: u64,
    pub gpu_upload_bytes: u64,
    pub gpu_download_bytes: u64,
    pub gpu_gemv_calls: u64,
    pub scratch_alloc_bytes: u64,
    pub scratch_reuse_hits: u64,
    pub kv_write_bytes: u64,
    pub kv_physical_pages: u64,
    pub kv_new_physical_pages: u64,
    pub kv_reused_physical_pages: u64,
    pub kv_quant_format_code: u64,
    pub model_load_ms: u64,
    pub model_loads: u64,
    pub scheduler_batches: u64,
    pub scheduler_batch_items: u64,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
    pub prefix_cache_bytes_saved: u64,
    pub speculative_draft_tokens: u64,
    pub speculative_verified_tokens: u64,
    pub speculative_accepted_tokens: u64,
    pub cuda_graph_replays: u64,
    pub scheduler_decode_waves: u64,
}

#[derive(Debug, Default)]
struct PerfCounters {
    quant_matvec_calls: AtomicU64,
    quant_matvec_rows: AtomicU64,
    quant_matvec_cols: AtomicU64,
    quant_matvec_ns: AtomicU64,
    gpu_upload_bytes: AtomicU64,
    gpu_download_bytes: AtomicU64,
    gpu_gemv_calls: AtomicU64,
    scratch_alloc_bytes: AtomicU64,
    scratch_reuse_hits: AtomicU64,
    kv_write_bytes: AtomicU64,
    kv_physical_pages: AtomicU64,
    kv_new_physical_pages: AtomicU64,
    kv_reused_physical_pages: AtomicU64,
    kv_quant_format_code: AtomicU64,
    model_load_ms: AtomicU64,
    model_loads: AtomicU64,
    scheduler_batches: AtomicU64,
    scheduler_batch_items: AtomicU64,
    prefix_cache_hits: AtomicU64,
    prefix_cache_misses: AtomicU64,
    prefix_cache_bytes_saved: AtomicU64,
    speculative_draft_tokens: AtomicU64,
    speculative_verified_tokens: AtomicU64,
    speculative_accepted_tokens: AtomicU64,
    cuda_graph_replays: AtomicU64,
    scheduler_decode_waves: AtomicU64,
    quant_by_type: Mutex<Vec<(u32, QuantTypeStats)>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct QuantTypeStats {
    calls: u64,
    rows: u64,
    ns: u64,
}

static PERF: OnceLock<PerfCounters> = OnceLock::new();

fn perf() -> &'static PerfCounters {
    PERF.get_or_init(PerfCounters::default)
}

pub fn record_quant_matvec(ggml_type: u32, rows: usize, cols: usize, elapsed_ns: u64) {
    let p = perf();
    p.quant_matvec_calls.fetch_add(1, Ordering::Relaxed);
    p.quant_matvec_rows
        .fetch_add(rows as u64, Ordering::Relaxed);
    p.quant_matvec_cols
        .fetch_add(cols as u64, Ordering::Relaxed);
    p.quant_matvec_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    let mut by_type = match p.quant_by_type.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some((_, stats)) = by_type.iter_mut().find(|(ty, _)| *ty == ggml_type) {
        stats.calls += 1;
        stats.rows += rows as u64;
        stats.ns += elapsed_ns;
    } else {
        by_type.push((
            ggml_type,
            QuantTypeStats {
                calls: 1,
                rows: rows as u64,
                ns: elapsed_ns,
            },
        ));
    }
}

pub fn record_gpu_transfer(upload_bytes: u64, download_bytes: u64, gemv_calls: u64) {
    let p = perf();
    p.gpu_upload_bytes
        .fetch_add(upload_bytes, Ordering::Relaxed);
    p.gpu_download_bytes
        .fetch_add(download_bytes, Ordering::Relaxed);
    p.gpu_gemv_calls.fetch_add(gemv_calls, Ordering::Relaxed);
}

pub fn record_scratch_alloc(bytes: usize) {
    perf()
        .scratch_alloc_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
}

pub fn record_scratch_reuse_hit() {
    perf().scratch_reuse_hits.fetch_add(1, Ordering::Relaxed);
}

pub fn record_kv_write(bytes: usize) {
    perf()
        .kv_write_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
}

pub fn record_kv_backend(
    physical_pages: usize,
    new_physical_pages: usize,
    reused_physical_pages: usize,
    quant_format_code: u64,
) {
    let p = perf();
    p.kv_physical_pages
        .store(physical_pages as u64, Ordering::Relaxed);
    p.kv_new_physical_pages
        .store(new_physical_pages as u64, Ordering::Relaxed);
    p.kv_reused_physical_pages
        .store(reused_physical_pages as u64, Ordering::Relaxed);
    p.kv_quant_format_code
        .store(quant_format_code, Ordering::Relaxed);
}

pub fn record_model_load(elapsed_ms: u64) {
    let p = perf();
    p.model_load_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
    p.model_loads.fetch_add(1, Ordering::Relaxed);
}

pub fn record_scheduler_batch(items: usize) {
    let p = perf();
    p.scheduler_batches.fetch_add(1, Ordering::Relaxed);
    p.scheduler_batch_items
        .fetch_add(items as u64, Ordering::Relaxed);
}

pub fn record_prefix_cache_hit(bytes_saved: usize) {
    let p = perf();
    p.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    p.prefix_cache_bytes_saved
        .fetch_add(bytes_saved as u64, Ordering::Relaxed);
}

pub fn record_prefix_cache_miss() {
    perf().prefix_cache_misses.fetch_add(1, Ordering::Relaxed);
}

pub fn record_cuda_graph_replay() {
    perf().cuda_graph_replays.fetch_add(1, Ordering::Relaxed);
}

pub fn record_scheduler_decode_wave(steps: usize) {
    perf()
        .scheduler_decode_waves
        .fetch_add(steps as u64, Ordering::Relaxed);
}

pub fn record_speculative(draft_tokens: u32, verified_tokens: u32, accepted_tokens: u32) {
    let p = perf();
    p.speculative_draft_tokens
        .fetch_add(draft_tokens as u64, Ordering::Relaxed);
    p.speculative_verified_tokens
        .fetch_add(verified_tokens as u64, Ordering::Relaxed);
    p.speculative_accepted_tokens
        .fetch_add(accepted_tokens as u64, Ordering::Relaxed);
}

pub fn snapshot() -> PerfSnapshot {
    let p = perf();
    PerfSnapshot {
        quant_matvec_calls: p.quant_matvec_calls.load(Ordering::Relaxed),
        quant_matvec_rows: p.quant_matvec_rows.load(Ordering::Relaxed),
        quant_matvec_cols: p.quant_matvec_cols.load(Ordering::Relaxed),
        quant_matvec_ns: p.quant_matvec_ns.load(Ordering::Relaxed),
        gpu_upload_bytes: p.gpu_upload_bytes.load(Ordering::Relaxed),
        gpu_download_bytes: p.gpu_download_bytes.load(Ordering::Relaxed),
        gpu_gemv_calls: p.gpu_gemv_calls.load(Ordering::Relaxed),
        scratch_alloc_bytes: p.scratch_alloc_bytes.load(Ordering::Relaxed),
        scratch_reuse_hits: p.scratch_reuse_hits.load(Ordering::Relaxed),
        kv_write_bytes: p.kv_write_bytes.load(Ordering::Relaxed),
        kv_physical_pages: p.kv_physical_pages.load(Ordering::Relaxed),
        kv_new_physical_pages: p.kv_new_physical_pages.load(Ordering::Relaxed),
        kv_reused_physical_pages: p.kv_reused_physical_pages.load(Ordering::Relaxed),
        kv_quant_format_code: p.kv_quant_format_code.load(Ordering::Relaxed),
        model_load_ms: p.model_load_ms.load(Ordering::Relaxed),
        model_loads: p.model_loads.load(Ordering::Relaxed),
        scheduler_batches: p.scheduler_batches.load(Ordering::Relaxed),
        scheduler_batch_items: p.scheduler_batch_items.load(Ordering::Relaxed),
        prefix_cache_hits: p.prefix_cache_hits.load(Ordering::Relaxed),
        prefix_cache_misses: p.prefix_cache_misses.load(Ordering::Relaxed),
        prefix_cache_bytes_saved: p.prefix_cache_bytes_saved.load(Ordering::Relaxed),
        speculative_draft_tokens: p.speculative_draft_tokens.load(Ordering::Relaxed),
        speculative_verified_tokens: p.speculative_verified_tokens.load(Ordering::Relaxed),
        speculative_accepted_tokens: p.speculative_accepted_tokens.load(Ordering::Relaxed),
        cuda_graph_replays: p.cuda_graph_replays.load(Ordering::Relaxed),
        scheduler_decode_waves: p.scheduler_decode_waves.load(Ordering::Relaxed),
    }
}

pub fn prometheus_text() -> String {
    let snap = snapshot();
    let mut s = String::new();
    macro_rules! counter {
        ($name:literal, $help:literal, $value:expr) => {{
            writeln!(s, "# HELP {} {}", $name, $help).unwrap();
            writeln!(s, "# TYPE {} counter", $name).unwrap();
            writeln!(s, "{} {}", $name, $value).unwrap();
        }};
    }
    counter!(
        "rbitnet_core_quant_matvec_calls_total",
        "Quantized matrix-vector calls executed by bitnet-core",
        snap.quant_matvec_calls
    );
    counter!(
        "rbitnet_core_quant_matvec_rows_total",
        "Logical output rows processed by quantized matrix-vector kernels",
        snap.quant_matvec_rows
    );
    counter!(
        "rbitnet_core_quant_matvec_ns_total",
        "Wall time spent in quantized matrix-vector kernels, nanoseconds",
        snap.quant_matvec_ns
    );
    counter!(
        "rbitnet_core_gpu_upload_bytes_total",
        "Bytes copied from host to GPU device by bitnet-core",
        snap.gpu_upload_bytes
    );
    counter!(
        "rbitnet_core_gpu_download_bytes_total",
        "Bytes copied from GPU device to host by bitnet-core",
        snap.gpu_download_bytes
    );
    counter!(
        "rbitnet_core_gpu_gemv_calls_total",
        "GPU GEMV calls issued by bitnet-core",
        snap.gpu_gemv_calls
    );
    counter!(
        "rbitnet_core_scratch_alloc_bytes_total",
        "Bytes allocated by runtime scratch arenas",
        snap.scratch_alloc_bytes
    );
    counter!(
        "rbitnet_core_scratch_reuse_hits_total",
        "Scratch arena buffer reuse hits",
        snap.scratch_reuse_hits
    );
    counter!(
        "rbitnet_core_kv_write_bytes_total",
        "Bytes written into KV cache backends",
        snap.kv_write_bytes
    );
    counter!(
        "rbitnet_core_kv_physical_pages",
        "Current physical KV pages tracked by the active backend",
        snap.kv_physical_pages
    );
    counter!(
        "rbitnet_core_kv_new_physical_pages_total",
        "Physical KV pages newly allocated by the active backend",
        snap.kv_new_physical_pages
    );
    counter!(
        "rbitnet_core_kv_reused_physical_pages_total",
        "Physical KV pages reused by the active backend",
        snap.kv_reused_physical_pages
    );
    counter!(
        "rbitnet_core_kv_quant_format_code",
        "Active KV quant format code: 0=f32/off, 1=q8, 2=q4",
        snap.kv_quant_format_code
    );
    counter!(
        "rbitnet_core_model_load_ms_total",
        "Model load wall time in milliseconds",
        snap.model_load_ms
    );
    counter!(
        "rbitnet_core_model_loads_total",
        "Model load attempts completed by bitnet-core",
        snap.model_loads
    );
    counter!(
        "rbitnet_core_scheduler_batches_total",
        "Scheduler batch waves planned by bitnet-core",
        snap.scheduler_batches
    );
    counter!(
        "rbitnet_core_scheduler_batch_items_total",
        "Scheduled request items handled by batch waves",
        snap.scheduler_batch_items
    );
    counter!(
        "rbitnet_core_prefix_cache_hits_total",
        "Prefix KV cache hits",
        snap.prefix_cache_hits
    );
    counter!(
        "rbitnet_core_prefix_cache_misses_total",
        "Prefix KV cache misses",
        snap.prefix_cache_misses
    );
    counter!(
        "rbitnet_core_prefix_cache_bytes_saved_total",
        "Estimated KV bytes saved by prefix cache hits",
        snap.prefix_cache_bytes_saved
    );
    counter!(
        "rbitnet_core_speculative_draft_tokens_total",
        "Tokens proposed by speculative draft paths",
        snap.speculative_draft_tokens
    );
    counter!(
        "rbitnet_core_speculative_verified_tokens_total",
        "Tokens verified or completed by the target model after speculative draft",
        snap.speculative_verified_tokens
    );
    counter!(
        "rbitnet_core_speculative_accepted_tokens_total",
        "Draft tokens accepted by the lightweight verifier path",
        snap.speculative_accepted_tokens
    );
    counter!(
        "rbitnet_core_cuda_graph_replays_total",
        "Decode steps recorded under CUDA graph mode",
        snap.cuda_graph_replays
    );
    counter!(
        "rbitnet_core_scheduler_decode_waves_total",
        "Decode wave steps planned by continuous batching scheduler",
        snap.scheduler_decode_waves
    );

    let by_type = match perf().quant_by_type.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    writeln!(
        s,
        "# HELP rbitnet_core_quant_matvec_by_type_total Quantized matvec calls by GGML type"
    )
    .unwrap();
    writeln!(s, "# TYPE rbitnet_core_quant_matvec_by_type_total counter").unwrap();
    for (ty, stats) in by_type.iter() {
        writeln!(
            s,
            "rbitnet_core_quant_matvec_by_type_total{{ggml_type=\"{}\"}} {}",
            ty, stats.calls
        )
        .unwrap();
        writeln!(
            s,
            "rbitnet_core_quant_matvec_rows_by_type_total{{ggml_type=\"{}\"}} {}",
            ty, stats.rows
        )
        .unwrap();
        writeln!(
            s,
            "rbitnet_core_quant_matvec_ns_by_type_total{{ggml_type=\"{}\"}} {}",
            ty, stats.ns
        )
        .unwrap();
    }
    s
}
