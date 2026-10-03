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
    pub gpu_attention_calls: u64,
    pub gpu_qwen_full_tokens: u64,
    pub gpu_gpt_full_tokens: u64,
    pub gpu_split_attention_queries: u64,
    pub gpu_tensor_gemm_calls: u64,
    pub scratch_alloc_bytes: u64,
    pub scratch_reuse_hits: u64,
    pub kv_write_bytes: u64,
    pub kv_physical_pages: u64,
    pub kv_new_physical_pages: u64,
    pub kv_reused_physical_pages: u64,
    pub kv_quant_format_code: u64,
    pub kv_pool_active_seqs: u64,
    pub kv_pool_allocated_pages: u64,
    pub kv_pool_free_pages: u64,
    pub kv_pool_fragmentation_permille: u64,
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
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub expert_cache_evictions: u64,
    pub expert_cache_upload_bytes: u64,
    pub expert_cache_upload_ns: u64,
    pub native_moe_resident_layers: u64,
    pub native_moe_fallback_layers: u64,
    pub native_moe_resident_ns: u64,
    pub native_moe_fallback_ns: u64,
    pub gpu_prefill_blocks: u64,
    pub gpu_prefill_tokens: u64,
    pub gpu_quant_gemm_calls: u64,
    pub speculative_verify_blocks: u64,
    pub speculative_target_tokens: u64,
    pub speculative_rollbacks: u64,
    pub speculative_draft_ns: u64,
    pub speculative_verify_ns: u64,
    pub scheduler_decode_waves: u64,
    pub scheduler_prefill_chunks: u64,
    pub scheduler_stall_free_iters: u64,
    pub scheduler_iteration_budget: u64,
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
    gpu_attention_calls: AtomicU64,
    gpu_qwen_full_tokens: AtomicU64,
    gpu_gpt_full_tokens: AtomicU64,
    gpu_split_attention_queries: AtomicU64,
    gpu_tensor_gemm_calls: AtomicU64,
    scratch_alloc_bytes: AtomicU64,
    scratch_reuse_hits: AtomicU64,
    kv_write_bytes: AtomicU64,
    kv_physical_pages: AtomicU64,
    kv_new_physical_pages: AtomicU64,
    kv_reused_physical_pages: AtomicU64,
    kv_quant_format_code: AtomicU64,
    kv_pool_active_seqs: AtomicU64,
    kv_pool_allocated_pages: AtomicU64,
    kv_pool_free_pages: AtomicU64,
    kv_pool_fragmentation_permille: AtomicU64,
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
    expert_cache_hits: AtomicU64,
    expert_cache_misses: AtomicU64,
    expert_cache_evictions: AtomicU64,
    expert_cache_upload_bytes: AtomicU64,
    expert_cache_upload_ns: AtomicU64,
    native_moe_resident_layers: AtomicU64,
    native_moe_fallback_layers: AtomicU64,
    native_moe_resident_ns: AtomicU64,
    native_moe_fallback_ns: AtomicU64,
    gpu_prefill_blocks: AtomicU64,
    gpu_prefill_tokens: AtomicU64,
    gpu_quant_gemm_calls: AtomicU64,
    speculative_verify_blocks: AtomicU64,
    speculative_target_tokens: AtomicU64,
    speculative_rollbacks: AtomicU64,
    speculative_draft_ns: AtomicU64,
    speculative_verify_ns: AtomicU64,
    scheduler_decode_waves: AtomicU64,
    scheduler_prefill_chunks: AtomicU64,
    scheduler_stall_free_iters: AtomicU64,
    scheduler_iteration_budget: AtomicU64,
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

pub(crate) fn record_gpu_prefill(tokens: usize, gemm_calls: u64) {
    let p = perf();
    p.gpu_prefill_blocks.fetch_add(1, Ordering::Relaxed);
    p.gpu_prefill_tokens
        .fetch_add(tokens as u64, Ordering::Relaxed);
    p.gpu_quant_gemm_calls
        .fetch_add(gemm_calls, Ordering::Relaxed);
}

pub(crate) fn record_qwen_full_token() {
    record_qwen_full_tokens(1);
}
pub(crate) fn record_qwen_full_tokens(count: usize) {
    perf()
        .gpu_qwen_full_tokens
        .fetch_add(count as u64, Ordering::Relaxed);
}
pub(crate) fn record_split_attention(queries: u64) {
    perf()
        .gpu_split_attention_queries
        .fetch_add(queries, Ordering::Relaxed);
}
pub(crate) fn record_gpt_full_token() {
    perf().gpu_gpt_full_tokens.fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn record_gpu_tensor_gemm(calls: u64) {
    perf()
        .gpu_tensor_gemm_calls
        .fetch_add(calls, Ordering::Relaxed);
}

pub fn record_gpu_transfer(upload_bytes: u64, download_bytes: u64, gemv_calls: u64) {
    let p = perf();
    p.gpu_upload_bytes
        .fetch_add(upload_bytes, Ordering::Relaxed);
    p.gpu_download_bytes
        .fetch_add(download_bytes, Ordering::Relaxed);
    p.gpu_gemv_calls.fetch_add(gemv_calls, Ordering::Relaxed);
}

/// Routed FFN stages only. Fallback can retain individual GPU projections;
/// resident timing includes cache fills. Full token graphs count layers but
/// cannot provide independent per-FFN host wall times (ns=0 there).
pub fn record_native_moe(resident: bool, layers: u64, ns: u64) {
    let p = perf();
    if resident {
        p.native_moe_resident_layers
            .fetch_add(layers, Ordering::Relaxed);
        p.native_moe_resident_ns.fetch_add(ns, Ordering::Relaxed);
    } else {
        p.native_moe_fallback_layers
            .fetch_add(layers, Ordering::Relaxed);
        p.native_moe_fallback_ns.fetch_add(ns, Ordering::Relaxed);
    }
}

pub(crate) fn record_gpu_verification(tokens: usize, gemm_calls: u64) {
    let p = perf();
    p.speculative_verify_blocks.fetch_add(1, Ordering::Relaxed);
    p.speculative_target_tokens
        .fetch_add(tokens as u64, Ordering::Relaxed);
    p.gpu_quant_gemm_calls
        .fetch_add(gemm_calls, Ordering::Relaxed);
}
pub(crate) fn record_speculative_cost(draft_ns: u64, verify_ns: u64, rollback: bool) {
    let p = perf();
    p.speculative_draft_ns
        .fetch_add(draft_ns, Ordering::Relaxed);
    p.speculative_verify_ns
        .fetch_add(verify_ns, Ordering::Relaxed);
    p.speculative_rollbacks
        .fetch_add(u64::from(rollback), Ordering::Relaxed);
}

pub(crate) fn record_expert_cache(hit: bool, evictions: u64, bytes: u64, ns: u64) {
    let p = perf();
    if hit {
        p.expert_cache_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        p.expert_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    p.expert_cache_evictions
        .fetch_add(evictions, Ordering::Relaxed);
    p.expert_cache_upload_bytes
        .fetch_add(bytes, Ordering::Relaxed);
    p.expert_cache_upload_ns.fetch_add(ns, Ordering::Relaxed);
}

pub(crate) fn record_gpu_attention() {
    perf().gpu_attention_calls.fetch_add(1, Ordering::Relaxed);
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

pub fn record_kv_pool(
    active_seqs: usize,
    allocated_pages: usize,
    free_pages: usize,
    fragmentation_permille: u64,
) {
    let p = perf();
    p.kv_pool_active_seqs
        .store(active_seqs as u64, Ordering::Relaxed);
    p.kv_pool_allocated_pages
        .store(allocated_pages as u64, Ordering::Relaxed);
    p.kv_pool_free_pages
        .store(free_pages as u64, Ordering::Relaxed);
    p.kv_pool_fragmentation_permille
        .store(fragmentation_permille, Ordering::Relaxed);
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

/// Alias used by the RadixAttention-style path; same counters as `record_prefix_cache_hit`,
/// plus the dedicated `prefix_hit` series for Akasha scrape naming.
pub fn record_prefix_hit(bytes_saved: usize) {
    record_prefix_cache_hit(bytes_saved);
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

pub fn record_scheduler_prefill_chunk(chunks: usize) {
    perf()
        .scheduler_prefill_chunks
        .fetch_add(chunks as u64, Ordering::Relaxed);
}

/// One stall-free iteration of the Sarathi-style schedule (decode-first, then prefill chunks).
pub fn record_scheduler_stall_free_iter(budget: usize) {
    let p = perf();
    p.scheduler_stall_free_iters.fetch_add(1, Ordering::Relaxed);
    p.scheduler_iteration_budget
        .store(budget as u64, Ordering::Relaxed);
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
        gpu_attention_calls: p.gpu_attention_calls.load(Ordering::Relaxed),
        gpu_qwen_full_tokens: p.gpu_qwen_full_tokens.load(Ordering::Relaxed),
        gpu_gpt_full_tokens: p.gpu_gpt_full_tokens.load(Ordering::Relaxed),
        gpu_split_attention_queries: p.gpu_split_attention_queries.load(Ordering::Relaxed),
        gpu_tensor_gemm_calls: p.gpu_tensor_gemm_calls.load(Ordering::Relaxed),
        scratch_alloc_bytes: p.scratch_alloc_bytes.load(Ordering::Relaxed),
        scratch_reuse_hits: p.scratch_reuse_hits.load(Ordering::Relaxed),
        kv_write_bytes: p.kv_write_bytes.load(Ordering::Relaxed),
        kv_physical_pages: p.kv_physical_pages.load(Ordering::Relaxed),
        kv_new_physical_pages: p.kv_new_physical_pages.load(Ordering::Relaxed),
        kv_reused_physical_pages: p.kv_reused_physical_pages.load(Ordering::Relaxed),
        kv_quant_format_code: p.kv_quant_format_code.load(Ordering::Relaxed),
        kv_pool_active_seqs: p.kv_pool_active_seqs.load(Ordering::Relaxed),
        kv_pool_allocated_pages: p.kv_pool_allocated_pages.load(Ordering::Relaxed),
        kv_pool_free_pages: p.kv_pool_free_pages.load(Ordering::Relaxed),
        kv_pool_fragmentation_permille: p.kv_pool_fragmentation_permille.load(Ordering::Relaxed),
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
        expert_cache_hits: p.expert_cache_hits.load(Ordering::Relaxed),
        expert_cache_misses: p.expert_cache_misses.load(Ordering::Relaxed),
        expert_cache_evictions: p.expert_cache_evictions.load(Ordering::Relaxed),
        expert_cache_upload_bytes: p.expert_cache_upload_bytes.load(Ordering::Relaxed),
        expert_cache_upload_ns: p.expert_cache_upload_ns.load(Ordering::Relaxed),
        native_moe_resident_layers: p.native_moe_resident_layers.load(Ordering::Relaxed),
        native_moe_fallback_layers: p.native_moe_fallback_layers.load(Ordering::Relaxed),
        native_moe_resident_ns: p.native_moe_resident_ns.load(Ordering::Relaxed),
        native_moe_fallback_ns: p.native_moe_fallback_ns.load(Ordering::Relaxed),
        gpu_prefill_blocks: p.gpu_prefill_blocks.load(Ordering::Relaxed),
        gpu_prefill_tokens: p.gpu_prefill_tokens.load(Ordering::Relaxed),
        gpu_quant_gemm_calls: p.gpu_quant_gemm_calls.load(Ordering::Relaxed),
        speculative_verify_blocks: p.speculative_verify_blocks.load(Ordering::Relaxed),
        speculative_target_tokens: p.speculative_target_tokens.load(Ordering::Relaxed),
        speculative_rollbacks: p.speculative_rollbacks.load(Ordering::Relaxed),
        speculative_draft_ns: p.speculative_draft_ns.load(Ordering::Relaxed),
        speculative_verify_ns: p.speculative_verify_ns.load(Ordering::Relaxed),
        scheduler_decode_waves: p.scheduler_decode_waves.load(Ordering::Relaxed),
        scheduler_prefill_chunks: p.scheduler_prefill_chunks.load(Ordering::Relaxed),
        scheduler_stall_free_iters: p.scheduler_stall_free_iters.load(Ordering::Relaxed),
        scheduler_iteration_budget: p.scheduler_iteration_budget.load(Ordering::Relaxed),
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
    let managed = crate::backend::cuda_managed_memory_stats();
    writeln!(s,"# TYPE rbitnet_core_cuda_managed_memory_available gauge\nrbitnet_core_cuda_managed_memory_available {}",u8::from(managed.is_some())).unwrap();
    if let Some(m) = managed {
        for (name, value) in [("limit", m.limit), ("live", m.live), ("peak", m.peak)] {
            writeln!(s,"# TYPE rbitnet_core_cuda_managed_{name}_bytes gauge\nrbitnet_core_cuda_managed_{name}_bytes {value}").unwrap();
        }
        counter!(
            "rbitnet_core_cuda_managed_allocations_total",
            "Successful model-managed CUDA allocations",
            m.allocations
        );
        counter!(
            "rbitnet_core_cuda_managed_refusals_total",
            "Managed cap or CUDA allocation refusals",
            m.refusals
        );
        writeln!(s, "# TYPE rbitnet_core_cuda_managed_category_bytes gauge").unwrap();
        for (name, value) in [
            "weights",
            "kv_state",
            "activations",
            "prefix",
            "experts",
            "scratch",
            "other",
        ]
        .iter()
        .zip(m.categories)
        {
            writeln!(
                s,
                "rbitnet_core_cuda_managed_category_bytes{{category=\"{name}\"}} {value}"
            )
            .unwrap();
        }
    }
    counter!(
        "rbitnet_core_gpu_qwen_full_tokens_total",
        "Tokens processed through the complete dense Qwen CUDA pipeline",
        snap.gpu_qwen_full_tokens
    );
    counter!(
        "rbitnet_core_gpu_gpt_full_tokens_total",
        "Tokens processed through the complete fixed-bank GPT-OSS CUDA pipeline",
        snap.gpu_gpt_full_tokens
    );
    counter!(
        "rbitnet_core_gpu_split_attention_queries_total",
        "Native split-KV query positions summed over actual split-enabled layers",
        snap.gpu_split_attention_queries
    );
    counter!(
        "rbitnet_core_gpu_tensor_gemm_calls_total",
        "Actual Tensor Core GEMM kernel launches",
        snap.gpu_tensor_gemm_calls
    );
    counter!(
        "rbitnet_core_speculative_verify_blocks_total",
        "Native target verification blocks",
        snap.speculative_verify_blocks
    );
    counter!(
        "rbitnet_core_speculative_target_tokens_total",
        "Target input positions evaluated in verification blocks including discarded tail",
        snap.speculative_target_tokens
    );
    counter!(
        "rbitnet_core_speculative_rollbacks_total",
        "Verification tails truncated after rejection or stop",
        snap.speculative_rollbacks
    );
    counter!(
        "rbitnet_core_speculative_draft_ns_total",
        "CPU wall nanoseconds proposing token drafts",
        snap.speculative_draft_ns
    );
    counter!(
        "rbitnet_core_speculative_verify_ns_total",
        "Wall nanoseconds in native target verification",
        snap.speculative_verify_ns
    );
    counter!(
        "rbitnet_core_gpu_prefill_blocks_total",
        "CUDA matrix prefill blocks",
        snap.gpu_prefill_blocks
    );
    counter!(
        "rbitnet_core_gpu_prefill_tokens_total",
        "Tokens evaluated through CUDA block prefill",
        snap.gpu_prefill_tokens
    );
    counter!(
        "rbitnet_core_gpu_quant_gemm_calls_total",
        "Shared-weight quantized CUDA matrix projections",
        snap.gpu_quant_gemm_calls
    );
    counter!(
        "rbitnet_core_expert_cache_hits_total",
        "Expert groups found in this model's device cache",
        snap.expert_cache_hits
    );
    counter!(
        "rbitnet_core_native_moe_resident_layers_total",
        "Routed FFN stages executed by resident native graphs",
        snap.native_moe_resident_layers
    );
    counter!(
        "rbitnet_core_native_moe_fallback_layers_total",
        "Routed FFN stages using individual expert fallback, potentially with GPU projections",
        snap.native_moe_fallback_layers
    );
    counter!(
        "rbitnet_core_native_moe_resident_ns_total",
        "Resident FFN wall time including cache fills, excluding whole-token graphs",
        snap.native_moe_resident_ns
    );
    counter!(
        "rbitnet_core_native_moe_fallback_ns_total",
        "Fallback routed FFN wall time including any failed native preparation",
        snap.native_moe_fallback_ns
    );
    counter!(
        "rbitnet_core_expert_cache_misses_total",
        "Expert groups absent from this model's device cache",
        snap.expert_cache_misses
    );
    counter!(
        "rbitnet_core_expert_cache_evictions_total",
        "Unleased expert groups evicted from device cache",
        snap.expert_cache_evictions
    );
    counter!(
        "rbitnet_core_expert_cache_upload_bytes_total",
        "Unchanged GGUF expert bytes uploaded on cache misses",
        snap.expert_cache_upload_bytes
    );
    counter!(
        "rbitnet_core_expert_cache_upload_ns_total",
        "Wall time allocating and uploading expert groups in nanoseconds",
        snap.expert_cache_upload_ns
    );
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
        "rbitnet_core_gpu_attention_calls_total",
        "Fused resident-KV CUDA attention calls issued by bitnet-core",
        snap.gpu_attention_calls
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
        "rbitnet_core_kv_pool_active_seqs",
        "Active sequences tracked by the process-wide paged KV pool (RBITNET_KV_POOL)",
        snap.kv_pool_active_seqs
    );
    counter!(
        "rbitnet_core_kv_pool_allocated_pages",
        "Physical KV pages allocated in the shared pool (including free-listed)",
        snap.kv_pool_allocated_pages
    );
    counter!(
        "rbitnet_core_kv_pool_free_pages",
        "Physical KV pages currently on the shared free list",
        snap.kv_pool_free_pages
    );
    counter!(
        "rbitnet_core_kv_pool_fragmentation_permille",
        "Shared KV free/allocated ratio in permille (0-1000)",
        snap.kv_pool_fragmentation_permille
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
        "rbitnet_core_prefix_hit",
        "Radix/prefix KV hits (Akasha prefix_hit alias of prefix_cache_hits)",
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
        "rbitnet_core_draft_accept",
        "Draft tokens accepted (Akasha draft_accept alias of speculative_accepted_tokens)",
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
    counter!(
        "rbitnet_core_scheduler_prefill_chunks_total",
        "Prefill chunks admitted under Sarathi-style token budget",
        snap.scheduler_prefill_chunks
    );
    counter!(
        "rbitnet_core_scheduler_stall_free_iters_total",
        "Stall-free schedule iterations (decode-first then prefill chunks)",
        snap.scheduler_stall_free_iters
    );
    counter!(
        "rbitnet_core_scheduler_iteration_budget_tokens",
        "Configured / last iteration token budget for continuous batching",
        snap.scheduler_iteration_budget
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
