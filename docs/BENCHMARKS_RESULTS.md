# Benchmark Results

This file is the append-only target for the small local benchmark matrix scripts:

```bash
scripts/bench_matrix.sh
scripts/bench_paged_kv.sh   # dense vs paged vs RBITNET_KV_POOL @ concurrency 1/4/8
scripts/bench_kv_q8.sh      # paged F32 vs Q8 resident_bytes (+ optional live RSS)
scripts/bench_sarathi.sh    # stall-free continuous batching unit gate (+ optional live)
```

```powershell
.\scripts\bench_matrix.ps1
```

By default the scripts start `rbitnet-server` in `RBITNET_STUB=1` mode, run a tiny reproducible HTTP benchmark through `scripts/bench_backend_compare.py`, and append a markdown section below. For real model numbers, start the server yourself with `RBITNET_MODEL` and `RBITNET_TOKENIZER`, then run:

```bash
NO_START_SERVER=1 MODEL=rbitnet-llama RUNS=12 scripts/bench_matrix.sh
```

```powershell
.\scripts\bench_matrix.ps1 -NoStartServer -Model rbitnet-llama -Runs 12
```

Record hardware, model basename, quantization, backend, and peak RSS when publishing a release-quality row. Keep stub results clearly labeled as API overhead smoke tests, not model throughput proof.

## Paged KV E2E — 2026-09-26 (unit gate)

**Methodology:** `cargo test -p bitnet-core --test kv_storage_paged` (no GGUF in this agent image). Live RSS/tok/s matrix requires TinyLlama Q4 + `scripts/bench_paged_kv.sh` with `RBITNET_MODEL` / `RBITNET_TOKENIZER`.

| Check | Result |
|-------|--------|
| Dense↔paged offset roundtrip | pass |
| Local free-list reuse after `clear` | pass |
| Shared pool multi-seq open/close reclaim | pass |
| Shared pool concurrency page count &lt; dense-equivalent | pass |
| Shared `attention_scores_cpu` vs dense | pass |

Enable production path: `RBITNET_LLAMA_PAGED_KV=1` and/or `RBITNET_KV_POOL=1` (see [USAGE.md](USAGE.md)).

## KV Q8 compact pages — 2026-09-27 (unit gate)

**Methodology:** `cargo test -p bitnet-core --test kv_storage_paged q8_` / `scripts/bench_kv_q8.sh` (no GGUF in this agent image). Live RSS/tok/s F32 vs Q8 requires TinyLlama Q4 + `RBITNET_MODEL` / `RBITNET_TOKENIZER`.

| Check | Result |
|-------|--------|
| Q8 paged roundtrip ≤ scale/127 | pass |
| Q8 `resident_bytes` &lt; ½ of F32 at same pages | pass |
| Q8 vs F32 attention relative drift &lt; 5% (toy) | pass |
| Encode/decode row unit (`q8_roundtrip_preserves_row_shape`) | pass |

**Tradeoffs:** Q8 saves ~4× KV page RSS vs F32; not bit-exact — keep `RBITNET_KV_QUANT=off` for golden bit-exact; enable `q8` for concurrency / memory. Metric: `rbitnet_core_kv_quant_format_code=1`. Enable: `RBITNET_LLAMA_PAGED_KV=1 RBITNET_KV_QUANT=q8` or `rbitnet tune throughput`.

## SlimAttention + KIVI decision — 2026-10-02 (spike #39)

| Check | Result |
|-------|--------|
| SlimAttention 1D tile vs baseline drift &lt; 1e-4 (toy, no GGUF) | pass — `llama::slim_attention` unit tests |
| KIVI 2-bit vs stay on Q8 | **No-go** this spike — criteria in [KIVI_DECISION.md](KIVI_DECISION.md) (PPL/RSS); Q8 remains default compact KV |

Opt-in decode path: `RBITNET_SLIM_ATTENTION=1` (optional `RBITNET_SLIM_ATTENTION_TILE`); Llama CPU/hybrid attention uses 1D tiled online-softmax.

## Sarathi stall-free schedule — 2026-09-27 (unit gate)

**Methodology:** `cargo test -p bitnet-core --test scheduler_speculative` / `scripts/bench_sarathi.sh` (no GGUF in this agent image).

| Check | Result |
|-------|--------|
| PrefillDecodeQueue starts prefill-only | pass |
| Stall-free iters + prefill chunks under budget | pass |
| Decode waves still incremented | pass |
| Batch order preserved for ≥2 requests | pass |

Enable: `RBITNET_CONTINUOUS_BATCHING=1 RBITNET_ITERATION_TOKEN_BUDGET=512` (or `rbitnet tune throughput`). GPU fused multi-seq remains off.

## Fused multi-seq CPU (#46) — 2026-10-02 (stall + kernel)

**Decision:** e2e concurrency ≥4 tok/s gain is **stalled** — executors still sequential. See [FUSED_MULTI_SEQ.md](FUSED_MULTI_SEQ.md).

| Check | Result |
|-------|--------|
| `dense_matvec_multi_seq` ≡ sequential reference (unit) | pass |
| Scheduler calls `generate_decode_batch` when flag on | pass |
| E2E tok/s @ concurrency 4/8 vs sequential | **stalled / non mesuré** (no Llama batched forward) |
| Kernel microbench fused vs sequential @ batch 4/8 | `./scripts/bench_fused_multi_seq.sh` / `cargo bench -p bitnet-core --bench kernels -- fused_multi_seq` |

Enable hook only: `RBITNET_CONTINUOUS_BATCHING=1 RBITNET_FUSED_MULTI_SEQ=1` (experimental; not a throughput claim).

## Manual Result Template

Use this template when you benchmark a real GGUF outside the helper scripts:

| Host | OS | Rust | Backend | Model basename | Quant | Prompt tok | max_tokens | p50 ms | p95 ms | mean tok/s | Peak RSS | Command | Notes |
|------|----|------|---------|----------------|-------|------------|------------|--------|--------|------------|----------|---------|-------|
| TBD | TBD | `rustc -V` | `cpu` | `model.Q4_K_M.gguf` | `Q4_K_M` | TBD | 64 | TBD | TBD | TBD | TBD | `NO_START_SERVER=1 ...` | tokenizer/template source |

## Reference models (frozen for #23)

| Role | Model | File | Tokenizer |
|------|-------|------|-----------|
| Small CPU gate | TinyLlama 1.1B Chat | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` / `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` `tokenizer.json` |
| 1B dense | Llama 3.2 1B Instruct | `unsloth/Llama-3.2-1B-Instruct-GGUF` / `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | matching HF tokenizer |
| BitNet | b1.58 2B4T | curated `microsoft-bitnet-b1.58-2b-4t` bundle | see [BITNET_NATIVE.md](BITNET_NATIVE.md) |

## TinyLlama Q4_K_M CPU — 2026-09-30T10:23:24Z (real GGUF)

**Methodology:** release `rbitnet-server` (`cpu`), GGUF + tokenizer on disk, sequential HTTP `/v1/chat/completions` (warmup + 3 runs). End-to-end latency includes prefill+decode; **not** streaming TTFT. `llama-bench` not installed in this environment (`SKIP_LLAMA`). Generation text quality was degraded on this host (repetitive tokens) — row is a **throughput/RSS** measurement, not a quality claim.

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-09-30T10:23:24Z |
| Host | Linux 6.12.94+ x86_64 |
| CPU | Intel(R) Xeon(R) Processor (`nproc=4`) |
| Rust | `rustc 1.89.0 (29483883e 2025-08-04)` |
| Rbitnet SHA | `31629f1` (main tip when measured) |
| Backend | `cpu` (`RBITNET_BACKEND=cpu`, `RBITNET_MAX_CONCURRENT=1`) |
| GGUF | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (TheBloke) |
| Tokenizer | TinyLlama `tokenizer.json` |
| Prompt | `Say hello.` (~6 prompt tokens) |
| max_tokens | 8 |
| Warmup latency | 10.764 s |
| p50 latency (3 runs) | 10.683 s |
| mean latency | 10.697 s |
| mean e2e tok/s | **0.75** (completion_tokens / wall) |
| Peak RSS observed | **~723 MiB** (`VmRSS` ≈ 740156 KiB) |
| llama-bench | skipped (binary absent) |

**Reproduce:**

```bash
# terminal 1
RBITNET_BACKEND=cpu RBITNET_MAX_CONCURRENT=1 \
  RBITNET_MODEL=/path/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  RBITNET_TOKENIZER=/path/tokenizer.json \
  rbitnet-server

# terminal 2 — short sequential probe (or compare_llamacpp_rbitnet.sh once llama-bench exists)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rbitnet-llama","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0.0}'
```

## Follow-up benches checklist (#41) — 2026-10-02

Published culture for release rows: **machine** (CPU/OS/`nproc`), **engine version** (git SHA + `rustc`), **GGUF basename**, **backend**, **peak RSS**. Stub/API rows stay labeled as overhead only.

| Slot | Status | Why / how to fill |
|------|--------|-------------------|
| Fair `llama-bench` vs Rbitnet (same TinyLlama Q4_K_M) | **non mesuré** | `llama-bench` binary absent in cloud agent images; set `LLAMA_BENCH` + unset `SKIP_LLAMA` then `RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh` |
| Llama-3.2-1B Instruct Q4_K_M e2e tok/s | **non mesuré** | Weights not present here; matrix reproduce: `rbitnet up unsloth/Llama-3.2-1B-Instruct-GGUF --file Llama-3.2-1B-Instruct-Q4_K_M.gguf` then `NO_START_SERVER=1 MODEL=rbitnet-llama … scripts/bench_matrix.sh` |
| BitNet b1.58 2B4T e2e tok/s | **non mesuré** | Bundle install per [BITNET_NATIVE.md](BITNET_NATIVE.md); kernel microbench already published below — e2e HTTP row still open (ties #45) |
| Prefix-KV live delta @ concurrency 1/4/8 | **non mesuré** | Unit/metrics exist (`rbitnet_core_prefix_hit`); live matrix needs TinyLlama + `RBITNET_PREFIX_KV=1` vs off |
| KV Q8 live RSS/tok/s @ 1/4/8 | **non mesuré** | Unit gate shipped; live: `scripts/bench_kv_q8.sh` with real GGUF |
| Sarathi stall-free live @ 1/4/8 | **non mesuré** | Unit gate shipped; live: `scripts/bench_sarathi.sh` with real GGUF |
| GPU columns | deferred | Owned by #22 — do not invent CPU rows as GPU |

**Acceptance for closing #41:** fill ≥1 of Llama-3.2 or BitNet **measured** e2e row **or** a fair `llama-bench` compare (not stub), plus keep this table updated so remaining cells stay explicit `non mesuré` rather than silent gaps. This PR documents the slots; measured fills are follow-up commits on real hardware.

## llama.cpp comparison — 2026-09-25T20:07:31Z

**Methodology:** `scripts/compare_llamacpp_rbitnet.sh` with `SKIP_LLAMA=1` against `RBITNET_STUB=1` HTTP (no GGUF weights in this cloud agent image; `llama-bench` not installed). Numbers below are **API / stub overhead only** — not a model tok/s claim vs llama.cpp.

**Limitations / how to reproduce a fair row:**

1. Download the same TinyLlama Q4_K_M GGUF for both engines (`RBITNET_GGUF=…`).
2. Build llama.cpp `llama-bench` and set `LLAMA_BENCH`.
3. Start `RBITNET_BACKEND=cpu RBITNET_MODEL=… RBITNET_TOKENIZER=… rbitnet-server --release`.
4. Run `RESULTS_MD=docs/BENCHMARKS_RESULTS.md ./scripts/compare_llamacpp_rbitnet.sh` (unset `SKIP_LLAMA`).

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-09-25T20:07:31Z |
| Host | Linux 6.12.94+ x86_64 |
| CPU | Intel(R) Xeon(R) Processor |
| Rust | `rustc 1.88.0 (6b00bc388 2025-06-23)` |
| Rbitnet SHA | `f5bc87e` (pre-change base; see PR for tip) |
| Threads | 4 |
| GGUF | `n/a` (stub) |
| llama-bench status | skipped_by_flag |
| Rbitnet HTTP rc | 0 |

### llama-bench output

```
(none — no llama.cpp binary / no GGUF in agent environment)
```

### Rbitnet HTTP output (stub)

```
{
  "runs": 5,
  "p50_ms": 1.525816999901508,
  "p95_ms": 2.069930200013914,
  "mean_ms": 1.4235430000098859,
  "mean_tok_s": 7093.551072528254
}
```

## Local bench 2026-09-25 20:07:40 +0000

- Git SHA: `f5bc87e`
- Rust: `rustc 1.88.0 (6b00bc388 2025-06-23)`
- Base URL: `http://127.0.0.1:8080`
- Model: `rbitnet-stub`
- Runs: `8` warmup: `1` max_tokens: `32`

| Host | Backend | Model | p50 ms | p95 ms | mean ms | mean tok/s | Notes |
|------|---------|-------|--------|--------|---------|------------|-------|
| Unix local | stub/http | `rbitnet-stub` | 0.64 | 1.32 | 0.76 | 23530.82 | Small reproducible smoke bench |

## BitNet ternary kernels — 2026-09-26 (NATIVE_FIRST)

| Shape / paths | ns/call | bit_exact | widest gap | notes |
|---------------|---------|-----------|------------|-------|
| ternary 64x1024 | i8=34680ns i2s=31834ns tl2=33787ns auto=32469ns | bit_exact=true | widest_gap=i8_vs_i2s | Rust SIMD/LUT (no FFI) |

## BitNet ternary kernels — 2026-10-02 (NATIVE_FIRST, #45 spike)

**Host:** Linux 6.12.94+ x86_64, Intel Xeon (`nproc=4`, AVX2+FMA), `rustc 1.99.0`, contended cloud agent (absolute ns noisy; use ratios).

**Measured progression (release, `N=64 K=1024 ITERS=500`, median of 3):**

| Shape / paths | ns/call | bit_exact | widest gap | notes |
|---------------|---------|-----------|------------|-------|
| ternary 64x1024 | i8≈137k i2s≈165k tl2≈34k **auto≈15.5k** | bit_exact≈true (auto approx) | widest_gap=i2s_vs_tl2 | **AVX2+FMA `auto` ~2.2× vs TL2** on this host |
| TQ row dots 64×256 | tq2_stack≈740ns tq2_heap≈255ns; tq1_stack≈725ns tq1_heap≈280ns | bit_exact≈true | stack vs heap | Stack scratch removes per-row heap; single-row microbench favors heap reuse — keep stack for parallel mmap GEMV |

**Reproduce:**

```bash
N=64 K=1024 ITERS=500 ./scripts/bench_bitnet_kernels.sh
# or
N=64 K=1024 ITERS=500 cargo run -p bitnet-core --example ternary_microbench --release --locked
TQ_ITERS=400 TQ_ROWS=64 cargo run -p bitnet-core --example tq_dot_microbench --release --locked
```

### Saturation / gap analysis (why not chase more I2_S alone)

1. **Layout mismatch:** Production Microsoft b1.58 GGUF uses **TQ1_0 / TQ2_0** blocks in `quant_dot` (`BITNET_NATIVE.md`). `kernels.rs` I2_S/TL2 is a research surface inspired by bitnet.cpp — **not wired** into the BitNet forward today. Beating bitnet.cpp I2_S on this microbench does not move e2e tok/s until TQ GEMV (or a repack) uses the same kernels.
2. **AVX2 auto is the cheap win on the research path:** previous `auto` stubbed to scalar I2_S; now real AVX2+FMA (~2.2× vs TL2 here). Further scalar LUT tweaks are in the noise next to that.
3. **Naive fuse-into-acc lost to decode+dot:** interleaved decode+FMA prevented autovec; stack-per-block decode + tight mul_add is the right production shape (no per-row `Vec`), even when a microbench with allocator reuse makes heap look faster.
4. **No FFI:** NATIVE_FIRST — do not link llama.cpp / bitnet.cpp for this gap.

### Explicit next slice

1. Publish a real **BitNet 2B4T e2e** tok/s + RSS row (MODEL_MATRIX still **unpublished**).
2. SIMD / wider tiles on **TQ2_0** `dot_row` (the actual hot path), optionally sharing decode tables with I2_S research kernels.
3. Optional offline repack TQ→I2_S only if e2e profiling shows decode dominance — still no FFI.

## CUDA Gate E kernels — 2026-10-02 (RTX 4080 Laptop)

**Hardware:** NVIDIA GeForce RTX 4080 Laptop GPU, Driver 610.88, CUDA Toolkit 13.3 (cudart64_13 / cublas64_13).
**Library:** 
ative/cuda_quant built via `scripts/build_cuda_quant.ps1`; `RBITNET_CUDA_QUANT_LIB` set; `RBITNET_BENCH_CUDA=1`.
**Method:** `cargo bench -p bitnet-core --bench kernels -- --warm-up-time 1 --measurement-time 3`.
**Correctness:** `RBITNET_CUDA_QUANT_SMOKE=1 cargo test -p bitnet-core --test cuda_quant_residency opt_in_device_resident_quant_kernel_when_lib_present` — `device_resident_quant_gemv_calls` rises; GPU Q4_0 matches CPU golden within 1e-3.

| Bench | Shape | time (mean) |
|-------|-------|-------------|
| `cuda_backend_matvec_f32` (host-upload W) | 512×4096 | 506 µs |
| `cuda_device_resident_f32` | 512×4096 | 41 µs |
| `cuda_device_quant_q4_0` (device-resident W) | 512×4096 | 173 µs |

Notes: host-upload f32 pays H2D of **W** each call; resident f32/quant keep **W** on device. Full greedy Llama tok/s vs CPU still recorded separately when `RBITNET_MODEL` is set (Gate D).

## Unsloth Llama 3.2 1B reference pack — 2026-10-03

The Windows CPU [export recipe](../recipes/exported-llama.recipe.json) serves the pinned Unsloth Q4_K_M GGUF without Python in the serving process. The [hash-checked HTTP response](validation/2026-10-03-unsloth-llama32-1b.json) is `Paris.` (22 prompt tokens, two completion tokens). This verifies the prebuilt pack and recipe; no training or local GGUF conversion was performed.

The [four-engine/model protocol and raw results](benchmarks/2026-10-03-parity-round2/README.md) include the exact same GGUF SHA-256 `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`, with real-model sequence checks. Ryzen 7 9800X3D, RTX 4080 SUPER, Windows, 16 CPU threads; three measured short-prompt repetitions after warmup, 32 output tokens.

| Pack | Rbitnet CPU decode median | Rbitnet CUDA decode median |
|---|---:|---:|
| Unsloth Llama-3.2-1B-Instruct Q4_K_M | 38.6 tok/s | 477.6 tok/s |

These are decoding rates from the dated report, not complete HTTP-request rates or long-context performance. The erroneous [2026-10-02 smoke](validation/2026-10-02-unsloth-llama32-1b.json) is retained as evidence of the earlier inference defect.
