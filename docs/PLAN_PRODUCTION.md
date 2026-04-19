# Plan — “prod ready” Rbitnet release

**Implementation status (what exists vs gaps, next todos):** [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md)

This document sets a **roadmap** so a Rbitnet release can be announced as **production-ready** with verifiable criteria. Phases may overlap; the order reflects logical dependencies.

## Target definition (exit criteria)

A version counts as **prod ready** when **all** of the following hold:

1. **Performance criteria** documented and measured on at least one reference setup (p50/p95 latency, tokens/s, peak RAM).
2. **Limits and guardrails**: prompts, `max_tokens`, concurrency, timeouts — explicit rejections and consistent HTTP codes.
3. **Minimal security** for network exposure: configurable auth or deployment guide behind a proxy; no obvious abuse surface on inputs.
4. **Observability**: structured logs, metrics (requests, errors, duration), health endpoint suited to deployment.
5. **Quality**: green CI (build, tests), integration tests on a reference GGUF; versioning policy (semver) and release notes.
6. **Documentation**: install, configuration, operations, troubleshooting, known limitations — current for that release.

---

## Phase 0 — Scoping (short)

### Prod scope (working definition)

Rbitnet targets **local and trusted-network** deployments first: **developer machines**, **intranet** services, and **Akasha** on the same host or LAN. It is **not** positioned as an internet-facing multi-tenant API without additional hardening (reverse proxy, TLS, rate limits, and operational monitoring). Edge / SaaS-style exposure is out of scope until Phase 3–6 items are routinely met.

### Reference GGUF models (for benchmarks and CI)


| Role                 | Example                                          | Notes                                                                      |
| -------------------- | ------------------------------------------------ | -------------------------------------------------------------------------- |
| Integration / API    | Stub (`RBITNET_STUB=1`) or toy (`RBITNET_TOY=1`) | No GGUF artifact; CI uses these by default.                                |
| Parser / smoke       | Any small llama-compatible GGUF you already have | Set `RBITNET_TEST_GGUF` for optional `bitnet-core` tests.                  |
| Performance baseline | A fixed BitNet or Llama GGUF you choose locally  | Record hardware + command in `docs/BENCHMARKS.md` when publishing numbers. |


Paths are **local** (not checked in): update the table in `docs/BENCHMARKS.md` when you freeze a baseline.

### Indicative SLOs (non-contractual)


| Metric                                           | Starting target                                         |
| ------------------------------------------------ | ------------------------------------------------------- |
| Error rate (5xx from process bugs)               | < 0.1% under normal load                                |
| p95 time-to-first-token (short prompt, CPU ref.) | Measure and publish per release in `docs/BENCHMARKS.md` |
| Availability (single instance)                   | No unbounded memory growth on moderate sustained load   |


---

## Phase 1 — Reliability and limits


| Action                                                                        | Deliverable                               |
| ----------------------------------------------------------------------------- | ----------------------------------------- |
| Caps on JSON body size, prompt length, `max_tokens`, effective context size   | `413` / `400` with clear messages         |
| Per-request timeouts (inference + I/O)                                        | No requests blocked indefinitely          |
| Explicit handling of **OOM** / failed mmap / missing file                     | Typed errors, logs, no user-facing panic  |
| **Concurrency** policy (semaphore / queue)                                    | Configurable max simultaneous generations |
| Light-load **integration** tests (several sequential / few parallel requests) | CI test or documented script              |


---

## Phase 2 — Performance and resources


| Action                                                                | Deliverable                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| Reproducible **benchmarks** (fixed prompt, fixed length, `--release`) | `docs/BENCHMARKS.md` + numbers for reference models    |
| Profile hot paths (matmul, attention, dequant)                        | Short report + prioritized issues                      |
| **CPU** optimization paths (SIMD, threads, fewer allocations)         | Incremental implementations behind measurable criteria |
| (Optional) Document **RAM ceiling** per model and hardware notes      | Doc section                                            |


*Note: GPU support or calling an external backend can be a **later phase** if the product stance is “Rust CPU only.”*

---

## Phase 3 — Security and exposure


| Action                                                                                  | Deliverable                                          |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Review **user inputs** (chat, file paths from env)                                      | No arbitrary traversal; strict validation            |
| **Auth**: API key via header or `--api-key`, or explicit doc “only behind Nginx + auth” | Documented choice + minimal implementation if needed |
| **CORS** and bind: default `127.0.0.1`; warning if `0.0.0.0`                            | README + log at startup                              |
| Dependency scan / `cargo audit` in CI                                                   | CI job or release checklist                          |


---

## Phase 4 — Observability and operations


| Action                                                                             | Deliverable                              |
| ---------------------------------------------------------------------------------- | ---------------------------------------- |
| Metrics (Prometheus or simple stats): requests, durations, errors, tokens          | `/metrics` endpoint or documented export |
| Structured logs (level, duration, `task_id` / `request_id`)                        | Stable format                            |
| **Health**: `GET /health` or `/ready` (liveness vs readiness when model is loaded) | Spec for orchestrators                   |
| **Summarized** environment variables validated at startup                          | Clear error if config is invalid         |


---

## Phase 5 — Quality and compatibility


| Action                                                               | Deliverable                                   |
| -------------------------------------------------------------------- | --------------------------------------------- |
| Extend / freeze **supported GGML types** or explicit errors          | Table in docs                                 |
| **Tensor name aliases** or tests on two exports (llama.cpp / BitNet) | Fewer “silent” failures on valid GGUFs        |
| **Regression** suite (golden logits or short-prompt text)            | CI with optional `RBITNET_TEST_GGUF` artifact |
| **Release** process: workspace version, tag, changelog               | `README` + short `docs/RELEASE.md`            |


---

## Phase 6 — Documentation and communication


| Action                                                                      | Deliverable                     |
| --------------------------------------------------------------------------- | ------------------------------- |
| Update **USAGE**, **TRAINING_AND_COMPATIBILITY**, README with prod criteria | Aligned with this plan          |
| **“Limitations”** page (perf, types, archs)                                 | Sets realistic expectations     |
| **Deployment** guide (systemd, optional Docker, reverse proxy)              | At least one reference scenario |


---

## Suggested prioritization (internal prod MVP)

1. Phase **1** (limits + timeouts + concurrency) — blocks the most common incidents.
2. Phase **4** (logs + health) — essential to operate.
3. Phase **2** (bench + reasonable CPU perf) — makes real use credible.
4. Phase **3** (security) — before any Internet exposure.
5. Phases **5** and **6** — in parallel once functional stability is there.

---

## Tracking metrics (examples)


| Indicator                                      | Indicative target (to adjust)                |
| ---------------------------------------------- | -------------------------------------------- |
| CI tests                                       | 100% on the release branch                   |
| Error-path coverage (input, file, tokenizer)   | All paths tested or explicitly “unsupported” |
| p95 latency (short prompt, ref model, ref CPU) | Fixed and published in `BENCHMARKS.md`       |
| Expected uptime (internal)                     | No memory leak under moderate sustained load |


---

## Revision

This plan should be **revised** after each major release or if product scope changes (e.g. GPU support, multi-model). Keep [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) in sync when deliverables move between *done* and *missing*.