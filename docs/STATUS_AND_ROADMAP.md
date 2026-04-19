# Rbitnet — implementation status and roadmap

This document complements `[PLAN_PRODUCTION.md](PLAN_PRODUCTION.md)`: it tracks **what is implemented**, **what is partial or missing**, and **suggested next steps**. It should be updated when major features land or scope changes.

---

## Summary


| Area                                                           | Status                                                        |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| HTTP server (OpenAI-shaped API, limits, auth, health, metrics) | **Done**                                                      |
| CI (build, test, clippy, audit)                                | **Done**                                                      |
| Release binaries (GitHub Actions on tag `v`*)                  | **Done** — see `[RELEASE.md](RELEASE.md)`                     |
| Core inference (GGUF, Llama forward, tokenizer, stub/toy)      | **Done** for supported layouts                                |
| Performance baselines (published numbers)                      | **Missing** — template in `[BENCHMARKS.md](BENCHMARKS.md)`    |
| Profiling report (hot paths, prioritized follow-ups)           | **Missing** — checklist in `[PROFILING.md](PROFILING.md)`     |
| “Prod ready” exit criteria (all of PLAN)                       | **Not claimed** — several doc-only / measurement items remain |


---

## By phase (vs `PLAN_PRODUCTION.md`)

### Phase 0 — Scoping


| Item                                                           | Status                                                     |
| -------------------------------------------------------------- | ---------------------------------------------------------- |
| Prod scope (local / trusted network)                           | Documented in `PLAN_PRODUCTION.md`                         |
| Reference GGUF roles (stub, toy, optional `RBITNET_TEST_GGUF`) | Documented; CI uses stub/toy                               |
| Indicative SLO table                                           | Documented; **not measured** against a frozen baseline yet |


### Phase 1 — Reliability and limits


| Item                                    | Status                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| JSON body size cap                      | **Done** — `RBITNET_MAX_BODY_BYTES`, HTTP 413                                               |
| Prompt length / `max_tokens` caps       | **Done** — `RBITNET_MAX_PROMPT_CHARS`, `RBITNET_MAX_TOKENS_CAP`, HTTP 400                   |
| Per-request inference timeout           | **Done** — `RBITNET_INFERENCE_TIMEOUT_SECS`, HTTP 504                                       |
| OOM / mmap / missing file               | **Typed errors** in core; no panics on normal API paths                                     |
| Concurrency limit                       | **Done** — `RBITNET_MAX_CONCURRENT`, HTTP 503 when saturated                                |
| Sequential + parallel integration tests | **Done** — `bitnet-server` tests (including parallel stub)                                  |
| Effective context vs model max          | **Partial** — bounded by prompt + runtime; no separate HTTP knob beyond prompt/`max_tokens` |


**Gap:** After HTTP 504, the blocking inference task may still run to completion in the pool (documented in `[LIMITATIONS.md](LIMITATIONS.md)`).

### Phase 2 — Performance and resources


| Item                                           | Status                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Reproducible benchmarks doc                    | `[BENCHMARKS.md](BENCHMARKS.md)` — **template only**; fill in measured p50/p95, tokens/s |
| Criterion kernel benches                       | `cargo bench -p bitnet-core`                                                             |
| Profiling write-up                             | `[PROFILING.md](PROFILING.md)` — **checklist only**; no archived report in-repo          |
| CPU optimizations (SIMD, threads, allocations) | **Ongoing** / best-effort in kernels and forward                                         |
| RAM ceiling per model                          | **Partial** — qualitative note in `BENCHMARKS.md` / `LIMITATIONS.md`, no fixed table     |


### Phase 3 — Security and exposure


| Item                   | Status                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Env paths without `..` | **Done** — `validate_no_parent_components` in core                                                                         |
| API key                | **Done** — `RBITNET_API_KEY`, `Authorization` / `X-API-Key` on API routes; `/health`, `/ready`, `/metrics` unauthenticated |
| CORS + bind warning    | **Done** — `RBITNET_CORS_ANY`, log warning on `0.0.0.0` / `[::]`                                                           |
| `cargo audit` in CI    | **Done** — `.github/workflows/ci.yml`                                                                                      |
| CLI `--api-key`        | **Not implemented** — env var only (documented in `[USAGE.md](USAGE.md)`)                                                  |


### Phase 4 — Observability


| Item                      | Status                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- |
| Prometheus-style metrics  | **Done** — `GET /metrics`                                                             |
| Structured tracing        | **Done** — `x-request-id` on outer layer; span fields `method`, `path`, `request_id`  |
| Liveness / readiness      | **Done** — `GET /health`, `GET /ready` (`Engine::is_ready`)                           |
| Startup summary of limits | **Done** — logged in `rbitnet-server` main                                            |
| Config validation         | **Partial** — invalid numeric env vars fail at startup; no full “schema” for all vars |


### Phase 5 — Quality and compatibility


| Item                                                       | Status                                                            |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| GGML types documented                                      | `[BITNET_SPEC.md](BITNET_SPEC.md)` + `types.rs`                   |
| Tensor name aliases                                        | **Done** — `tensor_first_of` (e.g. `lm_head`, `attn_out`)         |
| Regression / golden tests                                  | Kernel goldens in CI; optional GGUF smoke via `RBITNET_TEST_GGUF` |
| Second exporter (e.g. llama.cpp vs BitNet) automated tests | **Missing** — manual / future                                     |
| Release process + semver                                   | `[RELEASE.md](RELEASE.md)`                                        |
| Prebuilt binaries on tag                                   | **Done** — `.github/workflows/release.yml`                        |


### Phase 6 — Documentation


| Item                                      | Status                                               |
| ----------------------------------------- | ---------------------------------------------------- |
| USAGE, TRAINING_AND_COMPATIBILITY, README | **Updated** with server envs and pointers            |
| Limitations                               | `[LIMITATIONS.md](LIMITATIONS.md)`                   |
| Deployment (systemd, nginx, Docker)       | `[DEPLOYMENT.md](DEPLOYMENT.md)` + root `Dockerfile` |


---

## Next todos (suggested order)

1. **Measurements** — Fill `[BENCHMARKS.md](BENCHMARKS.md)` with at least one reference CPU + model (or stub/toy) and p50/p95 for HTTP; optional RSS snapshot.
2. **Profiling** — Run perf/flamegraph once per major release; add a short “findings” subsection to `PROFILING.md` or a linked note.
3. **Changelog** — Add a root `CHANGELOG.md` (or rely on GitHub release notes only) and reference it from `[RELEASE.md](RELEASE.md)`.
4. **Timeouts** — Optionally cancel or isolate long-running `spawn_blocking` work after HTTP 504 (design trade-off with thread pool).
5. **Tokenizer** — If needed: support `tokenizer.model` (SentencePiece) in the same way as `tokenizer.json`, or document conversion only (already hinted in README).
6. **Compatibility** — Add more tensor aliases as real GGUF exports fail; optional integration test with a tiny public GGUF in CI (artifact/cache).
7. **Security** — For internet-facing: rate limiting and TLS remain out of repo (reverse proxy); document as today in `DEPLOYMENT.md`.

---

## Related docs


| Doc                                        | Role                                    |
| ------------------------------------------ | --------------------------------------- |
| `[PLAN_PRODUCTION.md](PLAN_PRODUCTION.md)` | Target exit criteria and phased roadmap |
| `[USAGE.md](USAGE.md)`                     | Runtime env vars and curl examples      |
| `[RELEASE.md](RELEASE.md)`                 | Tags, semver, GitHub release binaries   |
| `[LIMITATIONS.md](LIMITATIONS.md)`         | Known constraints                       |
| `[BENCHMARKS.md](BENCHMARKS.md)`           | Where to record performance numbers     |
| `[PROFILING.md](PROFILING.md)`             | How to profile CPU hot paths            |
