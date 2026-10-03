# Archived profiling snapshots

The [3 October 2026 real-model diagnosis](2026-10-03/README.md) profiles Llama CPU/CUDA and traces the three unsupported architectures from the cross-engine benchmark. Six measured ablations, raw timings and CPU code generation evidence are archived with it.

Per [docs/PLAN_PRODUCTION.md](../PLAN_PRODUCTION.md) Phase 2, record **one snapshot per major release** (or after significant kernel changes):

1. Hardware (CPU model, RAM), Rust toolchain version, **git commit SHA**.
2. Command used (`perf record`, `cargo flamegraph`, etc.).
3. **Top 3 hot spots** and **1–3 follow-ups**.

Keep files short and dated, for example:

- `snapshot-YYYY-MM.md` — findings for that month / release tag.

Link each snapshot from [docs/PROFILING.md](../PROFILING.md) and from the matching row in [docs/BENCHMARKS.md](../BENCHMARKS.md).

See [snapshot-2026-05.md](snapshot-2026-05.md) for the template used after Sprint 3 work (hot paths summary carried over from the main profiling doc).
