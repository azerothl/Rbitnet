# On-device kernel autotune (design note — deferred)

**Status:** deferred (issue [#40](https://github.com/azerothl/Rbitnet/issues/40)). No kernel autotune is implemented in-tree.

## Motivation

Magnitude-class local runtimes advertise decode gains from **compile + tune kernels on the target chip** before serving. Rbitnet already has Rust CPU matvec paths, optional OpenBLAS, and CUDA quant helpers (`RBITNET_QUANT_KERNEL`), but parameters are env/static — not profiled per machine.

## Scope (when revisited)

1. **Microbench harness** at first model load or `rbitnet tune`: timed GEMV / attention score kernels for the active backend (`cpu`, `cuda`, hybrid).
2. **Local param cache** under a machine-scoped path (e.g. tile size, thread split, CUDA block dims) keyed by CPU/GPU id + kernel name + dtype.
3. **Safe fallback:** if the cache is missing or a bench fails, keep today’s `auto` defaults; never block serving on a failed tune.
4. **Honesty:** do not claim “+X% vs llama.cpp” without frozen benches (`scripts/` + documented baseline).

## Non-goals (this spike)

- Shipping autotune for Metal / ROCm before those backends leave MVP stubs.
- Replacing ggml / external compilers.
- Productizing Magnitude-style agent connectors.

## Relation to sticky / prefix work

Sticky session affinity and multi-session prefix KV (see [DEPLOYMENT.md](DEPLOYMENT.md#multiple-replicas-prefix-locality)) improve **agent hit rate** independently of kernel autotune. Autotune is a separate latency lever and remains a design-only note until a POC lands.
