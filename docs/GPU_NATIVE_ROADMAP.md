# Feuille de route GPU native

Objectif: accelerer Rbitnet dans `bitnet-core` sans imposer Python, vLLM, Ollama, llama.cpp server ou autre daemon d'inference externe au runtime.

Refs: issue [#22](https://github.com/azerothl/Rbitnet/issues/22).

## Acceptance slice (spike #22 — residency / parity)

First vertical that must land before claiming “CUDA works” beyond stubs:

| Gate | Definition of done | CI expectation |
|------|--------------------|----------------|
| **A. CPU golden matvec** | Fixed `f32` 3×4 vector in `backend_conformance` matches CPU reference (and CUDA/ROCm/Vulkan/Metal stubs via CPU fallback). | Always on in default `cargo test` — **no GPU required**. |
| **B. Device-resident weight GEMV** | `CudaDeviceMatrix` / `gemv_device_weight_f32` keeps **W** on device across calls; only **x**/**y** traffic each GEMV. Counter: `CudaRuntimeMetrics::device_resident_gemv_calls`. | Manual / opt-in with real CUDA; CI never fails if CUDA missing. |
| **C. Host-upload GEMV** | `matvec_cuda` / `CudaBackend::matvec` may still H2D **W** each call (pooled buffers). Counts toward `gemv_calls` but **not** `device_resident_gemv_calls`. | Same — optional hardware. |
| **D. Token smoke** | One curated Llama-shaped GGUF produces correct greedy tokens on `RBITNET_BACKEND=cuda` when CUDA+cuBLAS load. | Documented script / local only; not a default CI job. |
| **E. Device-resident quantized matvec** | `CudaDeviceQuantMatrix` uploads GGML Q4_0/Q8_0/Q4_K/Q6_K payload once; matvec prefers `rbitnet_cuda_*_matvec_device` symbols, else CPU golden via `matvec_payload_quant`. Counter: `device_resident_quant_gemv_calls`. Llama `cuda`/`hybrid` offload prefers quant residency over densify when type is supported. | Default CI: `cuda_quant_residency` tests (CPU fallback only). Hardware: `scripts/smoke_cuda_quant_residency.sh` + `RBITNET_BENCH_CUDA=1`. |

**Still out of scope for default CI:** shipping `librbitnet_cuda_quant` itself, FlashAttention, ROCm/Vulkan/Metal beyond parity stubs, published tok/s rows without a measured GPU box.

### Residency checklist (operator / reviewer)

Use this when validating a CUDA box (not CI):

1. Confirm `CudaRuntime::try_load()` succeeds (`backend_accelerated` / logs).
2. Prefer paths that upload weights once (`CudaDeviceQuantMatrix` / `CudaDeviceMatrix` / `upload_u8` / `upload_f32`) over per-token `matvec` host **W**.
3. After a short generate, inspect `CudaRuntime::metrics_snapshot()`:
   - `device_resident_quant_gemv_calls` should rise when `librbitnet_cuda_quant` device symbols load;
   - else CPU fallback still produces correct tokens while `device_resident_quant_gemv_calls` stays 0;
   - `device_resident_gemv_calls` rises on densified f32 resident paths;
   - `gemv_calls ≥ device_resident_gemv_calls`;
   - `upload_bytes` should not grow linearly with token count for resident **W** (only activations).
4. Compare greedy first token vs CPU on the same GGUF (`RBITNET_BACKEND=cpu` vs `cuda`) — optional golden via [GOLDEN_TESTS.md](GOLDEN_TESTS.md).
5. Record GPU / driver / `RBITNET_*` in [BENCHMARKS_RESULTS.md](BENCHMARKS_RESULTS.md) when publishing numbers.
6. Run `scripts/smoke_cuda_quant_residency.sh` (always runs CPU goldens; set `RBITNET_BENCH_CUDA=1` for Criterion rows including `cuda_device_quant_q4_0`).

## Etat actuel

- `bitnet-core::backend::CudaRuntime` charge dynamiquement CUDA Runtime et cuBLAS depuis les bibliotheques systeme quand `RBITNET_BACKEND=cuda` est choisi.
- Le chemin CUDA general sait executer un GEMV `f32` natif via cuBLAS quand les symboles sont disponibles, puis retombe sur le CPU si CUDA/cuBLAS est absent.
- **Llama `cuda` / `hybrid` offload** prefers `CudaDeviceQuantMatrix` for Q4_0 / Q8_0 / Q4_K / Q6_K when staging layers; unsupported types still densify to `CudaDeviceMatrix`. Without CUDA or without `*_matvec_device` symbols, matvec falls back to host quant CPU (correctness first).
- Les chemins Qwen/GLM/GPT-OSS/DeepSeek experimentaux utilisent `CudaRuntime` comme facade native, mais une partie importante du graphe reste a porter et documente explicitement les limites.
- Les kernels BitNet ternaires restent principalement reference CPU; `bitnet_cuda_matvec_mvp` garde une entree stable pour les futurs kernels natifs.
- **Residency status:** pooled scratch for host-upload GEMV exists; durable device-resident **f32** weights via `device_resident_gemv_calls`; durable **quant** weights via `device_resident_quant_gemv_calls` when the optional quant library provides device symbols.

## Amelioration incrementale livree

- Les operations `CudaBackend::copy_from_host` et `CudaBackend::copy_to_host` ne font plus de faux aller-retour host -> device -> host. Tant que le trait expose des `Vec<f32>` cote host et pas encore de buffers device reutilisables, ces copies CUDA ne conservaient aucun etat GPU et ajoutaient seulement de la latence.
- `CudaRuntime::matvec_cuda` et `gemv_device_weight_f32` reutilisent un petit jeu de buffers device (`d_w`, `d_x`, `d_y`) avec capacites qui grossissent au besoin, au lieu d'allouer/liberer a chaque GEMV.
- Le benchmark `cargo bench -p bitnet-core` contient un hook opt-in `RBITNET_BENCH_CUDA=1` pour `cuda_backend_matvec_f32`, `cuda_device_resident_f32`, et `cuda_device_quant_q4_0` lorsque CUDA est disponible.
- Spike checklist + CPU-golden parity for f32 (A) and quant residency (E); CI stays CUDA-free.

## Phases suivantes

1. ~~Ajouter un type de buffer device explicite dans `bitnet-core` pour reutiliser `cudaMalloc` entre appels et eviter les allocations par GEMV.~~ *(premier palier: pool interne `CudaRuntime` pour le GEMV `f32` generique — etendre aux autres chemins si besoin.)*
2. **Hardware:** validate Gate E on a real NVIDIA box — ship or load `librbitnet_cuda_quant` with `*_matvec_device` symbols, confirm `device_resident_quant_gemv_calls` rises, publish a `RBITNET_BENCH_CUDA=1` row + greedy token parity vs CPU.
3. Porter les kernels BitNet ternaires chauds vers CUDA natif, avec tests de parite contre `matvec_ternary_i8` — *apres* parite CPU vs patterns bitnet.cpp ([2502.11880](https://arxiv.org/abs/2502.11880)).
4. Introduire un KV cache GPU natif par pages pour Llama-lineage, puis seulement ensuite evaluer un ordonnanceur de batching continu (aligné [INFERENCE_STACK_V2.md](INFERENCE_STACK_V2.md) Phase A→D).
5. Ajouter des benchmarks reproductibles CPU/CUDA dans `docs/BENCHMARKS_RESULTS.md` avec modele, backend, GPU, driver et variables `RBITNET_*`.
6. **(Research, non-default)** Evaluer attention fusee type FlashAttention-2 / FA3 ([2205.14135](https://arxiv.org/abs/2205.14135), [2307.08691](https://arxiv.org/abs/2307.08691)) et variantes INT ([2409.16997](https://arxiv.org/abs/2409.16997), [2412.08585](https://arxiv.org/abs/2412.08585)) **uniquement** une fois le KV device + residency stables. Ce n'est **pas** le chemin par defaut: le produit reste **native-first CPU**; le chemin CPU tiled (SlimAttention [2407.07304](https://arxiv.org/abs/2407.07304)) a la priorite serving locale.
7. ROCm / Vulkan / Metal: keep parity stubs until after the CUDA quant vertical is measured.

## Contraintes

- Aucune nouvelle dependance runtime obligatoire sur Python ou un binaire externe.
- Les bibliotheques CUDA restent chargees dynamiquement et optionnelles: un systeme sans CUDA doit continuer a utiliser le backend CPU.
- **CI must not require CUDA hardware** — default tests use CPU golden / stub parity only; GPU checks are opt-in (`RBITNET_BENCH_CUDA`, local smokes).
- Toute delegation vers un serveur externe reste experimentale, compilee uniquement via `experimental-external-backends`, et ne doit jamais devenir le chemin par defaut.
- **FlashAttention-2 / FA3 ne sont pas le chemin serving par defaut** — recherche GPU_NATIVE seulement; prioriser CPU + GGUF/BitNet (voir [STATUS_AND_ROADMAP.md — Research-backed priorities](STATUS_AND_ROADMAP.md#research-backed-priorities-2026-09)).
