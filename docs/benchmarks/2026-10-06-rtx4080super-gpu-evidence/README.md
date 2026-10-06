# Preuves GPU — RTX 4080 SUPER — 2026-10-06

Run local uniquement (pas CI). **Ne ferme pas #24 #83-86 #88-89 #92-98** : ce dossier
apporte des preuves, pas une cloture. Branche : `cursor/gpu-evidence-rtx4080super-f34e`.

Binaire mesure : `target/release/rbitnet-server.exe` + `target/release/examples/gpu_residency_probe.exe`
compiles depuis cette branche (`cargo build --release -p bitnet-server -p bitnet-core --example gpu_residency_probe`).
DLL : `native/cuda_quant/build/rbitnet_cuda_quant64.dll` (4 643 840 octets, build du jour via
`scripts/build_cuda_quant.ps1`, nvcc CUDA 13.3 + MSVC 14.38). Logs bruts : memes noms dans ce dossier.

## 1. nvidia-smi (doit montrer RTX 4080 SUPER)

```text
GPU 0: NVIDIA GeForce RTX 4080 SUPER (UUID: GPU-d9116be8-13c2-ea3b-b790-e27e8c7fe969)
name, driver_version, memory.total [MiB], compute_cap
NVIDIA GeForce RTX 4080 SUPER, 610.88, 16376 MiB, 8.9
```

Driver 610.88, CUDA UMD 13.3, `nvcc` 13.3.73, CPU AMD Ryzen 7 9800X3D, Windows 10 Pro (build 26220).
VRAM systeme bruittee par le desktop (3.9–14 Go utilises selon le moment) : l'attribution
precise vient des compteurs `rbitnet_core_cuda_managed_*` de `/metrics`, pas du total systeme.

## 2. `smoke_cuda_quant_residency.ps1 -BuildLib` — PASS

`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke_cuda_quant_residency.ps1 -BuildLib`
(log : `rbitnet_smoke_residency.log`).

| Etape | Resultat |
|---|---|
| Build `rbitnet_cuda_quant64.dll` (compute 89/86/80/75) | PASS (4 643 840 octets) |
| CPU golden `cuda_quant_residency` (7 tests) | PASS |
| CPU golden `backend_conformance` (7 tests) | PASS |
| Smoke opt-in `opt_in_device_resident_quant_kernel_when_lib_present` (`RBITNET_CUDA_QUANT_SMOKE=1`, DLL presente, `is_device_resident()==true`, `device_resident_quant_gemv_calls` augmente, parite GPU/CPU 1e-3) | PASS |

## 3. Llama 3.2 1B Q4_K_M — greedy CPU vs `RBITNET_BACKEND=cuda`

GGUF : `models/exported-llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (807 694 368 octets,
sha256 `3f5a22…cc1`, aussi blob ollama `rbitnet-bench-llama32-1b:q4_k_m`).
Tokenizer : `models/exported-llama/tokenizer.json`. Prompt : `What is the capital of France? Answer in one word.`

### 3a. Sonde dediee (`gpu_residency_probe`, 13 tokens prompt, 16 tokens max)

| | CPU | CUDA (`RBITNET_CUDA_QUANT_LIB`=DLL du jour) |
|---|---|---|
| Tokens generes | `[12366, 13]` → ` Paris.` (+ EOS) | **identiques** `[12366, 13]` → ` Paris.` (+ EOS) |
| Matrices `CudaQuant` residentes | 0/0 (114 mmap) | **113/113** (1 mmap restante) |
| `device_resident_quant_gemv_calls` (delta prefill+decode) | 0 | **1695** |
| Prefill 13 tokens | 557.2 ms | **222.5 ms** |
| Decode | 26.4 tok/s (2 tokens) | **78.2 tok/s** |
| Load | 0.3 ms (mmap) | 866.7 ms (upload 800 Mo) |

Parite greedy exacte CPU/CUDA au niveau `LlamaModel` (raw prompt, sans template de chat).

### 3b. `/v1/chat/completions` greedy (`temperature=0`, `max_tokens=16`, meme prompt)

| | CPU (`RBITNET_BACKEND=cpu`, port 18081) | CUDA (`RBITNET_BACKEND=cuda`, port 18082) |
|---|---|---|
| Reponse | `{"content":"Paris","finish_reason":"stop","completion_tokens":1,"prompt_tokens":47}` | **identique** `Paris`, stop, 1/47 |
| Wall 1re requete (incl. load) | 2012.9 ms (load 559 + prefill 1823 + decode 40) | **568.8 ms** |
| Wall requete tiede | — | **102.8 ms** (TTFT moy. 89 ms, decode 500 tok/s sur 2 requetes) |
| `/metrics` backend | `backend="cpu",family="llama"` 1, `native_accelerated` 0, `gpu_gemv` 0, `quant_matvec` CPU 5378 | `backend="cuda",family="llama"` 2, **`native_accelerated` 2/2**, `gpu_gemv` **10756**, `gpu_attention` **1536**, `cuda_graph_replays` 96, `quant_matvec` CPU **0** |
| RSS processus | 1353.7 Mo (`rbitnet_process_rss_bytes` 1 419 452 416) | 1888.3 Mo (`rbitnet_process_rss_bytes` 1 980 067 840) |
| VRAM attribuee (`managed_live/peak`) | 0 (pas d'alloc CUDA) | **1 337 361 460 octets** : weights 799 592 448 + kv 536 870 912 + activations 898 100 ; upload H2D 800 379 264, download 16 |

Fichiers : `rbitnet_v1_cpu.json`, `rbitnet_v1_cuda.json`, `rbitnet_v1_cuda_warm.json`.

## 4. Validations GGUF (DLL du jour via `-Library`) — PASS

| Suite | Commande | Resultat |
|---|---|---|
| Qwen3.5-2B Q8_0 (`D:/Rbitnet-benchmark-models/qwen35-2b/Qwen3.5-2B-Q8_0.gguf`) | `validate_qwen_full.ps1 -Library native/cuda_quant/build/rbitnet_cuda_quant64.dll` | **PASS 4/4** en 543.8 s (greedy/sequence/reset, prefix checkpoints, prefill cancel, teacher-forcing) |
| GPT-OSS 20B Q4_K_M (`D:/Rbitnet-benchmark-models/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf`) | `validate_gpt_full.ps1 -Library …` | **PASS 4/4 synthese** (25.4 s) + **PASS 1/1 reel** (1488.1 s) : resident weights 10 659 Mo, 24/24 couches experts fixes, graphs 0/1 + split, pire KL **3.465e-12**, delta NLL **6.442e-6** |
| GLM-4.7-Flash Q4_K_M (`D:/Rbitnet-benchmark-models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf`) | `validate_mla_full.ps1 -Library …` | **PASS 3/3 synthese** (1.2 s) + **PASS 2/2 reel** (254.2 s) : resident 1 267 Mo, 46/47 couches routees, graphs/split/prefix, pire KL **2.363e-11**, delta NLL **1.526e-5** ; fallback CPU verifie (KL 1.780e-11) |

Logs : `rbitnet_validate_qwen.log`, `rbitnet_validate_gpt.log`, `rbitnet_validate_mla.log`.

## 5. Changements de code (observabilite uniquement, additifs)

- `crates/bitnet-core/src/backend.rs` : `CudaDeviceQuantMatrix::runtime_metrics()` —
  snapshot des compteurs du runtime partage (dont `device_resident_quant_gemv_calls`).
- `crates/bitnet-core/src/llama/mod.rs` : re-export `MatrixWeights` (lecture des stats de residence).
- `crates/bitnet-core/examples/gpu_residency_probe.rs` : sonde locale (pas CI) GGUF + backend +
  compteurs + timings en JSON.

Aucun changement de kernel, de comportement inference, ou de CI.
