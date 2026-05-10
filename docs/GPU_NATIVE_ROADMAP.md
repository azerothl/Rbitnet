# Feuille de route GPU native

Objectif: accelerer Rbitnet dans `bitnet-core` sans imposer Python, vLLM, Ollama, llama.cpp server ou autre daemon d'inference externe au runtime.

## Etat actuel

- `bitnet-core::backend::CudaRuntime` charge dynamiquement CUDA Runtime et cuBLAS depuis les bibliotheques systeme quand `RBITNET_BACKEND=cuda` est choisi.
- Le chemin CUDA general sait executer un GEMV `f32` natif via cuBLAS quand les symboles sont disponibles, puis retombe sur le CPU si CUDA/cuBLAS est absent.
- Les chemins Qwen/GLM/GPT-OSS/DeepSeek experimentaux utilisent `CudaRuntime` comme facade native, mais une partie importante du graphe reste a porter et documente explicitement les limites.
- Les kernels BitNet ternaires restent principalement reference CPU; `bitnet_cuda_matvec_mvp` garde une entree stable pour les futurs kernels natifs.

## Amelioration incrementale livree

- Les operations `CudaBackend::copy_from_host` et `CudaBackend::copy_to_host` ne font plus de faux aller-retour host -> device -> host. Tant que le trait expose des `Vec<f32>` cote host et pas encore de buffers device reutilisables, ces copies CUDA ne conservaient aucun etat GPU et ajoutaient seulement de la latence.
- Le benchmark `cargo bench -p bitnet-core` contient maintenant un hook opt-in `RBITNET_BENCH_CUDA=1` pour mesurer `cuda_backend_matvec_f32 512x4096` lorsque CUDA est disponible, sans lancer de processus externe.

## Phases suivantes

1. Ajouter un type de buffer device explicite dans `bitnet-core` pour reutiliser `cudaMalloc` entre appels et eviter les allocations par GEMV.
2. Porter les kernels BitNet ternaires chauds vers CUDA natif, avec tests de parite contre `matvec_ternary_i8`.
3. Garder les poids quantifies en memoire mappee ou device selon le format GGUF, au lieu de densifier en `f32` quand ce n'est pas necessaire.
4. Introduire un KV cache GPU natif par pages pour Llama-lineage, puis seulement ensuite evaluer un ordonnanceur de batching continu.
5. Ajouter des benchmarks reproductibles CPU/CUDA dans `docs/BENCHMARKS_RESULTS.md` avec modele, backend, GPU, driver et variables `RBITNET_*`.

## Contraintes

- Aucune nouvelle dependance runtime obligatoire sur Python ou un binaire externe.
- Les bibliotheques CUDA restent chargees dynamiquement et optionnelles: un systeme sans CUDA doit continuer a utiliser le backend CPU.
- Toute delegation vers un serveur externe reste experimentale, compilee uniquement via `experimental-external-backends`, et ne doit jamais devenir le chemin par defaut.
