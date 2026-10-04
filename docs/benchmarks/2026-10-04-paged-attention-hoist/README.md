# Attention KV paginée : lecture des pointeurs par page

Le décodage Llama F32 paginé relisait la table des pointeurs K/V dans les boucles de tokens et de dimensions. Les kernels chargent maintenant le pointeur une fois par page, puis conservent le même ordre des FMA et des réductions. Le changement livré concerne seulement `paged_kernels.cuh`.

Le chemin paginé reste optionnel, via `RBITNET_CUDA_KV_PAGE_LIMIT` ; le chemin dense et le pool original conservent leur fonctionnement. La variante d’admission exclusive du pool est mesurée ici mais n’est pas intégrée : son effet isolé sur le débit est faible dans ces échantillons.

## Mesures

Même CLI figé, même Llama 3.2 1B Q4_K_M et tokenizer, RTX 4080 SUPER 16 Gio / Ryzen 7 9800X3D / 64 Gio / Windows. Contexte 2048, 24 notes communes, sortie plafonnée à 128 tokens, trois cycles dont une chauffe exclue. Médiane et plage de deux répétitions, séparées par prompt et mode. Ces deux répétitions ne permettent pas de conclure à un gain statistiquement établi.

| DLL / variante | Mode | Récit, tokens/s [min–max] | Code, tokens/s [min–max] |
|---|---|---:|---:|
| baseline | dense | 380.39 [379.82–380.95] | 380.39 [379.82–380.95] |
| baseline | paged | 363.13 [361.58–364.67] | 364.15 [363.64–364.67] |
| baseline | paged-prefix | 362.11 [359.55–364.67] | 365.19 [364.67–365.71] |
| exclusive | dense | 381.53 [379.82–383.23] | 379.82 [379.82–379.82] |
| exclusive | paged | 363.64 [362.61–364.67] | 364.15 [363.64–364.67] |
| exclusive | paged-prefix | 367.30 [365.71–368.88] | 367.82 [366.76–368.88] |
| hoisted | dense | 382.09 [382.09–382.09] | 380.95 [380.95–380.95] |
| hoisted | paged | 385.55 [384.38–386.71] | 384.97 [383.23–386.71] |
| hoisted | paged-prefix | 389.66 [387.88–391.44] | 389.12 [384.38–393.85] |
| combined | dense | 379.26 [378.70–379.82] | 380.39 [379.82–380.95] |
| combined | paged | 386.71 [386.71–386.71] | 386.13 [384.38–387.88] |
| combined | paged-prefix | 389.66 [387.88–391.44] | 389.07 [386.71–391.44] |

Le récit paginé mesure 363.13 → 385.55 tokens/s (+6.17 %) ; avec préfixe, 362.11 → 389.66 (+7.61 %). Le débit dense varie peu entre DLL dans ce lot. Les captures contiennent aussi préremplissage, latence HTTP, premier contenu SSE, mémoire et compteurs ; les transferts/allocation ne sont pas assimilés à un gain de tokens/s.

## Correction et portée

- Deux fixtures GPU de pages sur le modèle réel sont exécutées sur chacune des trois variantes avec et sans attention partitionnée. Elles comparent les logits au chemin GPU dense bit à bit et contrôlent ownership, préfixes, réinitialisation et rejeu.
- Les quatre captures de mesures contiennent 108 réponses JSON, 36 comparaisons JSON/SSE et 12 arrêts explicites, avec sorties identiques aux références de chaque capture.
- La variante combinée passe dix cas réseau. La variante finalement livrée, lecture des pointeurs seule avec le pool original, passe ensuite ses propres dix cas de déconnexion/reprise, sampling/pénalités, arrêt et concurrence sérialisée, avec les deux modes de prefill Llama.
- Cette optimisation porte sur le cache paginé CUDA F32 de Llama. Elle ne valide pas la combinaison avec F16/Q8 de #115 ni des kernels paginés Qwen/GPT/GLM, et ne constitue pas une nouvelle comparaison avec Ollama ou llama.cpp.

CLI : `5d937ad9f035c20548e12bebf7a4121ff389a2d09f3d1034f706f8194cf91470`. DLL hoisted : `05de803b837cf0bd78f54337d6ae5107d29d983dd0b538002fd975f020a6bd77`. Les données sont liées à leurs empreintes dans `evidence-index.json` ; les chemins internes restent ceux de la mesure.

## Reproduction

Compiler le CLI et la DLL CUDA de cette branche, adapter les modèles du manifeste, puis exécuter depuis la racine :

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'scripts'
python -B docs/benchmarks/2026-10-04-paged-attention-hoist/source/benchmark.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/paged-kv/cuda/rbitnet_cuda_quant64.dll --model llama32-1b --backend gpu --paged --modes dense,paged,paged-prefix --cycles 3 --notes 24 --max-tokens 128 --device-mib 12288 --output-dir target/page-hoist-benchmark
python -B docs/benchmarks/2026-10-04-paged-attention-hoist/source/live.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --binary target/release/rbitnet.exe --library target/paged-kv/cuda/rbitnet_cuda_quant64.dll --paged --split-kv --device-mib 12288 --output-dir target/page-hoist-live
```

Pour une ablation A/B, utiliser le même CLI avec une DLL reconstruite depuis la base #114 puis depuis cette branche, et des dossiers de sortie distincts. Refs [#92](https://github.com/azerothl/Rbitnet/issues/92), [#98](https://github.com/azerothl/Rbitnet/issues/98).
