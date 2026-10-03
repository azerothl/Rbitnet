# GEMM quantifié Tensor Cores compensé — 3 octobre 2026

`RBITNET_CUDA_PREFILL_TF32X3=1` accélère les grandes projections du préremplissage Llama résident, avec `RBITNET_CUDA_PREFILL=1`. Sur la RTX 4080 SUPER, le préremplissage long passe de **1 512,5 à 923,5 ms**, soit environ 39 % de temps en moins. La latence HTTP passe de 1 860,0 à 1 273,3 ms. Le décodage n'est pas accéléré par cette option. Elle reste expérimentale et désactivée par défaut.

## Méthode et périmètre

La tuile de sortie 32×32 partage chaque poids GGUF déquantifié entre 32 tokens. Les entrées F32 sont séparées en une partie TF32 et un résidu TF32. Trois produits calculent le terme principal et les deux corrections croisées, en omettant le produit des deux résidus. Les sous-totaux K=32 sont additionnés en FP32 en dehors de l'accumulation Tensor Core. Le principe s'inspire de [Ootomo et Yokota, §3.2–3.5](https://arxiv.org/html/2203.03341v3). Il s'agit d'une adaptation propre à Rbitnet, sans CUTLASS et sans promesse de SGEMM bit-exact ou d'émulation F32 universelle.

Les huit formats existants restent ceux du GGUF : F32, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4. Aucun miroir complet des poids déquantifiés n'est ajouté. Le chemin exige SM80+, au moins 64 tokens, 1 024 colonnes et 131 072 éléments de sortie ; sinon la projection reste SIMT. Les petites projections et le décodage d'un token conservent le chemin antérieur. Les scripts de build ajoutent SM80 et PTX compute_80 aux cibles existantes. Seule la RTX 4080 SUPER a été mesurée ici.

Rust configure le contexte par une nouvelle ABI optionnelle avant son premier bloc. Le compteur natif rapporte les lancements réellement sélectionnés et réussis ; Prometheus publie `rbitnet_core_gpu_tensor_gemm_calls_total`. Une ancienne DLL conserve le repli et un compteur nul.

## Ablation HTTP/SSE sur le binaire final

Llama3.2-1B Q4_K_M, même CLI/DLL pour les trois modes ; attention partitionnée active dans tous les modes. Un cycle de chauffe puis deux cycles mesurés avec deux prompts longs, et un contrôle court exclu du tableau. Entrée de 1 729 tokens, sortie de 128 tokens. Les chronométrages sont isolés des compilations et autres tests GPU.

| Mode | Préremplissage médian (min–max), ms | Décodage, ms | tok/s décodage | HTTP, ms |
|---|---:|---:|---:|---:|
| SIMT par blocs + split-KV | 1 512,5 (1 508–1 519) | 337,0 | 379,8 | 1 860,0 |
| Projections compensées + split-KV | 923,5 (911–933) | 338,5 | 378,1 | 1 273,3 |
| Projections + préfixe réutilisé | 3,0 (3–3) | 335,0 | 382,1 | 353,6 |

Les 27 réponses correspondent à la référence, ainsi que les neuf paires unary/SSE en glouton, sampling avec seed et pénalités. Les trois arrêts explicites passent. Les différences de décodage sont faibles et ne constituent pas un gain. Le préfixe chaud mesure une entrée déjà calculée. Les réponses, compteurs, mémoire et répétitions sont dans [ablation.json](ablation.json) et [summary.json](summary.json).

## Trois moteurs fraîchement remesurés

Ollama 0.35.0, llama.cpp b11351 / 631109b34, RTX 4080 SUPER, mêmes GGUF et prompts bruts. Température 0, concurrence 1, un échauffement puis trois mesures. Aucun préfixe réutilisé : Rbitnet prefix off, llama.cpp `cache_prompt=false`, déchargement/préchargement Ollama avant chaque requête chronométrée pour avoir un KV vide et un modèle résident. Qualité : 3/3 dans chaque moteur pour les sondes conservées.

| Prompt / moteur | tok/s décodage | HTTP, ms | Préremplissage, ms | Premier contenu SSE, ms |
|---|---:|---:|---:|---:|
| Court / llama.cpp | 492,4 | 93,0 | 5,0 | 30,7 |
| Court / Ollama | 481,0 | 136,2 | 44,8 | 68,5 |
| Court / Rbitnet | 477,6 | 125,6 | 42,0 | 57,8 |
| Long / llama.cpp | 417,9 | 366,8 | 44,4 | 63,4 |
| Long / Ollama | 485,5 | 374,3 | 83,5 | 100,8 |
| Long / Rbitnet | 375,4 | 1 258,2 | 914,0 | 940,1 |

Courts : 32 tokens de sortie ; longs : 128, avec les mêmes notes système que l'ablation. La sonde SSE utilise 16 tokens et une requête distincte. La parité globale n'est pas atteinte : le préremplissage demeure le principal écart, et le décodage long Llama reste plus lent. Ces trois sondes de qualité ne représentent pas une évaluation générale.

Capacité de contexte : 8 192 Rbitnet, 2 048 demandés aux références ; longueurs d'entrée effectivement identiques et sous les limites. K/V F32 pour Rbitnet et llama.cpp, F16 par défaut pour Ollama. RSS de l'arbre de processus ; pics GPU globaux incluant le bureau Windows, sans attribution de VRAM par processus. [comparison-short.json](comparison-short.json), [comparison-long.json](comparison-long.json) et les fixtures publient ces détails.

## Correction et résultats négatifs

- 240 tests workspace réussis, 0 échec, 1 ignoré, 25 suites ; Clippy tous targets passe avec les avertissements préexistants. Build CUDA/CLI et séquence réelle avec l'ancienne DLL passent.
- Oracle indépendant FP64 après déquantification CPU du GGUF : huit formats, blocs 7/16/33/128, bords irréguliers, annulation et produits à grande/petite amplitude. Les plus grands F32 finis ne deviennent pas infinis à la conversion. Borne de vérification `8e-7 * sum(abs(w*x)) + 1e-37` ; ceci borne ces fixtures, pas tous les domaines numériques. [oracle.log](oracle.log).
- Deux corpus, français et code, préfixes 63/64/129/257 et 48 positions avec tokens imposés : argmax identiques, logits dans `0.003*(1+abs(reference))`, pire KL `2.880e-11`, pire écart absolu de log-vraisemblance `4.761e-6`. Le test exige un compteur natif positif et compare deux contextes configurés explicitement, SIMT et Tensor Cores. [teacher-forcing.log](teacher-forcing.log).
- Llama réel : 43 IDs de référence, logits de vérification/rollback, seed/pénalités/annulation, checkpoints divergents/éviction. HTTP/SSE : trois interruptions/reprises, stop et quatre requêtes identiques concurrentes, encore sérialisées. Les requêtes chaudes peuvent légitimement afficher zéro GEMM Tensor Core ; le rapport exige leur exécution réelle sur les cas froids. [streaming.json](streaming.json).
- Le premier test de comparaison de contextes n'exécutait pas les Tensor Cores : `SetEnvironmentVariable` depuis Rust ne modifiait pas la copie `getenv` du CRT de la DLL. L'assertion sur le compteur l'a détecté. L'ABI explicite corrige cette divergence ; [initial-configuration-gap.log](initial-configuration-gap.log) garde la preuve du défaut, distincte du test final réussi.
- Le prototype 64×16 ralentissait plusieurs matrices, y compris certains blocs de 64 tokens. Il a été remplacé par 32×32 avec strides partagés décalés. [initial-64x16-gemm.json](initial-64x16-gemm.json) est une mesure historique, pas le noyau livré. Le noyau final forcé reste plus lent sur les petites projections, ce qui justifie le repli SIMT.

Les mesures de [real-gemm.json](real-gemm.json) emploient les matrices réelles de la couche 0, deux échauffements et 30 lancements par appel, trois cycles, avec des événements CUDA excluant les transferts hôte. Elles forcent les deux noyaux pour observer aussi les mauvaises configurations ; la sélection de production exclut les petites matrices. À 128 tokens, médianes : projection Q 0,431 → 0,233 ms ; gate FFN 1,679 → 0,777 ; down FFN 1,628 → 0,911. À 16 tokens, projection Q 0,080 → 0,142 et down 0,328 → 0,552 : régressions conservées dans les données.

## Reproduire

[manifest.json](manifest.json) relie les empreintes sources, modèle/tokenizer, CLI/DLL et preuves. Les rapports HTTP/SSE et comparatifs emploient les mêmes binaires finaux. Les sources sont celles du checkout Windows physique sur `e358b63` avec les changements du lot ; les deux fichiers historiques sont identifiés séparément. Le [manifeste du comparatif](comparison-manifest.json) donne matériel, versions, chemins et options.

```powershell
pwsh -NoProfile -File scripts/build_cuda_quant.ps1 -OutDir "$PWD/target/tensorcore/cuda"
cargo build --release -p rbitnet-cli
python scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-tensorcore-prefill/comparison-manifest.json --model llama32-1b --backend gpu --binary target/release/rbitnet.exe --library target/tensorcore/cuda/rbitnet_cuda_quant64.dll --cycles 3 --tf32x3 --output-dir target/repro/tensor
python scripts/validate_cache_streaming.py --config docs/benchmarks/2026-10-03-tensorcore-prefill/comparison-manifest.json --binary target/release/rbitnet.exe --library target/tensorcore/cuda/rbitnet_cuda_quant64.dll --tf32x3 --split-kv --output-dir target/repro/streaming
```

Adapter les chemins du [driver GPU](gpu-validation.ps1) et du [driver du comparatif](comparison-driver.ps1). Le test de tokens imposés se lance séparément avec `RBITNET_CUDA_TENSOR_TEST=1` et `cargo test --release -p bitnet-core optional_real_tensor_prefill --lib -- --nocapture`. Le driver comparatif utilise un daemon Ollama isolé, arrêté dans `finally`. Ne pas chronométrer pendant une compilation ou un autre test GPU. Qwen par blocs et les autres architectures restent dans #95/#88/#89 ; ce lot ne clôt pas ces tickets.
