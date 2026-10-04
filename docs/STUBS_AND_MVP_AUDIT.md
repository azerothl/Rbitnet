# Audit des chemins réels, prototypes et replis — #24

Revu le **3 octobre 2026**, après les lots cache, spéculation et Qwen résident. Un ancien ticket fermé ou une bibliothèque détectée ne prouve pas la prise en charge matérielle. Les recherches `stub/MVP/TODO/unimplemented/not implemented/fake/parity` dans core, serveur et CLI ont été recoupées avec les dispatchers, executors, APIs natives et chemins SSE.

| Chemin | Comportement réel | Limite / suivi |
|---|---|---|
| CPU | GGUF natif et kernels quantifiés mmap. | Layouts listés dans ARCHITECTURE_GGUF_MATRIX ; pas tous les exports. |
| CUDA | Kernels quantifiés, attention, graphes Llama ; récurrence Qwen, pipeline dense Qwen complet opt-in ; FFN experts GPT-OSS/GLM résidents en partie. | [Preuves matérielles](benchmarks/2026-10-03-qwen-full/README.md). Pipelines MoE complets : #88/#89. |
| Hybrid | Placement CUDA réel avec budget et repli CPU. Qwen3/Mixtral restent CPU, avec accelerated=false et métadonnée explicite. | Préchargement et choix par coût : #84/#86. |
| ROCm | Prototype hipBLAS F32 ; pas de kernels quantifiés/attention/pipeline device portés. | Aucun benchmark GGUF AMD validé sur le matériel actuel ; #22 ouvert. |
| Vulkan / Metal / Intel | Probes diagnostiques et calculs de bas niveau CPU ; une bibliothèque trouvée ne signifie pas accélération. Chargement GGUF explicite refusé. | Kernels GPU non implémentés ; Intel/oneAPI/Level Zero restent aliases de prototype Vulkan ; #22. |
| Auto | CUDA/ROCm implémentés puis CPU ; ignore probes Vulkan/Metal. Mixtral et Qwen3 dense choisissent CPU sans probe GPU. | Demande GPU explicite pour ces architectures refusée au chargement, pas réinterprétée en auto. |
| Llama / Qwen3 / Mixtral | Inférence réelle ; Llama goldens réels, Qwen3/Mixtral goldens synthétiques CI. Readiness Qwen3/Mixtral validée au chargement. | Qwen3/Mixtral CPU uniquement ; SSE attend la complétion puis un delta, sans streaming incrémental. |
| Qwen3.5 | GDN + attention gated CPU/CUDA/hybrid ; [pipeline dense complet](benchmarks/2026-10-03-qwen-full/README.md) opt-in. | MoE non validé sur GGUF réel ; prefill matriciel Qwen absent (#95). |
| GPT-OSS / deepseek2 | GPT-OSS20B et GLM4.7Flash split MLA/MoE réellement chargés et comparés. | Autres layouts non automatiquement compatibles. Tag glm4moe distinct, limité aux tensors Llama compatibles. #88/#89. |
| Cache de réponses | PREFIX_CACHE garde des réponses complètes. | Ne réutilise aucun tenseur KV. |
| Cache de préfixes | Llama CPU dense/pagé, GPU K/V ; Qwen checkpoints K/V + GDN + convolution. | [Preuves](benchmarks/2026-10-03-cache-foundation/README.md). Qwen exige la longueur exacte du checkpoint récurrent. Pas de SSD (#94). |
| Experts à la demande | Groupes gate/up/down quantifiés, leases, budget, LRU/LFU/Least-Stale, repli FFN CPU. | [Pilote et régressions](benchmarks/2026-10-03-cache-foundation/README.md). Pas de préchargement asynchrone/choix par coût ; budget non global (#84/#86). |
| Prefill matriciel | Llama GEMM SIMT opt-in avec attention causale et état device. | Pas de Tensor Cores ni Qwen/MoE matriciel ; attention entièrement tuilée à traiter (#95). |
| Spéculation | Llama PLD tokens, vérification matricielle par position, sampler cible, graphes, correction/rollback. Architectures/DLL incompatibles : génération ordinaire, compteurs spéculatifs faux retirés. | [No-go activation générale](benchmarks/2026-10-03-speculative-llama/README.md). Draft GGUF et rollback Qwen absents (#97). |
| Scheduling continu | Decode-first, budget d'itération, hooks et compteurs de vagues. Primitive CPU dense_matvec_multi_seq réelle. | Executors sérialisés, parfois régénération de préfixe ; pas de batch GPU. Ancien #46 fermé ne termine pas #96. |
| KV paginé / Q8 / SlimAttention | Prototypes CPU et Llama CPU/hybrid opt-in. | Graphes GPU : K/V F32 denses. Pages/formats device : #92/#93. KIVI historique no-go ne valide pas un cache GPU compressé. |
| Tokenizer | tokenizer.json et SentencePiece tokenizer.model directs. | Manquant/invalide bloque readiness ; templates propres au modèle. |
| BitNet | b1.58 Llama-shaped, mmap TQ1_0/TQ2_0 CPU ; helpers I2_S/TL2 de microbench. | bitnet_cuda_matvec_mvp appelle CPU. Registry de recherche distinct de production GGUF ; pas de TQ1/TQ2 CUDA validé. |

La disponibilité d'un backend de bas niveau reste une capacité, pas la preuve que toutes les opérations du modèle s'exécutent sur GPU. Les métadonnées de placement, compteurs réels et scripts matériels précisent les chemins partiels. Une erreur native est remontée ; aucun état CPU divergent n'est substitué au milieu d'une séquence.

Les modes intentionnels restent explicites : STUB fournit une réponse synthétique marquée ; TOY un petit LM F32 sans GGUF. Leur backend effectif est CPU. LoadFailed et unload idle remplacent l'engine sans prouver la validité d'un modèle ; readiness bloque les erreurs de chargement. Le SSE natif Llama/Qwen3.5/GPT-OSS/GLM suit la génération, tandis que le fallback ModelExecutor attend la complétion puis un delta.

Les métriques utilisent le working set Linux /proc ou Windows PSAPI. La VRAM par processus est encore **non mesurée** : le placeholder constant process_vram_bytes=0 est supprimé, remplacé par une disponibilité à zéro. [Migration du contrat](AKASHA_METRICS.md).

#24 reste ouvert pendant les conversions requises, avec #22 et #83–#97 ; il ne se résume plus au seul matériel #22. Les tests CPU avec early return GPU, les probes et les tickets historiques fermés ne suffisent pas à promouvoir un chemin matériel. Chaque livraison publie code, contrôles actifs, mesures et replis.
