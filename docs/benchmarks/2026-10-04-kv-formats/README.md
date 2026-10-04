# Cache KV CUDA Llama F16/Q8 — option de capacité

Cette branche ajoute `RBITNET_CUDA_KV_FORMAT=f32|f16|q8` au cache natif Llama, dense ou paginé, et conserve **F32 par défaut**. Les clés et valeurs sont encodées ; les poids, le calcul de l'attention, les états récurrents Qwen et le cache d'experts MoE ne sont pas quantifiés par cette option.

La branche a passé sa propre validation après intégration : `cargo check`, Clippy et les tests du workspace, construction CUDA neuve, 64 cas F64, contrôles de propriétaires/préfixes, corpus NLL, six captures JSON/SSE et quatre suites réseau. Les sources vérifiées restent exactement celles de la livraison `e906a06`. Les preuves fraîches sont dans [`production/validation.json`](production/validation.json), avec les sources, les empreintes des exécutables, les logs et les captures brutes compressées. Les changements de la PR #114 sont conservés. Le tableau historique ci-dessous garde les résultats du prototype privé ; les nouvelles mesures sont séparées dans [`production/summary.json`](production/summary.json).

## Résultat mesuré

Sur la RTX 4080 SUPER, Llama-3.2-1B, cache dense de capacité 8192, quatre requêtes longues chaudes par mode, sortie limitée à 128 tokens :

| Format | Décode, tokens/s | Préremplissage, ms | HTTP, ms | Pic de mémoire CUDA gérée, octets |
|---|---:|---:|---:|---:|
| F32 | 360,06 | 2126,5 | 2505,03 | 1 386 390 580 |
| F16 | 324,06 | 1962,0 | 2361,40 | 1 117 955 128 |
| Q8 | 233,58 | 2145,5 | 2714,08 | 992 126 008 |

Ce sont des médianes, sauf le pic mémoire. La mémoire gérée comprend les allocations de poids, KV, activations et scratch comptées par le moteur ; elle exclut les allocations opaques du pilote et des graphes. Q8 réduit ce pic d'environ 28,4 % mais ralentit le décodage d'environ 35,1 %. **Ces modes répondent à une limite de VRAM ; aucun gain général en tokens/s n'est établi.**

Les six captures comparent dense/paginé aux capacités 512/2048/8192, avec et sans préfixes. Chaque capture conserve 54 réponses JSON, 18 streams et 6 contrôles de stop. Les longueurs de prompt varient entre capacités : ces captures ne mesurent pas l'effet isolé du seul contexte. Le serveur du prototype sérialise les requêtes ; les essais de propriétaires 1/4/8 ne prouvent pas un forward multi-séquences.

## Format et reprise

F16 emploie deux octets par valeur, avec refus explicite des valeurs non finies ou hors plage F16. Q8 emploie des entiers signés, une échelle F32 par tête et par token pour K et V, et un arrondi au plus proche. Il ne s'agit pas du schéma KIVI asymétrique à clés par canal et fenêtre résiduelle.

Les échelles Q8 font partie des pages, des copies de préfixes et de leur comptabilité. Les snapshots conservent l'identité du format et du propriétaire. Les couples format/propriétaire incompatibles sont refusés. Les modes encodés emploient, au préremplissage, l'ordre d'accumulation du décodeur : cette correction est nécessaire pour retrouver exactement les logits d'un même format après des partitions ou une reprise de préfixe. Une demande TF32 avec KV encodé est refusée.

L'échec initial de reprise F16 est conservé dans `initial-failed-warmed-prefix.json.gz` et dans le log du reproducteur avec l'ancienne DLL. Les logits de partitions et de successeurs sont ensuite identiques bit à bit **au sein du même format**. Une identité F32/F16/Q8 n'est pas requise.

## Qualité et portée

Les seuils de `quality-plan.json` ont été fixés avant l'expérience. Les contrôles incluent un oracle d'attention F64, les frontières de pages et de contexte, les graphes et l'attention découpée activés/désactivés, les refus numériques, la troncature, la copie et la libération des propriétaires.

Deux corpus imposés, répétitifs et synthétiques, contiennent chacun 1024 cibles. L'écart NLL moyen absolu maximal observé est d'environ 0,0000379 en F16 et 0,00157 en Q8 ; les écarts maximaux par cible sont 0,00763 et 0,12706. Ces contrôles passent les seuils prévus. Ils ne constituent pas une évaluation générale de langage ni une garantie sur d'autres modèles. Les quatre suites réseau conservent aussi neuf réponses factuelles strictes chacune, les seeds, les pénalités, les déconnexions et les requêtes concurrentes sérialisées. La correction des raisons de fin HTTP est livrée séparément dans la PR #117 ; elle ne fait pas partie de cette branche.

`summary.json` contient les valeurs agrégées. Les fichiers `.json.gz` et `.log.gz` conservent les réponses et échantillons complets, y compris leurs différences entre formats. `manifest.json` distingue les sources de la branche, les exécutables privés mesurés et les artefacts compressés. Aucun binaire ni modèle n'est ajouté au dépôt.

## Utilisation et reproduction

Après construction de la DLL CUDA et du CLI :

```powershell
$env:RBITNET_CUDA_KV_FORMAT = 'f16' # ou q8 ; f32 par défaut
$env:RBITNET_CUDA_PREFILL = '1'
$env:RBITNET_CUDA_PREFILL_TF32X3 = '0'
# Le mode natif Llama et une DLL proposant create_kv sont requis.
# RBITNET_CUDA_KV_PAGE_LIMIT active facultativement les pages natives.
```

Les tests GPU sont facultatifs et demandent un vrai GGUF Llama et son tokenizer ; ils doivent être exécutés seuls, avec `--test-threads=1` :

```powershell
$env:RBITNET_TEST_GGUF = 'CHEMIN_ABSOLU/model.gguf'
$env:RBITNET_TOKENIZER = 'CHEMIN_ABSOLU/tokenizer.json'
$env:RBITNET_CUDA_QUANT_LIB = 'CHEMIN_ABSOLU/rbitnet_cuda_quant64.dll'
$env:RBITNET_CUDA_KV_TEST = '1'
$env:RBITNET_KV_CANONICAL_TEST = '1'
$env:RBITNET_CUDA_PAGES_TEST = '1'
$env:RBITNET_KV_QUALITY_PLAN = (Resolve-Path 'docs/benchmarks/2026-10-04-kv-formats/quality-plan.json').Path
$env:RBITNET_CUDA_SPLIT_KV = '1' # répéter aussi avec 0
cargo test -p bitnet-core --release --lib resident::quantized_tests -- --nocapture --test-threads=1
cargo test -p bitnet-core --release --lib resident::canonical_tests -- --nocapture --test-threads=1
$env:RBITNET_KV_TEST_CONTEXT = '8192' # répéter aussi avec 2048
cargo test -p bitnet-core --release --lib resident::quantized_long_tests -- --nocapture --test-threads=1
```

Pour les tests de pages, répéter `resident::paged_tests` avec `RBITNET_CUDA_KV_FORMAT=f16` puis `q8`. Les modèles, seeds et formats utilisés doivent figurer dans toute nouvelle comparaison. CPU, Qwen et les autres architectures sont hors du périmètre de cette option native Llama. KIVI reste à implémenter et évaluer ; l'issue #93 reste ouverte.

## Limite de l’ABI mutable

Le chemin Rust refuse TF32 avec KV encodé dès la configuration. L’API C++ mutable `configure_tensor_prefill` doit encore recevoir un refus équivalent lorsqu’un appelant natif tente de modifier un contexte encodé déjà créé. Cette correction distincte reste à valider ; la PR demeure en brouillon pendant ce travail. Les preuves ci-dessus concernent les configurations encodées sans TF32.
