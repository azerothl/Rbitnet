# GPT-OSS : segments d'experts et préfixes immuables

Implémentation : `84f686e`, proposée dans [PR #109](https://github.com/azerothl/Rbitnet/pull/109), sur PR #108. Les ablations ci-dessous utilisent le même CLI et la même DLL que les validations finales. Les options restent désactivées par défaut.

Le graphe de token entier reste disponible quand toutes les banques fixes sont résidentes. Un cache dynamique ou un placement partiel utilise des segments : attention/KV, normes, routeur et tête restent sur CUDA ; les IDs et probabilités passent sur l'hôte pour admettre les experts. Les leases restent vivantes jusqu'à la fin du FFN. Le repli routé est explicite dans les compteurs et les métadonnées. `RBITNET_REQUIRE_GPT_FULL` exige ce pipeline résident compatible, y compris un backbone qui tient dans le budget ; il autorise les FFN routés hors GPU dans les segments.

Les snapshots copient les K/V utilisés, peuvent restaurer un préfixe tronqué, et conservent une identité de génération du contexte. Une restauration ne produit pas d'activation de sortie : le diagnostic de hidden state refuse sa lecture avant la fin d'un nouveau token. Les changements d'ABI restent optionnels pour le graphe fixe d'une ancienne DLL.

Sur Windows, Ryzen 7 9800X3D et RTX 4080 SUPER, le build final passe 256 tests workspace, un test ignoré, Clippy et les builds release/CUDA. La régression native de 30 tests inclut les oracles F64 pour huit formats quantifiés, les routes fixes/dynamiques/CPU, graphes/split-KV, snapshots, reset, annulation et identité de génération. Certains autres tests réels conditionnels de cette commande restent désactivés ; les quatre essais GPT réels sont exécutés séparément.

Les quatre processus GPT-OSS Q4_K_M utilisent les couples cache/plafond géré, en Mio : `0/12288`, `8192/12288`, `16/12288` et `0/6144`. À contexte 2048, ils confrontent 544 positions observées, 96 générations et 24 replays de préfixes au chemin partiel, dont les positions presque ex aequo 145–149. La KL maximale est `1.952e-11` et l'écart absolu de NLL cible maximal `2.289e-5`, pour des bornes de `1e-5` et `1e-3`. Les textes gloutons, avec seed et pénalités, sont identiques. Le graphe fixe original passe en plus 72 positions et 18 générations. Dans les logs, « fixed expert layers » compte les contextes FFN présents : avec cache dynamique, ce nombre ne désigne pas des banques fixes entièrement chargées.

Les dix scénarios HTTP couvrent chargement/rechargement, déchargement, fixe/segments/cache/CPU routé/partiel/hybride, trois comportements d'ancienne DLL et refus d'un état hors budget. Les trois suites SSE — cache 8192, cache 16/plafond 6144, banques partielles/plafond 6144 — passent chacune déconnexion/replay en glouton, seed et pénalités, arrêt et quatre demandes concurrentes. Ces demandes restent sérialisées par l'exécutant ; elles ne constituent pas un batch natif.

Le [manifeste](validation/manifest.json) contient les SHA256 des sources, du CLI et de la DLL, les budgets et les empreintes des captures originales/publiées. CLI : `016ed0f518a0125b3441ab2945f3ece0f2ce99be95e0c9cc59788801c77e6f45`. DLL : `b8fb68b35761e2fc13a373846c21e9ffd3e5e62eb83aa060ecedc86cbf73ed43`. Les logs publiés sont normalisés en UTF-8/LF sans lignes blanches finales. Les durées de tests incluent la validation et ne servent pas de benchmark. Les premiers essais antérieurs à la protection du diagnostic, avec une autre DLL, ne sont pas confondus avec ces captures.

Reproduction : construire le CLI release et la DLL avec `scripts/build_cuda_quant.ps1`, puis définir `RBITNET_MAX_SEQ=2048` et lancer `scripts/validate_gpt_segmented.ps1` avec les chemins du GGUF, du tokenizer et de la DLL. Le [script de cycle de vie](scripts/lifecycle.py) et `scripts/validate_cache_streaming.py --gpt-full --gpt-segmented` reproduisent les contrôles HTTP/SSE avec leurs paramètres de budget.

## Ablations du build final

GPT-OSS-20B Q4_K_M, contexte 2048, plafond géré 12288 Mio et marge 256 Mio, sur la même machine. Chaque mode utilise trois cycles, dont le premier chauffe le modèle/cache ; les cycles 1 et 2 donnent quatre observations longues (deux prompts) et deux courtes. Les prompts longs partagent des notes communes et demandent des récits différents, avec au plus 128 tokens générés. Les captures conservent les tokens, réponses, durées, compteurs, plafonds et empreintes. Les commandes attendent la fin des builds et validations : aucun autre build ni moteur d'inférence ne fonctionne pendant ces ablations.

| Cache d'experts | Mode | Prefill long médiane [min–max], ms | Décodage médian, tok/s | HTTP médiane, ms | Prefill court, ms |
|---:|---|---:|---:|---:|---:|
| 0 Mio | baseline | 31354.5 [30660–31613] | 38.88 | 34638.6 | 1419.5 |
| 0 Mio | full-fixed | 18804.0 [18794–18807] | 59.01 | 20982.9 | 788.0 |
| 0 Mio | full | 21059.0 [20970–21173] | 54.28 | 23451.2 | 900.0 |
| 0 Mio | full-split | 16663.5 [16623–16760] | 80.84 | 18274.6 | 887.0 |
| 0 Mio | full-split-prefix | 74.0 [74–76] | 80.86 | 1662.5 | 221.5 |
| 8192 Mio | baseline | 27160.5 [26833–28646] | 40.92 | 30372.5 | 1446.5 |
| 8192 Mio | full | 22763.5 [22426–22940] | 46.67 | 25534.8 | 1201.0 |
| 8192 Mio | full-split | 18117.0 [18007–18172] | 67.89 | 20027.0 | 1162.5 |
| 8192 Mio | full-split-prefix | 79.0 [75–94] | 78.05 | 1724.6 | 232.0 |

`baseline` garde le chemin partiel. `full-fixed` conserve le graphe de token entier, sans Split-KV ; il est testé seulement avec cache 0. `full` force les segments, puis `full-split` ajoute l'attention partitionnée et `full-split-prefix` les snapshots. La comparaison fixe/segments à attention identique montre leur coût d'admission sur l'hôte : 59,01 contre 54,28 tok/s. La sélection automatique conserve donc le graphe entier quand toutes les banques fixes tiennent. Ce tableau n'inclut pas une nouvelle ligne de graphe entier avec Split-KV ; sa précédente ablation est publiée dans [le lot GPT fixe](../2026-10-03-gpt-full/README.md).

Les 81 requêtes HTTP mesurées, 27 paires requête unary/SSE supplémentaires et neuf arrêts donnent les mêmes textes. Une paire comprend une requête HTTP supplémentaire et un flux SSE : il y a donc 117 demandes unary et 27 flux au total. Le premier contenu visible SSE glouton long passe de 31496,8 ms à 16937,1 ms avec les segments/Split-KV et à 313,5 ms au préfixe chaud, sans cache d'experts. Ce temps inclut les tokens éventuels masqués par le traitement Harmony, et désigne le premier contenu visible au client.

Sans cache d'experts, les compteurs logiques par demande longue mesurée passent de 3,196 Go H2D/2,402 Go D2H à 18,427 Mo H2D/1,229 Mo D2H pour les segments, puis 1,533 Mo/0,103 Mo avec préfixe chaud. Le graphe entier ne renvoie que 512 octets pour ces 128 tokens gloutons. Ce sont des octets aux API, sans mesure du trafic PCIe physique. Les valeurs par mode, hits/misses, chargements d'experts et distributions complètes sont dans [le manifeste de performance](performance/manifest.json) et les rapports [cache 0](performance/cache0/results.json) / [cache 8192](performance/cache8192/results.json).

Le cache de 8 Gio accélère le préremplissage du chemin partiel mais ralentit les segments/Split-KV : 67,89 contre 80,84 tok/s et 18,12 contre 16,66 s. Il ne devient pas le défaut pour ce corpus. Le préfixe chaud réutilise du KV ; il ne cache pas la réponse et ses 74 ms ne représentent pas un prompt froid.

Les plafonds gérés et pics RSS/VRAM sont enregistrés séparément. Sans cache d'experts, le pic géré est d'environ 10891 Mio pour les segments/Split-KV et 11037 Mio avec snapshots ; le pic RSS du processus est proche de 23 Go. Le delta VRAM global observé avoisine 11 Gio selon le mode et comprend les allocations du pilote et des autres applications : il ne remplace pas le registre d'allocations gérées.

Le champ `source_dirty` des captures vaut vrai à cause des fichiers `__pycache__` non suivis et de la publication documentaire entre les deux essais ; aucune source du moteur ou du harness n'a changé. Les commits de provenance peuvent donc différer entre cache 0 et 8192, avec les mêmes SHA256 source/CLI/DLL vérifiés avant et après chaque processus. Le manifeste d'entrée conserve des paramètres historiques du comparatif, notamment un contexte 1024 ; les environnements effectifs des lignes et le manifeste de cette ablation font autorité : contexte 2048. Les scripts figés et [la chaîne terminée](performance/chain.log) sont publiés sans binaires. Les captures publiées sont normalisées en UTF-8/LF ; le manifeste distingue leurs SHA256 de ceux des fichiers originaux conservés dans `target`. Les attributs Git de ce dossier préservent ces octets lors des checkouts.

Pour reproduire chaque budget, utiliser `scripts/benchmark_cache_stack.py --config docs/benchmarks/2026-10-03-parity-round2/manifest.json --model gpt-oss-20b --binary <CLI> --library <DLL> --output-dir <dossier> --gpt-full --gpt-segmented --split-kv --moe-cache 0 --device-mib 12288 --cycles 3`, puis répéter avec `--moe-cache 8192` sur les mêmes binaires. Utiliser un port libre avec `--port`.

Le préremplissage GPT demeure token par token. Ce lot ne livre pas les transferts asynchrones, les pages ou formats KV, la persistance sur SSD ni le batching continu. Ces ablations ne comparent pas de nouvelles versions d'Ollama/llama.cpp et n'établissent pas leur parité.
