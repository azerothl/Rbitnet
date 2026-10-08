# Cache d'experts asynchrone GPT-OSS/GLM — 4 octobre 2026

Le cache optionnel garde des groupes d'experts READY et PENDING dans un seul pool physique borné. Les octets quantifiés GGUF restent inchangés. Un flux CUDA privé copie depuis une ou deux zones RAM épinglées ; un groupe ne devient visible qu'après la réussite de son événement de fin. Les experts sélectionnés et les copies en cours conservent leurs propriétaires jusqu'à la fin du calcul. Le déchargement draine les événements avant de libérer ces buffers.

`RBITNET_MOE_ASYNC=1` active cette variante avec `RBITNET_MOE_EXECUTION=cache`, CUDA/hybrid et un budget d'experts positif. `RBITNET_MOE_PINNED_SLOTS=1|2` borne la RAM épinglée. `RBITNET_MOE_PREFETCH=off|previous-pass` compare la demande seule aux experts observés à la couche suivante lors de la passe précédente. Il s'agit d'une heuristique ; les prédicteurs entraînés de [Mira](https://arxiv.org/abs/2609.38090) restent une étude séparée. Les combinaisons incompatibles et valeurs invalides demandées explicitement sont refusées. Le défaut reste synchrone, sans préchargement.

## Mesures intégrées

RTX 4080 SUPER 16 Gio, mêmes GGUF/tokenizers que le manifeste de référence, contexte 2048, plafond CUDA 12 Gio/marge 256 Mio. Huit notes communes, sortie 32 tokens, trois cycles : le premier chauffe chaque mode. La table utilise quatre observations longues des cycles suivants ; `summary.json` conserve aussi les bornes min/max du débit. Le protocole intégral conserve le prompt court, les réponses, métriques, températures, RAM/VRAM et erreurs éventuelles. GPT utilise son graphe segmenté ; GLM utilise MLA/split-KV. Aucun résultat de débit ne provient de Nsight.

| Modèle | Cache Mio | Mode | Préremplissage ms | Décodage tok/s | HTTP ms |
|---|---:|---|---:|---:|---:|
| gpt-oss-20b | 8192 | sync | 2988.5 | 72.74 | 3431.3 |
| gpt-oss-20b | 8192 | async-demand | 3013.5 | 71.19 | 3471.8 |
| gpt-oss-20b | 8192 | async-previous-pass | 2973.0 | 71.99 | 3438.3 |
| glm47-flash | 8192 | sync | 9848.5 | 24.14 | 11205.6 |
| glm47-flash | 8192 | async-demand | 8300.5 | 28.29 | 9427.8 |
| glm47-flash | 8192 | async-previous-pass | 8489.0 | 27.49 | 9679.0 |

Ces mesures portent sur le binaire intégré à 8192 Mio. Elles ne justifient pas un changement automatique du défaut. Le cache asynchrone réserve des slots ayant la capacité maximale des groupes de projections ; les allocations physiques, les payloads READY/PENDING et les octets utiles sont mesurés séparément. Un format mixte Q4/Q6 peut donc avoir une réserve supérieure aux seuls payloads utiles. Les métriques exposent préchargements demandés/utilisés/inutiles/tardifs, copies, octets gaspillés, attente, slots et RAM épinglée.

## Première ablation, prototype distinct

Même protocole, mais binaire/DLL et empreintes distincts des mesures intégrées. La comparaison des budgets a précédé l'adoption. Ces lignes ne sont pas agrégées aux lignes intégrées.

| Modèle | Cache Mio | Mode | Préremplissage ms | Décodage tok/s | HTTP ms |
|---|---:|---|---:|---:|---:|
| gpt-oss-20b | 512 | sync | 29681.0 | 8.92 | 33243.0 |
| gpt-oss-20b | 512 | async-demand | 32105.5 | 8.26 | 35990.4 |
| gpt-oss-20b | 512 | async-previous-pass | 34711.5 | 7.83 | 38850.6 |
| gpt-oss-20b | 8192 | sync | 3071.5 | 71.27 | 3530.7 |
| gpt-oss-20b | 8192 | async-demand | 3117.5 | 69.87 | 3578.8 |
| gpt-oss-20b | 8192 | async-previous-pass | 3109.0 | 70.18 | 3574.7 |
| glm47-flash | 512 | sync | 45934.0 | 4.41 | 53253.1 |
| glm47-flash | 512 | async-demand | 25107.0 | 8.17 | 28922.3 |
| glm47-flash | 512 | async-previous-pass | 30096.5 | 7.41 | 34402.0 |
| glm47-flash | 8192 | sync | 9862.5 | 24.10 | 11193.5 |
| glm47-flash | 8192 | async-demand | 8332.0 | 27.49 | 9510.7 |
| glm47-flash | 8192 | async-previous-pass | 8111.5 | 28.93 | 9249.0 |

Le petit cache crée beaucoup de rechargements. L'augmentation de recouvrement ne garantit donc pas une baisse de latence : demande asynchrone et préchargement peuvent aider ou ralentir selon le modèle et le budget. Les timings courts ne démontrent pas la parité avec Ollama/llama.cpp ; aucun nouveau comparatif de ces moteurs n'est effectué ici.

## Correction, mémoire et serveur observés

- 291 tests workspace réussis, 1 ignoré(s), Clippy et CLI release ; trois fixtures CUDA explicites vérifient événements/publication, protections READY/PENDING, budget unique, pointeurs stables, drainage et refill des octets originaux Q4/Q6. Les warnings sont conservés.
- Neuf refus de configuration exécutés sur le vrai GGUF GPT-OSS ; toutes les catégories CUDA restent inchangées et le chemin CPU désactivé reste chargeable.
- GPT-OSS et GLM, budgets 512/8192 Mio, références/variants réelles : teacher forcing des logits, générations avec seed/pénalités, préfixes, événements/prefetch et durée de vie des modèles. Tolérances et maxima sont dans les logs dédiés. La préférence `previous-pass` peut être inutile avec un grand cache ; aucune obligation artificielle d'obtenir un hit n'est imposée.
- Sur le binaire intégré : 54 observations, 18 paires unary/SSE et six arrêts exacts ; deux suites réseau à 512 Mio testent sampling, seed/pénalités, déconnexion/reprise et quatre requêtes concurrentes sérialisées.
- Deux cycles serveur de déchargement/rechargement à 512 Mio : budgets READY/PENDING/pool/pinned bornés, registre de modèles vide et toutes catégories libérées sauf scratch persistant ; requête sur modèle déchargé refusée, nouveau propriétaire après rechargement et même sortie.
- Nsight capture une seconde génération chaude GPT à 512 Mio, après sortie identique au témoin et nonce de fin validé : 28746 copies H2D privées, 126666374400 octets ; 5443 copies croisent effectivement des kernels sur un autre flux du même GPU/contexte. L'intersection fusionnée vaut 0.544 s, soit 10.90 % de la durée cumulée de ces copies. Le rapport de trace distingue temps cumulés et unions ; aucun débit n'en est déduit.

Les captures brutes et l'analyse de trace sont publiées avec leurs empreintes. Les lourds fichiers SQLite/NSYS restent locaux, leurs SHA sont dans l'analyse. L'absence d'allocation chaude, lorsqu'observée, ne porte que sur la génération capturée. Les requêtes concurrentes restent sérialisées : ce lot ne réalise pas le batching GPU. Une injection d'erreur matérielle CUDA n'a pas été réalisée.

## Reproduction et portée

`manifest.json` lie les sources intégrées, le binaire, la DLL et chaque capture. Les `.log`/`.json` ne normalisent que BOM/CRLF ; leurs SHA originaux sont conservés. Les commandes et scripts historiques sont archivés sous `reproduction/`. Pour répéter les mesures après checkout, construire la CLI de cette révision, placer les deux harnesses sous `target/performance-cache/followup-harness/`, puis passer `--binary`, `--library`, `--async`, les chemins GGUF/tokenizer du manifeste, `--moe-cache 512|8192`, les mêmes notes/sorties/cycles. Les générateurs de prototype sont des archives de la préparation précédente, pas des migrations à réappliquer sur ces sources déjà intégrées.

#84 reste ouvert pour l'étude séparée des prédicteurs entraînés et des autres architectures. #85 traite le choix de politique ; [#86 documente le no-go du placement mixte par expert](../../MOE_HYBRID_DECISION.md) tant qu'il n'a pas un gain mesuré ; #98 conserve le comparatif final des moteurs. Le préchargement n'altère jamais les IDs du routeur pour augmenter artificiellement les hits.
