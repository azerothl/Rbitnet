# GLM et cache d’experts : diagnostic conservé du 4 octobre 2026

L’ancienne divergence GLM Least-Stale reste inexpliquée. Les observations ci-dessous renforcent les contrôles de correction ; elles ne démontrent ni sa résolution ni un gain de débit. Le défaut reste LRU, les transferts asynchrones et le préchargement sont désactivés dans la reproduction HTTP.

## Trois vérifications distinctes

| Vérification | Version réellement exécutée | Résultat et portée |
| --- | --- | --- |
| HTTP GLM sans lecture supplémentaire des logits | CLI compilée depuis `084122f1c9195a46ea7e48db8d0247c679bb9ae5`, sources compilées équivalentes à main `5c4f009f3839b6323e57008518152b4bd181c935` ; bibliothèque Native actuelle | 24 réponses exactes : 2 politiques × 4 cycles × 3 prompts. Chaque répétition et Least-Stale sont comparés à la première réponse LRU de chaque prompt. |
| Compute Sanitizer sur le bloc MLA complet synthétique | Binaire de test historique et bibliothèque historique liés à la capture négative | memcheck, racecheck, initcheck et synccheck : un test réellement exécuté par outil, aucune erreur ou course signalée. Attention, routeur, experts fixes/dynamiques, modes eager/graphes, repli FFN CPU, reset et restauration de préfixe ; 8 formats de poids. |
| Octets des experts réellement présents en VRAM | Nouveau binaire de test compilé depuis `0da733e41f4ab32ab6abf569e7c37c632d501fbe` ; bibliothèque Native actuelle réutilisée après égalité des sources | GPT-OSS et GLM : 864 groupes vérifiés, soit 2 592 projections gate/up/down, bit à bit contre les tranches GGUF attendues. LRU/LFU/Least-Stale, 4 séquences, 3 passes, 2 couches routées, budget de 2 experts. |

Le nouveau test impose une vraie réutilisation d’adresses, protège les deux experts détenus et charge un expert absent avant un expert sélectionné déjà présent. Il relit aussi le slot conservé après le remplacement d’un autre slot. Ces lectures synchrones sont réservées aux tests et modifient l’ordonnancement ; elles ne constituent pas une reproduction silencieuse de l’erreur initiale.

Les commandes fraîches `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets` et `cargo test --workspace` sont passées : 349 tests réussis, 1 ignoré. Clippy conserve les avertissements préexistants du projet ; aucun nettoyage global n’est inclus. Les tests matériels sur les deux modèles sont exécutés séparément avec leurs gardes activés.

## Capture négative conservée

La capture originale échoue dans Least-Stale au cycle 1, prompt 0, après trois requêtes du premier cycle. Son CLI a l’empreinte `5d937ad9f035c20548e12bebf7a4121ff389a2d09f3d1034f706f8194cf91470` et sa bibliothèque Native `b86c8fa6d5cf34f500c01e860bae1f87fcb2411862d1ed2e0f68dc08f9411e38`. Les résultats et journaux négatifs sont conservés dans `raw/original-failure-*` ; ils ne sont pas remplacés par les répétitions réussies.

Les diagnostics précédents qui relisaient tous les logits avaient déjà échoué à reproduire la divergence. La répétition HTTP présentée ici n’ajoute ni lecture complète des logits ni trace du routeur. Elle conserve les métriques HTTP ordinaires.

## Reproduction du contrôle des octets en VRAM

Sur une machine CUDA, lancer séparément le test pour chaque GGUF GPT-OSS ou GLM :

```powershell
$env:RBITNET_MOE_CACHE_TEST = "1"
$env:RBITNET_TEST_GGUF = "chemin/vers/le/modele.gguf"
$env:RBITNET_CUDA_QUANT_LIB = "chemin/vers/rbitnet_cuda_quant64.dll"
$env:RBITNET_MOE_ASYNC = "0"
$env:RBITNET_MOE_PREFETCH = "off"
cargo test -p bitnet-core --release --lib optional_expert_cache_all_policies_preserve_device_bytes_across_refills_and_passes -- --nocapture --test-threads=1
```

Le marqueur attendu par modèle est `EXPERT_DEVICE_BYTES_DONE groups=432 projections=1296`. Sans `RBITNET_MOE_CACHE_TEST=1`, ce test matériel est volontairement ignoré par son garde : un succès Cargo seul ne prouve pas son exécution GPU. Les journaux publiés contiennent les marqueurs des trois politiques et des quatre séquences.

## Traçabilité et limites

`publication.json` contient les empreintes SHA-256 des captures publiées et, pour les fichiers compressés, celles des octets originaux. `compiled-source-index.json` lie les sources Rust/Native et manifestes figés lors des commandes de validation aux blobs Git normalisés. Les manifestes bruts conservent commandes, environnements, modèles et empreintes des exécutables/bibliothèques. Les chemins absolus sont ceux de la machine de mesure ; les helpers sont conservés tels qu’exécutés, à adapter pour une autre installation.

Les modèles sont GPT-OSS-20B Q4_K_M et GLM-4.7-Flash Q4_K_M. Le contrôle des experts lit leurs véritables poids mappés et compare directement chaque projection aux octets transférés depuis la VRAM. Les fichiers GGUF sont identifiés par chemin, taille, mtime et référence de téléchargement ; aucun nouveau hash intégral des GGUF n’est revendiqué.

Le test complet Compute Sanitizer utilise des matrices synthétiques et une bibliothèque historique, pas le modèle GLM de 30 milliards de paramètres. Aucun résultat ne valide AMD, Intel ou Metal. Aucun benchmark de vitesse ni parité avec Ollama/llama.cpp n’est effectué ici. Les critères de résolution de [#85](https://github.com/azerothl/Rbitnet/issues/85) et la recherche de cause de [#116](https://github.com/azerothl/Rbitnet/pull/116) restent ouverts.
