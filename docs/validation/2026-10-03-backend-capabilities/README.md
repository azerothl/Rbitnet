# Backends, état HTTP et mesures mémoire — 3 octobre 2026

Une bibliothèque Vulkan/Metal présente ne constitue pas une implémentation GPU. La sélection automatique utilise CUDA/ROCm pour les architectures compatibles et le CPU pour Mixtral/Qwen3, dont les exécuteurs utilisent actuellement le CPU. Une demande explicite de backend non pris en charge est refusée au chargement ; elle ne crée plus un moteur marqué prêt qui échoue à la première génération. Les prototypes Vulkan/Metal déclarent `backend_accelerated=false`. Les modes explicites stub/toy déclarent CPU.

Le contrôle du serveur Windows réel a découvert une seconde erreur : après un échec de chargement Vulkan, `/ready` renvoyait 503, mais le chat renvoyait une réponse factice avec HTTP 200. [initial-regression.json](initial-regression.json) conserve cet échec de développement. Le correctif interdit cette réponse sur les six combinaisons chat/completions/messages × unary/SSE. Après l'éviction d'un modèle réel, les mêmes routes et `/ready` renvoient 503 `ModelUnloaded`. Une requête garde le moteur qu'elle a validé, y compris pour le template, le comptage des tokens et le streaming ; une sélection dans le registre garde son propre moteur même si une autre requête change la sélection globale.

Les rechargements réussis rétablissent l'inférence. Un rechargement raté conserve le modèle réel précédemment chargé et permet de continuer à l'utiliser. Un serveur de smoke explicitement configuré en stub reste utilisable. Ces règles sont testées indépendamment de l'option de vérification du nom du modèle.

`rbitnet_process_rss_bytes` mesure maintenant le working set Windows via PSAPI (Linux utilise toujours `/proc`). La mesure est comparée à `psutil` dans chacun des quatre serveurs réels. L'ancienne valeur VRAM constamment égale à zéro est supprimée : la mesure par processus n'est pas implémentée et `rbitnet_process_vram_measurement_available=0` indique cette indisponibilité. Aucun chiffre de VRAM n'est inventé.

## Preuves

- [results.json](results.json) : quatre cas réels, avec empreintes SHA-256 du CLI et de la DLL ; CPU stub malgré une demande CUDA, CUDA automatique avec Qwen3.5-2B Q8_0 répondant « Paris », puis refus Vulkan et Metal. Le cas CUDA comprend rechargement raté, maintien de la réponse réelle, déchargement, six erreurs 503, rechargement réussi et nouvelle réponse réelle. Les refus comprennent aussi les six erreurs attendues.
- [workspace-auto.log](workspace-auto.log) : 236 tests réussis, 0 échec, 1 ignoré, 25 suites ; workspace sans `RBITNET_BACKEND`, sur la machine équipée CUDA. Sélection automatique CPU des fixtures Qwen3/Mixtral et refus explicites vérifiés. Les tests de sélection/probes sont des tests logiciels ; aucun GPU AMD, Intel ou Apple n'est présent.
- [clippy.log](clippy.log) et [cli-build.log](cli-build.log) : vérification workspace/all-targets et compilation CLI release réussies. Les avertissements existants restent visibles.
- [live.log](live.log) et journaux par mode : exécution de `scripts/validate_backend_capabilities.py`, sans proxy vers un autre moteur.

Machine : Ryzen 7 9800X3D, RTX 4080 SUPER 16 Go, Windows, pilote 610.88, Rust 1.94.1. Modèle/tokenizer Qwen et DLL identiques à la [validation du pipeline résident](../../benchmarks/2026-10-03-qwen-full/README.md). Ceci valide l'état et les contrats HTTP ; aucune mesure de débit supplémentaire n'est revendiquée.

Les issues #22 et #24 restent ouvertes : le support natif AMD/Intel/Apple et les fonctionnalités encore incomplètes exigent leurs propres implémentations et preuves matérielles. Voir [l'audit actualisé](../../STUBS_AND_MVP_AUDIT.md).
