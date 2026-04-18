# Plan — version « prod ready » de Rbitnet

Ce document fixe une **feuille de route** pour qu’une release Rbitnet puisse être annoncée comme **prête production** avec des critères vérifiables. Les phases peuvent se chevaucher ; l’ordre reflète une dépendance logique.

## Définition cible (critères de sortie)

Une version est considérée **prod ready** lorsque **toutes** les conditions suivantes sont remplies :

1. **Critères de performance** documentés et mesurés sur au moins une configuration de référence (latence p50/p95, tokens/s, RAM max).
2. **Limites et garde-fous** : prompts, `max_tokens`, concurrence, timeouts — refus explicites et codes HTTP cohérents.
3. **Sécurité minimale** pour une exposition réseau : auth configurable ou guide de déploiement derrière proxy ; pas de surface d’abus évidente sur les entrées.
4. **Observabilité** : logs structurés, métriques (requêtes, erreurs, durée), endpoint de santé adapté au déploiement.
5. **Qualité** : CI verte (build, tests), tests d’intégration sur un GGUF de référence ; politique de version (semver) et notes de release.
6. **Documentation** : installation, configuration, exploitation, dépannage, limites connues — à jour pour cette release.

---

## Phase 0 — Cadrage (court)

| Action | Livrable |
|--------|----------|
| Définir le **périmètre prod** visé (intranet, edge, SaaS, Akasha uniquement) | Paragraphe dans ce doc ou ADR |
| Choisir **1–3 modèles GGUF de référence** (taille, quant, arch) pour bench et tests | Liste figée + emplacement des artefacts |
| Fixer des **SLO indicatifs** (ex. p95 latence première token, erreurs &lt; 0,1 %) | Tableau cible non contractuel |

---

## Phase 1 — Fiabilité et limites

| Action | Livrable |
|--------|----------|
| Plafonds sur taille de corps JSON, longueur prompt, `max_tokens`, taille contexte effective | Rejet `413` / `400` avec message clair |
| Timeouts par requête (inférence + I/O) | Pas de requêtes bloquées indéfiniment |
| Gestion explicite **OOM** / mmap échoué / fichier manquant | Erreurs typées, logs, pas de panic utilisateur |
| Politique de **concurrence** (semaphore / file) | Nombre max de générations simultanées configurable |
| Tests d’intégration charge **légère** (plusieurs requêtes séquentielles / peu parallèles) | Test CI ou script documenté |

---

## Phase 2 — Performance et ressources

| Action | Livrable |
|--------|----------|
| **Benchmarks** reproductibles (prompt fixe, longueur fixe, `--release`) | `docs/BENCHMARKS.md` + chiffres pour modèles de référence |
| Profiler hot paths (matmul, attention, déquant) | Rapport court + issues priorisées |
| Pistes d’optimisation **CPU** (SIMD, threads, réduction allocations) | Implémentations incrémentales derrière critères mesurables |
| (Optionnel) Documenter **plafond RAM** par modèle et recommandations matériel | Section doc |

*Note : le GPU ou l’appel à un backend externe peut être une **phase ultérieure** si le positionnement produit est « Rust CPU seul ».*

---

## Phase 3 — Sécurité et exposition

| Action | Livrable |
|--------|----------|
| Revue **entrées utilisateur** (chat, chemins de fichiers env) | Pas de traversal arbitraire ; validation stricte |
| **Auth** : clé API via header ou `--api-key`, ou doc explicite « uniquement derrière Nginx + auth » | Choix documenté + implémentation minimale si besoin |
| **CORS** et binding : défaut `127.0.0.1` ; avertissement si `0.0.0.0` | README + log au démarrage |
| Scan dépendances / `cargo audit` en CI | Job CI ou checklist release |

---

## Phase 4 — Observabilité et exploitation

| Action | Livrable |
|--------|----------|
| Métriques (Prometheus ou stats simples) : requêtes, durées, erreurs, tokens | Endpoint `/metrics` ou export documenté |
| Logs structurés (niveau, durée, `task_id` / `request_id`) | Format stable |
| **Health** : `GET /health` ou `/ready` (liveness vs readiness si modèle chargé) | Spécification pour orchestrateurs |
| Variables d’environnement **récapitulées** et validées au démarrage | Message d’erreur clair si config invalide |

---

## Phase 5 — Qualité et compatibilité

| Action | Livrable |
|--------|----------|
| Étendre / figurer les **types GGML** supportés ou erreurs explicites | Tableau dans la doc |
| **Alias** de noms de tenseurs ou tests sur 2 exports (llama.cpp / BitNet) | Moins d’échecs « silencieux » sur des GGUF valides |
| Suite de **tests de non-régression** (golden logits ou texte sur prompts courts) | CI avec artefact optionnel `RBITNET_TEST_GGUF` |
| Processus **release** : version workspace, tag, changelog | `README` + `docs/RELEASE.md` (court) |

---

## Phase 6 — Documentation et communication

| Action | Livrable |
|--------|----------|
| Mettre à jour **USAGE**, **TRAINING_AND_COMPATIBILITY**, README avec critères prod | Cohérence avec ce plan |
| Page **« Limitations »** (perf, types, archs) | Évite les attentes irréalistes |
| Guide **déploiement** (systemd, Docker optionnel, reverse proxy) | Au moins un scénario de référence |

---

## Priorisation suggérée (MVP prod interne)

1. Phase **1** (limites + timeouts + concurrence) — bloque les incidents les plus fréquents.  
2. Phase **4** (logs + health) — indispensable pour opérer.  
3. Phase **2** (bench + perf CPU raisonnable) — crédibilise l’usage réel.  
4. Phase **3** (sécurité) — avant toute exposition Internet.  
5. Phases **5** et **6** — en parallèle dès que la stabilité fonctionnelle est là.

---

## Métriques de suivi (exemples)

| Indicateur | Cible indicative (à ajuster) |
|------------|-------------------------------|
| Tests CI | 100 % sur la branche release |
| Couverture des cas d’erreur (entrée, fichier, tokenizer) | Tous les chemins documentés testés ou explicitement « non supporté » |
| Latence p95 (prompt court, modèle ref., CPU ref.) | Fixée et publiée dans `BENCHMARKS.md` |
| Uptime attendu (interne) | Pas de fuite mémoire sur charge modérée longue durée |

---

## Révision

Ce plan doit être **révisé** après chaque release majeure ou si le périmètre produit change (ex. support GPU, multi-modèles).
