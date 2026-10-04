# Rejouer une trace de placement d'experts

`scripts/simulate_expert_cache.py` compare LRU, LFU et l'heuristique Least-Stale sur les mêmes choix d'experts et des budgets d'octets identiques. Le simulateur protège **tous** les experts sélectionnés dans une couche avant le premier chargement, y compris ceux qui seront acquis plus tard. Cela reproduit la réservation du cache natif : une demande manquante ne doit pas évincer un expert déjà disponible et sélectionné dans le même forward.

Le test de régression utilise un cache de deux experts : `0`, puis `1` deux fois, puis la sélection `[2, 0]`. L'ancien simulateur pouvait évincer `0` avant sa lecture ; le résultat correct est deux hits, trois misses et une éviction, avec les trois politiques. Cette correction porte sur le simulateur. Elle ne change pas les kernels ni le moteur d'inférence.

```powershell
python scripts/simulate_expert_cache.py TRACE.jsonl.gz --budgets-mib 512 2048 4096 8192 --output replay.json --cache-dir CACHE_REPLAY
python -m unittest discover -s scripts -p 'test_*replay_cache.py' -v
python -m unittest discover -s scripts -p 'test_simulate_expert_cache.py' -v
```

`--cache-dir` est facultatif. Il conserve uniquement les résultats déterministes du simulateur. L'identité inclut les octets du simulateur et de l'implémentation du cache, les champs de routage (`pass`, `position`, `phase`, `layer`, `selected`, `group_bytes`), le budget et la politique. Un résultat altéré ou tronqué est recalculé. Les fichiers sont publiés après écriture, flush et synchronisation. Aucune durée d'inférence HTTP/GPU n'est enregistrée ni réutilisée par ce cache.

Sans cette option, le fonctionnement reste celui d'un rejeu complet. Un budget nul représente le repli CPU des couches sélectionnées ; les budgets négatifs et les politiques inconnues sont refusés.

Une simulation seule ne prouve pas un gain du moteur. Les compteurs doivent être confrontés à une exécution native sur la trace capturée, et les mesures de latence doivent être collectées séparément sans instrumentation de routage. Une différence de budget effectif doit également rester visible. Les études matérielles des deux modèles MoE sont suivies dans l'issue #85 ; cette correction et son cache de calcul ne clôturent pas la comparaison.
