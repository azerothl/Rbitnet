# Démarrage en 5 minutes (Rbitnet)

Ce guide s’adresse aux personnes qui veulent **un serveur local** type OpenAI, sans lire tout
`ENV_REFERENCE.md` au premier lancement.

## Prérequis

- **Windows, Linux ou macOS** avec ~4 Go de RAM libres pour un petit modèle quantifié (Q4_K_M).
- Un dossier disque pour les poids (≈700 Mo–1,5 Go pour TinyLlama Q4_K_M selon la source).

## Étapes

1. **Installer** le CLI et le serveur depuis une [release GitHub](https://github.com/azerothl/Rbitnet/releases)
   ou depuis les binaires que vous avez compilés (`rbitnet`, `rbitnet-server`).

2. **Télécharger un modèle** recommandé pour débuter :

   ```bash
   rbitnet models install tinyllama-1.1b-chat-q4-k-m --dir ./models
   ```

3. **Écrire `rbitnet.toml`** avec les chemins résolus (évite de répéter les variables d’environnement) :

   ```bash
   rbitnet up tinyllama-1.1b-chat-q4-k-m --dir ./models
   ```

   Sous PowerShell, vous pouvez aussi utiliser `rbitnet quickstart … --write-config`.

4. **Lancer le serveur** :

   ```bash
   rbitnet serve
   ```

   Interface web locale : URL affichée au démarrage (`/ui`).

5. **Tester sans GPU ni GGUF** (stub HTTP) :

   ```bash
   RBITNET_STUB=1 rbitnet serve
   ```

## Attentes réalistes

- L’inférence **CPU pure** est lente sur les grands modèles ; TinyLlama reste raisonnable pour un test.
- Les modèles du catalogue exposent **`golden_tier`** dans `data/compatible_models.json` :
  `verified_golden` lorsqu’une preuve de parité (golden llama.cpp) est tenue à jour ;
  `best_effort` sinon (charge utile toujours Llama-compatible, sans golden obligatoire).
  Voir `docs/GOLDEN_TESTS.md`.

## Dépannage express

- **« tokenizer missing »** : placez `tokenizer.json` à côté du `.gguf` ou définissez `RBITNET_TOKENIZER`.
- **« model not loaded »** : vérifiez `model = "…"` dans `rbitnet.toml` ou `RBITNET_MODEL`.
- **Mémoire** : réduisez la taille du modèle (quantization plus petite) ou fermez les autres applications.

Pour les limites produit connues : `docs/LIMITATIONS.md`.
