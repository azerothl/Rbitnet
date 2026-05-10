# Politique native-first

Rbitnet ne doit pas dependre fonctionnellement d'un moteur d'inference externe pour son fonctionnement normal. Le chemin par defaut reste l'execution native dans `bitnet-core`, exposee par `bitnet-server`, `rbitnet-server`, `rbitnet-runner` et `rbitnet-proxy`.

## Audit des backends externes

- `rbitnet-proxy` contenait un mode `RBITNET_INFERENCE_BACKEND=vllm` qui transferait `/v1/chat/completions`, `/v1/completions` et `/v1/models` vers un serveur vLLM externe. Ce mode est maintenant compile uniquement avec la feature Cargo `experimental-external-backends`, desactivee par defaut.
- Aucune dependance Ollama, llama.cpp server, TensorRT-LLM ou autre daemon HTTP d'inference n'a ete trouvee dans le chemin runtime normal.
- Les references a `llama.cpp` dans la documentation et les scripts concernent la conversion/export GGUF, les conventions de tenseurs ou des comparaisons techniques. Elles ne sont pas des dependances runtime.
- `scripts/bench_ollama_multi.py` est un outil de comparaison manuel et n'est pas appele par Rbitnet.

## Regle de produit

- Par defaut, Rbitnet doit demarrer et inferer avec des binaires du workspace et `bitnet-core`, sans lancer ni exiger vLLM, Ollama, llama.cpp server, Python ou un service HTTP externe.
- Les outils externes sont acceptes seulement pour le developpement, la conversion de modeles, les benchmarks comparatifs ou l'experimentation explicite.
- Tout backend d'inference externe ajoute en code doit etre derriere `#[cfg(feature = "experimental-external-backends")]`, feature desactivee par defaut, et documente comme dev-only.
- `rbitnet-proxy` doit superviser uniquement des workers natifs (`rbitnet-runner` / binaires du workspace) dans le chemin standard et nettoyer l'environnement transmis aux enfants pour eviter la delegation implicite.
