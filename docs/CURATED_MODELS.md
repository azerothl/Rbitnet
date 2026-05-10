# Curated Models

`data/compatible_models.json` is a small reviewed catalog, not a full Hugging Face index. It is meant to help new users pick a first GGUF that matches the loaders Rbitnet can actually start today.

## Schema

Each model entry keeps the existing download fields and may include:

- `tier`: `recommended_beginner` for small first-run models, or `power` for larger models that need more RAM or more careful tokenizer/chat-template setup.
- `use_case`: short tags such as `chat` or `code`.
- `min_ram_gb`: conservative minimum host RAM for the listed quantization plus runtime headroom.
- `verified`: `true` only after project-owned evidence shows the exact entry works with Rbitnet, such as a committed smoke log, CI fixture, or documented maintainer test. Architecture metadata from the Hugging Face API alone is useful, but it is not enough for `verified: true`.

The legacy `min_ram` and `tested` fields remain accepted for older clients. New tooling should prefer `min_ram_gb` and `verified`.

## Curation Policy

Entries should be added only when:

- The repo exposes at least one `.gguf` file and its `gguf.architecture` is supported by the current loader, or the entry clearly documents the required override.
- The catalog names a specific primary GGUF file instead of asking users to guess among many quantizations.
- Tokenizer requirements are explicit: bundled tokenizer, separate tokenizer repo, or `RBITNET_TOKENIZER` instructions.
- RAM guidance is conservative enough for a laptop or workstation user to avoid immediate out-of-memory failures.

Keep `recommended_beginner` small and boring. Prefer one or two CPU-friendly models that make `/ui` and OpenAI-compatible clients easy to try before listing larger models.

## Verification Levels

- `verified: false`: reviewed metadata and docs only, or generated catalog output that still needs a maintainer run.
- `verified: true`: the exact repo/file combination has been run through Rbitnet with a tokenizer and produced a successful `/v1/chat/completions` response, with the evidence linked in `notes` or project docs.

When in doubt, leave `verified` false and explain the known evidence in `notes`.
