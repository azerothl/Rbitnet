# Get started in 5 minutes

English quickstart for a local OpenAI-shaped server. French secondary guide: [DEMARRAGE_5MIN.md](DEMARRAGE_5MIN.md).

For the full runtime guide see [USAGE.md](USAGE.md). Env index: [ENV_REFERENCE.md](ENV_REFERENCE.md).

## Prerequisites

- Windows, Linux, or macOS with ~4 GB free RAM for a small quantized model (Q4_K_M)
- Disk space for weights (~700 MB–1.5 GB for TinyLlama Q4_K_M depending on source)

## Steps

1. **Install** the CLI and server from a [GitHub Release](https://github.com/azerothl/Rbitnet/releases) or build from source (`rbitnet`, `rbitnet-server`).

   ```bash
   curl -fsSL https://raw.githubusercontent.com/azerothl/Rbitnet/main/scripts/install.sh | sh
   ```

   Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`

2. **Download** a starter model (stable tag or catalog id):

   ```bash
   rbitnet models install tinyllama:q4 --dir ./models
   ```

3. **Write `rbitnet.toml`** with resolved paths:

   ```bash
   rbitnet up tinyllama:q4 --dir ./models
   ```

   Or: `rbitnet quickstart tinyllama:q4 --write-config`. Tags also include `bitnet:2b` (see `data/compatible_models.json`).

4. **Serve**:

   ```bash
   rbitnet serve
   ```

   Local web UI URL is printed at startup (`/ui`). Default API: `http://127.0.0.1:8080/v1`.

5. **Stub mode** (no GPU / no GGUF) for HTTP smoke tests:

   ```bash
   RBITNET_STUB=1 rbitnet serve
   ./scripts/smoke_openai.sh
   ```

≤3 commands for a reference model: `install` → `up` → `serve` (or `up` alone then `serve`).
## Realistic expectations

- Pure **CPU** inference is slow on large models; TinyLlama is fine for a first run.
- Catalog entries expose `golden_tier` in `data/compatible_models.json` (`verified_golden` vs `best_effort`). See [GOLDEN_TESTS.md](GOLDEN_TESTS.md).

## Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| tokenizer missing | Put `tokenizer.json` next to the `.gguf` or set `RBITNET_TOKENIZER` |
| model not loaded | Check `model = "…"` in `rbitnet.toml` or `RBITNET_MODEL` |
| out of memory | Smaller quant, or close other apps |

Known product limits: [LIMITATIONS.md](LIMITATIONS.md).
