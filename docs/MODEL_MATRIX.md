# Model Matrix

This matrix is intentionally conservative. Entries in `data/compatible_models.json` remain `verified: false` until someone records a local smoke result with a GGUF and tokenizer on disk.

## Reproduction Commands

Download and write a local config:

```bash
rbitnet up TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --write-config
rbitnet models download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dir models/tinyllama-tokenizer \
  --file tokenizer.json
```

Run a smoke chat:

```bash
rbitnet serve --open-ui
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"temperature":0.7}'
```

Record benchmark rows:

```bash
NO_START_SERVER=1 MODEL=rbitnet-llama RUNS=12 MAX_TOKENS=64 scripts/bench_matrix.sh
```

```powershell
.\scripts\bench_matrix.ps1 -NoStartServer -Model rbitnet-llama -Runs 12 -MaxTokens 64
```

## Current Matrix

| Model id | Repo / file | Status | RAM placeholder | tok/s placeholder | Reproduce |
|----------|-------------|--------|-----------------|-------------------|-----------|
| `tinyllama-1.1b-chat-q4-k-m` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` / `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | API metadata checked, local generation not recorded | `>= 4 GB` | TBD | `rbitnet up ...` then tokenizer download from `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `llama-3.2-1b-instruct-q4-k-m` | `unsloth/Llama-3.2-1B-Instruct-GGUF` / `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | API metadata checked, local generation not recorded | `>= 4 GB` | TBD | `rbitnet up unsloth/Llama-3.2-1B-Instruct-GGUF --file Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `llama-3.2-3b-instruct-q4-k-m` | `unsloth/Llama-3.2-3B-Instruct-GGUF` / `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | API metadata checked, local generation not recorded | `>= 8 GB` | TBD | `rbitnet up unsloth/Llama-3.2-3B-Instruct-GGUF --file Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| `hermes-2-pro-llama-3-8b-q4-k-m` | `NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF` / `Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf` | API metadata checked, local generation not recorded | `>= 16 GB` | TBD | `rbitnet up NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF --file Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf --chat-format chatml` |

## Promotion Rule

Before changing `verified` to `true`, attach:

- OS, CPU, RAM, `rustc -V`, git SHA.
- Exact GGUF basename, tokenizer source, and chat format.
- `/v1/models` output and one `/v1/chat/completions` response.
- One row appended by `scripts/bench_matrix.*` or an equivalent command using `scripts/bench_backend_compare.py`.
