# Using Rbitnet — models, tokenizer, and runtime

Consolidated variable index: **[ENV_REFERENCE.md](ENV_REFERENCE.md)**.

## Akasha `llm_router.yaml` (BitNet / local)

Point a **BitNet** or OpenAI-compatible route at `http://127.0.0.1:<port>/v1` (see `rbitnet serve` / `rbitnet-server` and `RBITNET_*` env vars in this repo). In [Akasha](https://github.com/azerothl/Akasha), add a provider entry in `llm_router.yaml` with that base URL and a small model id matching `RBITNET_MODEL`. Use **`akasha services doctor`** (akasha-models compose) plus **`GET /api/router/metrics`** after traffic to validate latency and errors.

### Example `llm_router.yaml` snippet (Akasha)

Akasha already documents a **BitNet / Rbitnet** provider block in [`spec/llm_router.example.yaml`](https://github.com/azerothl/Akasha/blob/main/spec/llm_router.example.yaml) (`providers.bitnet.base_url`). Point it at your listener (default `http://127.0.0.1:8080` — no `/v1` suffix in that field; the router adds the API path). Then set a task type’s `primary` to `provider: bitnet` and `model: <id>` where `<id>` matches **`RBITNET_MODEL`** / `GET /v1/models`. Prefer **`rbitnet models install <bundle>`** so `rbitnet.manifest.json` paths stay consistent.

## Prometheus metrics (`GET /metrics`)

`rbitnet-server` (and `rbitnet serve`) expose **`GET /metrics`** as **Prometheus text** alongside **`GET /health`** and **`GET /ready`** (no API key required for these paths — see `crates/bitnet-server/tests/openai_compat.rs`). Counters include end-to-end wall time plus **phase-oriented** sums when using a real executor: `rbitnet_inference_encode_ms_sum` (tokenizer), `rbitnet_inference_prefill_ms_sum`, `rbitnet_inference_decode_ms_sum`, `rbitnet_inference_ttft_ms_sum` (encode+prefill), `rbitnet_inference_itl_us_sum` and `rbitnet_inference_tpot_us_sum` (per-request average inter-token / TPOT in decode, summed per completed call), and `rbitnet_completion_tokens_total` (generated token count from the runtime when available). Use them for:

- **Hermes-style self-hosted ops:** scrape with Prometheus / Grafana or a simple `curl -sS http://127.0.0.1:8080/metrics | head`.
- **Correlation with Akasha:** when Akasha routes traffic here, watch Rbitnet request counters and errors while observing **`GET /api/router/metrics`** on the Akasha daemon.

**Tested profiles:** treat bundles from **`rbitnet models install --list`** / **`data/compatible_models.json`** as the supported matrix; Hub `search` hits remain best-effort until promoted to the curated list.

## Do you need Python?

**No — not for running Rbitnet.** Inference is implemented in **Rust** (`bitnet-core`): GGUF is memory-mapped, weights are dequantized in-process, and text is generated via the [`Engine`](../crates/bitnet-core/src/inference.rs) or the HTTP server.

You **only need Python (or another stack)** if you are **converting** checkpoints from Hugging Face Safetensors into **GGUF** using upstream tools (for example [microsoft/BitNet](https://github.com/microsoft/BitNet) scripts). That is a one-time **export** step on the machine where you build the file, not a runtime dependency of `rbitnet-server`.

### Fine-tuning helper (optional Python)

Training still happens **outside** `bitnet-core`. This repo includes an optional **[`training/`](../training/README.md)** tree (Transformers + TRL + PEFT) and CLI helpers:

| Command | Purpose |
|--------|---------|
| `rbitnet train --repo-root DIR --recipe recipes/sft_lora.py -- …` | Run a Python recipe under `DIR/training/`; arguments after `--` are forwarded to the script (example in [`training/README.md`](../training/README.md)). Env **`RBITNET_REPO_ROOT`** defaults to `.`. Override interpreter with **`RBITNET_PYTHON`**. |
| `rbitnet export-gguf [--checkpoint DIR]` | Print a short **HF checkpoint → GGUF** checklist (llama.cpp); optional `--checkpoint` fills in an example `convert_hf_to_gguf.py` line. |

After you have a `.gguf` and tokenizer files, inference is unchanged: **`RBITNET_MODEL`**, `rbitnet serve`. See **[TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md)** for [ml-intern](https://github.com/huggingface/ml-intern), LoRA/SFT, and compatibility rules.

## Hugging Face: curated list, search, and download (no Python)

The **`rbitnet`** binary (crate `rbitnet-cli`) lists a **curated** model index, can **search** the Hugging Face Hub for repos that expose **`.gguf`** files, and **downloads** files into a directory using the same cache layout as the Python hub (`HF_TOKEN` / `--token` for gated models).

**Disk space (cache vs `--dir`):** Downloads go through the Hugging Face Hub cache first (`hf-hub`, same roots as Python `huggingface_hub`). By default, `rbitnet` places files under your `--dir` with a **hard link** to the cached blob when the OS allows it (same volume as the cache), so weights are **not** duplicated. If a hard link cannot be created (different drive, filesystem, or permissions), the CLI **falls back to a full copy**. Use **`--symlink`** on `models download`, `models install`, and interactive `models list` / `models search` to create a **symbolic link** to the cache instead (Unix-friendly; on Windows you may need Developer Mode or an elevated shell). Clearing the Hub cache later can break symlink targets.

**Cache location:** You can point the Hub cache with **`HF_HOME`**, **`XDG_CACHE_HOME`**, or **`HUGGINGFACE_HUB_CACHE`** (see Hugging Face docs). That does not remove the need for a destination `--dir` when you want a project-local layout or `rbitnet.manifest.json` paths—it only changes where `hf-hub` stores blobs.

**Tokenizers:** There is no separate Hub API for “the tokenizer”—you download **`tokenizer.json`**, **`tokenizer.model`**, etc., like any other repo file; Rbitnet resolves those filenames from the Hub model API (`siblings`).

**Why many HF BitNet repos do not “just work”:** Rbitnet loads **GGUF + Llama-shaped** graphs and a **tokenizer file** on disk; Hugging Face often splits **Safetensors vs GGUF** across repos, or documents **AutoTokenizer** from another (sometimes **gated**) repository. See **[HF_BITNET_RBITNET_GAP.md](HF_BITNET_RBITNET_GAP.md)** for the full gap table, readiness labels, and `models install` bundles.

| Command | Purpose |
|--------|---------|
| `rbitnet quickstart <repo_id> [--file NAME] [--dir DIR] [--write-config]` | Resolve/download a Hugging Face GGUF repo, infer local `RBITNET_MODEL` / `RBITNET_TOKENIZER` paths when available, and print exact PowerShell + bash commands plus `/v1/models` and chat `curl` examples. Use `--no-download` to print commands only. `--write-config` writes `rbitnet.toml` (or `--user-config`) so `serve` works without model env vars. |
| `rbitnet up <repo_id> [--file NAME] [--dir DIR]` | Same resolver/downloader as `quickstart`, but writes `rbitnet.toml` by default. |
| `rbitnet models list` | Print the curated catalog (default: raw `data/compatible_models.json` on GitHub). Override with `RBITNET_MODELS_INDEX_URL`. |
| `rbitnet models list --interactive` (`-i`) | Same catalog in a **terminal UI** (table + detail panel + download with `d`). Target directory: `--download-dir` or `RBITNET_DOWNLOAD_DIR` (default `models`); optional `HF_TOKEN` for gated downloads. |
| `rbitnet models search <query>` | Query the Hub API and show repos that have at least one `.gguf` (not project-tested — see stderr warning). Includes a heuristic `confidence` label for BitNet likelihood and an **`rbitnet=`** readiness hint (`ready`, `needs_tokenizer`, `needs_external_tokenizer`, `unsupported_arch_likely`, `experimental_gguf` — see [HF_BITNET_RBITNET_GAP.md](HF_BITNET_RBITNET_GAP.md)). **Default mode is strict BitNet filtering** (`likely`/`possible` only). |
| `rbitnet models search <query> --all-gguf` | Disable strict filtering and show all GGUF repos, including `generic-gguf`. |
| `rbitnet models search <query> -i` | Same search as an **interactive** table (detail + `d` download like `models download` without `--file`). Press `f` to toggle between the default strict BitNet filter and `all-gguf`. Readiness appears in the **rbitnet** column. |
| `rbitnet models install --list` | Print curated **bundle** ids (paired GGUF repo + tokenizer source). |
| `rbitnet models install <bundle-id> --dir DIR [--symlink]` | Download the bundle into `DIR` and write **`rbitnet.manifest.json`** with suggested `RBITNET_MODEL` / `RBITNET_TOKENIZER` paths (relative). Uses `HF_TOKEN` when the Hub requires it. **`--symlink`** optional, same semantics as `models download`. |
| `rbitnet models generate-catalog` | Build a `compatible_models.json` **draft** from Hub search (one GGUF + tokenizer per repo when found). Review before commit — see below. |
| `rbitnet models download <repo_id> [--dir DIR] [--file NAME]... [--symlink]` | Download files (repeat `--file`; if omitted, all `.gguf` plus tokenizer files when present). Optional **`--symlink`** : symlink into `--dir` instead of hard link / copy. |
| `rbitnet models inspect <PATH>` | Inspect a local model file or directory and print model, tokenizer, and template-source paths. |
| `rbitnet models rm <PATH> --yes` | Remove a local model file or directory. The command refuses deletion without `--yes`. Alias: `models remove`. |
| `rbitnet serve` | Same HTTP server as `rbitnet-server` (same `RBITNET_*` env vars). Optional **`--api-key`** / **`--bind`** apply only when `RBITNET_API_KEY` / `RBITNET_BIND` are unset (CLI does not override existing env). |
| `rbitnet chat` | Terminal chatbot TUI for fast local tests. Connects to an existing server with `--base-url`, or launches one with `--serve`. |
| `rbitnet-proxy` | Parent OpenAI-compatible proxy. Requires `RBITNET_MODEL_REGISTRY`; spawns one native `rbitnet-runner` child per requested model id. No external inference engine is required. |

**Compatibility:** Only entries in the **curated** list are maintained for Rbitnet testing. Search hits are **best-effort** Hub results based on `.gguf` file presence only. **Important:** `.gguf` does **not** imply BitNet 1-bit weights nor guaranteed Rbitnet compatibility.

**TLS / networking (Windows):** Hub calls and `raw.githubusercontent.com` use **native-tls** (the OS certificate store) and honor **`HTTPS_PROXY` / `HTTP_PROXY`** when set. If you see errors such as "connection closed by remote host" (10054) or TLS initialization failures, check your proxy, antivirus, or corporate HTTPS inspection settings.

**Regenerate the catalog file without filling it in by hand:** `models generate-catalog` queries the Hugging Face API (like `search`), selects **one** `.gguf` file per repo (using a `Q4_K_M`-style heuristic when present), and adds `tokenizer.json` / `tokenizer.model` if they are listed in the repo. The default query is **`gguf`** (`llama`-style searches often return Safetensors repos first, without `.gguf` files to inspect). The output is a **draft** to review, then commit to `data/compatible_models.json`.

```bash
cargo build -p rbitnet-cli --release
./target/release/rbitnet models list
# Interactive table: ↑/↓ or j/k (row), PgUp/PgDn (detail), d (download), q or Esc (quit)
./target/release/rbitnet models list -i --download-dir ./models
./target/release/rbitnet models search llama
./target/release/rbitnet models search llama --all-gguf
./target/release/rbitnet models search gguf -i
./target/release/rbitnet models search bitnet -i
./target/release/rbitnet models install --list
./target/release/rbitnet models install microsoft-bitnet-b1.58-2b-4t --dir ./models
./target/release/rbitnet quickstart TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
./target/release/rbitnet up TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
./target/release/rbitnet models inspect ./models
# JSON draft to stdout (or --output data/compatible_models.json)
./target/release/rbitnet models generate-catalog --max-entries 40 --output data/compatible_models.json
# (default: `--query gguf`; for a specific family: `--query llama` + `--max-inspect 400`)
./target/release/rbitnet models download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --dir ./models --file tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --file tokenizer.json
```

**Interactive mode:** The catalog or search results are displayed in a **table** (ratatui). The selected row fills the **Detail** panel; **`d`** downloads the listed files (curated catalog: JSON file list; search: `.gguf` + `tokenizer.json` / `tokenizer.model` **if** present in Hub siblings). In interactive search mode, **`f`** toggles the filter between strict BitNet and `all-gguf`. Target directory: **`--download-dir`** or environment variable **`RBITNET_DOWNLOAD_DIR`** (default `models`). Optional Hub token: **`HF_TOKEN`** / **`--token`** (for search and private repos). Optional **`--symlink`** (with `-i`) uses symlinks to the Hub cache instead of hard links / copy.

**BitNet heuristic (`confidence`):** `likely-bitnet`, `possible-bitnet`, and `generic-gguf` are **textual hints** (repo/file name matching) and not a formal validation. Strict mode (the default) combines the Hub `other=bitnet` filter with this heuristic to reduce noise; entries should still be manually validated before production use.

**Tokenizers on the Hub:** Many BitNet / Transformers repos document `AutoTokenizer.from_pretrained(...)` without publishing a `tokenizer.json` or `tokenizer.model` in the same repo as the GGUF (e.g., tokenizer loaded from a different repo, or only Safetensors weights). Rbitnet requires a **file** `tokenizer.json` or `tokenizer.model` alongside the GGUF, or **`RBITNET_TOKENIZER`** pointing to one of those files. The `models search` command can only list what the `siblings` API exposes; if the model card points to a different Hugging Face id for the tokenizer, download that file from that repo or set `RBITNET_TOKENIZER` accordingly.
## Requirements to run a real model

1. A **`.gguf`** file with **Llama-compatible** layout (see [BITNET_SPEC.md](BITNET_SPEC.md) and [TRAINING_AND_COMPATIBILITY.md](TRAINING_AND_COMPATIBILITY.md)), **or** a roadmap-tagged GGUF (`glm4moe`, `gptoss`, `deepseek2`) with **`RBITNET_BACKEND=cuda`** whose tensors still match the Llama loader. If not, the server fails at **load** with an explicit error — see [ARCHITECTURE_GGUF_MATRIX.md](ARCHITECTURE_GGUF_MATRIX.md).
2. A **tokenizer** file that the Hugging Face `tokenizers` crate can load:
   - Prefer **`tokenizer.json`** next to the GGUF, **or**
   - **`tokenizer.model`** (SentencePiece) in the same directory, **or**
   - Set **`RBITNET_TOKENIZER`** to the absolute path of either file.

Without a tokenizer, the engine returns `TokenizerMissing` when you try to generate text.

**Chat templates:** By default, Rbitnet builds a simple **plain-text** prompt from `messages` (`role: content` lines). Set **`RBITNET_CHAT_FORMAT=llama3`**, **`chatml`**, or **`raw`** to select a built-in template. If neither `RBITNET_CHAT_TEMPLATE` nor `RBITNET_CHAT_FORMAT` is set, the server looks for `chat_template` in `tokenizer_config.json` next to the resolved tokenizer/model and maps common Llama 3 / ChatML Jinja templates onto the built-ins. For small custom prompts, set **`RBITNET_CHAT_TEMPLATE`**; it supports placeholder replacement for `{messages}` / `{{messages}}`, `{prompt}`, `{system}`, `{user}`, and `{assistant}`. This is intentionally a small subset, not a full Jinja engine. Bundles from **`rbitnet models install`** record tokenizer-relative paths in `rbitnet.manifest.json`; align temperature and stop tokens with the upstream recommendation.

## Quick start — HTTP server with a GGUF

```bash
# Linux / macOS
export RBITNET_MODEL=/absolute/path/to/model.gguf
# Optional if tokenizer.json is not beside the GGUF:
# export RBITNET_TOKENIZER=/absolute/path/to/tokenizer.json
export RBITNET_BIND=127.0.0.1:8080
cargo run -p bitnet-server --bin rbitnet-server --release
```

```powershell
# Windows PowerShell
# Set RBITNET_MODEL to one .gguf file (not a folder). Example: C:\models\weights.Q4_K_M.gguf
$env:RBITNET_MODEL="C:\path\to\model.gguf"
$env:RBITNET_BIND="127.0.0.1:8080"
cargo run -p bitnet-server --bin rbitnet-server --release
```

**CLI overrides (same as `rbitnet serve`):** `rbitnet-server --bind 127.0.0.1:8080 --api-key your-secret` only fills env when those variables are **not** already set.

**Local config:** `rbitnet serve` / `rbitnet-server` read flat `rbitnet.toml` defaults from the current directory, `RBITNET_CONFIG`, then the user config directory (`%APPDATA%\Rbitnet\rbitnet.toml` on Windows, `$XDG_CONFIG_HOME/rbitnet/rbitnet.toml` or `~/.config/rbitnet/rbitnet.toml` on Unix). Supported keys: `model`, `tokenizer`, `bind`, `chat_format`, `model_registry`, `active_model_id`. Environment variables keep priority.

## Multi-process proxy

Use `rbitnet-proxy` when you want one parent OpenAI-compatible base URL with isolated child processes per model. Each child is a real `rbitnet-runner` server with its own `RBITNET_MODEL`, bind port, mmap, tokenizer, and crash boundary.

Example registry:

```json
{
  "default": "tiny",
  "models": {
    "tiny": {
      "gguf": "C:/models/tiny.gguf",
      "tokenizer": "C:/models/tokenizer.json",
      "architecture": "llama"
    },
    "other": {
      "gguf": "C:/models/other.gguf"
    }
  }
}
```

PowerShell:

```powershell
$env:RBITNET_MODEL_REGISTRY="C:\path\to\rbitnet-registry.json"
$env:RBITNET_PROXY_BIND="127.0.0.1:8080"
cargo run -p rbitnet-proxy --release
```

The proxy routes `/v1/chat/completions` and `/v1/completions` by the JSON `model` field. `GET /v1/models` lists the registry and marks a model loaded once its child is running. `RBITNET_API_KEY` is enforced at the proxy and forwarded to children.

Native-first policy: the proxy's normal mode supervises only workspace binaries (`rbitnet-runner` / `rbitnet-server` internals) and routes to `bitnet-core`. Experimental delegation to external HTTP inference servers is not built by default; see [NATIVE_FIRST.md](NATIVE_FIRST.md).

**Health and metrics (operations):**

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
curl -s http://127.0.0.1:8080/metrics
```

**Check models list:**

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Use the reported `id` (for example `rbitnet-llama` when `general.architecture` is `llama`) as the `model` field in chat requests.

## Hot Reload and Terminal Chat TUI

Set `RBITNET_ADMIN_TOKEN` to enable runtime model management without restarting
the server process:

```powershell
$env:RBITNET_ADMIN_TOKEN = "dev-secret"
```

Unload the active model and keep the HTTP process alive:

```powershell
Invoke-WebRequest `
  "http://127.0.0.1:8080/v1/admin/unload" `
  -Method POST `
  -Headers @{ "X-Rbitnet-Admin-Token" = "dev-secret" }
```

Reload the current env/config model:

```powershell
Invoke-WebRequest `
  "http://127.0.0.1:8080/v1/admin/reload" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ "X-Rbitnet-Admin-Token" = "dev-secret" } `
  -Body "{}"
```

Reload a registry model:

```powershell
Invoke-WebRequest `
  "http://127.0.0.1:8080/v1/admin/reload" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ "X-Rbitnet-Admin-Token" = "dev-secret" } `
  -Body '{"active_model_id":"tiny"}'
```

Reload a single GGUF path:

```powershell
Invoke-WebRequest `
  "http://127.0.0.1:8080/v1/admin/reload" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ "X-Rbitnet-Admin-Token" = "dev-secret" } `
  -Body '{"model":"C:/models/tiny.gguf","tokenizer":"C:/models/tokenizer.json","architecture":"llama"}'
```

Metrics include `rbitnet_model_reloads_total`,
`rbitnet_model_reload_failures_total`, and `rbitnet_model_reload_ms_sum`.

### `rbitnet chat`

Connect to an already running server:

```powershell
cargo run -p rbitnet-cli -- chat `
  --base-url http://127.0.0.1:8080/v1 `
  --model rbitnet-llama `
  --admin-token dev-secret
```

Launch a managed local server and open the TUI:

```powershell
cargo run -p rbitnet-cli -- chat --serve `
  --model-path C:\models\tiny.gguf `
  --tokenizer C:\models\tokenizer.json `
  --chat-format raw `
  --admin-token dev-secret
```

Useful keys in the TUI:

| Key | Action |
|-----|--------|
| `Enter` | Send the current prompt. |
| `Ctrl+R` | Call `POST /v1/admin/reload`. |
| `Ctrl+U` | Call `POST /v1/admin/unload`. |
| `m` | Fetch `/v1/models` and show model ids in the status bar. |
| `F2` / `F3` | Decrease / increase `max_tokens`. |
| `-` / `+` | Decrease / increase `temperature`. |
| `q` with empty prompt or `Ctrl+C` | Quit. |

Use `--transcript chat.jsonl` to append prompt/reply rows for quick regression
checks.

**Chat completion (non-streaming):**

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rbitnet-llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0.8,"stop":["</s>"]}'
```

OpenAI compatibility note: `stop` is applied to returned text after generation. `temperature`, `top_p`, `seed`, `frequency_penalty`, and `presence_penalty` are promoted into the core sampler for real Llama/Qwen generation. Stub/toy modes keep their lightweight deterministic behavior and only use `temperature` where applicable.

## Modes without a full GGUF

| Mode | Purpose |
|------|---------|
| **`RBITNET_STUB=1`** | Integration smoke tests; no weights; canned reply. |
| **`RBITNET_TOY=1`** | Tiny built-in toy LM; no file; useful for CI and API checks. |

Do **not** set stub/toy if you want real generation from `RBITNET_MODEL`.

## Environment variables

### Engine

| Variable | Meaning |
|----------|---------|
| `RBITNET_BIND` | Listen address (default `127.0.0.1:8080`). |
| `RBITNET_MODEL` | Path to a `.gguf` file (must not contain `..` path components). |
| `RBITNET_TOKENIZER` | Path to `tokenizer.json` if not next to the GGUF (must not contain `..`). |
| `RBITNET_ARCHITECTURE` | Force the architecture dispatch key (ASCII, case-insensitive); wins over `general.architecture` and `RBITNET_MODEL_FAMILY`. Use to experiment or to force `llama` when a file advertises an unsupported arch (e.g. MoE). |
| `RBITNET_MODEL_FAMILY` | `llama`, `bitnet`, or `auto` (default): with `auto`, the key is `bitnet` when the GGUF says so, otherwise `general.architecture` (lowercased), else `llama` if that metadata is missing (legacy files). |
| `RBITNET_CHAT_FORMAT` | Built-in chat prompt format: `raw` (default), `llama3`, or `chatml`. |
| `RBITNET_CHAT_TEMPLATE` | Simple custom template string. Supports `{messages}`, `{prompt}`, `{system}`, `{user}`, `{assistant}` (also `{{...}}` forms). Overrides `RBITNET_CHAT_FORMAT`. |
| `RBITNET_STUB` | `1` / `true` / `yes` — stub responses (overrides real inference when set). |
| `RBITNET_TOY` | `1` — toy LM instead of GGUF. |
| `RBITNET_TOY_SEED` | Integer seed for the toy LM (default `42`). |
| `RBITNET_TEST_GGUF` | Used only by the `optional_gguf_from_env_smoke` test in `bitnet-core`. |
| `RBITNET_CACHE_OUTPUT_F32` | **Default on** for the experimental **`qwen35moe`** CUDA path: load `output.weight` / `lm_head` as F32 once and reuse for logits (much lower per-token dequant overhead). Costs extra **host RAM** on the order of `vocab × hidden × 4` bytes. Set to `0`, `false`, or `no` to keep weights quantized in memory during logits (slower logits, less RAM). |
| `RBITNET_BACKEND=hybrid` | CPU/GPU hybrid mode for Llama-lineage GGUFs. The CPU keeps orchestration and fallback while selected f32-dequantized layer weights are uploaded once to CUDA device buffers. |
| `RBITNET_QUANT_KERNEL` | Quantized matvec backend: `auto` (CPU parallel), `scalar`, or `cuda` to call optional `rbitnet_cuda_quant*` native symbols for `Q4_K`, `Q6_K`, `Q4_0`, and `Q8_0` with CPU fallback. |
| `RBITNET_QUANT_PAR_MIN_ROWS` | Output-row threshold for CPU parallel quant matvec (default `256`). |
| `RBITNET_HYBRID_POLICY` | `layers`, `hotcold`, or `auto`. `layers` preserves explicit/early-layer behavior; `hotcold` selects deeper decode-hot layers first within the VRAM budget; `auto` honors `RBITNET_HYBRID_LAYERS` when set, otherwise hot/cold selection. |
| `RBITNET_HYBRID_LAYERS` | Optional comma/range list of Llama layers to offload (`0`, `0-3`, `0,2,4`). If unset, early layers are selected within `RBITNET_HYBRID_MAX_VRAM_MB`. |
| `RBITNET_HYBRID_MAX_VRAM_MB` | Soft upload budget for automatic hybrid layer selection (default `512`). |
| `RBITNET_HYBRID_MIN_ROWS` | Minimum matrix output rows for hybrid upload (default `512`). Smaller matrices stay on CPU. |
| `RBITNET_HYBRID_OUTPUT` | Set to `1` to try offloading the Llama output head. This can consume substantial VRAM. |
| `RBITNET_KV_BACKEND` | KV backend hint: `cpu` by default; `gpu`/`cuda` marks paged KV as GPU-planned while preserving CPU fallback. |
| `RBITNET_KV_QUANT` | Paged Llama KV format: `off`/`f32`, `q8`, or `q4`. Quantized pages store K/V compactly and decode heads on demand for attention; keep `off` for maximum numerical safety. |
| `RBITNET_PREFILL_CHUNK_TOKENS` | Positive integer (default `128`). Llama and Qwen3 runtimes now call explicit `prefill_chunk(tokens)` and `decode_one(token)` APIs; each chunk still runs one forward step per token until batched multi-position kernels are added. |
| `RBITNET_STRUCTURED_OUTPUT` | `off`, `json`, or `tool`. JSON/tool enables a lightweight FSM mask before sampling for byte/ASCII-compatible tokenizers. |
| `RBITNET_PREFIX_CACHE` | `1` / `true` / `yes` enables an **exact-match cache of prior completions**: same prompt string, `max_tokens`, and `temperature`. This is **not** Hugging Face / OpenAI–style **prompt caching** that reuses **KV blocks** for a shared prefix across requests. See [LIMITATIONS.md](LIMITATIONS.md). |
| `RBITNET_PREFIX_CACHE_MAX_ENTRIES` | LRU-ish cap for prefix-cache entries (default `64`). |
| `RBITNET_LLAMA_WEIGHT_MODE` | `auto` | **`dense`** (full `f32` weight materialization, high RAM), **`mmap_quant`** (quantized weights in mmap + row GEMV; fails if a matrix uses an unsupported GGML type), **`auto`** (mmap when all Llama weight tensors are supported, else `dense`). |
| `RBITNET_LLAMA_PAGED_KV` | `0` | **`1` / `true` / `yes`** — use **paged KV slabs** for Llama-lineage GGUF (Inference stack v2 phase A). Uses `RBITNET_PAGED_KV_PAGE_TOKENS` and `RBITNET_PAGED_KV_MAX_PAGES` (see [`paged_kv.rs`](../crates/bitnet-core/src/paged_kv.rs)). Default remains dense buffers. |
| `RBITNET_PAGED_KV_PAGE_TOKENS`, `RBITNET_PAGED_KV_MAX_PAGES` | `16`, `4096` | Page size and per-layer physical page cap when **`RBITNET_LLAMA_PAGED_KV`** is enabled; also tune experimental Qwen35 attention metadata (see code). |
| `RBITNET_CONTINUOUS_BATCHING`, `RBITNET_SPECULATIVE`, `RBITNET_SPEC_DRAFT_RATIO_NUM`, `RBITNET_SPEC_DRAFT_RATIO_DEN` | Scheduler flags (see `crates/bitnet-core/src/scheduler.rs`). Batching is still sequential per request; speculative mode now supports `RBITNET_DRAFT_PATH=ngram|toy` and records draft/verify counters. |
| `RBITNET_DRAFT_PATH`, `RBITNET_DRAFT_MODEL` | Draft source for speculative decoding. `ngram` and `toy` are local lightweight drafts; `RBITNET_DRAFT_MODEL` accepts a GGUF path and currently falls back to n-gram while the separate draft executor is completed. |

**Programmatic phase stats:** `Engine::complete_detailed` returns `InferenceOutput.stats` with `encode_ms`, `prefill_ms`, `decode_ms`, `ttft_ms`, `itl_us`, `tpot_us`, and tokenizer token counts when a real `ModelExecutor` is loaded.

### Server limits and security (`rbitnet-server`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `RBITNET_MAX_BODY_BYTES` | `1048576` | Max JSON body size for `/v1/chat/completions` (HTTP 413 if exceeded). |
| `RBITNET_MAX_PROMPT_CHARS` | `256000` | Max UTF-8 characters in the built prompt (HTTP 400). |
| `RBITNET_MAX_PROMPT_TOKENS` | unset | Optional cap on tokenizer-encoded prompt length (HTTP 400 when exceeded). Checked after character limit; stub/toy modes use a rough `len/4` estimate. |
| `RBITNET_MAX_TOKENS_CAP` | `8192` | Hard ceiling on client `max_tokens` (HTTP 400 if higher). |
| `RBITNET_MAX_CONCURRENT` | `4` | Simultaneous blocking inference tasks (HTTP 503 when saturated). |
| `RBITNET_INFERENCE_TIMEOUT_SECS` | `600` | Wall-clock limit per completion (HTTP 504). |
| `RBITNET_API_KEY` | unset | If set, `Authorization: Bearer <key>` or `X-API-Key: <key>` is required on `/`, `/v1/models`, and `/v1/chat/completions` (not on `/health`, `/ready`, `/metrics`). |
| `RBITNET_CORS_ANY` | unset | Set to `1` only for dev to allow any CORS origin. |

Binding to `0.0.0.0` or `[::]` logs a warning: use a reverse proxy and TLS for untrusted networks ([DEPLOYMENT.md](DEPLOYMENT.md)).

## GGUF general.architecture dispatch

Rbitnet resolves a **normalized architecture key** from the environment and from `general.architecture` in the GGUF, then selects an executor builder (Atlas-style factory in [`crates/bitnet-core/src/loaders/`](../crates/bitnet-core/src/loaders/)).

| Example `general.architecture` | Loader / outcome |
|----------------------------------|------------------|
| `llama`, `mistral`, (typical Llama-shaped family) | Llama GGUF stack (`LlamaExecutor`) |
| `bitnet` | Error: BitNet weights forward not implemented yet (same message as before). |
| `qwen35moe` | **Experimental** native text path when `RBITNET_BACKEND=cuda` (NVIDIA CUDA runtime + cuBLAS must load). CPU dispatch returns an explicit error. Expect high VRAM usage and slow first-token latency: weights are dequantized on the host per matmul; logits use `cuBLAS` GEMV in chunks by default after a one-shot F32 **`output.weight`** cache (`RBITNET_CACHE_OUTPUT_F32`, on unless disabled). Pair the GGUF with a Hugging Face tokenizer (`tokenizer.json` next to the checkpoint or `RBITNET_TOKENIZER`). Vision and MTP are out of scope for this first path. To force the Llama loader instead, set `RBITNET_ARCHITECTURE=llama` (likely to fail on tensor layout). |

Extend the match table in [`registry.rs`](../crates/bitnet-core/src/loaders/registry.rs) when adding a new family.

## Inspect a GGUF (no server)

```bash
cargo run -p bitnet-core --example inspect_gguf -- /path/to/model.gguf
```

## Programmatic use (Rust)

Depend on `bitnet-core` and build an [`Engine`](../crates/bitnet-core/src/inference.rs) from the environment or from a path:

```rust
use bitnet_core::Engine;

let engine = Engine::load_path(std::path::Path::new("/path/to/model.gguf"))?;
let text = engine.complete("Hello", 64, 0.8)?;
```

The first call that needs generation will load the tokenizer (same rules as above). Ensure `RBITNET_TOKENIZER` or a tokenizer beside the GGUF is available at runtime.

## Limitations and troubleshooting

- **Quantization types:** Some rare GGML types may not be implemented yet; loading can fail with `UnsupportedGgmlType`. Prefer widely used types (for example Q4_K, Q8_0, or F16 layers) or re-export with a supported layout.
- **Tensor names:** The loader expects **llama.cpp-style** names, with a few **aliases** (for example `lm_head.weight` vs `output.weight`). See [BITNET_SPEC.md](BITNET_SPEC.md). Odd exports may still need renaming or loader tweaks.
- **Context length:** Generation is bounded by `llama.context_length` (capped internally for safety). Very long prompts can hit limits or run slowly on CPU.
- **Performance:** Pure Rust + dequantized matmuls is correct but not as fast as highly optimized C++/GPU stacks; for production throughput, profile on your hardware.

For golden tests and regression expectations, see [GOLDEN_TESTS.md](GOLDEN_TESTS.md). For HF → GGUF conversion workflows, see [MODEL_TESTING.md](MODEL_TESTING.md). For **what is implemented vs still planned** (production roadmap), see [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md).
