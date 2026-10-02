# Golden tests (Llama / Qwen3 numerical regression)

## Goal

Prove that the Llama-compatible and dense **Qwen3** forwards in Rbitnet match a **reproducible
reference** on at least one small open GGUF (or a synthetic fixture in CI), so text quality
regressions are caught in CI or in a manual pre-release step.

## Reference model (recommended)

| Role | Bundle id | GGUF | Tokenizer source |
|------|-----------|------|-------------------|
| Primary Llama smoke | `tinyllama-1.1b-chat-q4-k-m` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`tokenizer.json`) |
| Qwen3 CI synthetic | (in-test fixture) | built by `qwen3_dense_golden_ci` | WordLevel `tokenizer.json` in-test |
| Mixtral MoE CI synthetic | (in-test fixture) | built by `mixtral::ci_fixture` / `mixtral_moe_golden_ci` | WordLevel `tokenizer.json` in-test |
| Qwen3 Hub (optional) | small dense Q4_K_M | e.g. Qwen3-0.6B/1.7B/4B | matching HF `tokenizer.json` |

Use the **same** GGUF file, tokenizer, and prompt string for both the reference export and Rbitnet.

## Exporting the reference greedy first token (llama.cpp)

1. Build [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and use `llama-cli` from that tree.
2. Run a **single-token greedy** completion after your fixed prompt (adjust paths):

```bash
./llama-cli -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --no-warmup \
  -p "Hello" \
  -n 1 \
  --temp 0 \
  --top-k 1 \
  --no-display-prompt
```

3. Note the **first generated token id** printed by your build (some `llama-cli` versions print
   token ids with `-v` / `--verbose` / log lines; if your build only prints text, use
   `--logit-bias` disabled and read the tokenizer id from debug output, or use the small Python
   snippet below).

### Python fallback (llama-cpp-python)

If you already have `llama_cpp` installed with the **same** GGUF:

```python
from llama_cpp import Llama
m = Llama("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", logits_all=True, verbose=False)
p = m.tokenize(b"Hello", add_bos=True)
out = m.create_completion(prompt="Hello", max_tokens=1, temperature=0.0, top_k=1)
# Inspect completion token id via model internals or echo `out` depending on binding version.
```

Record the **integer token id** you obtain.

## Golden file for Rbitnet

1. Create `tests/data/golden/<name>.golden.json` (or any path) with:

```json
{
  "format": "rbitnet-golden-v1",
  "architecture": "llama",
  "prompt": "Hello",
  "expected_greedy_first_token": 12345
}
```

Replace `12345` with the id from llama.cpp / Python. Optional `architecture` selects the
runtime (`llama` default, `qwen3`, or `mixtral`); `RBITNET_ARCHITECTURE` overrides when set.

2. Run the optional integration test:

```bash
export RBITNET_GOLDEN_JSON=tests/data/golden/<name>.golden.json
export RBITNET_TEST_GGUF=/abs/path/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
export RBITNET_TOKENIZER=/abs/path/tokenizer.json
cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
```

Or use `scripts/run-golden-test.sh` / `scripts/run-golden-test.ps1`.

## Tolerance and scope

- Optional **`RBITNET_BLAS=1`** (OpenBLAS) and **`RBITNET_LLAMA_MATMUL=ggml`** (experimental hook) should keep the same greedy first token on CPU for a given GGUF; if a golden run fails after enabling them, treat it as a regression in the fast path or document a deliberate numerical change.

- **`RBITNET_KV_QUANT=q8`:** not bit-exact vs F32 KV. Run goldens with `RBITNET_KV_QUANT=off` (default). Q8 quality gate is the unit suite in `kv_storage_paged` (INT8 round-trip + &lt;5% relative attention drift on toy tensors) plus optional live smoke via `scripts/bench_kv_q8.sh` — do **not** fail the greedy first-token golden on Q8 alone.

- **Greedy token id** is exact: no float tolerance.
- For **logit vectors** (optional future extension), start with `max(abs diff)) < 5e-2` on CPU
  after prefill for the last prompt position, then tighten once kernels are stable.

## Qwen3 dense golden (#25)

Dense **`general.architecture=qwen3`** is the first non-Llama family with an in-tree forward path.

### Default CI (synthetic)

`cargo test -p bitnet-core --test qwen3_dense_golden_ci` builds a tiny 1-layer F32 Qwen3 GGUF
plus a WordLevel tokenizer, runs `Qwen3Runtime::greedy_next_token_id_after_prompt`, and checks
`tests/data/golden/qwen3-dense-synthetic.golden.json`. This runs in the main CI job (no Hub
download).

### Optional Hub Qwen3 golden

1. Prefer a small dense Qwen3 quant (e.g. 0.6B / 1.7B / 4B Q4_K_M) + matching `tokenizer.json`.
2. Export greedy first-token id with llama.cpp the same way as Llama (section above).
3. Use format `rbitnet-golden-v1` with `"architecture": "qwen3"` (see
   `tests/data/golden/qwen3-dense.example.golden.json`) and run
   `optional_golden_greedy_first_token_matches` with `RBITNET_TEST_GGUF` /
   `RBITNET_TOKENIZER` / `RBITNET_GOLDEN_JSON`.
4. Default CI does **not** download Hub Qwen3 weights — keep Hub goldens optional like Llama.
   Workflow **Golden (optional)** accepts an optional `architecture` input.

### Mixtral MoE (#25)

`cargo test -p bitnet-core --test mixtral_moe_golden_ci` builds a tiny Mixtral MoE GGUF
(4 experts, top-2) and checks `tests/data/golden/mixtral-moe-synthetic.golden.json`.
HTTP e2e: `cargo test -p bitnet-server --test mixtral_moe_v1` hits `/v1/chat/completions`.

DeepSeek MLA / non-Mixtral MoE graphs remain refused or CUDA-experimental; see
[LIMITATIONS.md](LIMITATIONS.md).

## Current state in Rbitnet

- Kernel-level matvec tests run in default CI (`tests/golden_kernels.rs`).
- **Qwen3 dense synthetic** greedy golden runs in default CI (`qwen3_dense_golden_ci`).
- **Mixtral MoE synthetic** greedy golden + `/v1` e2e run in default CI.
- Llama / Hub Qwen3 **end-to-end** goldens are **optional** (env vars above) so CI stays lightweight.
- GitHub Actions: workflow **Golden (optional)** — `.github/workflows/golden-optional.yml` — manual
  `workflow_dispatch` to run the same test on a runner where you attach a cached GGUF + JSON.

## Llama 3.2 real-model sequence regression

`tests/data/golden/llama32-1b-instruct-q4-k-m.sequence.json` records five greedy sequences
from **llama.cpp b11351 on CPU**, including the final end-of-turn token. The prompts cover
a fact, arithmetic, translation, an explanation and conversation history. The fixture pins
the GGUF and tokenizer SHA-256 values; use those exact files.

The comparison sends **token ID arrays** to llama.cpp `/completion`. Sending an already
formatted BOS-prefixed string to that endpoint can prepend another BOS. Check `/tokenize`
with `add_special: false` and pass its IDs to `/completion` to keep the inputs identical.
The reference uses one CPU slot, context 512 and F32 K/V caches (`-ngl 0 -c 512 -np 1
--cache-type-k f32 --cache-type-v f32 --no-warmup`), with `temperature: 0`, `top_k: 1`
and `n_probs: 5` in each completion request. Read generated IDs from
`completion_probabilities`, including the final EOT.

The optional integration test checks the tokenizer IDs, every greedy generated ID, the
decoded response, the prompt token count and stopping before EOT. It runs the Rust engine
directly; HTTP JSON and SSE were checked separately in the [validation record](validation/2026-10-03-llama32-inference-fix.json).

From the repository root in PowerShell, with the pinned model and tokenizer already present:

```powershell
$env:RBITNET_LLAMA_SEQUENCE_JSON = (Resolve-Path tests/data/golden/llama32-1b-instruct-q4-k-m.sequence.json).Path
$env:RBITNET_TEST_GGUF = (Resolve-Path models/exported-llama/model.gguf).Path
$env:RBITNET_TOKENIZER = (Resolve-Path models/exported-llama/tokenizer.json).Path
$env:RBITNET_BACKEND = "cpu"
$env:RBITNET_LLAMA_WEIGHT_MODE = "auto"
$env:RBITNET_KV_QUANT = "off"
cargo test --release --locked -p bitnet-core --test optional_llama_sequence -- --nocapture
```

Repeat with `RBITNET_LLAMA_WEIGHT_MODE=dense` when changing dense decoding. Without the
three file variables, this test skips model I/O. Default CI still exercises independent
Q6_K signed-scale values, RoPE factor validation and Llama 3 BOS/EOT handling without a download.

## Related docs

- `tests/data/golden/README.md` — schema and file layout.
- `docs/PLAN_PRODUCTION.md` — quality gate for releases.
- `docs/RELEASE_PACKAGING.md` — pre-publish checklist including golden when claiming parity.
