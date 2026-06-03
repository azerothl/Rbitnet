# Golden tests (Llama numerical regression)

## Goal

Prove that the Llama-compatible forward in Rbitnet matches a **reproducible external reference**
(typically **llama.cpp** built from a known commit) on at least one small open GGUF, so text
quality regressions are caught in CI or in a manual pre-release step.

## Reference model (recommended)

| Role | Bundle id | GGUF | Tokenizer source |
|------|-----------|------|-------------------|
| Primary Llama smoke | `tinyllama-1.1b-chat-q4-k-m` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`tokenizer.json`) |

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
  "prompt": "Hello",
  "expected_greedy_first_token": 12345
}
```

Replace `12345` with the id from llama.cpp / Python.

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

- **Greedy token id** is exact: no float tolerance.
- For **logit vectors** (optional future extension), start with `max(abs diff)) < 5e-2` on CPU
  after prefill for the last prompt position, then tighten once kernels are stable.

## Current state in Rbitnet

- Kernel-level matvec tests run in default CI (`tests/golden_kernels.rs`).
- Llama **end-to-end** golden is **optional** (env vars above) so CI stays lightweight.
- GitHub Actions: workflow **Golden (optional)** — `.github/workflows/golden-optional.yml` — manual
  `workflow_dispatch` to run the same test on a runner where you attach a cached GGUF + JSON.

## Related docs

- `tests/data/golden/README.md` — schema and file layout.
- `docs/PLAN_PRODUCTION.md` — quality gate for releases.
- `docs/RELEASE_PACKAGING.md` — pre-publish checklist including golden when claiming parity.
