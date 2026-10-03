# Golden vectors (Llama / Qwen3 parity)

This directory holds **optional** regression artefacts for comparing Rbitnet’s Llama and dense
Qwen3 forwards against a reference (typically **llama.cpp** `llama-cli`), plus one
**checked-in synthetic** Qwen3 golden used by default CI.

## Layout

- `*.golden.json` — machine-readable spec (checked in or generated locally).
- `*.example.json` / `*.example.golden.json` — hand-editable templates only (no committed secrets or large logits).
- `qwen3-dense-synthetic.golden.json` — **default CI** expected greedy id for the in-test tiny Qwen3 fixture.

## JSON schema (`rbitnet-golden-v1`)

```json
{
  "format": "rbitnet-golden-v1",
  "architecture": "llama",
  "prompt": "Hello",
  "expected_greedy_first_token": 0
}
```

- **`architecture`** — optional; `llama` (default), `qwen3`, or `mixtral`. Overridden by `RBITNET_ARCHITECTURE` when set.
- **`prompt`** — plain text passed to the tokenizer with the same special-token policy as Rbitnet’s default (`RBITNET_LLAMA_ENCODE_ADD_SPECIAL` unset ⇒ specials added for HF tokenizer).
- **`expected_greedy_first_token`** — vocabulary id of the argmax after full prompt prefill (`temperature = 0`).

Record how you produced the id in `docs/GOLDEN_TESTS.md` (llama.cpp build id, flags, platform).

## Running the optional test

```bash
export RBITNET_GOLDEN_JSON=tests/data/golden/my-model.golden.json
export RBITNET_TEST_GGUF=/path/model.gguf
export RBITNET_TOKENIZER=/path/tokenizer.json
cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
```

PowerShell:

```powershell
$env:RBITNET_GOLDEN_JSON="tests\data\golden\my-model.golden.json"
$env:RBITNET_TEST_GGUF="C:\path\model.gguf"
$env:RBITNET_TOKENIZER="C:\path\tokenizer.json"
cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
```

Default CI Qwen3 synthetic golden (no env vars):

```bash
cargo test -p bitnet-core --test qwen3_dense_golden_ci -- --nocapture
```

See also `scripts/run-golden-test.sh` and `scripts/run-golden-test.ps1`.

## Llama 3.2 multi-token reference

`llama32-1b-instruct-q4-k-m.sequence.json` uses `rbitnet-llama-sequence-v1`: five cases
with `name`, formatted `prompt`, `prompt_ids`, reference `greedy_ids` (including EOT) and
decoded `text`. These are outputs from llama.cpp b11351 on the pinned GGUF, not example IDs.
The file records the GGUF and tokenizer SHA-256 values.

`optional_llama_sequence` checks every greedy token, text, stop behavior and BOS count.
See [the run instructions](../../../docs/GOLDEN_TESTS.md) for the three required absolute
file paths. Without those environment variables the real-model test skips; the default CI
kernel and tokenizer regressions still run.
