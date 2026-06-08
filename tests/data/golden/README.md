# Golden vectors (Llama parity)

This directory holds **optional** regression artefacts for comparing Rbitnet’s Llama forward
against a reference build (typically **llama.cpp** `llama-cli`).

## Layout

- `*.golden.json` — machine-readable spec (checked in or generated locally; **not** required for default CI).
- `*.example.json` — hand-editable templates only (no committed secrets or large logits).

## JSON schema (`rbitnet-golden-v1`)

```json
{
  "format": "rbitnet-golden-v1",
  "prompt": "Hello",
  "expected_greedy_first_token": 0
}
```

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

See also `scripts/run-golden-test.sh` and `scripts/run-golden-test.ps1`.
