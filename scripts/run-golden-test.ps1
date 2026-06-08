# Run optional Llama golden test (requires GGUF + tokenizer + golden JSON on disk).
# Example:
#   $env:RBITNET_GOLDEN_JSON = "tests\data\golden\my.golden.json"
#   $env:RBITNET_TEST_GGUF = "C:\path\model.gguf"
#   $env:RBITNET_TOKENIZER = "C:\path\tokenizer.json"
#   .\scripts\run-golden-test.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
cargo test -p bitnet-core optional_golden_greedy_first_token_matches -- --nocapture
