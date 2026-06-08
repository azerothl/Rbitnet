$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:MODEL_ID = if ($env:MODEL_ID) { $env:MODEL_ID } else { "TinyLlama/TinyLlama-1.1B-Chat-v1.0" }
$outDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { Join-Path $Root "out/sft-example" }
$maxSteps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "20" }
python (Join-Path $Root "training/recipes/sft_lora.py") `
  --model-id $env:MODEL_ID `
  --dataset-jsonl (Join-Path $Root "training/data/example.jsonl") `
  --output-dir $outDir `
  --max-steps $maxSteps `
  --bf16 `
  @args
