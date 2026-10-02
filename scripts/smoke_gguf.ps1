# Verify a running server against a local GGUF and save its actual response.
# Run after starting recipes/exported-llama.recipe.json (see docs/UNSLOTH_TO_RBITNET.md).
param(
    [Parameter(Mandatory = $true)] [string] $ModelPath,
    [Parameter(Mandatory = $true)] [string] $ExpectedSha256,
    [string] $BaseUrl = "http://127.0.0.1:18079",
    [string] $ApiKey = $env:RBITNET_API_KEY,
    [string] $Prompt = "What is the capital of France? Answer in one word.",
    [ValidateRange(1, 128)] [int] $MaxTokens = 16,
    [string] $OutputPath = "target/gguf-smoke.json"
)

$ErrorActionPreference = "Stop"
$modelFile = (Resolve-Path -LiteralPath $ModelPath).Path
if ($ExpectedSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw "ExpectedSha256 must be the 64-character SHA-256 from your export record or publisher."
}
$actualSha = (Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "GGUF SHA-256 mismatch. Refusing the smoke request."
}

$base = $BaseUrl.TrimEnd('/')
$headers = @{}
if ($ApiKey) { $headers.Authorization = "Bearer $ApiKey" }
$null = Invoke-RestMethod -Uri "$base/ready" -Headers $headers -TimeoutSec 30
$models = Invoke-RestMethod -Uri "$base/v1/models" -Headers $headers -TimeoutSec 30
$loaded = @($models.data | Where-Object { $_.loaded -and $_.ready -and $_.metadata.model_path })
if ($loaded.Count -ne 1) {
    throw "Expected exactly one ready GGUF model; stub/toy servers cannot validate an export."
}
$entry = $loaded[0]
$servedPath = (Resolve-Path -LiteralPath $entry.metadata.model_path).Path
if ($servedPath -ne $modelFile) {
    throw "Server is serving a different GGUF: $servedPath"
}
if ($entry.id -in @('rbitnet-stub', 'rbitnet-toy') -or
    $entry.metadata.architecture -in @('stub', 'toy')) {
    throw "A stub/toy response cannot validate an export."
}

$request = @{
    model = $entry.id
    messages = @(@{ role = "user"; content = $Prompt })
    max_tokens = $MaxTokens
    temperature = 0
    stream = $false
}
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$reply = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
    -Headers $headers -ContentType "application/json; charset=utf-8" `
    -Body ($request | ConvertTo-Json -Depth 6) -TimeoutSec 300
$watch.Stop()
if ($reply.model -in @('rbitnet-stub', 'rbitnet-toy') -or
    [string]::IsNullOrWhiteSpace([string] $reply.choices[0].message.content) -or
    $reply.usage.prompt_tokens -le 0 -or $reply.usage.completion_tokens -le 0) {
    throw "The GGUF returned no generated text/tokens."
}

$record = [ordered] @{
    kind = "rbitnet-gguf-http-smoke-v1"
    captured_at_utc = [DateTime]::UtcNow.ToString('o')
    model_path = $modelFile
    sha256 = $actualSha
    metadata = $entry.metadata
    request = $request
    response = $reply
    elapsed_ms = $watch.ElapsedMilliseconds
    validation = "GGUF path, SHA-256 and non-empty HTTP generation checked; quality and golden parity require review."
}
$outputFile = [System.IO.Path]::GetFullPath($OutputPath)
$null = New-Item -ItemType Directory -Path (Split-Path -Parent $outputFile) -Force
$record | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $outputFile -Encoding utf8
Write-Host "GGUF loading/API smoke passed. Review the generated text. Evidence: $outputFile"
Write-Output $reply.choices[0].message.content
