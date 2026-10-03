param([switch]$ShortOnly)
$ErrorActionPreference='Stop'
$taskRoot='E:\devs\Rbitnet\target\gpt-full\comparison'
New-Item -ItemType Directory -Force -Path $taskRoot | Out-Null
$env:OLLAMA_HOST='127.0.0.1:11439'
$env:OLLAMA_MODELS='D:\Rbitnet-benchmark-models\ollama'
$env:OLLAMA_NUM_PARALLEL='1'
$env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_NOPRUNE='1'
$daemon=Start-Process -FilePath 'C:\Users\azero\AppData\Local\Programs\Ollama\ollama.exe' -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskRoot\ollama.out.log" -RedirectStandardError "$taskRoot\ollama.err.log"
try {
 $ready=$false
 for ($attempt=0;$attempt -lt 60;$attempt++) {
  if ($daemon.HasExited) {throw 'Isolated daemon exited'}
  try {$null=Invoke-RestMethod 'http://127.0.0.1:11439/api/version';$ready=$true;break} catch {Start-Sleep -Milliseconds 300}
 }
 if (!$ready) {throw 'Ollama did not become ready'}
 $config=Get-Content -LiteralPath "$taskRoot\manifest-prelaunch.json" -Raw | ConvertFrom-Json
 $config.environment.rbitnet_exe_sha256=(Get-FileHash -LiteralPath $config.rbitnet -Algorithm SHA256).Hash.ToLowerInvariant()
 $config.environment.cuda_quant_library_sha256=(Get-FileHash -LiteralPath $config.cuda_quant_library -Algorithm SHA256).Hash.ToLowerInvariant()
 $config | Add-Member -NotePropertyName ollama_pid -NotePropertyValue $daemon.Id -Force
 $config | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath "$taskRoot\manifest.json" -Encoding utf8NoBOM
 & C:/Users/azero/anaconda3/python.exe scripts/benchmark_engines.py --manifest "$taskRoot\manifest.json" --output "$taskRoot\short\results.json" --models gpt-oss-20b --backends gpu --repeats 3 --tokens 32 *> "$taskRoot\short.log"
 if ($LASTEXITCODE -ne 0) {throw 'Short comparison failed'}
 if (!$ShortOnly) {
  & C:/Users/azero/anaconda3/python.exe scripts/benchmark_engines.py --manifest "$taskRoot\manifest.json" --output "$taskRoot\long\results.json" --models gpt-oss-20b --backends gpu --repeats 3 --tokens 128 --long-notes 72 *> "$taskRoot\long.log"
  if ($LASTEXITCODE -ne 0) {throw 'Long comparison failed'}
 }
} finally {
 if (!$daemon.HasExited) {Stop-Process -Id $daemon.Id -ErrorAction SilentlyContinue}
}
