# Wait for scripts/kaggle/chain.py to report RUN COMPLETE (checkpoint fully copied), then run every benchmark.
#
#   powershell -ExecutionPolicy Bypass -File scripts\after_chain.ps1 -Run leo-1.7b-v3 -Data data/processed-v3
param(
    [string]$Run = "leo-1.7b-v3",
    [string]$Data = "data/processed-v3"
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$chainLog = "results\logs\kaggle_chain_$Run.log"
while ($true) {
    $text = if (Test-Path $chainLog) { Get-Content $chainLog -Raw -Encoding utf8 } else { "" }
    if ($text -match "RUN COMPLETE") { break }
    if ($text -match "stopping the chain|max sessions reached") { Write-Output "chain failed; not benchmarking"; exit 1 }
    Start-Sleep -Seconds 120
}
Write-Output "chain complete at $(Get-Date -Format 'HH:mm:ss'); benchmarking checkpoints\$Run"
$env:HF_HUB_OFFLINE = "1"
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bench_all.ps1 -Run "checkpoints\$Run" -Name $Run -Data $Data -Multilingual -Browser -Dtype bf16
exit $LASTEXITCODE
