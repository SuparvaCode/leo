# Select a run's checkpoint on dev data (leo.select), then run every benchmark on it.
# Safe to rerun: each step overwrites its own result files.
#
#   powershell -ExecutionPolicy Bypass -File scripts\bench_all.ps1 -Run checkpoints\leo-0.6b-v0 -Name leo-0.6b-v0
#   add -WaitForTraining to block until the training run in -Run has finished and calibrated.
param(
    [string]$Run = "checkpoints\leo-0.6b-v0",
    [string]$Name = "leo-0.6b-v0",
    [string]$Data = "data/processed",
    [switch]$WaitForTraining,
    [switch]$SkipProbes,
    [switch]$Multilingual,
    [switch]$Browser,
    [string]$BrowserName = "",
    [string]$Dtype = "auto",
    [switch]$AllowCpu
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = Join-Path $root "hf_cache"
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"
$env:USE_TF = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
# A GPU hidden in the parent shell (e.g. for CPU-only tests) is inherited by child processes. Unhide it.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$py = Join-Path $root ".venv\Scripts\python.exe"

$cuda = & $py -c "import torch; print(torch.cuda.is_available())"
Write-Output "CUDA available: $cuda"
if ($cuda -ne "True" -and -not $AllowCpu) {
    Write-Output "No GPU visible; the benchmarks would take hours on the CPU. Pass -AllowCpu to run anyway."
    exit 1
}

if ($WaitForTraining) {
    $cfg = Join-Path $Run "best\leo_config.json"
    while ($true) {
        if (Test-Path $cfg) {
            $done = (Get-Content $cfg -Raw | ConvertFrom-Json).completed
            if ($done) { break }
        }
        Start-Sleep -Seconds 30
    }
    Write-Output "training complete: $cfg"
}

$steps = @(
    @("-m", "leo.select", "--run", $Run, "--data", $Data),
    @("-m", "leo.bench.heldout", "--backend", "leo", "--model", $Run, "--name", $Name, "--dtype", $Dtype),
    @("-m", "leo.bench.heldout", "--report"),
    @("-m", "leo.bench.jevbench", "--model", $Run, "--name", $Name, "--dtype", $Dtype)
)
# Probes include a live Jev column: order/parity answers come from the on-disk cache, latency is re-measured (~140 calls).
$probeDtypes = if ($Dtype -eq "auto") { "bf16" } else { $Dtype }
if (-not $SkipProbes) { $steps += , @("-m", "leo.bench.probes", "--leo", $Run, "--name", $Name, "--leo_dtypes", $probeDtypes, "--order_views", "1,2") }
if ($Multilingual) { $steps += , @("-m", "leo.bench.multilingual", "--backend", "leo", "--model", $Run, "--name", $Name, "--dtype", $Dtype) }
if ($Browser) {
    # Both arms again, alternating within each repeat, so live sites are compared at the same time of day.
    $bn = if ($BrowserName) { $BrowserName } else { "final-$Name" }
    $bd = if ($Dtype -eq "auto") { "bf16" } else { $Dtype }
    $steps += , @("-m", "leo.bench.browser", "--backends", "jev,leo", "--leo", $Run, "--leo-dtype", $bd, "--name", $bn, "--repeats", "3")
}
foreach ($s in $steps) {
    Write-Output ("=== python " + ($s -join " "))
    & $py @s
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED (exit $LASTEXITCODE): $($s -join ' ')"; exit $LASTEXITCODE }
}
Write-Output "all benchmarks finished"
