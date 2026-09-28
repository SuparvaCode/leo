# Build data, train, select and benchmark one Leo run. Every step is resumable or idempotent, so after a
# crash or power cut just run the same command again.
#
#   powershell -ExecutionPolicy Bypass -File scripts\pipeline.ps1 -Recipe v1 -Name leo-0.6b-v1
param(
    [string]$Recipe = "v1",
    [string]$Name = "leo-0.6b-v1",
    [string]$Base = "Qwen/Qwen3-0.6B-Base",
    [double]$Epochs = 2,
    [int]$DataSeed = 13,
    [int]$TrainSeed = 0,
    [int]$TrainStateTokens = 384,
    [int]$MaxRowTokens = 1536,
    [int]$MaxBatchTokens = 6144,
    [int]$Accum = 2,
    [switch]$SkipProbes = $true
)

# No $ErrorActionPreference = "Stop": Windows PowerShell 5.1 can turn a native command's stderr (progress
# bars, warnings) into terminating errors. Failures are detected from exit codes instead.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HOME = Join-Path $root "hf_cache"
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$py = Join-Path $root ".venv\Scripts\python.exe"
$data = if ($DataSeed -eq 13) { "data/processed-$Recipe" } else { "data/processed-$Recipe-s$DataSeed" }
$run = "checkpoints/$Name"

function Invoke-Step([string[]]$StepArgs) {
    Write-Output ("=== python " + ($StepArgs -join " ") + "   [" + (Get-Date -Format "HH:mm:ss") + "]")
    & $py @StepArgs
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE): $($StepArgs -join ' ')" }
}

# 1. Data: rebuild unless a complete build for this recipe is already there.
$manifest = Join-Path $data "manifest.json"
$haveData = $false
if (Test-Path $manifest) {
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    $haveData = ($m.recipe -eq $Recipe) -and ($m.seed -eq $DataSeed)
}
if ($haveData) { Write-Output "=== data: $data already built for recipe $Recipe, seed $DataSeed" }
else { Invoke-Step @("-m", "leo.data.build", "--out", $data, "--recipe", $Recipe, "--seed", "$DataSeed") }

# 2. Training (resumes from resume.pt if a previous attempt was interrupted).
$cfg = Join-Path $run "best\leo_config.json"
$trained = (Test-Path $cfg) -and ((Get-Content $cfg -Raw | ConvertFrom-Json).completed)
if ($trained) { Write-Output "=== training: $run already complete" }
else {
    Invoke-Step @("-m", "leo.train", "--data", $data, "--base", $Base, "--out", $run, "--name", $Name,
                  "--epochs", "$Epochs", "--seed", "$TrainSeed", "--eval_every", "200", "--save_every", "50",
                  "--log_every", "25", "--train_state_tokens", "$TrainStateTokens", "--max_row_tokens", "$MaxRowTokens",
                  "--max_batch_tokens", "$MaxBatchTokens", "--accum", "$Accum", "--resume")
}

# 3. Selection on dev and benchmarks.
$benchArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts\bench_all.ps1", "-Run", $run, "-Name", $Name, "-Data", $data)
if ($SkipProbes) { $benchArgs += "-SkipProbes" }
& powershell @benchArgs
if ($LASTEXITCODE -ne 0) { throw "benchmarks failed (exit $LASTEXITCODE)" }
Write-Output "=== pipeline finished for $Name   [$(Get-Date -Format 'HH:mm:ss')]"
