# Browser suite for leo-4b-v5.1 (served on Modal, account suparvacode), Leo only; same settings as final-leo-4b-v5.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HUB_OFFLINE = "1"; $env:HF_HOME = "$root\hf_cache"; $env:PYTHONPATH = "$root"; $env:PYTHONIOENCODING = "utf-8"
$env:LEO_TEXT_HELPER_DTYPE = "bf16"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:LEO_API_KEY = (Get-Content D:\Leo-API\api_key.txt -Raw).Trim()
& "$root\.venv\Scripts\python.exe" -m leo.bench.browser --backends leo --leo checkpoints\leo-4b-v5.1 --leo-url https://suparvacode--leo-bench-serve.modal.run --name final-leo-4b-v5.1 --repeats 3
Write-Output "browser v5.1 finished"
