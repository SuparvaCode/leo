# Browser benchmark runs for v5 (one at a time: they share the automation Chrome port).
#   1. Wikipedia recheck, v3 and v5-1.7B, text helper in bf16 (the nf4 helper returned no valid value today)
#   2. Full suite, Jev vs leo-4b-v5 (served on Modal, account suparvacode), arms alternating, bf16 helper
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HUB_OFFLINE = "1"; $env:HF_HOME = "$root\hf_cache"; $env:PYTHONPATH = "$root"; $env:PYTHONIOENCODING = "utf-8"
$env:LEO_TEXT_HELPER_DTYPE = "bf16"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = "$root\.venv\Scripts\python.exe"
foreach ($r in "leo-1.7b-v3", "leo-1.7b-v5") {
    & $py -m leo.bench.browser --backends leo --leo "checkpoints\$r" --leo-dtype bf16 --name wiki-bf16-helper --tasks wikipedia-godel --repeats 3
}
$env:LEO_API_KEY = (Get-Content D:\Leo-API\api_key.txt -Raw).Trim()
& $py -m leo.bench.browser --backends jev,leo --leo checkpoints\leo-4b-v5 --leo-url https://suparvacode--leo-bench-serve.modal.run --name final-leo-4b-v5 --repeats 3
Write-Output "browser runs finished"
