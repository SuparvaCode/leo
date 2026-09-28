"""Kaggle capability check for Leo training: GPU, precision support, memory, disk, internet, rough speed."""
import os
import shutil
import subprocess
import time
import urllib.request

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu{i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm {p.major}.{p.minor}, bf16={torch.cuda.is_bf16_supported()}")

print(f"cpus {os.cpu_count()}, disk free /kaggle/working {shutil.disk_usage('/kaggle/working').free / 2**30:.0f} GiB")
with open("/proc/meminfo") as fh:
    print(fh.readline().strip())

for url in ("https://huggingface.co/api/models/Qwen/Qwen3-1.7B-Base", "https://pypi.org/simple/peft/"):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            print("internet ok:", url, r.status)
    except Exception as e:
        print("internet FAILED:", url, type(e).__name__, e)

# Dense matmul throughput, the dominant cost of LoRA training on a 0.6B-4B base.
n = 8192
for dtype in (torch.float16, torch.bfloat16):
    try:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{dtype}: {2 * n**3 * 20 / dt / 1e12:.1f} TFLOP/s")
    except Exception as e:
        print(f"{dtype}: failed ({type(e).__name__}: {e})")

for pkg in ("transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
    try:
        mod = __import__(pkg)
        print(pkg, mod.__version__)
    except Exception:
        print(pkg, "not installed")
