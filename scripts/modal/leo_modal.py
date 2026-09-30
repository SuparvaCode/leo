"""Train Leo on Modal. One Volume per workspace holds data, the Hugging Face cache and runs.

    set MODAL_PROFILE=<profile>
    modal volume create leo-v5
    modal volume put leo-v5 data/processed-v5 /data/processed-v5
    modal run scripts/modal/leo_modal.py::download --base Qwen/Qwen3-4B-Base          # CPU only
    modal run scripts/modal/leo_modal.py::speed --gpu H100 --tag h100-ckpt --extra "..."
    modal run --detach scripts/modal/leo_modal.py::full --gpu H100 --run leo-4b-v5 --budget-min 300 --extra "..."
    modal run scripts/modal/leo_modal.py::status --run leo-4b-v5

Safety: every GPU function has a hard timeout, so a run cannot bill past it. Training writes resume state
every --save_every steps and a background thread commits the Volume every 5 minutes, so a lost container
costs at most a few minutes (rerun `full` with the same arguments; leo.train resumes). No automatic retries.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")  # the container has no repo
app = modal.App("leo-v5")
vol = modal.Volume.from_name("leo-v5", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # the exact versions leo was developed and tested with locally
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers==5.17.0", "peft==0.21.0", "safetensors==0.8.0", "numpy==2.5.3", "pydantic==2.13.5",
                 "huggingface_hub==1.33.0", "tokenizers==0.23.2")
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/src", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(ROOT / "leo"), remote_path="/root/src/leo", ignore=["**/__pycache__/**"])
)
V = "/vol"


def _commit_loop(stop: threading.Event, every_s: int = 300) -> None:
    while not stop.wait(every_s):
        try:
            vol.commit()
        except Exception as e:  # a failed commit is retried at the next tick
            print(f"volume commit failed: {e}", flush=True)


def _train(run: str, data: str, base: str, extra: str, budget_min: float) -> int:
    vol.reload()
    out = f"{V}/runs/{run}"
    cmd = (f"python -m leo.train --data {V}/data/{data} --base {base} --out {out} --name {run} "
           f"--precision bf16 --resume --time_budget_min {budget_min} {extra}")
    print("+", cmd, flush=True)
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    try:
        r = subprocess.run(cmd, shell=True, cwd="/root/src")
    finally:
        stop.set()
        vol.commit()
    print("exit code", r.returncode, flush=True)
    return r.returncode


GPU_KW = dict(image=image, volumes={V: vol}, cpu=4, memory=32768)


@app.function(image=image, volumes={V: vol}, cpu=2, memory=8192, timeout=3600)
def download(base: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"])
    vol.commit()
    return path


@app.function(gpu="H100", timeout=1500, **GPU_KW)
def speed_h100(tag: str, data: str, base: str, extra: str) -> int:
    return _train(f"speed-{tag}", data, base, extra, 10)


@app.function(gpu="B200", timeout=1500, **GPU_KW)
def speed_b200(tag: str, data: str, base: str, extra: str) -> int:
    return _train(f"speed-{tag}", data, base, extra, 10)


@app.function(gpu="H100", timeout=int(5.5 * 3600), **GPU_KW)
def full_h100(run: str, data: str, base: str, extra: str, budget_min: float) -> int:
    return _train(run, data, base, extra, budget_min)


@app.function(gpu="B200", timeout=int(5.5 * 3600), **GPU_KW)
def full_b200(run: str, data: str, base: str, extra: str, budget_min: float) -> int:
    return _train(run, data, base, extra, budget_min)


@app.function(image=image, volumes={V: vol}, cpu=1, memory=2048, timeout=600)
def read_log(run: str) -> dict:
    vol.reload()
    d = Path(f"{V}/runs/{run}")
    log = d / "train_log.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    recs = [json.loads(x) for x in lines if x.strip()]
    files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else []
    return {"records": recs, "files": files}


@app.local_entrypoint()
def main(action: str = "status", gpu: str = "H100", run: str = "", tag: str = "", data: str = "processed-v5",
         base: str = "Qwen/Qwen3-4B-Base", extra: str = "", budget_min: float = 300.0) -> None:
    if action == "download":
        print(download.remote(base))
    elif action == "speed":
        fn = speed_b200 if gpu.upper() == "B200" else speed_h100
        print("exit", fn.remote(tag, data, base, extra))
        _summary(read_log.remote(f"speed-{tag}"))
    elif action == "full":
        fn = full_b200 if gpu.upper() == "B200" else full_h100
        print("exit", fn.remote(run, data, base, extra, budget_min))
    elif action == "status":
        _summary(read_log.remote(run))
    else:
        raise SystemExit(f"unknown action {action}")


def _summary(info: dict) -> None:
    recs = info["records"]
    steps = [r for r in recs if "step" in r and "loss" in r]
    devs = [r["dev"] for r in recs if "dev" in r]
    for r in steps[-6:]:
        print(json.dumps(r))
    for d in devs[-3:]:
        print("DEV", json.dumps({k: round(v, 4) for k, v in d.items() if not k.startswith("acc_src")}))
    tps = [r["tok_per_s"] for r in steps[2:]] or [r["tok_per_s"] for r in steps]
    if tps:
        tps.sort()
        print(f"median tok/s {tps[len(tps) // 2]}, max mem GB {max(r.get('mem_gb', 0) for r in steps)}, "
              f"last step {steps[-1]['step']}, elapsed {steps[-1]['elapsed_min']} min")
    print("files:", [f for f in info["files"] if not f.startswith("best/adapter")][:20])
