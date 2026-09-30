"""Benchmarks and an HTTP endpoint for checkpoints too big for the local 8 GB GPU (a separate app from training).

    set MODAL_PROFILE=suparvabaranwal4
    modal volume put leo-v5 checkpoints/leo-4b-v5/best /ckpt/leo-4b-v5/best         # after training
    modal volume put leo-v5 data/external/jevbench /data/external/jevbench
    modal secret create leo-bench HF_TOKEN=... LEO_API_KEY=...
    modal deploy scripts/modal/leo_bench_modal.py
    python -c "import modal; print(modal.Function.from_name('leo-bench','bench').spawn('leo-4b-v5').object_id)"
    modal volume get leo-v5 /results/leo-4b-v5 results-modal/        # then merge into results/

The endpoint (`serve`) runs leo.serve's app behind a bearer key (secret LEO_API_KEY) and scales to zero when
idle, so the local browser and naturalcodz benchmarks can use the 4B model at bf16.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("leo-bench")
vol = modal.Volume.from_name("leo-v5")
secret = modal.Secret.from_name("leo-bench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers==5.17.0", "peft==0.21.0", "safetensors==0.8.0", "numpy==2.5.3", "pydantic==2.13.5",
                 "huggingface_hub==1.33.0", "tokenizers==0.23.2", "datasets==5.0.1", "pandas==3.0.6", "pyarrow==25.0.1",
                 "fastapi==0.141.1", "uvicorn==0.54.0", "httpx==0.28.1", "h2==4.4.1", "python-dotenv==1.2.3",
                 "browser-harness==0.1.13")
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/src:/root/src/external/jev-ultrafast", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false", "BH_TELEMETRY": "0"})
    .add_local_dir(str(ROOT / "leo"), remote_path="/root/src/leo", ignore=["**/__pycache__/**"])
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/src/scripts", ignore=["**/__pycache__/**", "kaggle/**"])
    .add_local_dir(str(ROOT / "external" / "jev-ultrafast" / "jev_ultrafast"),
                   remote_path="/root/src/external/jev-ultrafast/jev_ultrafast", ignore=["**/__pycache__/**"])
)
SRC = Path("/root/src")


def _link_workspace(name: str) -> Path:
    """Point the repo-relative paths the benchmark code uses at the Volume."""
    vol.reload()
    data = SRC / "data"
    if not data.exists():
        data.symlink_to("/vol/data")
    ck = SRC / "checkpoints"
    if not ck.exists():
        ck.symlink_to("/vol/ckpt")
    res = Path("/vol/results") / name
    res.mkdir(parents=True, exist_ok=True)
    (SRC / "results").mkdir(exist_ok=True)
    traces = Path("/vol/data/v3_flights_traces")  # the recorded Google Flights states used by done_probe.py
    if traces.exists():
        shutil.copytree(traces, SRC / "results/browser/final-leo-1.7b-v3/traces", dirs_exist_ok=True)
    return res


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": vol}, secrets=[secret], timeout=4 * 3600)
def bench(name: str, parts: str = "short,heldout,jevbench,multilingual,done,probes") -> dict:
    out = _link_workspace(name)
    model = f"/vol/runs/{name}" if Path(f"/vol/runs/{name}/selected.json").exists() else f"/vol/ckpt/{name}"
    steps = {
        "short": f"python -m leo.bench.short --backend leo --model {model} --name {name}",
        "short_canon": f"python -m leo.bench.short --backend leo --model {model} --name {name}-canon --canonicalize input",
        "heldout": f"python -m leo.bench.heldout --backend leo --model {model} --name {name} --dtype bf16",
        "jevbench": f"python -m leo.bench.jevbench --model {model} --name {name} --dtype bf16",
        "multilingual": f"python -m leo.bench.multilingual --backend leo --model {model} --name {name} --dtype bf16",
        "done": f"python scripts/done_probe.py --leo {model} --name {name} --n 400 --dtype bf16",
        "probes": f"python -m leo.bench.probes --leo {model} --name {name} --leo_dtypes bf16 --order_views 1,2 --no-jev",
    }
    status = {}
    for part in parts.split(","):
        print(f"=== {part}: {steps[part]}", flush=True)
        r = subprocess.run(f"set -o pipefail; {steps[part]} 2>&1 | tee -a /vol/results/{name}/bench_{part}.log",
                           shell=True, cwd=str(SRC), executable="/bin/bash")
        status[part] = r.returncode
        shutil.copytree(SRC / "results", out, dirs_exist_ok=True)  # keep partial results if a later part fails
        vol.commit()
    print(status, flush=True)
    (out / "bench_status.json").write_text(__import__("json").dumps(status), encoding="utf-8")
    vol.commit()
    return status


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": vol}, secrets=[secret], timeout=3600)
def calibrate(name: str, data: str = "processed-v5.1") -> dict:
    """Fit per-bucket temperatures for /vol/ckpt/<name> on the dev set (used for merged checkpoints)."""
    import json

    import torch
    from transformers import AutoConfig, AutoTokenizer

    from leo.infer import make_tokenize
    from leo.model import supports_block_mask
    from leo.select import dev_scores
    from leo.train import load_jsonl, prepare

    vol.reload()
    d = Path(f"/vol/ckpt/{name}")
    cfg = json.loads((d / "leo_config.json").read_text(encoding="utf-8"))
    tok = make_tokenize(AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_revision"]))
    packable = supports_block_mask(AutoConfig.from_pretrained(cfg["base_model"], revision=cfg["base_revision"]))
    dev = prepare(load_jsonl(Path(f"/vol/data/{data}/dev.jsonl")))
    res = dev_scores(d, dev, torch.device("cuda"), torch.bfloat16, tok, packable, cfg["train"])
    cfg.update(temperatures=res["temperatures"], dev={"uncalibrated": res["raw"], "calibrated": res["calibrated"]},
               completed=True)
    (d / "leo_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    vol.commit()
    keep = ("loss", "ece_top", "acc/choice", "acc/noul", "acc/score")
    return {"raw": {k: round(v, 4) for k, v in res["raw"].items() if k in keep},
            "calibrated": {k: round(v, 4) for k, v in res["calibrated"].items() if k in keep}}


@app.function(image=image, gpu="L4", cpu=2, memory=16384, volumes={"/vol": vol}, secrets=[secret], timeout=3600,
              scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def public():
    """Leo-1 behind the public trial site (leo.kognare.com): TypeSafe-exact responses, named leo-1, bare yes/no
    conditions rewritten into questions, states over 8,192 tokens rejected with a 422. One L4 at most; it
    scales to zero after 5 idle minutes. Only the trial site holds the bearer key."""
    from leo.infer import Leo
    from leo.serve import create_app

    vol.reload()
    leo = Leo.load("/vol/ckpt/leo-4b-soup3", dtype="bf16", jev_exact=True, max_state_tokens=8192,
                   encoder_overrides={"canonicalize": "input", "long_state": "reject"})
    leo.name = "leo-1"
    return create_app(leo, api_key=os.environ["LEO_API_KEY"], max_body_bytes=512 * 1024, max_questions=64)


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": vol}, secrets=[secret], timeout=3600,
              scaledown_window=600, max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def serve():
    from leo.infer import Leo
    from leo.serve import create_app

    vol.reload()
    pick = Path("/vol/serve_model.txt")  # which checkpoint to serve; a new container reads it at start
    name = pick.read_text().strip() if pick.exists() else os.environ.get("LEO_SERVE_MODEL", "leo-4b-v5.1")
    path = f"/vol/runs/{name}" if Path(f"/vol/runs/{name}/selected.json").exists() else f"/vol/ckpt/{name}"
    leo = Leo.load(path, dtype="bf16", jev_exact=True, max_state_tokens=12288,
                   max_row_tokens=20480, max_batch_tokens=20480)
    leo.encoder.max_question_tokens = 6144
    leo.encoder.max_instruction_tokens = 1024
    return create_app(leo, api_key=os.environ["LEO_API_KEY"])
