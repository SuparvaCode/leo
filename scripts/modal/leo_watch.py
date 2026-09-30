"""Unattended pipeline for Modal training runs: a scheduled watchdog (every 15 min, server-side).

Per job file in the Volume (/vol/jobs/<run>.json), it moves the run through
    train -> select -> bench -> done
* train: while the training call runs, nothing. If it ends without a calibrated best/ (crash, preemption,
  time budget), it relaunches the same training (leo.train resumes from resume.pt), at most max_relaunches times.
* select: leo.select on the run's dev set (L4), writes selected.json.
* bench: leo-bench's `bench` function (L4) on the selected checkpoint; results land in /vol/results/<run>.

It is a separate app from leo-v5 so deploying it never touches running training containers.

    set MODAL_PROFILE=<profile>
    modal deploy scripts/modal/leo_watch.py
    python scripts/modal/leo_watch.py register --run leo-4b-v5 --call fc-... --base Qwen/Qwen3-4B-Base --extra "..."
    python scripts/modal/leo_watch.py show
    python scripts/modal/leo_watch.py tick          # run one watchdog pass now
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("leo-watch")
vol = modal.Volume.from_name("leo-v5")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers==5.17.0", "peft==0.21.0", "safetensors==0.8.0", "numpy==2.5.3", "pydantic==2.13.5",
                 "huggingface_hub==1.33.0", "tokenizers==0.23.2")
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/src", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT / "leo"), remote_path="/root/src/leo", ignore=["**/__pycache__/**"])
)
V = Path("/vol")
JOBS = V / "jobs"


def _call_state(call_id: str) -> tuple[str, str]:
    """("running" | "done" | "failed", detail) for a spawned function call."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        return "done", repr(call.get(timeout=0))
    except TimeoutError:  # the builtin TimeoutError means the call has not finished yet
        return "running", ""
    except Exception as e:  # remote exception, function timeout, expired output, ...
        return "failed", f"{type(e).__name__}: {e}"[:300]


def _completed(run: str) -> bool:
    cfg = V / "runs" / run / "best" / "leo_config.json"
    try:
        return bool(json.loads(cfg.read_text(encoding="utf-8")).get("completed"))
    except (OSError, ValueError):
        return False


def _last_step(run: str) -> int | None:
    log = V / "runs" / run / "train_log.jsonl"
    step = None
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            step = rec.get("step", step)
    except OSError:
        pass
    return step


def _note(job: dict, msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    job.setdefault("history", []).append(f"{stamp} {msg}")
    print(f"[{job['run']}] {msg}", flush=True)


def _advance(job: dict) -> None:
    run = job["run"]
    stage = job["stage"]
    if stage in ("done", "failed"):
        return
    state, detail = _call_state(job["call_id"])
    if state == "running":
        job["last_seen_step"] = _last_step(run)
        return
    if stage == "train":
        _note(job, f"training call {job['call_id']} ended ({state} {detail}); last logged step {_last_step(run)}")
        if _completed(run):
            call = select_run.spawn(run, job["data"])
            job.update(stage="select", call_id=call.object_id)
            _note(job, f"training complete; selection started ({call.object_id})")
        elif job["relaunches"] < job["max_relaunches"]:
            fn = modal.Function.from_name("leo-v5", "full_b200" if job["gpu"].upper() == "B200" else "full_h100")
            call = fn.spawn(run, job["data"], job["base"], job["extra"], job["budget_min"])
            job["relaunches"] += 1
            job["call_id"] = call.object_id
            _note(job, f"not finished: relaunched training, attempt {job['relaunches']} ({call.object_id}); it resumes from resume.pt")
        else:
            job["stage"] = "failed"
            _note(job, "not finished and out of relaunches; stopping (resume state is in the run folder)")
    elif stage == "select":
        _note(job, f"selection ended ({state} {detail})")
        if job.get("bench_parts"):
            fn = modal.Function.from_name("leo-bench", "bench")
            call = fn.spawn(run, job["bench_parts"])
            job.update(stage="bench", call_id=call.object_id)
            _note(job, f"benchmarks started ({call.object_id}): {job['bench_parts']}")
        else:
            job["stage"] = "done"
    elif stage == "bench":
        job["stage"] = "done"
        _note(job, f"benchmarks ended ({state} {detail}); results in /vol/results/{run}")


@app.function(image=image, volumes={"/vol": vol}, cpu=0.25, memory=512, timeout=600, schedule=modal.Period(minutes=15))
def watchdog() -> list[dict]:
    vol.reload()
    JOBS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(JOBS.glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        try:
            _advance(job)
        except Exception as e:  # never let one job break the others; retried next tick
            _note(job, f"watchdog error: {type(e).__name__}: {e}")
        path.write_text(json.dumps(job, indent=1), encoding="utf-8")
        jobs.append(job)
    vol.commit()
    return jobs


@app.function(image=image, volumes={"/vol": vol}, cpu=0.25, memory=512, timeout=120)
def register(job: dict) -> dict:
    vol.reload()
    JOBS.mkdir(parents=True, exist_ok=True)
    base = {"data": "processed-v5", "gpu": "H100", "budget_min": 300.0, "relaunches": 0, "max_relaunches": 2,
            "stage": "train", "bench_parts": "short,short_canon,heldout,jevbench,multilingual,done,probes", "history": []}
    job = {**base, **job}
    _note(job, f"registered at stage {job['stage']} with call {job['call_id']}")
    (JOBS / f"{job['run']}.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    vol.commit()
    return job


@app.function(image=image, volumes={"/vol": vol}, cpu=0.25, memory=512, timeout=120)
def show() -> list[dict]:
    vol.reload()
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(JOBS.glob("*.json"))] if JOBS.exists() else []


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": vol}, timeout=2 * 3600)
def select_run(run: str, data: str) -> int:
    vol.reload()
    cmd = f"python -m leo.select --run /vol/runs/{run} --data /vol/data/{data}"
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd="/root/src")
    vol.commit()
    return r.returncode


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--run", required=True)
    r.add_argument("--call", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--extra", required=True)
    r.add_argument("--data", default="processed-v5")
    r.add_argument("--budget-min", type=float, default=300.0)
    r.add_argument("--stage", default="train")
    r.add_argument("--bench-parts", default="short,short_canon,heldout,jevbench,multilingual,done,probes")
    r.add_argument("--max-relaunches", type=int, default=2)
    sub.add_parser("show")
    sub.add_parser("tick")
    a = ap.parse_args()
    if a.cmd == "register":
        job = {"run": a.run, "call_id": a.call, "base": a.base, "extra": a.extra, "data": a.data,
               "budget_min": a.budget_min, "stage": a.stage, "bench_parts": a.bench_parts,
               "max_relaunches": a.max_relaunches}
        print(json.dumps(modal.Function.from_name("leo-watch", "register").remote(job), indent=1))
    elif a.cmd == "show":
        for job in modal.Function.from_name("leo-watch", "show").remote():
            print(f"{job['run']}: stage={job['stage']} call={job['call_id']} relaunches={job['relaunches']} "
                  f"last_step={job.get('last_seen_step')}")
            for h in job.get("history", [])[-6:]:
                print("   ", h)
    else:
        for job in modal.Function.from_name("leo-watch", "watchdog").remote():
            print(job["run"], job["stage"], job.get("last_seen_step"))


if __name__ == "__main__":
    main()
