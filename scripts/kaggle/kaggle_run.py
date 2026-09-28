"""Train Leo on Kaggle's free GPUs (2x T4) from this machine.

    python scripts/kaggle/kaggle_run.py bundle --data data/processed-v3          # upload code + data (private dataset)
    python scripts/kaggle/kaggle_run.py launch --kernel leo-train-a --run leo-1.7b-v3 --data processed-v3 --budget 470
    python scripts/kaggle/kaggle_run.py status --kernel leo-train-a
    python scripts/kaggle/kaggle_run.py fetch  --kernel leo-train-a --run leo-1.7b-v3   # -> checkpoints/<run>

A session stops cleanly at --budget minutes and leaves resume.pt in its output. The next session (launched
with --resume-from <previous kernel>) copies that state in and continues, so a run can span many sessions.
Everything stays private: the dataset and kernels are created private, and no token is uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results" / "kaggle"
USER = os.environ.get("KAGGLE_USERNAME", "suparvabaranwal")  # set KAGGLE_USERNAME to use your own account
BUNDLE = "leo-bundle"


def kaggle(*args: str) -> str:
    # UTF-8 mode: the CLI writes kernel logs with the default encoding, which is cp1252 on Windows and fails
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def bundle(data_dirs: list[str], message: str, init_models: list[str] | None = None) -> None:
    d = WORK / "bundle"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for m in init_models or []:  # exported models for --init warm starts, as init/<run name>/
        src = ROOT / m
        shutil.copytree(src, d / "init" / src.parent.name, ignore=shutil.ignore_patterns("*.tmp"))
    for p in (ROOT / "leo").rglob("*.py"):
        dst = d / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for data in data_dirs:
        src = ROOT / data
        dst = d / src.name
        dst.mkdir()
        for f in ("train.jsonl", "dev.jsonl", "manifest.json"):
            shutil.copy2(src / f, dst / f)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": BUNDLE, "id": f"{USER}/{BUNDLE}", "licenses": [{"name": "other"}]}, indent=1))
    try:
        kaggle("datasets", "status", f"{USER}/{BUNDLE}")
        exists = True
    except SystemExit:
        exists = False
    if exists:
        print(kaggle("datasets", "version", "-p", str(d), "-m", message, "--dir-mode", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip"))
    # wait until the new version is ready, or a kernel may mount the old one
    for _ in range(120):
        s = kaggle("datasets", "status", f"{USER}/{BUNDLE}")
        if "ready" in s.lower():
            print("dataset ready")
            return
        time.sleep(10)
    print("dataset still processing; check `kaggle datasets status`")


KERNEL = r'''
import glob, json, os, shutil, subprocess, sys, zipfile
CFG = json.loads(r"""__CFG__""")
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/" + pattern, recursive=True))
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):  # in case Kaggle kept an archive
        x = "/kaggle/working/unz/" + os.path.basename(z)[:-4]
        if not os.path.exists(x):
            zipfile.ZipFile(z).extractall(x)
    return hits or sorted(glob.glob("/kaggle/working/unz/" + pattern, recursive=True))

code = find("**/leo/train.py")
assert code, "leo-bundle dataset not mounted: " + str(glob.glob("/kaggle/input/*/*"))
src = "/kaggle/working/src"
shutil.rmtree(src, ignore_errors=True)
shutil.copytree(os.path.dirname(code[0]), src + "/leo")
hits = find("**/" + CFG["data"] + "/train.jsonl")
assert hits, "data folder not found: " + CFG["data"]
data = os.path.dirname(hits[0])
print("code:", code[0], "data:", data, flush=True)

sh("pip install -q transformers==5.17.0 peft==0.21.0 2>&1 | tail -n 3")
sh("pip uninstall -y -q torchao 2>&1 | tail -n 1")  # Kaggle's torchao 0.10 makes peft 0.21 refuse to wrap layers
out = "/kaggle/working/" + CFG["run"]
prev = [p for p in glob.glob("/kaggle/input/**/" + CFG["run"] + "/resume.pt", recursive=True)]
if CFG.get("resume_from") and not prev:
    # a continuation session must never quietly start the run over from step 0
    raise SystemExit("resume state from " + CFG["resume_from"] + " is not mounted: " + str(glob.glob("/kaggle/input/*")))
if prev and not os.path.exists(os.path.join(out, "resume.pt")):
    shutil.copytree(os.path.dirname(prev[0]), out, dirs_exist_ok=True)
    print("continuing from", prev[0], flush=True)

from huggingface_hub import snapshot_download
snapshot_download(CFG["base"], allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])

env = dict(os.environ, PYTHONPATH=src, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
ngpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
launcher = f"torchrun --standalone --nproc_per_node {ngpu}" if ngpu > 1 else sys.executable
init = ""
if CFG.get("init"):
    hit = find("**/init/" + CFG["init"] + "/leo_config.json")
    assert hit, "init model not in the bundle: " + CFG["init"]
    init = " --init " + os.path.dirname(hit[0])
cmd = (f"{launcher} -m leo.train --data {data} --base {CFG['base']} --out {out} --name {CFG['run']} "
       f"--precision fp16 --time_budget_min {CFG['budget']} --resume{init} " + CFG["extra"])
print("+", cmd, flush=True)
r = subprocess.run(cmd, shell=True, env=env)
for f in ("resume.pt.tmp",):
    p = os.path.join(out, f)
    if os.path.exists(p):
        os.remove(p)
print("exit code", r.returncode, flush=True)
sys.exit(r.returncode)
'''


def launch(kernel: str, run: str, data: str, base: str, budget: float, extra: str, resume_from: str | None,
           init: str | None = None) -> None:
    d = WORK / "kernels" / kernel
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    cfg = {"run": run, "data": data, "base": base, "budget": budget, "extra": extra, "resume_from": resume_from,
           "init": init}
    (d / "train.py").write_text(KERNEL.replace("__CFG__", json.dumps(cfg)), encoding="utf-8")
    meta = {
        "id": f"{USER}/{kernel}", "title": kernel, "code_file": "train.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{USER}/{BUNDLE}"], "competition_sources": [],
        "kernel_sources": [f"{USER}/{resume_from}"] if resume_from else [], "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(d)))


def status(kernel: str) -> str:
    return kaggle("kernels", "status", f"{USER}/{kernel}")


def fetch(kernel: str, run: str | None, log_only: bool) -> None:
    d = WORK / "output" / kernel
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    args = ["kernels", "output", f"{USER}/{kernel}", "-p", str(d)]
    if log_only:
        args += ["--file-pattern", r".*\.log$"]
    kaggle(*args)
    for log in d.glob("*.log"):
        lines = [e.get("data", "") for e in json.loads(log.read_text(encoding="utf-8"))]
        (d / "console.txt").write_text("".join(lines), encoding="utf-8")
        print(f"log: {d / 'console.txt'}")
    if run and (d / run).exists():
        dst = ROOT / "checkpoints" / run
        shutil.copytree(d / run, dst, dirs_exist_ok=True)
        print(f"checkpoint copied to {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--data", action="append", required=True)
    b.add_argument("--message", default="update")
    b.add_argument("--init-model", action="append", help="exported model dir to ship for --init, e.g. checkpoints/leo-1.7b-v3/final")
    l = sub.add_parser("launch")
    l.add_argument("--kernel", required=True)
    l.add_argument("--run", required=True)
    l.add_argument("--data", required=True, help="data folder name inside the bundle, e.g. processed-v3")
    l.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    l.add_argument("--budget", type=float, default=470, help="training minutes before a clean stop")
    l.add_argument("--extra", default="", help="extra leo.train arguments")
    l.add_argument("--resume-from", help="kernel whose output holds the run's resume.pt")
    l.add_argument("--init", help="warm start from bundled init/<name> (see bundle --init-model)")
    s = sub.add_parser("status")
    s.add_argument("--kernel", required=True)
    s.add_argument("--wait", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("--kernel", required=True)
    f.add_argument("--run")
    f.add_argument("--log-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "bundle":
        bundle(a.data, a.message, a.init_model)
    elif a.cmd == "launch":
        launch(a.kernel, a.run, a.data, a.base, a.budget, a.extra, a.resume_from, a.init)
    elif a.cmd == "status":
        while True:
            s = status(a.kernel)
            print(s, flush=True)
            if not a.wait or not any(x in s for x in ("QUEUED", "RUNNING")):
                break
            time.sleep(60)
    else:
        fetch(a.kernel, a.run, a.log_only)


if __name__ == "__main__":
    sys.exit(main())
