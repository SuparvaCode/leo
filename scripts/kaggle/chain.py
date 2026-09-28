"""Watch a Kaggle training session and chain the next one until the run finishes, then fetch the checkpoint.

    python scripts/kaggle/chain.py --run leo-1.7b-v3 --data processed-v3 --running leo-train-a --extra "..."

Sessions alternate between two kernels (a kernel cannot mount its own output). A session that stopped at its
time budget is continued by the other kernel with --resume-from; one that errored is retried once from its
last saved resume state. The --extra arguments must be identical across sessions or leo.train refuses to
resume. Do not upload a new leo-bundle while a chain is running: a changed train.jsonl also blocks resuming.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("kaggle_run", Path(__file__).with_name("kaggle_run.py"))
kr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kr)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait(kernel: str) -> str:
    while True:
        try:
            s = kr.status(kernel)
        except SystemExit as e:  # transient API error
            log(f"status failed ({e}); retrying")
            time.sleep(120)
            continue
        if not any(x in s for x in ("QUEUED", "RUNNING")):
            return s
        time.sleep(180)


def console(kernel: str) -> str:
    kr.fetch(kernel, None, log_only=True)
    p = kr.WORK / "output" / kernel / "console.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--budget", type=float, default=470)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--running", required=True, help="kernel of the session currently running")
    ap.add_argument("--kernels", default="leo-train-a,leo-train-b")
    ap.add_argument("--max-sessions", type=int, default=4)
    a = ap.parse_args()
    ks = a.kernels.split(",")
    current = a.running
    retried = False
    for session in range(1, a.max_sessions + 1):
        log(f"session {session}: waiting for {current}")
        s = wait(current)
        text = console(current)
        tail = "\n".join(text.strip().splitlines()[-6:])
        log(f"{current} finished: {s}\n{tail}")
        if "done: model in" in text:
            kr.fetch(current, a.run, log_only=False)
            log(f"RUN COMPLETE: checkpoints/{a.run}")
            return
        nxt = ks[(ks.index(current) + 1) % len(ks)]
        if "TIME BUDGET reached" in text:
            retried = False
        elif not retried and "resume state saved" in text:
            retried = True
            log("session errored after saving resume state; retrying once from it")
        else:
            log("session failed without a usable resume state; stopping the chain")
            raise SystemExit(1)
        kr.launch(nxt, a.run, a.data, a.base, a.budget, a.extra, resume_from=current)
        log(f"launched {nxt} (resume from {current})")
        current = nxt
        time.sleep(120)
    log("max sessions reached without completion")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
