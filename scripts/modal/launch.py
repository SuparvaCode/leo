"""Start a training run on the *deployed* leo-v5 app, server-side: the call keeps running whatever happens
to this PC (a local `modal run` client that dies takes its run down with it, even with --detach).

    set MODAL_PROFILE=<profile>
    modal deploy scripts/modal/leo_modal.py
    python scripts/modal/launch.py --gpu H100 --run leo-4b-v5 --base Qwen/Qwen3-4B-Base --budget-min 300 --extra "..."
    python scripts/modal/launch.py --status <call id>
"""
from __future__ import annotations

import argparse

import modal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--run")
    ap.add_argument("--data", default="processed-v5")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--budget-min", type=float, default=300.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--status", help="function call id from an earlier launch")
    a = ap.parse_args()
    if a.status:
        call = modal.FunctionCall.from_id(a.status)
        try:
            print("finished, exit code", call.get(timeout=0))
        except TimeoutError:
            print("still running")
        return
    fn = modal.Function.from_name("leo-v5", "full_b200" if a.gpu.upper() == "B200" else "full_h100")
    call = fn.spawn(a.run, a.data, a.base, a.extra, a.budget_min)
    print("spawned", call.object_id)


if __name__ == "__main__":
    main()
