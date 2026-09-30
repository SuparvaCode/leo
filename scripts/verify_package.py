"""Self-check shipped inside a packaged Leo folder: load it offline and re-answer the blind short-input suite.

    python verify.py              # from inside the folder (GPU if it fits, else CPU)
    python verify.py --device cpu

It compares every P(yes) with the value recorded when the model was benchmarked (results/short__<name>.json).
bf16 numerics differ slightly between GPUs and CPU, so a difference up to 0.05 is expected; accuracy is recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    a = ap.parse_args()
    from leo.infer import Leo

    cfg = json.loads((HERE / "leo_config.json").read_text(encoding="utf-8"))
    src = cfg["packaged_from"]["results"]
    rec = json.loads((HERE / "results" / f"short__{src}.json").read_text(encoding="utf-8"))["items"]
    by_state: dict[str, list[dict]] = defaultdict(list)
    for it in rec:
        by_state[it["state"]].append(it)
    t0 = time.perf_counter()
    leo = Leo.load(HERE, device=a.device, dtype=a.dtype)
    print(f"loaded {leo.name} ({cfg['base_model']} from bundled base/) on {leo.device} in {time.perf_counter() - t0:.0f} s")
    reqs = [{"state": s, "questions": {f"c{i}": {"type": "noul", "instructions": it["condition"]} for i, it in enumerate(items)}}
            for s, items in by_state.items()]
    t0 = time.perf_counter()
    resp = []
    for i in range(0, len(reqs), 8):
        resp += leo.predict_many(reqs[i:i + 8])
    secs = time.perf_counter() - t0
    diffs, ok_new, ok_old = [], 0, 0
    for (s, items), r in zip(by_state.items(), resp):
        for i, it in enumerate(items):
            p = r["answers"][f"c{i}"]["noul"]
            diffs.append(abs(p - it["p"]))
            ok_new += (p >= 0.5) == it["gold"]
            ok_old += it["ok"]
    n = len(diffs)
    print(f"{n} conditions in {secs:.0f} s: accuracy now {ok_new / n:.3f}, recorded {ok_old / n:.3f}; "
          f"max |dP| {max(diffs):.3f}, mean |dP| {sum(diffs) / n:.4f}")
    print("OK" if max(diffs) <= 0.05 and abs(ok_new - ok_old) <= 3 else "MISMATCH: check the files against SHA256SUMS.txt")


if __name__ == "__main__":
    main()
