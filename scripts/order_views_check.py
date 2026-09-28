"""Effect of Leo's order_views (option-order averaging) on flip rate, accuracy, calibration and latency.

    python scripts/order_views_check.py --model checkpoints/leo-0.6b-v1 --views 1,2,3,4
"""
import argparse
import time

import numpy as np
import torch

from leo.bench import heldout, probes
from leo.infer import Leo
from leo.metrics import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--views", default="1,2,3,4")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rows", type=int, default=800)
    a = ap.parse_args()
    leo = Leo.load(a.model, dtype=a.dtype)
    ad = probes.LeoAdapter.__new__(probes.LeoAdapter)
    ad.leo = leo
    from leo.schema import to_spec

    ad._to_spec = to_spec
    for v in (int(x) for x in a.views.split(",")):
        leo.order_views = v
        cells = []
        for key in ("emotion", "fin_topic"):
            o = probes.order_probe(ad, key, n_rows=300, n_perm=5)
            spec = heldout.SPECS[key]
            rows = heldout.load_rows(spec, limit=a.rows)
            p = ad.choice_probs([t for t, _ in rows], spec.instruction, list(spec.labels))
            m = summarize(p, np.asarray([g for _, g in rows]), len(spec.labels))
            cells.append(f"{key}: flip {o['flip_rate']:.3f} acc {m['accuracy']:.3f} ece {m['ece']:.3f}")
        qs = probes.make_questions(10)
        for _ in range(3):
            leo.system_one(probes.LONG_STATE, qs)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            leo.system_one(probes.LONG_STATE, qs)
        torch.cuda.synchronize()
        print(f"views={v} | " + " | ".join(cells) + f" | 10q long-state latency {(time.perf_counter() - t) * 100:.0f} ms",
              flush=True)


if __name__ == "__main__":
    main()
