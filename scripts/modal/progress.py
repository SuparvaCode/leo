"""Progress of a Modal training run: step, speed, time left, dev evals.   python scripts/modal/progress.py leo-4b-v5"""
import sys

import modal

run = sys.argv[1]
total = int(sys.argv[2]) if len(sys.argv) > 2 else 10981
recs = modal.Function.from_name("leo-v5", "read_log").remote(run)["records"]
r = [x for x in recs if "loss" in x]
d = [x["dev"] for x in recs if "dev" in x]
a, b = r[max(0, len(r) - 20)], r[-1]
rate = (b["step"] - a["step"]) / max(1e-6, b["elapsed_min"] - a["elapsed_min"])
left = (total - b["step"]) / rate
print(f"{run}: step {b['step']}/{total}, {rate:.0f} steps/min, about {left / 60:.1f} h of training left, "
      f"elapsed {b['elapsed_min']:.0f} min, train loss {b['loss']:.3f}")
for x in d:
    print("   DEV", {k: round(v, 4) for k, v in x.items() if k in ("step", "loss", "acc/choice", "acc/noul", "acc/score")})
