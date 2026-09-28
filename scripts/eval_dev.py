"""Score a Leo checkpoint on a dev file, with a breakdown for choice questions that offer a none/other option.

Used to compare runs on the same in-distribution data, e.g. v0 and v1 both on data/processed-v1/dev.jsonl,
which contains "none" options that are right and "none" options that are wrong.

    python scripts/eval_dev.py --model checkpoints/leo-0.6b-v0 --data data/processed-v1/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from leo.data import example_specs
from leo.data.augment import RECIPES
from leo.infer import Leo

NONE_KEYS = {k for r in RECIPES.values() for k in r["none_keys"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", help="write the summary JSON here")
    args = ap.parse_args()

    examples = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    requests = [(ex["state"], example_specs(ex)) for ex in examples]
    leo = Leo.load(args.model)
    probs = leo.probabilities(requests)

    hits: dict[str, list[float]] = defaultdict(list)
    for ex, (_, specs), p in zip(examples, requests, probs):
        for s in specs:
            y = np.asarray(s.target)
            if y.max() < 0.99:  # soft / unknowable targets: skip for accuracy
                continue
            ok = float(np.argmax(p[s.qid]) == np.argmax(y))
            hits[f"type/{s.type}"].append(ok)
            hits[f"source/{ex['source']}"].append(ok)
            hits["all"].append(ok)
            if s.type == "choice":
                none_slots = [i for i, k in enumerate(s.keys) if k.lower() in NONE_KEYS]
                if none_slots:
                    gold_is_none = int(np.argmax(y)) in none_slots
                    hits["none_option/gold_is_none" if gold_is_none else "none_option/gold_is_real"].append(ok)
                    if not gold_is_none:
                        hits["none_option/picked_none_when_wrong"].append(float(int(np.argmax(p[s.qid])) in none_slots))
    summary = {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(hits.items())}
    summary["picked_none_rate_when_real_is_gold"] = summary.pop("none_option/picked_none_when_wrong", {"n": 0, "acc": 0.0})
    name = Path(args.model).name
    for k, v in summary.items():
        if not k.startswith("source/"):
            print(f"{name:>14} {k:<40} n={v['n']:<5} {v['acc']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "data": args.data, "summary": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
