"""Continuation data for leo-4b-v5.1 (warm start from leo-4b-v5).

leo-4b-v5 retyped "COPENHAGEN" into a field whose goal said "Copenhagen" until jev-ultrafast's no-progress rule
stopped it (travel-glasshouse 0/3). The simulators now write typed values in varied case and include states
where the value is already typed, possibly repeated with no page change. This set is:

  * the two browser families regenerated with that change (new seeds),
  * a replay sample of the full v5 training set (every source), so nothing else drifts.

    python scripts/build_v51.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "jev-ultrafast"))

from leo.data import browser, browser_evidence, example_specs  # noqa: E402

SEED = 551


def read(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def main() -> None:
    out = ROOT / "data" / "processed-v5.1"
    out.mkdir(parents=True, exist_ok=True)
    v5_train, v5_dev = read(ROOT / "data/processed-v5/train.jsonl"), read(ROOT / "data/processed-v5/dev.jsonl")
    rng = random.Random(SEED)
    replay = rng.sample(v5_train, 24000)
    ev = browser_evidence.examples(random.Random(f"{SEED}:ev"), 8000)
    ev_dev = browser_evidence.examples(random.Random(f"{SEED}:ev:dev"), 300)
    cat = browser.examples(random.Random(f"{SEED}:cat"), 4000)
    for ex in ev + ev_dev:
        ex["source"] = browser_evidence.SOURCE
    for ex in ev + ev_dev + cat:
        example_specs(ex)
    train = replay + ev + cat
    rng.shuffle(train)
    dev = v5_dev + ev_dev
    retype = sum(1 for ex in ev + cat if any(a.get("kind") == "fill" and a.get("page_changed") is False
                                            for a in ex["state"].get("recent_actions", [])))
    case = sum(1 for ex in ev + cat for e in ex["state"].get("elements", [])
               if e.get("operations") and "TYPE_TEXT" in e["operations"] and e.get("value", "").isupper() and len(e["value"]) > 2)
    manifest = {"recipe": "v5.1", "seed": SEED, "replay_from": "processed-v5", "replay": len(replay),
                "browser_evidence": len(ev), "browser_catalog": len(cat),
                "states_with_repeated_no_change_fill": retype, "states_with_upper_case_field_value": case,
                "labels_new": dict(Counter(ex["questions"]["operation"]["label"] for ex in ev + cat))}
    for split, rows in (("train", train), ("dev", dev)):
        p = out / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        manifest[split] = {"requests": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("train", "dev")}, indent=1))
    print("train", len(train), "dev", len(dev))


if __name__ == "__main__":
    main()
