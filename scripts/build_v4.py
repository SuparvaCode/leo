"""Data for leo-1.7b-v4: a warm-started continuation of leo-1.7b-v3 that fixes false DONE on browser tasks.

train = a random replay share of v3's training requests (every source, so nothing is forgotten)
      + leo.data.browser_evidence (new: DONE only with visible evidence; see that module).
dev   = all of v3's dev set + a dev split of the new family, so model selection sees both.

    python scripts/build_v4.py --replay 30000 --evidence 9000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "jev-ultrafast"))

from leo.data import browser_evidence, example_specs  # noqa: E402


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed-v3")
    ap.add_argument("--out", default="data/processed-v4")
    ap.add_argument("--replay", type=int, default=30000)
    ap.add_argument("--evidence", type=int, default=9000)
    ap.add_argument("--evidence_dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    src, out = ROOT / a.src, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    v3_train, v3_dev = read(src / "train.jsonl"), read(src / "dev.jsonl")
    replay = rng.sample(v3_train, min(a.replay, len(v3_train)))
    ev_train = browser_evidence.examples(random.Random(f"{a.seed}:evidence"), a.evidence)
    ev_dev = browser_evidence.examples(random.Random(f"{a.seed}:evidence:dev"), a.evidence_dev)
    for ex in ev_train + ev_dev:
        ex["source"] = browser_evidence.SOURCE
        example_specs(ex)
    train = replay + ev_train
    rng.shuffle(train)
    dev = v3_dev + ev_dev
    v3_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"recipe": "v4", "seed": a.seed, "built_from": a.src, "v3_manifest_train_sha256": v3_manifest["train"]["sha256"],
                "replay_requests": len(replay), "replay_by_source": dict(Counter(ex["source"] for ex in replay)),
                "new_sources": {browser_evidence.SOURCE: {"generator": "leo.data.browser_evidence.examples",
                                                          "n_train": len(ev_train), "n_dev": len(ev_dev),
                                                          "train_themes": [t.brand for t in browser_evidence.TRAIN_THEMES],
                                                          "held_out_themes": list(browser_evidence.PROBE_THEMES)}},
                "new_labels": dict(Counter(ex["questions"]["operation"]["label"] for ex in ev_train))}
    for split, rows in (("train", train), ("dev", dev)):
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        qtypes = Counter(q["type"] for ex in rows for q in ex["questions"].values())
        manifest[split] = {"requests": len(rows), "questions": sum(qtypes.values()), "by_type": dict(qtypes),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{split}: {len(rows)} requests {dict(qtypes)}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["new_labels"]))


if __name__ == "__main__":
    main()
