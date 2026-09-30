"""Data for leo-4b-v5 (Qwen3-4B-Base): the whole v3 mixture plus the v5 families.

train = every v3 training request (replayed in full, so nothing regresses)
      + leo.data.browser_evidence with direct "open X" goals (fixes v4's search shortcut)
      + leo.data.short: bare conditions, tiny states, GoEmotions, civil_comments rules, PII, severity scales,
        naturalcodz phrasings
      + a second draw of the code-generated reasoning families (new seed)
dev   = v3 dev + a dev split of every new family (for checkpoint selection and calibration).

    python scripts/build_v5.py
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

from leo.data import browser_evidence, example_specs, reasoning, short  # noqa: E402

SEED = 505


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_dev(rows: list[dict], n_dev: int) -> tuple[list[dict], list[dict]]:
    return rows[n_dev:], rows[:n_dev]


def main() -> None:
    out = ROOT / "data" / "processed-v5"
    out.mkdir(parents=True, exist_ok=True)
    v3_train, v3_dev = read(ROOT / "data/processed-v3/train.jsonl"), read(ROOT / "data/processed-v3/dev.jsonl")
    rng = lambda name: random.Random(f"{SEED}:{name}")  # noqa: E731

    fams: dict[str, tuple[list[dict], list[dict]]] = {}

    def add(name: str, tr: list[dict], dv: list[dict]) -> None:
        for ex in tr + dv:
            example_specs(ex)  # validates labels
        fams[name] = (tr, dv)
        print(f"ok   {name:<22} train={len(tr):<6} dev={len(dv)}", flush=True)

    # Rows derived from v3 rows keep their split: train from v3 train, dev from v3 dev.
    add("short_choices", short.from_choices(v3_train, rng("choices"), 30000), short.from_choices(v3_dev, rng("choices:dev"), 150))
    add("short_ratings", short.from_ratings(v3_train, rng("ratings"), 2500), short.from_ratings(v3_dev, rng("ratings:dev"), 60))
    add("short_binary", short.from_binary(v3_train, rng("binary")), short.from_binary(v3_dev, rng("binary:dev")))
    add("short_generic", short.generic_instructions(v3_train, rng("generic"), 8000),
        short.generic_instructions(v3_dev, rng("generic:dev"), 120))
    add("go_emotions", short.go_emotions(rng("goemo"), 15000), short.go_emotions(rng("goemo:dev"), 150, split="validation"))
    add("civil_comments", *(split_dev(short.civil_comments(rng("civil"), 15150), 150)))
    add("syn_pii", *(split_dev(short.pii(rng("pii"), 4100), 100)))
    add("syn_tiny", short.tiny_and_mentions(rng("tiny"), repeat=26), short.tiny_and_mentions(rng("tiny:dev"), repeat=2))
    add("syn_severity", *(split_dev(short.severity(rng("severity"), 3100), 100)))
    ev = browser_evidence.examples(rng("evidence"), 9300)
    for ex in ev:
        ex["source"] = browser_evidence.SOURCE
    add("syn_browser_evidence", *(split_dev(ev, 300)))
    extra = []
    r = rng("reasoning")
    for name, (fn, n) in reasoning.FAMILIES.items():
        extra += [fn(r) for _ in range(n)]
    add("reasoning_v5", extra, [])

    train = list(v3_train)
    dev = list(v3_dev)
    for tr, dv in fams.values():
        train += tr
        dev += dv
    random.Random(SEED).shuffle(train)
    manifest = {"recipe": "v5", "seed": SEED, "v3_train_requests": len(v3_train),
                "families": {k: {"n_train": len(v[0]), "n_dev": len(v[1])} for k, v in fams.items()},
                "train_by_source": dict(Counter(ex["source"] for ex in train))}
    for split, rows in (("train", train), ("dev", dev)):
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        qtypes = Counter(q["type"] for ex in rows for q in ex["questions"].values())
        manifest[split] = {"requests": len(rows), "questions": sum(qtypes.values()), "by_type": dict(qtypes),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{split}: {len(rows)} requests {dict(qtypes)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
