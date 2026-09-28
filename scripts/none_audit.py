"""How often a none/other/unknown option is offered in a dataset's choice questions, and how often it is right.

v0's shortcut came from a none option that was correct every time it appeared (1,078 of 1,078). A healthy
mixture offers such options often and makes them wrong most of the time.

    python scripts/none_audit.py data/processed data/processed-v1 data/processed-v3
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

NONE_KEYS = {"other", "none of the above", "none", "something else", "unknown", "not stated", "no match"}


def is_right(label, key: str) -> bool:
    if isinstance(label, dict):
        return label.get(key, 0) >= max(label.values()) and label.get(key, 0) > 0
    return label == key


def audit(path: Path) -> dict:
    per_src = defaultdict(lambda: [0, 0, 0])  # choice questions, none offered, none right
    blocked = [0, 0]  # browser operation questions offering BLOCKED, BLOCKED right
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            for q in ex["questions"].values():
                if q["type"] != "choice":
                    continue
                s = per_src[ex.get("source", "?")]
                s[0] += 1
                keys = [k for k in q["criteria"] if k.strip().lower() in NONE_KEYS]
                if keys:
                    s[1] += 1
                    s[2] += any(is_right(q["label"], k) for k in keys)
                if "BLOCKED" in q["criteria"]:
                    blocked[0] += 1
                    blocked[1] += is_right(q["label"], "BLOCKED")
    tot = [sum(v[i] for v in per_src.values()) for i in range(3)]
    return {"total": tot, "per_src": dict(per_src), "blocked": blocked}


def main() -> None:
    for d in sys.argv[1:]:
        for split in ("train", "dev"):
            r = audit(Path(d) / f"{split}.jsonl")
            q, off, right = r["total"]
            print(f"{d}/{split}: {q} choice questions | none offered in {off} ({off / q:.1%}) | "
                  f"none is the answer in {right} of those ({right / max(off, 1):.0%}) | "
                  f"BLOCKED offered {r['blocked'][0]}, right {r['blocked'][1]} ({r['blocked'][1] / max(r['blocked'][0], 1):.1%})")
            if split == "train" and len(sys.argv) > 1 and d == sys.argv[-1]:
                rows = sorted(((s, *v) for s, v in r["per_src"].items() if v[1]), key=lambda x: -x[2])
                for s, n, o, k in rows:
                    print(f"      {s:<22} offered {o:>5} of {n:>6}, right {k:>5} ({k / o:.0%})")


if __name__ == "__main__":
    main()
