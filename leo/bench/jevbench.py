"""JevBench public items (231 typed decisions), run live on Jev and on Leo, compared item by item.

JevBench (github.com/fstandhartinger/jevbench, MIT) publishes 231 public decisions (easy 48, standard 72,
hard 111) in TypeSafe's wire format. Both systems get the identical request per item and are scored with
JevBench's rule: argmax over the item's label set. These items are evaluation-only.

    python -m leo.bench.jevbench --backend jev --name jev-live
    python -m leo.bench.jevbench --model checkpoints/leo-0.6b-v1 --name leo-0.6b-v1
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "jevbench"
OUT = ROOT / "results" / "jevbench"
FILES = {"easy": "easy", "original": "standard", "hard": "hard"}
TIERS = ("easy", "standard", "hard")
JEV_LIVE = "jev-live"
# Pinned upstream revision (2026-09-25) and the files read from it; --fetch downloads exactly these.
JEVBENCH_COMMIT = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"
REMOTE_FILES = {
    "datasets/public/easy.jsonl": "easy.jsonl",
    "datasets/public/original.jsonl": "original.jsonl",
    "datasets/public/hard.jsonl": "hard.jsonl",
    "results/v1.2/jevbench-v1.2-per-task.json": "jevbench-v1.2-per-task.json",
    "LICENSE": "LICENSE",
}


def fetch() -> None:
    import httpx

    SRC.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH_COMMIT}/"
    for remote, local in REMOTE_FILES.items():
        r = httpx.get(base + remote, timeout=120, follow_redirects=True)
        r.raise_for_status()
        (SRC / local).write_bytes(r.content)
        print(f"fetched {remote} ({len(r.content)} bytes)")


def load_items() -> list[dict[str, Any]]:
    items = []
    for fname, tier in FILES.items():
        with open(SRC / f"{fname}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    items.append({**json.loads(line), "tier": tier})
    return items


def jev_published() -> dict[str, bool]:
    """Jev 1.13.0's per-item outcomes as published by JevBench (their own run)."""
    data = json.loads((SRC / "jevbench-v1.2-per-task.json").read_text(encoding="utf-8"))
    return {tid: v[0] == "c" for tid, v in data["systems"]["jev-1.13.0"]["public_tasks"].items()}


def predicted_label(item: dict[str, Any], answer: dict[str, Any]) -> tuple[str | int, float]:
    """JevBench's rule (argmax over the label set) plus the probability given to that answer."""
    qtype = item["question"]["type"]
    if qtype == "noul":
        p = float(answer["noul"])
        return ("yes", p) if p > 0.5 else ("no", 1.0 - p)
    probs = answer["probabilities"]
    best = max(probs, key=lambda k: probs[k])
    total = sum(probs.values()) or 1.0
    return (answer["choice"] if qtype == "choice" else int(best)), float(probs[best]) / total


def score(items: list[dict[str, Any]], responses: list[dict[str, Any]]) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    correct, per_item = {}, []
    for it, r in zip(items, responses):
        ans = r["answers"]["q"]
        pred, p = predicted_label(it, ans)
        ok = pred == it["expected"]
        correct[it["id"]] = ok
        per_item.append({"id": it["id"], "tier": it["tier"], "family": it["family"], "type": it["question"]["type"],
                         "expected": it["expected"], "predicted": pred, "p_predicted": round(p, 4), "correct": ok,
                         "answer": ans})
    return correct, per_item


def tier_accuracy(correct: dict[str, bool], items: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for tier in TIERS + ("all",):
        ids = [it["id"] for it in items if tier == "all" or it["tier"] == tier]
        out[tier] = sum(correct[i] for i in ids) / len(ids)
    return out


def ece(per_item: list[dict[str, Any]], bins: int = 10) -> float:
    conf = np.asarray([x["p_predicted"] for x in per_item])
    hit = np.asarray([float(x["correct"]) for x in per_item])
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def load_run(name: str) -> dict[str, Any] | None:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["leo", "jev"], default="leo")
    ap.add_argument("--model", help="Leo checkpoint (leo) or Jev model name (jev, default jev-latest)")
    ap.add_argument("--name")
    ap.add_argument("--max_state_tokens", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--limit", type=int, help="smoke test: first N items per tier (results not saved)")
    ap.add_argument("--fetch", action="store_true", help=f"download the public items at commit {JEVBENCH_COMMIT[:8]}")
    args = ap.parse_args()

    if args.fetch or not (SRC / "hard.jsonl").exists():
        fetch()
    if args.backend == "leo" and not args.model:
        return
    name = args.name or (JEV_LIVE if args.backend == "jev" else None)
    if not name:
        ap.error("--name is required")

    items = load_items()
    if args.limit:
        items = [it for t in TIERS for it in [x for x in items if x["tier"] == t][: args.limit]]
    requests = [{"state": it["state"], "questions": {"q": it["question"]}} for it in items]
    t0 = time.perf_counter()
    if args.backend == "jev":
        from leo.bench.jev import JevRunner, summarize_latency

        runner = JevRunner(model=args.model or "jev-latest")
        by_id = runner.run("jevbench", [(it["id"], req) for it, req in zip(items, requests)], legacy_ok=True)
        responses = [by_id[it["id"]] for it in items]
        extra = {"jev_model": sorted({r.get("model", "?") for r in responses}), **summarize_latency(by_id)}
    else:
        from leo.infer import Leo

        leo = Leo.load(args.model, device=args.device, max_state_tokens=args.max_state_tokens, dtype=args.dtype)
        responses = leo.predict_many(requests)
        extra = {"max_state_tokens": args.max_state_tokens}
    seconds = time.perf_counter() - t0
    correct, per_item = score(items, responses)

    # Reference: Jev live if it has been run, else Jev as published by JevBench.
    ref_run = load_run(JEV_LIVE) if name != JEV_LIVE else None
    ref_name = "Jev 1.13 (live)" if ref_run else "Jev 1.13 (published by JevBench)"
    ref = {x["id"]: x["correct"] for x in ref_run["items"]} if ref_run else jev_published()
    rows = {f"{name}": correct, ref_name: ref}
    if name == JEV_LIVE:
        rows["Jev 1.13 (published by JevBench)"] = jev_published()
    table = {k: tier_accuracy(v, items) for k, v in rows.items()}
    eces = {name: ece(per_item)}
    if ref_run:
        eces[ref_name] = ece([x for x in ref_run["items"] if x["id"] in correct])
    paired = Counter((ref[i], correct[i]) for i in correct)
    fam = defaultdict(lambda: [0, 0, 0])
    for it in items:
        f = fam[f"{it['tier']}/{it['family']}"]
        f[0] += 1
        f[1] += correct[it["id"]]
        f[2] += ref[it["id"]]

    lines = ["| system | easy (48) | standard (72) | hard (111) | all (231) | ECE |", "|---|---|---|---|---|---|"]
    for k, acc in table.items():
        e = f"{eces[k]:.3f}" if k in eces else "-"
        lines.append(f"| {k} | " + " | ".join(f"{acc[t]:.3f}" for t in TIERS + ("all",)) + f" | {e} |")
    lines += ["", f"Paired with {ref_name} on the same items: both right {paired[(True, True)]}, only Jev "
              f"{paired[(True, False)]}, only {name} {paired[(False, True)]}, both wrong {paired[(False, False)]}.", "",
              f"| tier/family | n | {name} | {ref_name} |", "|---|---|---|---|"]
    for key in sorted(fam):
        n, a, j = fam[key]
        lines.append(f"| {key} | {n} | {a / n:.2f} | {j / n:.2f} |")
    report = "\n".join(lines)
    print(report)
    if args.limit:
        print(f"\nsmoke run on {len(items)} items in {seconds:.1f}s (not saved)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({
        "backend": args.backend, "model": args.model, "jevbench_commit": JEVBENCH_COMMIT, "seconds": round(seconds, 1),
        "accuracy": table[name], "ece": eces[name], **extra, "reference": ref_name,
        "paired_with_reference": {f"ref={k[0]},this={k[1]}": v for k, v in paired.items()}, "items": per_item,
    }, indent=1, default=str), encoding="utf-8")
    (OUT / f"{name}.md").write_text(report + "\n", encoding="utf-8")
    print(f"\n{len(items)} items in {seconds:.1f}s")


if __name__ == "__main__":
    main()
