"""Held-out zero-shot benchmark on four public datasets, run live against Jev and against Leo.

The four datasets, pinned revisions, instructions and bare label names follow the frozen protocol of
elcronos/jev-vs-open-decision-models: raw text as ``state``, one ``choice`` question per row, criteria
``{label: ""}``. None of these datasets is in Leo's training data.

    python -m leo.bench.heldout --backend jev --name jev-live                  # TypeSafe API, cached
    python -m leo.bench.heldout --backend leo --model checkpoints/leo-0.6b-v1 --name leo-0.6b-v1
    python -m leo.bench.heldout --report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from leo.metrics import summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "bench"


@dataclass(frozen=True)
class Spec:
    key: str
    repo: str
    revision: str
    labels: tuple[str, ...]
    instruction: str
    qid: str
    n: int
    split: str = "test"
    config: str | None = None
    file: str | None = None


SPECS: dict[str, Spec] = {
    "emotion": Spec(
        "emotion", "dair-ai/emotion", "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        ("sadness", "joy", "love", "anger", "fear", "surprise"),
        "Which single primary emotion is expressed in this text?", "emotion", 2000, config="split",
    ),
    "tweet_topic": Spec(
        "tweet_topic", "cardiffnlp/tweet_topic_single", "87b7a0d1c402dbb481db649569c556d9aa27ac05",
        ("arts & culture", "business & entrepreneurs", "pop culture", "daily life", "sports & gaming", "science & technology"),
        "Which single topic does this tweet belong to?", "topic", 1693,
        split="test_2021", file="dataset/split_temporal/test_2021.single.json",
    ),
    "fin_topic": Spec(
        "fin_topic", "zeroshot/twitter-financial-news-topic", "acbc8af2a35ccf0916124efcbe9e6cf25f191012",
        ("Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend",
         "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials",
         "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change",
         "Stock Commentary", "Stock Movement"),
        "Which single topic does this financial news tweet belong to?", "topic", 4117, split="validation",
    ),
    "daily_dialog": Spec(
        "daily_dialog", "OpenRL/daily_dialog", "1668faf0c0dc44664f108c489fd0666128db2c48",
        ("no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"),
        "Which single emotion is expressed in this utterance?", "emotion", 7740,
    ),
}

# Jev 1.13 as published by elcronos/jev-vs-open-decision-models (through OpenRouter, 2026-09-20). Kept as a
# check that the live runs here reproduce it.
PUBLISHED: dict[str, dict[str, dict[str, float]]] = {
    "Jev 1.13 (published by elcronos)": {
        "emotion": {"accuracy": 0.587, "macro_f1": 0.500, "brier": 0.667, "ece": 0.281, "nll": 2.845},
        "tweet_topic": {"accuracy": 0.793, "macro_f1": 0.694, "brier": 0.294, "ece": 0.063, "nll": 0.703},
        "fin_topic": {"accuracy": 0.670, "macro_f1": 0.630, "brier": 0.509, "ece": 0.166, "nll": 1.788},
        "daily_dialog": {"accuracy": 0.710, "macro_f1": 0.385, "brier": 0.460, "ece": 0.156, "nll": 1.451},
    },
}
MAJORITY = {"emotion": 0.348, "tweet_topic": 0.396, "fin_topic": 0.207, "daily_dialog": 0.817}


# ------------------------------------------------------------------------------ data

def load_rows(spec: Spec, limit: int | None = None) -> list[tuple[str, int]]:
    if spec.key == "tweet_topic":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(spec.repo, spec.file, repo_type="dataset", revision=spec.revision)
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rows.append((rec["text"], int(rec["label"])))
    elif spec.key == "daily_dialog":
        from datasets import load_dataset

        ds = load_dataset(spec.repo, split=spec.split, revision=spec.revision)
        rows = [(str(u), int(e)) for d, es in zip(ds["dialog"], ds["emotion"]) for u, e in zip(d, es)]
    else:
        from datasets import load_dataset

        ds = load_dataset(spec.repo, spec.config, split=spec.split, revision=spec.revision)
        rows = list(zip(ds["text"], (int(x) for x in ds["label"])))
    if limit is None and len(rows) != spec.n:
        raise RuntimeError(f"{spec.key}: {len(rows)} rows, expected {spec.n}")
    return rows[:limit] if limit else rows


def question(spec: Spec, empty: Any = "") -> dict[str, Any]:
    return {spec.qid: {"type": "choice", "instructions": spec.instruction, "criteria": {l: empty for l in spec.labels}}}


# ------------------------------------------------------------------------------ backends

Predictor = Callable[[Spec, list[str]], np.ndarray]
LAST_EXTRA: dict[str, Any] = {}  # per-dataset extras (e.g. Jev latency) picked up by run()


def leo_backend(model_path: str, batch_requests: int = 256, dtype: str = "auto") -> Predictor:
    from leo.infer import Leo
    from leo.schema import to_spec

    leo = Leo.load(model_path, dtype=dtype)

    def predict(spec: Spec, texts: list[str]) -> np.ndarray:
        qspec = to_spec(spec.qid, question(spec)[spec.qid])
        out = []
        for i in range(0, len(texts), batch_requests):
            chunk = texts[i : i + batch_requests]
            probs = leo.probabilities([(t, [qspec]) for t in chunk])
            out += [p[spec.qid] for p in probs]
        return np.asarray(out)

    return predict


def jev_backend(model: str = "jev-latest", per_second: float = 15.0) -> Predictor:
    from leo.bench.jev import JevRunner, summarize_latency

    runner = JevRunner(model=model, per_second=per_second)

    def predict(spec: Spec, texts: list[str]) -> np.ndarray:
        q = question(spec)
        responses = runner.run(f"heldout-{spec.key}", [(str(i), {"state": t, "questions": q}) for i, t in enumerate(texts)],
                               legacy_ok=True)
        probs = []
        for i in range(len(texts)):
            dist = responses[str(i)]["answers"][spec.qid]["probabilities"]
            probs.append([float(dist.get(l, 0.0)) for l in spec.labels])
        versions = sorted({r.get("model", "?") for r in responses.values()})
        LAST_EXTRA[spec.key] = {"jev_model": versions, **summarize_latency(responses),
                                "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in responses.values())}
        return np.asarray(probs)

    return predict


def remote_backend(base_url: str, model: str, api_key_env: str | None, workers: int = 8) -> Predictor:
    """Any other /v1/systemone server, e.g. a Leo server running elsewhere."""
    from leo.client import SystemOneClient

    client = SystemOneClient(base_url, os.environ.get(api_key_env) if api_key_env else None, model)

    def one(args: tuple[Spec, str]) -> list[float]:
        spec, text = args
        a = client.system_one(text, question(spec))["answers"][spec.qid]["probabilities"]
        return [float(a[l]) for l in spec.labels]

    def predict(spec: Spec, texts: list[str]) -> np.ndarray:
        with ThreadPoolExecutor(workers) as pool:
            return np.asarray(list(pool.map(one, [(spec, t) for t in texts])))

    return predict


# ------------------------------------------------------------------------------ run + report

def run(predict: Predictor, name: str, datasets: list[str], limit: int | None) -> dict[str, Any]:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for key in datasets:
        spec = SPECS[key]
        rows = load_rows(spec, limit)
        texts = [t for t, _ in rows]
        gold = np.asarray([g for _, g in rows])
        t0 = time.perf_counter()
        probs = predict(spec, texts)
        dt = time.perf_counter() - t0
        probs = np.clip(probs, 0, None)
        sums = probs.sum(1, keepdims=True)
        probs = np.where(sums > 0, probs / np.where(sums > 0, sums, 1), 1.0 / probs.shape[1])
        m = summarize(probs, gold, len(spec.labels))
        m["seconds"] = round(dt, 1)
        m["ms_per_row"] = round(1000 * dt / len(texts), 2)
        m["limit"] = limit
        m.update(LAST_EXTRA.pop(key, {}))
        summary[key] = m
        with open(out_dir / f"{key}.jsonl", "w", encoding="utf-8") as fh:
            for i, (p, g) in enumerate(zip(probs, gold)):
                fh.write(json.dumps({"i": i, "gold": int(g), "probs": [round(float(x), 6) for x in p]}) + "\n")
        print(f"{name:>24} {key:<13} n={m['n']:<5} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} "
              f"ece={m['ece']:.3f} brier={m['brier']:.3f} nll={m['nll']:.3f} ({m['ms_per_row']} ms/row)", flush=True)
    prev = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    prev.update(summary)
    (out_dir / "summary.json").write_text(json.dumps(prev, indent=2), encoding="utf-8")
    return summary


def measured_runs() -> dict[str, dict[str, Any]]:
    """Jev and Leo runs only (other systems' result folders are ignored)."""
    runs = {}
    for d in sorted(OUT.glob("*/summary.json")):
        name = d.parent.name
        if name.startswith(("jev", "leo")):
            runs[name] = json.loads(d.read_text())
    return runs


def report() -> str:
    systems: dict[str, dict[str, Any]] = dict(PUBLISHED)
    systems |= {f"{name} (measured here)": res for name, res in measured_runs().items()}
    keys = list(SPECS)
    lines = ["| system | " + " | ".join(f"{k} acc / F1 / ECE" for k in keys) + " | mean acc |",
             "|---|" + "---|" * (len(keys) + 1)]
    lines.append("| majority class | " + " | ".join(f"{MAJORITY[k]:.3f} / - / -" for k in keys) + " | - |")
    for name, res in systems.items():
        cells, accs = [], []
        for k in keys:
            m = res.get(k)
            if m and not m.get("limit"):
                cells.append(f"{m['accuracy']:.3f} / {m['macro_f1']:.3f} / {m['ece']:.3f}")
                accs.append(m["accuracy"])
            elif m:
                cells.append(f"({m['accuracy']:.3f}, n={m['n']})")
            else:
                cells.append("-")
        mean = f"{np.mean(accs):.3f}" if len(accs) == len(keys) else "-"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {mean} |")
    table = "\n".join(lines)
    (OUT / "compare.md").write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["leo", "jev", "remote"])
    ap.add_argument("--model", help="Leo checkpoint dir, or model name for jev/remote (default jev-latest)")
    ap.add_argument("--name", help="results/bench/<name>/")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key-env", default="LEO_API_KEY")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.backend:
        datasets = list(SPECS) if args.datasets == "all" else args.datasets.split(",")
        if args.backend == "leo":
            predict = leo_backend(args.model, dtype=args.dtype)
        elif args.backend == "jev":
            predict = jev_backend(args.model or "jev-latest")
        else:
            predict = remote_backend(args.base_url, args.model or "leo-latest", args.api_key_env)
        run(predict, args.name or args.backend, datasets, args.limit)
    if args.report or not args.backend:
        print(report())


if __name__ == "__main__":
    main()
