"""Robustness and latency probes (PLAN.md §7), run the same way on Leo and on live Jev.

* option order: rerun held-out choice questions with shuffled option order; report how often the
  top answer changes and the mean total-variation distance between the two distributions.
* question independence: ask four questions together and one at a time; report the largest
  probability difference and how many top answers changed.
* latency: model time for 1, 10 and 50 questions on a short and a longer state. For Leo this is
  in-process time on the local GPU; for Jev it is TypeSafe's server time (x-envoy-upstream-service-time),
  with the end-to-end client time recorded next to it.

    python -m leo.bench.probes --leo checkpoints/leo-0.6b-v1 --name leo-0.6b-v1          # Leo + live Jev
    python -m leo.bench.probes --leo checkpoints/leo-0.6b-v1 --name leo-0.6b-v1 --no-jev
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leo.bench import heldout

OUT = heldout.ROOT / "results" / "probes"

PARITY_QUESTIONS: dict[str, Any] = {
    "emotion": {"type": "choice", "instructions": heldout.SPECS["emotion"].instruction,
                "criteria": {l: "" for l in heldout.SPECS["emotion"].labels}},
    "happy": {"type": "noul", "instructions": "Is the writer happy?"},
    "intensity": {"type": "score", "instructions": "How strong is the emotion expressed?",
                  "criteria": ["mild or barely noticeable", "clearly present", "very strong"]},
    "topic": {"type": "choice", "instructions": "What is the text mainly about?",
              "criteria": {"work or school": None, "family or relationships": None, "health": None, "something else": None}},
}

SHORT_STATE = "i feel like i have been dragging my feet all week and nothing is getting done"
LONG_STATE = (
    "Subject: Second duplicate charge this month\n\n"
    "Hello, I am writing again about my account (customer number TS-48213). On the 3rd I was charged twice for the "
    "annual Pro plan, once at 09:12 and once at 09:14, both for the full amount. I contacted support the same day and "
    "was told the duplicate would be reversed within five business days. It has now been nine business days and the "
    "second charge is still on my card statement. Yesterday I noticed a third charge for the same plan. My bank says "
    "all three are settled payments, not holds. I have attached screenshots of the statement and of the email thread "
    "with your agent. I rely on your product for my small bakery's online orders, so I do not want to cancel, but I "
    "cannot keep floating money for charges I did not authorise. Please refund the two extra charges today and "
    "confirm in writing that it will not happen again at the next renewal. If this is not resolved by Friday I will "
    "have to dispute the charges with my bank and look at other providers. Thank you, Maria."
)


def _question(i: int) -> dict[str, Any]:
    kind = i % 3
    if kind == 0:
        return {"type": "choice", "instructions": f"Which team should handle this (routing rule {i})?",
                "criteria": {"billing": "charges, refunds", "technical": "bugs, outages", "sales": "plans, pricing", "other": None}}
    if kind == 1:
        return {"type": "noul", "instructions": f"Does the customer mention a deadline? (check {i})"}
    return {"type": "score", "instructions": f"How frustrated is the customer? (scale {i})",
            "criteria": ["calm", "frustrated but civil", "very angry"]}


def make_questions(n: int) -> dict[str, Any]:
    return {f"q{i}": _question(i) for i in range(n)}


def dist(answer: dict[str, Any]) -> list[float]:
    if answer["type"] == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    # Jev returns probabilities in no particular key order, so compare by key, not by position.
    probs = answer["probabilities"]
    return [float(probs[k]) for k in sorted(probs)]


# ------------------------------------------------------------------------------ adapters

class LeoAdapter:
    name = "leo"

    def __init__(self, path: str, dtype: str = "auto", order_views: int = 1) -> None:
        from leo.infer import Leo
        from leo.schema import to_spec

        self.leo = Leo.load(path, dtype=dtype, order_views=order_views)
        self._to_spec = to_spec

    def choice_probs(self, texts: list[str], instruction: str, labels: list[str]) -> np.ndarray:
        spec = self._to_spec("q", {"type": "choice", "instructions": instruction, "criteria": {l: "" for l in labels}})
        out = []
        for i in range(0, len(texts), 256):
            out += [p["q"] for p in self.leo.probabilities([(t, [spec]) for t in texts[i : i + 256]])]
        return np.asarray(out)

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self.leo.system_one(state, questions)

    def system_one_many(self, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.leo.system_one(b["state"], b["questions"]) for b in bodies]

    def timed(self, state: Any, questions: dict[str, Any]) -> dict[str, float]:
        t0 = time.perf_counter()
        self.leo.system_one(state, questions)
        return {"model_ms": (time.perf_counter() - t0) * 1000}


class JevAdapter:
    """Live TypeSafe Jev. Probability probes go through the on-disk cache; latency calls never do."""

    name = "jev"

    def __init__(self, model: str = "jev-latest") -> None:
        from leo.bench.jev import JevRunner

        self.runner = JevRunner(model=model)
        self._n = 0

    def _run(self, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = self.runner.run("probes", [(str(i), b) for i, b in enumerate(bodies)])
        return [by_id[str(i)] for i in range(len(bodies))]

    def choice_probs(self, texts: list[str], instruction: str, labels: list[str]) -> np.ndarray:
        q = {"q": {"type": "choice", "instructions": instruction, "criteria": {l: "" for l in labels}}}
        res = self._run([{"state": t, "questions": q} for t in texts])
        return np.asarray([[float(r["answers"]["q"]["probabilities"].get(l, 0.0)) for l in labels] for r in res])

    def system_one_many(self, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._run(bodies)

    def timed(self, state: Any, questions: dict[str, Any]) -> dict[str, float]:
        self.runner.limiter.wait()
        r = self.runner.client.system_one(state, questions)
        out = {"client_ms": float(r["client_latency_ms"])}
        if "server_ms" in r:
            out["model_ms"] = float(r["server_ms"])
        return out


# ------------------------------------------------------------------------------ probes

def order_probe(adapter, key: str, n_rows: int = 300, n_perm: int = 5, seed: int = 0) -> dict[str, Any]:
    spec = heldout.SPECS[key]
    texts = [t for t, _ in heldout.load_rows(spec, limit=n_rows)]
    labels = list(spec.labels)
    base = adapter.choice_probs(texts, spec.instruction, labels)
    rng = random.Random(seed)
    flips, tvds = [], []
    for _ in range(n_perm):
        perm = labels[:]
        rng.shuffle(perm)
        p = adapter.choice_probs(texts, spec.instruction, perm)[:, [perm.index(l) for l in labels]]
        flips.append(float((p.argmax(1) != base.argmax(1)).mean()))
        tvds.append(float(0.5 * np.abs(p - base).sum(1).mean()))
    return {"dataset": key, "n_rows": len(texts), "n_perm": n_perm, "flip_rate": float(np.mean(flips)),
            "mean_tvd": float(np.mean(tvds))}


def parity_probe(adapter, n_rows: int = 100) -> dict[str, Any]:
    texts = [t for t, _ in heldout.load_rows(heldout.SPECS["emotion"], limit=n_rows)]
    bodies = []
    for t in texts:
        bodies.append({"state": t, "questions": PARITY_QUESTIONS})
        bodies += [{"state": t, "questions": {qid: q}} for qid, q in PARITY_QUESTIONS.items()]
    res = adapter.system_one_many(bodies)
    max_diff, changed, n = 0.0, 0, 0
    step = 1 + len(PARITY_QUESTIONS)
    for i in range(len(texts)):
        together = res[i * step]["answers"]
        for j, qid in enumerate(PARITY_QUESTIONS, start=1):
            a = dist(together[qid])
            b = dist(res[i * step + j]["answers"][qid])
            max_diff = max(max_diff, max(abs(x - y) for x, y in zip(a, b)))
            changed += int(np.argmax(a) != np.argmax(b))
            n += 1
    return {"n_questions": n, "max_abs_diff": max_diff, "argmax_changes": changed}


def latency_probe(adapter, repeats: int = 20, warmup: int = 3, counts: tuple[int, ...] = (1, 10, 50)) -> list[dict[str, Any]]:
    rows = []
    for label, state in (("short", SHORT_STATE), ("long", LONG_STATE)):
        for nq in counts:
            qs = make_questions(nq)
            for _ in range(warmup):
                adapter.timed(state, qs)
            samples = [adapter.timed(state, qs) for _ in range(repeats)]
            row: dict[str, Any] = {"state": label, "questions": nq}
            for key in samples[0]:
                vals = [s[key] for s in samples if key in s]
                row[f"{key.removesuffix('_ms')}_p50_ms"] = float(np.percentile(vals, 50))
                row[f"{key.removesuffix('_ms')}_p95_ms"] = float(np.percentile(vals, 95))
            rows.append(row)
    return rows


def run_all(adapter, quick: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_rows, n_perm, n_par, reps = (8, 2, 3, 2) if quick else (300, 5, 100, 20)
    res = {
        "order": [order_probe(adapter, "emotion", n_rows, n_perm), order_probe(adapter, "fin_topic", n_rows, n_perm)],
        "parity": parity_probe(adapter, n_par),
        "latency": latency_probe(adapter, repeats=reps, warmup=1 if quick else 3, counts=(1, 3) if quick else (1, 10, 50)),
    }
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(results: dict[str, dict[str, Any]]) -> str:
    names = list(results)
    lines = ["| probe | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, key in enumerate(("emotion", "fin_topic")):
        lines.append(f"| option-order flip rate, {key} (300 rows x 5 shuffles) | "
                     + " | ".join(f"{results[n]['order'][i]['flip_rate']:.3f}" for n in names) + " |")
        lines.append(f"| mean total-variation shift, {key} | "
                     + " | ".join(f"{results[n]['order'][i]['mean_tvd']:.3f}" for n in names) + " |")
    lines.append("| 4 questions together vs one at a time: max prob. difference (100 states) | "
                 + " | ".join(f"{results[n]['parity']['max_abs_diff']:.4f}" for n in names) + " |")
    lines.append("| ... top answers that changed | " + " | ".join(str(results[n]['parity']['argmax_changes']) for n in names) + " |")
    def cell(row: dict[str, Any], kind: str) -> str:
        if f"{kind}_p50_ms" not in row:
            return "-"
        return f"{row[f'{kind}_p50_ms']:.0f} / {row[f'{kind}_p95_ms']:.0f}"

    for j, row in enumerate(results[names[0]]["latency"]):
        where = f"{row['state']} state, {row['questions']} question(s)"
        lines.append(f"| model p50 / p95 ms, {where} | " + " | ".join(cell(results[n]["latency"][j], "model") for n in names) + " |")
        if any("client_p50_ms" in results[n]["latency"][j] for n in names):
            lines.append(f"| end-to-end p50 / p95 ms incl. network, {where} | "
                         + " | ".join(cell(results[n]["latency"][j], "client") for n in names) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--leo_dtypes", default="bf16,fp32", help="comma list of Leo precisions to probe")
    ap.add_argument("--order_views", default="1", help="comma list of Leo order-view counts to probe, e.g. 1,2")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--no-jev", action="store_true", help="skip the live Jev column")
    ap.add_argument("--quick", action="store_true", help="tiny sizes to check the plumbing (results not saved)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    systems: list[tuple[str, Any]] = [
        (f"{args.name} {dt}" + (f" {v} views" if v != "1" else ""), lambda dt=dt, v=v: LeoAdapter(args.leo, dt, int(v)))
        for dt in args.leo_dtypes.split(",") for v in args.order_views.split(",")
    ]
    if not args.no_jev:
        systems.append(("Jev (live)", lambda: JevAdapter(args.jev_model)))
    results: dict[str, dict[str, Any]] = {}
    for label, factory in systems:
        adapter = factory()
        results[label] = run_all(adapter, quick=args.quick)
        print(label, json.dumps(results[label]), flush=True)
        del adapter  # drop the only reference before the next model loads
        release_memory()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    table = report(results)
    if args.quick:
        print(table)
        return
    (OUT / f"{args.name}.json").write_text(json.dumps({"device": device, "results": results}, indent=1), encoding="utf-8")
    (OUT / f"{args.name}.md").write_text(f"Device: {device}\n\n{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
