"""Does a checkpoint answer DONE only with visible evidence? Held-out probe for the false-DONE failure.

Three parts, none of them in any training set:

* evidence: leo.data.browser_evidence episodes on the two held-out themes (other seed), scored per screen kind;
* catalog: leo.data.browser catalog episodes (v3's simulator) on a new seed, to catch regressions;
* flights: the decision states from the v3 Google Flights runs (results/browser/final-leo-1.7b-v3), measured
  only; P(DONE) on each, with the page it saw. None of these pages show flight results, so DONE is wrong on all.

    python scripts/done_probe.py --leo checkpoints/leo-1.7b-v3 --name leo-1.7b-v3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leo.bench.browser import load_leo  # noqa: E402
from leo.data import browser, browser_evidence  # noqa: E402

OUT = ROOT / "results" / "probes" / "done"
TRACES = ROOT / "results" / "browser" / "final-leo-1.7b-v3" / "traces"


def strip(ex: dict) -> dict:
    return {"state": ex["state"], "questions": {k: {kk: vv for kk, vv in q.items() if kk != "label"}
                                                  for k, q in ex["questions"].items()}}


def score(leo, exs: list[dict], group) -> dict:
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i in range(0, len(exs), 16):
        chunk = exs[i:i + 16]
        for ex, resp in zip(chunk, leo.predict_many([strip(e) for e in chunk])):
            gold = ex["questions"]["operation"]["label"]
            op = resp["answers"]["operation"]
            ok = op["choice"] == gold
            tq = next((k for k in ex["questions"] if k != "operation"), None)
            if ok and tq:
                ok = resp["answers"][tq]["choice"] == ex["questions"][tq]["label"]
            p_done = op["probabilities"].get("DONE", 0.0)
            for g in ("all", group(ex)):
                s = stats[g]
                s["n"] += 1
                s["step_acc"] += ok
                if gold == "DONE":
                    s["done_gold"] += 1
                    s["done_recall"] += op["choice"] == "DONE"
                else:
                    s["neg"] += 1
                    s["false_done"] += op["choice"] == "DONE"
                    s["p_done_neg"] += p_done
    out = {}
    for g, s in sorted(stats.items()):
        out[g] = {"n": int(s["n"]), "step_acc": round(s["step_acc"] / s["n"], 4),
                  "false_done_rate": round(s["false_done"] / s["neg"], 4) if s["neg"] else None,
                  "mean_p_done_when_wrong": round(s["p_done_neg"] / s["neg"], 4) if s["neg"] else None,
                  "done_recall": round(s["done_recall"] / s["done_gold"], 4) if s["done_gold"] else None}
    return out


def flights(leo) -> list[dict]:
    rows = []
    for path in sorted(TRACES.glob("google-flights-leo_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for i, dec in enumerate(d["decisions_full"]):
            if dec.get("operation") != "DONE" and dec["operation_probabilities"].get("DONE", 0) < 0.05:
                continue  # keep the states v3 found tempting
            req = {"state": dec["state"], "questions": None}
            rows.append((path.name, i, req, dec))
    out = []
    for name, i, req, dec in rows:
        # rebuild the request exactly as jev-ultrafast sent it (the trace stores the state, not the questions)
        page = {"url": dec["state"]["page"]["url"], "title": dec["state"]["page"]["title"], "text": dec["state"]["page"]["text"]}
        body = _rebuild(dec["state"], d_goal(name), list(dec["operation_probabilities"]))
        resp = leo.system_one(body["state"], body["questions"])
        out.append({"trace": name, "decision": i, "page_text": page["text"][:80], "n_elements": len(dec["state"]["elements"]),
                    "v3_p_done": dec["operation_probabilities"].get("DONE"),
                    "p_done": resp["answers"]["operation"]["probabilities"].get("DONE"),
                    "choice": resp["answers"]["operation"]["choice"]})
    return out


def d_goal(name: str) -> str:
    return json.loads((TRACES / name).read_text(encoding="utf-8"))["goal"]


def _rebuild(state: dict, goal: str, offered: list[str]) -> dict:
    """The operation question for a traced state, with the operations it offered, in order (the trace keeps
    the state and the answer, not the request). Labels are jev-ultrafast's (model.choose, snapshot.js)."""
    from jev_ultrafast.questions import NEXT_ACTION

    labels = {"CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
              "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
              "SELECT": "Select an observed dropdown value.", "SCROLL_DOWN": "Scroll down", "SCROLL_UP": "Scroll up",
              "WAIT": "Wait for the page to update", "DONE": "Every requirement is visibly satisfied.",
              "BLOCKED": "No supported operation can progress."}
    ops = {k: labels[k] for k in offered}
    return {"state": state, "questions": {"operation": {"type": "choice", "criteria": ops,
                                                         "instructions": {"goal": goal, "rules": NEXT_ACTION}}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--dtype", default="fp32")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "external" / "jev-ultrafast"))
    leo = load_leo(a.leo, a.dtype)
    ev = browser_evidence.examples(random.Random("probe:evidence"), a.n, themes=browser_evidence.HELD_OUT_THEMES,
                                   with_kind=True)
    cat = browser.examples(random.Random("probe:catalog"), a.n // 2)
    res = {"checkpoint": a.leo,
           "evidence_heldout_themes": score(leo, ev, lambda e: e["kind"]),
           "catalog_v3_sim": score(leo, cat, lambda e: e["questions"]["operation"]["label"]),
           "flights_states": flights(leo)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for part in ("evidence_heldout_themes", "catalog_v3_sim"):
        print(part)
        for g, m in res[part].items():
            print(f"  {g:<12} {m}")
    print("flights states (v3 P(DONE) -> this checkpoint)")
    for r in res["flights_states"]:
        print(f"  {r['trace'][-8:]} #{r['decision']:<2} {r['v3_p_done']:.3f} -> {r['p_done']:.3f} {r['choice']:<10} "
              f"{r['n_elements']:>2} el  {r['page_text']!r}")


if __name__ == "__main__":
    main()
