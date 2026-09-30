"""Side-by-side table of every benchmark for v3, the v5 models and live Jev (reads results/ only).

    python scripts/compare_v5.py            -> prints and writes results/compare_v5.md
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
NAMES = ["leo-1.7b-v3", "leo-4b-v5", "leo-4b-v5.1", "leo-4b-soup3", "jev-live"]
CANON = {"leo-1.7b-v3": "leo-1.7b-v3.1-canon", "leo-1.7b-v5": "leo-1.7b-v5-canon", "leo-4b-v5": "leo-4b-v5-canon", "leo-4b-v5.1": "leo-4b-v5.1-canon", "leo-4b-soup3": "leo-4b-soup3-canon", "leo-4b-soup3": "leo-4b-soup3-canon", "leo-1.7b-v5": "leo-1.7b-v5-canon"}
H = ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


rows = []


def row(label, f):
    cells = []
    for n in NAMES:
        try:
            cells.append(f(n))
        except Exception:
            cells.append("-")
    rows.append((label, cells))


row("Short suite, 211 conditions (as sent)", lambda n: f"{load(f'short/{n}.json')['summary']['all']['accuracy']:.3f}")
row("Short suite (short-condition rewrite on)", lambda n: f"{load(f'short/{CANON[n]}.json')['summary']['all']['accuracy']:.3f}")
row("Short suite ECE (as sent)", lambda n: f"{load(f'short/{n}.json')['summary']['all']['ece']:.3f}")
for t in ["all", "easy", "standard", "hard"]:
    row(f"JevBench {t}", lambda n, t=t: f"{load(f'jevbench/{n}.json')['accuracy'][t]:.3f}")
row("JevBench ECE", lambda n: f"{load(f'jevbench/{n}.json')['ece']:.3f}")
for h in H:
    row(f"Held-out {h} (acc)", lambda n, h=h: f"{load(f'bench/{n}/summary.json')[h]['accuracy']:.3f}")
row("Held-out mean accuracy", lambda n: f"{sum(load(f'bench/{n}/summary.json')[h]['accuracy'] for h in H) / 4:.3f}")
row("Held-out mean macro-F1", lambda n: f"{sum(load(f'bench/{n}/summary.json')[h]['macro_f1'] for h in H) / 4:.3f}")
row("Held-out mean ECE", lambda n: f"{sum(load(f'bench/{n}/summary.json')[h]['ece'] for h in H) / 4:.3f}")
for p in ["belebele", "mmmlu", "include", "all"]:
    row(f"Multilingual {p}", lambda n, p=p: f"{load(f'multilingual/{n}.json')['summary'][p]['accuracy']:.3f}")
row("False DONE, held-out browser screens", lambda n: f"{load(f'probes/done/{n}.json')['evidence_heldout_themes']['all']['false_done_rate']:.3f}")
row("Browser step accuracy, held-out screens", lambda n: f"{load(f'probes/done/{n}.json')['evidence_heldout_themes']['all']['step_acc']:.3f}")
row("Browser step accuracy, v3 simulator", lambda n: f"{load(f'probes/done/{n}.json')['catalog_v3_sim']['all']['step_acc']:.3f}")


def flights(n, i):
    s = load(f"probes/done/{n}.json")["flights_states"][i]
    return f"{s['p_done']:.2f} {s['choice']}"


for i, label in [(0, "Flights: unsubmitted form P(DONE)"), (1, "Flights: seat-class list P(DONE)"), (5, "Flights: blank page P(DONE)")]:
    row(label, lambda n, i=i: flights(n, i))

NC = Path("D:/Leo-API/naturalcodz/leo-compare")
NC_FILE = {"leo-1.7b-v3": "out-leo.json", "leo-1.7b-v5": "out-leo-1.7b-v5.json", "leo-4b-v5": "out-leo-4b-v5.json", "leo-4b-v5.1": "out-leo-4b-v5.1.json", "leo-4b-soup3": "out-leo-4b-soup3.json", "leo-4b-soup3": "out-leo-4b-soup3.json",
           "jev-live": "out-jev.json"}


def naturalcodz(n):
    rows_ = json.loads((NC / NC_FILE[n]).read_text(encoding="utf-8"))
    return f"{sum(r['ok'] for r in rows_)}/{len(rows_)}"


def browser(n):
    run = "final-leo-4b-v5" if n == "jev-live" else f"final-{n}"
    latest = {}
    for line in (R / "browser" / run / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task"], r["backend"], r["repeat"])] = r
    want = "jev" if n == "jev-live" else f"leo:{n}"
    rr = [r for r in latest.values() if r["backend"] == want]
    return f"{sum(r['passed'] for r in rr)}/{len(rr)}"


rows.insert(0, ("naturalcodz drop-in, 37 scenarios", [naturalcodz(n) for n in NAMES]))
row_b = []
for n in NAMES:
    try:
        row_b.append(browser(n))
    except Exception:
        row_b.append("-")
rows.insert(1, ("Browser tasks passed (7 tasks x 3)", row_b))

lines = ["| benchmark | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
lines += [f"| {l} | " + " | ".join(c) + " |" for l, c in rows]
text = "\n".join(lines)
(R / "compare_v5.md").write_text(text + "\n", encoding="utf-8")
print(text)
