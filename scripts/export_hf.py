"""Package a trained Leo checkpoint as a self-contained Hugging Face model repo, and optionally upload it.

    # recommended release on main, with v4 shown next to it
    python scripts/export_hf.py --run checkpoints/leo-1.7b-v3 --name leo-1.7b-v3 --compare leo-1.7b-v4 \
        --out release/leo-1.7b --results-md docs/RESULTS.md --push Suparva/leo-1.7b --public
    # experimental version on a branch
    python scripts/export_hf.py --run checkpoints/leo-1.7b-v4 --name leo-1.7b-v4 --compare leo-1.7b-v3 \
        --out release/leo-1.7b-v4 --push Suparva/leo-1.7b --revision v4-experimental

The repo holds the LoRA adapter, the pointer head + marker embeddings, the calibrated config, the inference
code (leo/, so `from leo.infer import Leo` works from the repo folder) and a model card. Every number in the
card is read from results/ at export time. The base model (Qwen3) is downloaded from its own repo when loading.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
HF_REPO = "Suparva/leo-1.7b"
GITHUB = "https://github.com/SuparvaCode/leo"
JEV = "Jev 1.13 (live API)"
MAIN, EXPERIMENTAL = "leo-1.7b-v3", "leo-1.7b-v4"
BRANCH = {MAIN: "main", EXPERIMENTAL: "v4-experimental"}
CODE = ["__init__.py", "schema.py", "render.py", "encode.py", "model.py", "infer.py", "calibrate.py",
        "checkpoint.py", "serve.py", "client.py"]
LICENSES = {
    "ag_news": ("fancyzhx/ag_news", "unknown"), "dbpedia_14": ("fancyzhx/dbpedia_14", "CC BY-SA 3.0"),
    "yahoo_answers": ("community-datasets/yahoo_answers_topics", "unknown"),
    "banking77": ("legacy-datasets/banking77", "CC BY 4.0"), "clinc_oos": ("clinc/clinc_oos", "CC BY 3.0"),
    "snli": ("stanfordnlp/snli", "CC BY-SA 4.0"), "multi_nli": ("nyu-mll/multi_nli", "mixed (see dataset card)"),
    "mrpc": ("nyu-mll/glue", "other (GLUE / MRPC terms)"), "boolq": ("google/boolq", "CC BY-SA 3.0"),
    "arc_easy": ("allenai/ai2_arc", "CC BY-SA 4.0"), "arc_challenge": ("allenai/ai2_arc", "CC BY-SA 4.0"),
    "commonsense_qa": ("tau/commonsense_qa", "MIT"), "imdb": ("stanfordnlp/imdb", "other"),
    "yelp": ("Yelp/yelp_review_full", "other (Yelp dataset terms)"), "sms_spam": ("ucirvine/sms_spam", "unknown on card"),
    "prompt_injections": ("deepset/prompt-injections", "Apache-2.0"),
    "jailbreak": ("jackhhao/jailbreak-classification", "Apache-2.0"), "helpsteer2": ("nvidia/HelpSteer2", "CC BY 4.0"),
    "massive": ("AmazonScience/massive", "CC BY 4.0"), "sib200": ("Davlan/sib200", "CC BY-SA 4.0"),
    "ml_sentiment": ("tyqiangz/multilingual-sentiments", "Apache-2.0"), "mind2web": ("osunlp/Mind2Web", "CC BY 4.0"),
}
SYNTHETIC = {
    "syn_browser": "browser-agent episodes on simulated sites, in jev-ultrafast's exact request format",
    "syn_browser_evidence": "browser screens where DONE needs visible evidence (overlays, pop-overs, blank and loading "
                            "renders, stale or unsubmitted results, values a site does not offer)",
    "syn_field_reference": "questions that point at a JSON field by path",
    "syn_policy": "policy compliance checks", "syn_unknowable": "questions the state cannot answer",
    "syn_conversation": "multi-turn conversations", "syn_multi_hop": "multi-hop lookups across records",
    "syn_long_state": "a relevant fact inside long irrelevant state", "syn_injection": "instructions injected into state",
    "syn_numeric": "counting, arithmetic and date comparison", "syn_policy_exceptions": "policies with exceptions",
}
LIVE = {"wikipedia-godel", "google-flights"}
HELDOUT = ["emotion", "tweet_topic", "fin_topic", "daily_dialog"]
TIERS = [("easy", 48), ("standard", 72), ("hard", 111), ("all", 231)]


# ------------------------------------------------------------------------------ result files

def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def heldout(n: str) -> dict | None:
    return _load(R / "bench" / n / "summary.json")


def jevbench(n: str) -> dict | None:
    return _load(R / "jevbench" / f"{n}.json")


def multilingual(n: str) -> dict | None:
    return _load(R / "multilingual" / f"{n}.json")


def done_probe(n: str) -> dict | None:
    return _load(R / "probes" / "done" / f"{n}.json")


def browser_runs(n: str) -> tuple[list[dict], list[dict]]:
    """(Leo rows, Jev rows) of the matched browser session for Leo version ``n`` (latest row per task/repeat)."""
    path = R / "browser" / f"final-{n}" / "runs.jsonl"
    if not path.exists():
        return [], []
    latest: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task"], r["backend"], r["repeat"])] = r
    rows = list(latest.values())
    return [r for r in rows if r["backend"] == f"leo:{n}"], [r for r in rows if r["backend"] == "jev"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def _f(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _passed(rows: list[dict]) -> str:
    return f"{sum(r['passed'] for r in rows)}/{len(rows)}" if rows else "-"


# ------------------------------------------------------------------------------ tables

def headline(names: list[str]) -> list[str]:
    head = "| benchmark | " + " | ".join(f"**{n}**" if i == 0 else n for i, n in enumerate(names)) + f" | {JEV} |"
    lines = [head, "|---|" + "---|" * (len(names) + 1)]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    jev_b = [_passed(browser_runs(n)[1]) for n in names]
    row("Browser tasks passed (jev-ultrafast, 7 tasks x 3 runs)",
        [_passed(browser_runs(n)[0]) for n in names] + [" / ".join(dict.fromkeys(jev_b)) + " (one per matched session)"])
    row("False DONE rate, held-out browser screens (lower is better)",
        [_f((done_probe(n) or {}).get("evidence_heldout_themes", {}).get("all", {}).get("false_done_rate")) for n in names]
        + ["not measured"])
    for tier, label in (("all", "JevBench, all 231 items"), ("standard", "JevBench, standard tier"), ("hard", "JevBench, hard tier")):
        row(label, [_f((jevbench(n) or {}).get("accuracy", {}).get(tier)) for n in names + ["jev-live"]])
    cells = []
    for n in names + ["jev-live"]:
        s = heldout(n)
        cells.append(_f(_mean([s[k]["accuracy"] for k in HELDOUT])) if s else "-")
    row("Held-out classification, mean accuracy (4 datasets, zero-shot)", cells)
    row("Multilingual, 2,049 items in 15 languages (blind)",
        [_f((multilingual(n) or {}).get("summary", {}).get("all", {}).get("accuracy")) for n in names + ["jev-live"]])
    cells = []
    for n in names + ["jev-live"]:
        s = heldout(n)
        cells.append(_f(_mean([s[k]["ece"] for k in HELDOUT])) if s else "-")
    row("Calibration error (ECE), held-out mean (lower is better)", cells)
    return lines


def browser_section(names: list[str]) -> list[str]:
    sessions = [(n, *browser_runs(n)) for n in names]
    sessions = [s for s in sessions if s[1]]
    if not sessions:
        return []
    tasks = list(dict.fromkeys(r["task"] for _, leo, _ in sessions for r in leo))
    lines = ["### Browser automation", "",
             "[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) is TypeSafe's own open-source "
             "browser agent: each step sends one `/v1/systemone` request with an `operation` question and one target "
             "question per operation. The agent loop, request builder, response validation, DOM snapshot and "
             "executor were left unchanged; only the model answering the request differs. A run passes only when "
             "the agent stops with DONE **and** independent checks on the final page hold (a DONE alone is never "
             "trusted). 3 runs per task; each Leo version had its own session with Jev, the two arms alternating "
             "within each repeat (same Chrome profile, viewport, 60-action budget, 120 s limit). Five tasks use "
             "jev-ultrafast's local fixture, two use live websites.", "",
             "Cells are passed runs, then median time / model decisions.", "",
             "| task | " + " | ".join(f"{n} | Jev (same session)" for n, _, _ in sessions) + " |",
             "|---|" + "---|---|" * len(sessions)]

    def cell(rows: list[dict]) -> str:
        if not rows:
            return "-"
        return (f"{_passed(rows)} · {_median([r['elapsed_ms'] for r in rows]) / 1000:.1f} s / "
                f"{_median([r['decisions'] for r in rows])}")

    for t in tasks:
        cells = []
        for _, leo, jev in sessions:
            cells += [cell([r for r in leo if r["task"] == t]), cell([r for r in jev if r["task"] == t])]
        lines.append(f"| {t}{' (live site)' if t in LIVE else ''} | " + " | ".join(cells) + " |")
    lines.append("| **all** | " + " | ".join(f"**{_passed(leo)}** | **{_passed(jev)}**" for _, leo, jev in sessions) + " |")
    lines += ["", "How the failed runs ended:", ""]
    for n, leo, jev in sessions:
        for who, rows in ((n, leo), (f"Jev (session with {n})", jev)):
            fails = [r for r in rows if not r["passed"]]
            by: dict[tuple[str, str], int] = defaultdict(int)
            for r in fails:
                how = "hit the 60-action budget" if "budget" in (r["error"] or "") else \
                    (r["error"] or f"stopped with {r['status'].upper()}")
                by[(r["task"], how)] += 1
            for (t, how), k in by.items():
                lines.append(f"- {who}, {t}: {k} run(s) {how}")
    lines.append("")
    return lines


def done_section(names: list[str]) -> list[str]:
    probes = {n: done_probe(n) for n in names}
    if not all(probes.values()):
        return []
    kinds = [k for k in probes[names[0]]["evidence_heldout_themes"] if k != "all"]
    what = {"blank": "blank page between two renders (right move: WAIT)", "loading": "results still loading (WAIT)",
            "overlay": "a dropdown list covers the page", "popover": "a counter pop-over with its own Done button",
            "stale": "results still show the previous search", "unsubmitted": "search form not submitted yet",
            "form": "settled results page (DONE is right if it matches)", "detail": "opened item (DONE is right if it matches)"}
    lines = ["### False-DONE probe", "",
             "`scripts/done_probe.py` replays 400 simulated browser steps from two website themes that never appear in "
             "any training set, on screens where the action history can look finished while the page proves nothing.", "",
             "| screen | n | " + " | ".join(f"{n} false DONE" for n in names) + " | " + " | ".join(f"{n} step accuracy" for n in names) + " |",
             "|---|---|" + "---|" * (2 * len(names))]
    for k in kinds + ["all"]:
        ms = [probes[n]["evidence_heldout_themes"][k] for n in names]
        fd = [_f(m["false_done_rate"]) if m["false_done_rate"] is not None else "n/a" for m in ms]
        lines.append(f"| {what.get(k, '**all screens**') if k != 'all' else '**all screens**'} | {ms[0]['n']} | "
                     + " | ".join(fd) + " | " + " | ".join(_f(m["step_acc"]) for m in ms) + " |")
    lines += ["", "Real Google Flights states from leo-1.7b-v3's recorded runs, replayed as sent:", "",
              "| recorded state | " + " | ".join(f"{n} P(DONE) / answer" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    seen = set()
    for i, st in enumerate(probes[names[0]]["flights_states"]):
        key = (st["page_text"], st["n_elements"])
        if key in seen:
            continue
        seen.add(key)
        page = "blank page (0 elements)" if not st["n_elements"] else \
            f"{st['n_elements']} elements: " + st["page_text"].replace("\n", " / ")[:55]
        cells = [f"{_f(probes[n]['flights_states'][i]['p_done'])} / {probes[n]['flights_states'][i]['choice']}" for n in names]
        lines.append(f"| {page} | " + " | ".join(cells) + " |")
    lines += ["", "None of these pages show flight results, so DONE is wrong on all of them. The held-out themes come "
              "from the same simulator family as v4's new training data, so the probe is easier than real sites; the "
              "browser suite above is the real test.", ""]
    return lines


def jevbench_section(names: list[str]) -> list[str]:
    sy = [(n, jevbench(n)) for n in names] + [(JEV, jevbench("jev-live"))]
    sy = [(n, s) for n, s in sy if s]
    if len(sy) < 2:
        return []
    lines = ["### JevBench (public items)", "",
             "231 typed decisions from [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) "
             f"(commit `{sy[0][1].get('jevbench_commit', '?')[:10]}`). JevBench items were read while designing the training "
             "data, so this is not a blind score.", "",
             "| system | " + " | ".join(f"{t} ({k})" for t, k in TIERS) + " | ECE |", "|---|" + "---|" * (len(TIERS) + 1)]
    for n, s in sy:
        lines.append(f"| {n} | " + " | ".join(_f(s["accuracy"][t]) for t, _ in TIERS) + f" | {_f(s['ece'])} |")
    fam: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for n, s in sy:
        for it in s["items"]:
            fam[f"{it['tier']}/{it['family']}"][n].append(bool(it["correct"]))
    lines += ["", "<details><summary>Accuracy by task family</summary>", "",
              "| tier / family | n | " + " | ".join(n for n, _ in sy) + " |", "|---|---|" + "---|" * len(sy)]
    for k in sorted(fam):
        n_items = len(next(iter(fam[k].values())))
        lines.append(f"| {k} | {n_items} | " + " | ".join(_f(_mean(fam[k][n]), 2) if fam[k][n] else "-" for n, _ in sy) + " |")
    lines += ["", "</details>", ""]
    return lines


def heldout_section(names: list[str]) -> list[str]:
    sy = [(n, heldout(n)) for n in names] + [(JEV, heldout("jev-live"))]
    sy = [(n, s) for n, s in sy if s]
    if len(sy) < 2:
        return []
    lines = ["### Held-out classification (zero-shot)", "",
             "Four datasets whose sources and task families were never trained on, with the protocol of "
             "[elcronos/jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models): "
             "emotion (6 labels), tweet_topic (6), fin_topic (20 financial-news topics), daily_dialog (7 dialogue "
             "emotions, 82% \"no emotion\"). Cells are accuracy / macro-F1 / ECE.", "",
             "| system | " + " | ".join(HELDOUT) + " | mean accuracy |", "|---|" + "---|" * (len(HELDOUT) + 1)]
    for n, s in sy:
        cells = [f"{s[k]['accuracy']:.3f} / {s[k]['macro_f1']:.3f} / {s[k]['ece']:.3f}" for k in HELDOUT]
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {_mean([s[k]['accuracy'] for k in HELDOUT]):.3f} |")
    lines.append("")
    return lines


def multilingual_section(names: list[str]) -> list[str]:
    sy = [(n, multilingual(n)) for n in names] + [(JEV, multilingual("jev-live"))]
    sy = [(n, s) for n, s in sy if s]
    if len(sy) < 2:
        return []
    parts = ["belebele", "mmmlu", "include"]
    lines = ["### Multilingual (blind, evaluation only)", "",
             "[Belebele](https://huggingface.co/datasets/facebook/belebele) reading comprehension, "
             "[MMMLU](https://huggingface.co/datasets/openai/MMMLU) (professionally translated MMLU) and "
             "[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44) (native regional exams): 50 items "
             "per language and suite, one `choice` request each. None were used for training; Belebele passages that "
             "share text with the SIB-200 training data were dropped.", "",
             "| system | Belebele | MMMLU (knowledge) | INCLUDE (knowledge) | all | ECE |", "|---|---|---|---|---|---|"]
    for n, s in sy:
        m = s["summary"]
        lines.append(f"| {n} | " + " | ".join(_f(m[p]["accuracy"]) for p in parts) + f" | {_f(m['all']['accuracy'])} | {_f(m['all']['ece'])} |")
    for p in parts:
        langs = list(sy[-1][1]["summary"][p].get("by_lang", {}))
        if not langs:
            continue
        lines += ["", f"<details><summary>{p} accuracy by language</summary>", "",
                  "| language | " + " | ".join(n for n, _ in sy) + " |", "|---|" + "---|" * len(sy)]
        for lang in langs:
            lines.append(f"| {lang} | " + " | ".join(_f(s["summary"][p]["by_lang"].get(lang), 2) for _, s in sy) + " |")
        lines += ["", "</details>"]
    lines.append("")
    return lines


def probes_section(name: str) -> list[str]:
    md = R / "probes" / f"{name}.md"
    if not md.exists():
        return []
    text = md.read_text(encoding="utf-8").splitlines()
    table = [l for l in text if l.startswith("|")]
    device = next((l for l in text if l.startswith("Device:")), "Device: local GPU").removeprefix("Device: ")
    return [f"### Robustness and speed ({name})", "",
            "Option-order sensitivity (how often the top answer changes when the options are shuffled), whether "
            f"questions sharing a request influence each other, and latency. Leo was timed on one {device} "
            "(8 GB laptop GPU). Jev's model time is the server time TypeSafe reports; its end-to-end time includes the "
            "network round trip from this machine. \"2 views\" is `--order-views 2`.", "", *table, ""]


def results_markdown(names: list[str]) -> str:
    out = headline(names) + [""] + browser_section(names) + done_section(names) + jevbench_section(names)
    out += heldout_section(names) + multilingual_section(names) + probes_section(names[0])
    return "\n".join(out)


# ------------------------------------------------------------------------------ model card

def _elapsed_min(run: Path) -> float | None:
    log = run / "train_log.jsonl"
    if not log.exists():
        return None
    last = None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = rec.get("elapsed_min", last)
    return last


def v4_story(v3: str = MAIN, v4: str = EXPERIMENTAL) -> list[str]:
    p3, p4 = done_probe(v3), done_probe(v4)
    b3, b4 = browser_runs(v3)[0], browser_runs(v4)[0]
    j3, j4 = jevbench(v3), jevbench(v4)
    if not (p3 and p4 and b3 and b4 and j3 and j4):
        return []
    seat3 = next(s for s in p3["flights_states"] if s["n_elements"] == 4)
    seat4 = next(s for s in p4["flights_states"] if s["n_elements"] == 4)
    fl4 = [r for r in b4 if r["task"] == "google-flights"]
    research4 = [r for r in b4 if r["task"].startswith("research")]
    return [
        "## The v4 experiment: fixing false DONE", "",
        f"**Problem in v3.** {v3} declares a browser task DONE once its action history covers every part of the goal, "
        "whatever the page shows. On Google Flights it stopped with DONE in all three runs without completing the search, "
        f"with P(DONE) = {_f(seat3['p_done'])} while only a seat-class list was on screen, and 0.999 on a blank render.", "",
        f"**What v4 changed.** {v4} continues v3's weights (1 epoch, half the learning rate) on 30,000 replayed v3 "
        "requests plus 9,000 steps from a new simulator family (`leo/data/browser_evidence.py`) in which DONE is only "
        "right with visible evidence: close the open dropdown, wait out blank or loading renders, resubmit when results "
        "show an older search, report BLOCKED when a requested value is not offered.", "",
        "**What happened.**", "",
        f"- The targeted flaw is fixed in simulation and largely on the real site. The false-DONE rate on held-out screens "
        f"fell from {_f(p3['evidence_heldout_themes']['all']['false_done_rate'])} to "
        f"{_f(p4['evidence_heldout_themes']['all']['false_done_rate'])}. On live Google Flights, v4 no longer claims success: "
        f"all {len(fl4)} runs ended with an honest {', '.join(sorted({r['status'].upper() for r in fl4}))} "
        f"(the seat-class state that fooled v3 went from P(DONE) {_f(seat3['p_done'])} to {_f(seat4['p_done'])}).",
        f"- It introduced a new failure. On jev-ultrafast's two reading-room tasks (open one article, no search) v4 "
        "opens the right article and then goes back to the list, over and over: "
        f"{_passed(research4)} passed, " + ("every run hit the 60-action budget" if all("budget" in (r["error"] or "") for r in research4)
                                            else "most runs hit the 60-action budget") +
        f". The likely cause: every goal in the new family includes a search, so an item page reached without a "
        f"search in the history always meant \"go back\" in training. Browser total: {_passed(b3)} for v3, "
        f"{_passed(b4)} for v4.",
        f"- JevBench fell from {_f(j3['accuracy']['all'])} to {_f(j4['accuracy']['all'])} (standard tier "
        f"{_f(j3['accuracy']['standard'])} to {_f(j4['accuracy']['standard'])}), mostly policy and trap items where v4 "
        "now answers \"yes\" or skips \"other\"/\"unknown\". Held-out classification and multilingual accuracy are "
        "unchanged within noise.", "",
        f"**Decision.** {v3} stays the recommended model on `main`. {v4} is published on the `v4-experimental` branch "
        "for research and for agent loops that need an honest BLOCKED more than they need open-an-item tasks. The fix "
        "for the next version is to mix goals without a search step into the evidence family and to re-check the "
        "reading-room tasks before release.", "",
    ]


def model_card(name: str, cfg: dict, run: Path, names: list[str], repo: str) -> str:
    main_cfg = json.loads((ROOT / "checkpoints" / MAIN / "final" / "leo_config.json").read_text(encoding="utf-8"))
    man = main_cfg.get("data_manifest") or {}
    srcs = man.get("sources", {})
    real = [(k, *LICENSES[k], v["n_train"]) for k, v in srcs.items() if k in LICENSES]
    syn = [(k, v["n_train"]) for k, v in srcs.items() if k not in LICENSES]
    tr, tr3 = cfg.get("train", {}), main_cfg.get("train", {})
    dev_cal = cfg.get("dev", {}).get("calibrated", {})
    v4man = json.loads((ROOT / "data" / "processed-v4" / "manifest.json").read_text(encoding="utf-8")) \
        if (ROOT / "data" / "processed-v4" / "manifest.json").exists() else {}
    is_main = name == MAIN
    branch = BRANCH.get(name, "main")
    gpu_h = sum(h for h in (_elapsed_min(ROOT / "checkpoints" / MAIN), None if is_main else _elapsed_min(run)) if h) / 60 * 2
    langs = ["en", "de", "fr", "es", "pt", "it", "nl", "pl", "ru", "ar", "hi", "bn", "zh", "ja", "ko", "id", "tr", "vi",
             "th", "sw", "multilingual"]
    datasets = sorted({r for r, _ in LICENSES.values()})
    jb, jbj = jevbench(name) or {}, jevbench("jev-live") or {}
    ml = (multilingual(name) or {}).get("summary", {})
    leo_b, jev_b = browser_runs(name)
    flights = [r for r in leo_b if r["task"] == "google-flights"]
    fl_status = ", ".join(sorted({r["status"].upper() for r in flights})) or "not run"
    load_rev = "" if is_main else f", revision=\"{branch}\""

    L = ["---", "license: apache-2.0", f"base_model: {cfg['base_model']}", "pipeline_tag: text-classification",
         "inference: false", "language:", *[f"- {l}" for l in langs], "datasets:", *[f"- {d}" for d in datasets],
         "tags:", "- decision-model", "- calibrated-probabilities", "- zero-shot-classification", "- browser-agent",
         "- lora", "- qwen3", "- typesafe-compatible", "---", "", f"# Leo 1.7B ({name})", ""]
    if not is_main:
        L += [f"> **Experimental branch.** This is `{branch}`. It fixes false DONE on browser tasks but regresses on "
              f"other ones (see *The v4 experiment*). For general use take `main` ({MAIN}).", ""]
    L += [
        "**Leo is an open-weight decision model.** You send a *state* (text or JSON) and typed questions (`choice`, "
        "`score`, or `noul` yes/no). It returns a calibrated probability distribution for every question from a single "
        "forward pass, with no text generation. Typical uses: routing, classification, moderation triage, grading on a "
        "rubric, policy checks, and the next-action choices of an agent.", "",
        "The request and response format follows the `POST /v1/systemone` API that TypeSafe documents for its Jev "
        "models, so existing clients work after a base-URL change. Leo is an independent project: it is **not "
        "affiliated with TypeSafe**, contains none of its weights, and was **not trained on Jev outputs**. Jev was called "
        "only to score it next to Leo on identical requests.", "",
        f"- **Code, training pipeline, benchmarks:** [{GITHUB.removeprefix('https://')}]({GITHUB})",
        f"- **Versions in this repo:** `main` = {MAIN} (recommended), `v4-experimental` = {EXPERIMENTAL}", "",
        "## Model details", "", "| | |", "|---|---|",
        "| Developer | Suparva Baranwal |",
        f"| Version | {name}" + ("" if is_main else f" (continues {MAIN})") + " |",
        "| Model type | Decoder-only transformer run prefill-only, LoRA adapter, listwise pointer head |",
        f"| Base model | [`{cfg['base_model']}`](https://huggingface.co/{cfg['base_model']}) (Apache-2.0), revision `{cfg.get('base_revision', '?')[:10]}` |",
        f"| Parameters | about 1.7B in the base; 20.3M trained (LoRA r{cfg['lora']['r']} on every attention and MLP projection, plus markers and head) |",
        "| Files | `adapter/` (LoRA, 70 MB), `leo_head.safetensors` (head and marker embeddings, 12 MB), `leo_config.json` (calibration and metadata), `leo/` (inference code) |",
        "| Inputs | a state (string, JSON object or array) and many typed questions per request |",
        "| Outputs | per question: `choice` + `probabilities` + `confidence`, or `score` + `probabilities` + `legend`, or `noul` = P(yes) |",
        "| Context | trained on states up to 2,048 tokens; browser states up to about 12k tokens were used in evaluation |",
        "| Languages | English plus 50+ languages via MASSIVE, SIB-200 and multilingual sentiment data; much weaker on low-resource languages |",
        "| Licence | Apache-2.0 (weights and code); training datasets keep their own terms |", "",
        "## Results at a glance", "",
        f"Measured on one machine, with {JEV} (`jev-1.13.0`, September 2026) answering exactly the same requests. "
        "Per-task tables and the protocol follow further down.", "", *headline(names), "",
    ]
    if is_main:
        L += [f"In short: {MAIN} ties Jev on the browser suite ({_passed(leo_b)} vs {_passed(jev_b)}), is close on "
              "JevBench's easy and standard tiers, and beats Jev on tweet topics. It is clearly behind on hard "
              "reasoning (long policies, ambiguity, trade-offs, multi-hop), knowledge-heavy exams and low-resource "
              "languages. Most of that gap comes from the size of the 1.7B base model.", ""]
    L += v4_story()
    L += [
        "## Quick start", "",
        "```bash", "pip install torch transformers peft safetensors numpy pydantic huggingface_hub", "```", "",
        "```python", "import sys", "from huggingface_hub import snapshot_download", "",
        f"path = snapshot_download(\"{repo}\"{load_rev})" + ("   # revision=\"v4-experimental\" for v4" if is_main else ""),
        "sys.path.insert(0, path)            # the repo ships its own inference code in leo/",
        "from leo.infer import Leo", "",
        "leo = Leo.load(path, dtype=\"bf16\")  # use \"fp32\" on CPU; the Qwen3 base is fetched on first use",
        "out = leo.system_one(", "    EXAMPLE_STATE,", "    EXAMPLE_QUESTIONS,", ")", "print(out)", "```", "",
        "__EXAMPLE__", "",
        "`leo.predict_many([{\"state\": ..., \"questions\": ...}, ...])` batches many requests.", "",
        "### HTTP server", "",
        "```bash", "pip install fastapi uvicorn httpx", "cd <snapshot path>",
        "python -m leo.serve --model . --port 8000                          # binds 127.0.0.1",
        "LEO_API_KEY=change-me python -m leo.serve --model . --host 0.0.0.0 # a key is required off loopback",
        "```", "",
        "`POST /v1/systemone` and `GET /v1/models` follow TypeSafe's API: bearer auth whenever `LEO_API_KEY` is set, "
        "422 on invalid requests, body-size and question-count limits. The server refuses to bind a public address "
        "without a key. Flags:", "",
        "- `--order-views 2` averages each choice over two option orders: about half the option-order sensitivity, "
        "for 1.4 to 2.8 times the latency.",
        "- `--dtype fp32` gives answers that do not depend on which other questions share the request (bf16 moves "
        "probabilities by up to about 0.02).",
        "- `--precise` returns 4-decimal probabilities plus `latency_ms` instead of the TypeSafe-exact shape.", "",
        "`leo.client.SystemOneClient` talks to a Leo server or any other `/v1/systemone` endpoint.", "",
        "### Question types", "", "| type | `criteria` | answer |", "|---|---|---|",
        "| `choice` | object mapping option key to a description (string, object, array or null), 2 to 255 options | `choice`, `probabilities` over the keys, `confidence` = (K·p_max − 1)/(K − 1) |",
        "| `score` | ordered array of 2 to 10 level descriptions | `score` = Σ i·pᵢ, `probabilities`, `confidence`, `legend` |",
        "| `noul` | optional `{\"true\": ..., \"false\": ...}` | `noul` = P(yes) |", "",
        "Instructions can be a string or any JSON value and can refer to parts of the state by path, such as "
        "`` `ticket.body` ``. Question IDs are never shown to the model. Questions cannot see each other, so adding or "
        "removing one does not change the others.", "",
        "## How it works", "",
        f"- **Backbone:** `{cfg['base_model']}` run prefill-only (no decoding), adapted with LoRA r{cfg['lora']['r']} / alpha {cfg['lora']['alpha']}.",
        "- **Packed layout:** the state is encoded once; each question follows it in the same sequence under a "
        "block-causal mask, with position ids restarting after the state. Questions share the state encoding but "
        "cannot attend to each other.",
        "- **Reserved markers:** state, question, option and decision boundaries are trainable embeddings outside the "
        "vocabulary, so text inside the state cannot forge them.",
        "- **Listwise pointer readout:** a small head scores each option's end marker against the question's decision "
        "marker, which comes after the full option list, so options are judged together.",
        "- **Proper-scoring-rule training:** log loss against the label distribution plus a ranked probability score "
        "term for `score` questions.",
        f"- **Calibration:** one temperature per question type and option-count bucket, fitted on dev data (dev top-label "
        f"ECE {_f(dev_cal.get('ece_top'))}). Confidence uses the formulas TypeSafe publishes.", "",
        "## Evaluation", "",
        "- Every Jev number comes from live calls on byte-identical requests (same state, instructions, option keys "
        "and option order), September 2026. Jev responses were used only for scoring, never for training, prompt "
        "tuning or checkpoint selection. Checkpoints were selected on dev loss alone.",
        "- Held-out classification and the multilingual suite are blind: none of those datasets or task families were "
        "trained on. JevBench items were read while designing the data, so treat that score as seen.",
        "- The browser suite runs TypeSafe's own agent unchanged and checks the final page independently. Live sites "
        "(Wikipedia, Google Flights) can change between sessions; Jev's own Google Flights result differed between "
        "the two sessions.",
        "- TYPE_TEXT values in the browser suite come from the same local Qwen3-1.7B helper for both arms.", "",
        results_markdown(names),
        "## Limitations and known flaws", "",
        "Please read these before relying on Leo.", "",
    ]
    if is_main:
        L.append(f"- **False DONE in browser agents.** On Google Flights, {name} stopped with {fl_status} in "
                 f"{len(flights)} of {len(flights)} runs without completing the search. It can declare a task done when "
                 "its action history looks complete even if the page shows no evidence (an open dropdown, a blank "
                 "render). Always verify outcomes independently, as jev-ultrafast recommends, and never let a DONE "
                 "trigger anything irreversible. The v4 experiment above targets this.")
    else:
        L.append("- **Never says DONE on item pages reached without a search.** v4 loops back to the list on "
                 "\"open this article\" tasks until the action budget runs out (see *The v4 experiment*). Use `main` "
                 "unless you need its more honest BLOCKED behaviour.")
        L.append("- **More \"yes\" on policy and trap items, less \"other\"/\"unknown\"** than v3 (JevBench standard "
                 f"tier {_f(jb.get('accuracy', {}).get('standard'))} vs {_f((jevbench(MAIN) or {}).get('accuracy', {}).get('standard'))}).")
    L += [
        f"- **Hard reasoning.** JevBench hard tier: {_f(jb.get('accuracy', {}).get('hard'))} vs Jev "
        f"{_f(jbj.get('accuracy', {}).get('hard'))}. Weakest on ambiguous items, long policies with exceptions, "
        "trade-offs and multi-hop lookups.",
        f"- **World knowledge.** MMMLU {_f(ml.get('mmmlu', {}).get('accuracy'))} and INCLUDE "
        f"{_f(ml.get('include', {}).get('accuracy'))}, far below Jev. Do not use it as a knowledge source; put the facts "
        "it needs into the state.",
        "- **Low-resource languages.** Accuracy drops steeply outside the major languages (Yoruba and Swahili are near "
        "chance on several suites). Test on your language before deploying.",
        "- **Option order.** The top answer changes more often than Jev's when options are shuffled, especially with "
        "many similar options. Use `--order-views 2` when that matters.",
        "- **Catch-all labels** such as \"other\" or \"no emotion\" are picked less often than they are right (low "
        "macro-F1 on daily_dialog).",
        "- **Calibration away from home.** Dev ECE is about 0.01; held-out ECE ranges from about 0.03 to 0.17. Re-fit "
        "the temperatures on a few hundred labelled examples of your own traffic before thresholding on probabilities.",
        "- **`score` questions** are the weakest type (dev accuracy about 0.65); treat the expected score as a soft signal.",
        "- **Many questions per request** cost more time on Leo than on Jev's servers (see the latency table).",
        "- **Long states** beyond the 2,048 training tokens work but are less tested.", "",
        "## Intended use", "",
        "Good fits: ticket and message routing, intent and topic classification, moderation triage, policy checks with "
        "the policy in the state, yes/no checks over documents, rubric grading, next-action choices for agents that "
        "verify outcomes, research on calibrated decision models, and a local, private backend for `/v1/systemone` "
        "clients.", "",
        "Out of scope: decisions about people's health, legal status, credit, housing or employment without qualified "
        "human review; sole-filter safety moderation; questions that need world knowledge the state does not contain; "
        "unattended agents that can spend money or change accounts.", "",
        "## Training", "",
        f"- **{MAIN}:** {man.get('train', {}).get('requests', 0):,} requests ({man.get('train', {}).get('questions', 0):,} "
        f"questions), {tr3.get('epochs')} epochs, 6,556 optimizer steps, learning rate {tr3.get('lr')}, from `{main_cfg['base_model']}`.",
    ]
    if v4man:
        L.append(f"- **{EXPERIMENTAL}:** warm start from {MAIN}; {v4man['train']['requests']:,} requests "
                 f"({v4man['replay_requests']:,} replayed from v3's training set, "
                 f"{v4man['new_sources']['syn_browser_evidence']['n_train']:,} new evidence-family steps), 1 epoch, learning "
                 "rate 1e-4; released checkpoint = step 1,200 of 1,911, chosen by calibrated dev loss.")
    L += [
        f"- fp16 mixed precision, data-parallel on 2x NVIDIA T4 (Kaggle). States cut at {tr3.get('train_state_tokens')} tokens, "
        f"packed rows at {tr3.get('max_row_tokens')} tokens. About {gpu_h:.0f} T4 GPU-hours"
        + ("" if is_main else " including v3") + ".",
        "- Option order shuffled; option keys and descriptions varied (bare, described, snake_case, letters, numbers); "
        "none/other options appear both when right and when wrong, so they are not a shortcut.", "",
        "### Training data", "",
        "Human-labelled public datasets (check each licence before commercial use; some are unknown or custom):", "",
        "| source | dataset | licence | requests |", "|---|---|---|---|",
        *[f"| {k} | [`{r}`](https://huggingface.co/datasets/{r}) | {lic} | {n:,} |" for k, r, lic, n in real], "",
        "Code-generated families (labels computed by code, not by a model):", "",
        "| family | what it teaches | requests |", "|---|---|---|",
        *[f"| {k} | {SYNTHETIC.get(k, '')} | {n:,} |" for k, n in syn],
    ]
    if v4man:
        L.append(f"| syn_browser_evidence (v4 only) | {SYNTHETIC['syn_browser_evidence']} | "
                 f"{v4man['new_sources']['syn_browser_evidence']['n_train']:,} |")
    L += [
        "", "Never used for training: any output of TypeSafe's models, the four held-out datasets, Belebele, MMMLU, "
        "INCLUDE, JevBench, and the sites and goals of the browser benchmark.", "",
        "## Versions", "", "| version | branch | notes |", "|---|---|---|",
        f"| {MAIN} | `main` | recommended |",
        f"| {EXPERIMENTAL} | `v4-experimental` | honest BLOCKED instead of false DONE; fails open-an-item tasks; lower JevBench |",
        "| leo-0.6b-v0 / v1 | not released | Qwen3-0.6B prototypes |", "",
        "## Licence", "",
        "Weights and code: Apache-2.0; the Qwen3 base is Apache-2.0 too. The training mix includes datasets under "
        "CC BY-SA, custom and unknown terms (listed above). For a clean licence chain, retrain on the permissive subset "
        "with the pipeline on GitHub.", "",
        "## Citation", "", "```bibtex", "@misc{leo2026,",
        "  title  = {Leo: an open-weight decision model with calibrated typed answers},",
        "  author = {Baranwal, Suparva},", "  year   = {2026},",
        f"  url    = {{https://huggingface.co/{repo}}},", f"  note   = {{Code: {GITHUB}}}", "}", "```", "",
        "## Acknowledgements", "",
        "[Qwen](https://huggingface.co/Qwen) for the base model; "
        "[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast), "
        "[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) and "
        "[elcronos/jev-vs-open-decision-models](https://github.com/elcronos/jev-vs-open-decision-models) for the "
        "benchmarks; TypeSafe for documenting the `/v1/systemone` format publicly.", "",
    ]
    return "\n".join(L)


EXAMPLE_STATE = {"ticket": {"subject": "Charged twice", "body": "I was billed twice this month. Please fix it today."}}
EXAMPLE_QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team should handle `ticket`?",
              "criteria": {"billing": "charges, refunds, invoices", "technical": "bugs, outages", "other": None}},
    "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"},
    "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "very angry"]},
}


def example_block(out: Path) -> str:
    """Run the quick-start request on the exported folder itself, so the card shows real output."""
    import subprocess
    import sys

    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); from leo.infer import Leo; "
            "leo = Leo.load(sys.argv[1], dtype='bf16'); "
            "print('JSON=' + json.dumps(leo.system_one(json.loads(sys.argv[2]), json.loads(sys.argv[3]))))")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code, str(out), json.dumps(EXAMPLE_STATE), json.dumps(EXAMPLE_QUESTIONS)],
                       capture_output=True, text=True, encoding="utf-8", cwd=str(out.parent), env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith("JSON=")), None)
    if not line:
        raise SystemExit(f"quick-start example failed on the exported folder:\n{r.stderr[-2000:]}")
    return ("Real output of this request (the exported folder, loaded on its own). The Python API returns 4 decimals; "
            "the HTTP server returns TypeSafe's exact shape by default (2 decimals summing to exactly 1).\n\n```json\n") + \
        json.dumps(json.loads(line[5:]), indent=2, ensure_ascii=False) + "\n```"


def fill_example(card: str, block: str) -> str:
    fmt = lambda o: json.dumps(o, ensure_ascii=False, indent=4).replace("null", "None").replace("\n", "\n    ")  # noqa: E731
    return card.replace("EXAMPLE_STATE", fmt(EXAMPLE_STATE)).replace("EXAMPLE_QUESTIONS", fmt(EXAMPLE_QUESTIONS)) \
        .replace("__EXAMPLE__", block)


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="", help="comma list of other Leo result names shown next to this one")
    ap.add_argument("--results-md", help="also write the results section to this file (e.g. docs/RESULTS.md)")
    ap.add_argument("--no-example", action="store_true", help="skip running the quick-start example")
    ap.add_argument("--push", help="Hugging Face repo id to upload to")
    ap.add_argument("--revision", default="main", help="branch to upload to (created if missing)")
    ap.add_argument("--public", action="store_true", help="make the Hugging Face repo public")
    a = ap.parse_args()
    import sys

    sys.path.insert(0, str(ROOT))
    from leo.checkpoint import resolve_checkpoint

    run = ROOT / a.run
    src = resolve_checkpoint(run)
    names = [a.name] + [n for n in a.compare.split(",") if n]
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(src / "adapter", out / "adapter", ignore=shutil.ignore_patterns("README.md"))
    shutil.copy2(src / "leo_head.safetensors", out / "leo_head.safetensors")
    cfg = json.loads((src / "leo_config.json").read_text(encoding="utf-8"))
    cfg["name"] = a.name
    (out / "leo_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "leo").mkdir()
    for f in CODE:
        shutil.copy2(ROOT / "leo" / f, out / "leo" / f)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    (out / "requirements.txt").write_text("torch>=2.6\ntransformers>=5.0\npeft>=0.15\nsafetensors\nnumpy\npydantic>=2\n"
                                          "huggingface_hub\nfastapi\nuvicorn\nhttpx\n", encoding="utf-8")
    repo = a.push or HF_REPO
    card = model_card(a.name, cfg, run, names, repo)
    block = "" if a.no_example else example_block(out)
    (out / "README.md").write_text(fill_example(card, block), encoding="utf-8")
    if a.results_md:
        text = (f"# Leo benchmark results\n\nGenerated by `scripts/export_hf.py` from `results/`. Weights: "
                f"[{repo}](https://huggingface.co/{repo}).\n\n" + results_markdown(names) + "\n")
        (ROOT / a.results_md).write_text(text, encoding="utf-8")
    print(f"exported {src} -> {out}")
    if a.push:
        from huggingface_hub import HfApi

        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.push, exist_ok=True, private=not a.public)
        if a.revision != "main":
            api.create_branch(a.push, branch=a.revision, exist_ok=True)
        api.upload_folder(repo_id=a.push, folder_path=str(out), revision=a.revision, commit_message=a.name)
        if a.public:
            api.update_repo_settings(a.push, private=False)
        print(f"uploaded to https://huggingface.co/{a.push}/tree/{a.revision}")


if __name__ == "__main__":
    main()
