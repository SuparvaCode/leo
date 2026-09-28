"""Blind multilingual benchmark: the same typed choice requests sent to live Jev and to Leo.

Three public evaluation sets, none used in training:

* Belebele (facebook/belebele, CC BY-SA 4.0): passage + question, 4 answers. The same (passage, question)
  pairs are used in every language, so languages are directly comparable. Passages come from FLORES, as do
  SIB-200's sentences (a training source), so any passage containing a sentence Leo trained on is dropped.
* MMMLU (openai/MMMLU, MIT): MMLU test questions professionally translated into 14 languages.
* INCLUDE (CohereLabs/include-base-44, Apache-2.0): regional exam questions written in the local language.

Every item is one /v1/systemone request: the passage or question as ``state``, one ``choice`` question whose
criteria map A-D to the answer texts. Scored by argmax, as JevBench does.

    python -m leo.bench.multilingual --backend jev --name jev-live
    python -m leo.bench.multilingual --backend leo --model checkpoints/leo-1.7b-v3 --name leo-1.7b-v3
    python -m leo.bench.multilingual --report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "multilingual"
PER_LANG = 50
SEED = 7
KEYS = ("A", "B", "C", "D")
INSTRUCTION = {"belebele": "Based on the passage, which answer to the question is correct?",
               "mmmlu": "Which answer to the question is correct?",
               "include": "Which answer to the question is correct?"}
# language label -> (Belebele code, MMMLU file, INCLUDE folder or None)
LANGS = {
    "Arabic": ("arb_Arab", "AR-XY", "Arabic"), "Bengali": ("ben_Beng", "BN-BD", "Bengali"),
    "German": ("deu_Latn", "DE-DE", "German"), "Spanish": ("spa_Latn", "ES-LA", "Spanish"),
    "French": ("fra_Latn", "FR-FR", "French"), "Hindi": ("hin_Deva", "HI-IN", "Hindi"),
    "Indonesian": ("ind_Latn", "ID-ID", "Indonesian"), "Italian": ("ita_Latn", "IT-IT", "Italian"),
    "Japanese": ("jpn_Jpan", "JA-JP", "Japanese"), "Korean": ("kor_Hang", "KO-KR", "Korean"),
    "Portuguese": ("por_Latn", "PT-BR", "Portuguese"), "Swahili": ("swh_Latn", "SW-KE", None),
    "Yoruba": ("yor_Latn", "YO-NG", None), "Chinese": ("zho_Hans", "ZH-CN", "Chinese"),
    "English": ("eng_Latn", None, None),
}


def _hf(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _trained_texts() -> list[str]:
    """SIB-200 sentences in any training set, to drop overlapping Belebele passages."""
    texts: set[str] = set()
    for d in [*ROOT.glob("data/processed*/train.jsonl"), *ROOT.glob("data/processed*/dev.jsonl")]:
        with open(d, encoding="utf-8") as fh:
            for line in fh:
                if '"source": "sib200"' not in line:
                    continue
                s = json.loads(line)["state"]
                t = s if isinstance(s, str) else next((v for v in s.values() if isinstance(v, str) and len(v) > 30), "")
                if len(t) > 30:
                    texts.add(t.strip())
    return sorted(texts)


def _item(ds: str, lang: str, iid: str, state: str, options: list[str], gold: int) -> dict[str, Any]:
    return {"id": f"{ds}:{lang}:{iid}", "dataset": ds, "lang": lang, "gold": KEYS[gold],
            "request": {"state": state, "questions": {"q": {"type": "choice", "instructions": INSTRUCTION[ds],
                                                                 "criteria": dict(zip(KEYS, options))}}}}


def load_items(per_lang: int = PER_LANG) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import pandas as pd

    rng = random.Random(SEED)
    items: list[dict[str, Any]] = []
    dropped = {"belebele_overlap_with_training": 0}

    # Belebele: pick question ids once (from English), reuse them in every language; drop trained-on passages.
    trained = _trained_texts()
    by_lang = {}
    for lang, (code, _, _) in LANGS.items():
        with open(_hf("facebook/belebele", f"data/{code}.jsonl"), encoding="utf-8") as fh:
            by_lang[lang] = {(r["link"], r["question_number"]): r for r in map(json.loads, fh)}
    common = sorted(set.intersection(*(set(v) for v in by_lang.values())))
    rng.shuffle(common)
    clean = []
    for key in common:
        if any(any(t in by_lang[lang][key]["flores_passage"] for t in trained) for lang in by_lang):
            dropped["belebele_overlap_with_training"] += 1
            continue
        clean.append(key)
        if len(clean) == per_lang:
            break
    for lang, rows in by_lang.items():
        for key in clean:
            r = rows[key]
            opts = [r[f"mc_answer{i}"] for i in range(1, 5)]
            state = {"passage": r["flores_passage"], "question": r["question"]}
            items.append(_item("belebele", lang, f"{key[1]}@{key[0].rsplit('/', 1)[-1]}", state, opts,
                               int(r["correct_answer_num"]) - 1))

    # MMMLU: the same question indices in every language.
    frames = {lang: pd.read_csv(_hf("openai/MMMLU", f"test/mmlu_{f}.csv")) for lang, (_, f, _) in LANGS.items() if f}
    n = min(len(f) for f in frames.values())
    idx = rng.sample(range(n), per_lang)
    for lang, df in frames.items():
        for i in idx:
            r = df.iloc[i]
            opts = [str(r[k]) for k in KEYS]
            if str(r["Answer"]).strip() not in KEYS or len(set(opts)) < 4:
                continue
            items.append(_item("mmmlu", lang, str(i), str(r["Question"]), opts, KEYS.index(str(r["Answer"]).strip())))

    # INCLUDE: written per language, so each language gets its own random sample.
    for lang, (_, _, folder) in LANGS.items():
        if not folder:
            continue
        df = pd.read_parquet(_hf("CohereLabs/include-base-44", f"{folder}/test-00000-of-00001.parquet"))
        for i in rng.sample(range(len(df)), min(per_lang, len(df))):
            r = df.iloc[i]
            opts = [str(r[f"option_{c}"]) for c in "abcd"]
            if len(set(opts)) < 4:
                continue
            items.append(_item("include", lang, str(i), str(r["question"]), opts, int(r["answer"])))
    return items, dropped


# ------------------------------------------------------------------------------ scoring and report

def score(items: list[dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it, a in zip(items, answers):
        probs = {k: float(v) for k, v in a["probabilities"].items()}
        total = sum(probs.values()) or 1.0
        pred = max(probs, key=probs.get)
        out.append({"id": it["id"], "dataset": it["dataset"], "lang": it["lang"], "gold": it["gold"], "pred": pred,
                    "correct": pred == it["gold"], "p_pred": probs[pred] / total, "p_gold": probs.get(it["gold"], 0.0) / total})
    return out


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    conf = np.asarray([r["p_pred"] for r in rows])
    hit = np.asarray([float(r["correct"]) for r in rows])
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi) if i else (conf >= 0) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list] = defaultdict(list)
    by_lang: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ds[r["dataset"]].append(r)
        by_lang[r["dataset"]][r["lang"]].append(r["correct"])
    return {"all": {"n": len(rows), "accuracy": float(np.mean([r["correct"] for r in rows])), "ece": ece(rows)},
            **{d: {"n": len(v), "accuracy": float(np.mean([r["correct"] for r in v])), "ece": ece(v),
                   "by_lang": {l: float(np.mean(x)) for l, x in sorted(by_lang[d].items())}} for d, v in by_ds.items()}}


def report() -> str:
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))}
    if not runs:
        return "no runs"
    names = sorted(runs, key=lambda n: (not n.startswith("jev"), n))
    lines = ["| system | Belebele | MMMLU | INCLUDE | all | ECE (all) |", "|---|---|---|---|---|---|"]
    for n in names:
        s = runs[n]["summary"]
        cell = lambda d: f"{s[d]['accuracy']:.3f}" if d in s else "-"  # noqa: E731
        lines.append(f"| {n} | {cell('belebele')} | {cell('mmmlu')} | {cell('include')} | {s['all']['accuracy']:.3f} | "
                     f"{s['all']['ece']:.3f} |")
    for d in ("belebele", "mmmlu", "include"):
        langs = sorted({l for n in names for l in runs[n]["summary"].get(d, {}).get("by_lang", {})})
        if not langs:
            continue
        lines += ["", f"{d} accuracy by language:", "", "| language | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for l in langs:
            lines.append(f"| {l} | " + " | ".join(f"{runs[n]['summary'].get(d, {}).get('by_lang', {}).get(l, float('nan')):.2f}"
                                                   for n in names) + " |")
    if len(names) > 1 and names[0].startswith("jev"):
        ref = {r["id"]: r["correct"] for r in runs[names[0]]["items"]}
        for n in names[1:]:
            pair = defaultdict(int)
            for r in runs[n]["items"]:
                if r["id"] in ref:
                    pair[(ref[r["id"]], r["correct"])] += 1
            lines += ["", f"{n} vs {names[0]} on the same items: both right {pair[(True, True)]}, only Jev "
                          f"{pair[(True, False)]}, only {n} {pair[(False, True)]}, both wrong {pair[(False, False)]}."]
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["jev", "leo"])
    ap.add_argument("--model", help="Leo checkpoint, or Jev model name (default jev-latest)")
    ap.add_argument("--name")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.backend:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        items, dropped = load_items()
        print(f"{len(items)} items; dropped {dropped}", flush=True)
        t0 = time.perf_counter()
        if args.backend == "jev":
            from leo.bench.jev import JevRunner

            runner = JevRunner(model=args.model or "jev-latest")
            by_id = runner.run("multilingual", [(it["id"], it["request"]) for it in items])
            answers = [by_id[it["id"]]["answers"]["q"] for it in items]
            model = sorted({by_id[it["id"]].get("model", "?") for it in items})
        else:
            from leo.infer import Leo

            leo = Leo.load(args.model, dtype=args.dtype, max_state_tokens=4096)
            answers = []
            for i in range(0, len(items), 128):
                answers += [r["answers"]["q"] for r in leo.predict_many([it["request"] for it in items[i:i + 128]])]
            model = [leo.name]
        rows = score(items, answers)
        summary = summarize(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        name = args.name or args.backend
        (OUT / f"{name}.json").write_text(json.dumps({"backend": args.backend, "model": model, "dropped": dropped,
                                                      "seconds": round(time.perf_counter() - t0, 1), "summary": summary,
                                                      "items": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: (v["accuracy"] if isinstance(v, dict) else v) for k, v in summary.items()}), flush=True)
    print(report())


if __name__ == "__main__":
    main()
