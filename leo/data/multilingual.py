"""Multilingual decision data from human-labelled public datasets (commercial-friendly licences only).

* MASSIVE (AmazonScience/massive, CC BY 4.0): 51 locales, 60 intents / 18 scenarios -> routing choices.
* SIB-200 (Davlan/sib200, CC BY-SA 4.0): 205 languages, 7 topics -> topic choice and yes/no.
* multilingual-sentiments (tyqiangz/multilingual-sentiments, Apache-2.0): 12 languages, 3-way sentiment
  -> choice, 3-level score and yes/no. Tweet-sourced rows are dropped: the tweet domain is held out
  (leo.bench.heldout's tweet_topic and fin_topic).

Instructions and option keys stay mostly English, the way most TypeSafe callers write them while the state
is in the user's language; a share of requests uses native-language option descriptions via the data
itself (the utterance / sentence text), never machine-translated labels.
"""
from __future__ import annotations

import csv
import random
import re
from typing import Any

from leo.data.augment import about_noul, choice_question, humanize, noul_question, qids, refer, score_question, wrap_state

PARQUET = "refs/convert/parquet"


def _req(source: str, state: Any, qs: list[dict[str, Any] | None], rng: random.Random) -> dict[str, Any] | None:
    qs = [q for q in qs if q is not None]
    if not qs:
        return None
    return {"source": source, "state": state, "questions": dict(zip(qids(len(qs), rng), qs))}


def _parquet(repo: str, path: str):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    return Dataset.from_parquet(hf_hub_download(repo, path, repo_type="dataset", revision=PARQUET))


# ------------------------------------------------------------------------------ MASSIVE

MASSIVE_LOCALES = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK", "de-DE", "el-GR", "en-US", "es-ES", "fa-IR",
    "fi-FI", "fr-FR", "he-IL", "hi-IN", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH",
    "kn-IN", "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU",
    "sl-SL", "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "zh-CN", "zh-TW",
]
SCENARIO_DESC = {
    "alarm": "alarms", "audio": "device volume", "calendar": "calendar events and reminders", "cooking": "recipes and cooking",
    "datetime": "dates, times and time zones", "email": "email and contacts", "general": "small talk and jokes",
    "iot": "smart-home devices", "lists": "to-do and shopping lists", "music": "music preferences and settings",
    "news": "news", "play": "playing music, radio, podcasts, audiobooks or games", "qa": "factual questions, maths, currency, stocks",
    "recommendation": "recommendations for places, events or movies", "social": "social media", "takeaway": "food delivery",
    "transport": "tickets, taxis, traffic and travel", "weather": "the weather",
}


def massive(rng: random.Random, n: int, split: str) -> list[dict[str, Any]]:
    per = max(1, n // len(MASSIVE_LOCALES) + 1)
    out: list[dict[str, Any]] = []
    for loc in MASSIVE_LOCALES:
        ds = _parquet("AmazonScience/massive", f"{loc}/{'train' if split == 'train' else 'validation'}/0000.parquet")
        intents, scenarios = ds.features["intent"].names, ds.features["scenario"].names
        idx = rng.sample(range(len(ds)), min(per, len(ds)))
        for i in idx:
            row = ds[i]
            intent, scen = intents[row["intent"]], scenarios[row["scenario"]]
            state, field = wrap_state(row["utt"], rng, ("utterance", "request", "message"))
            qs: list[dict[str, Any] | None] = []
            r = rng.random()
            if r < 0.55:  # fine intent, many options (routing with 10-60 labels)
                max_opts = 60 if rng.random() < 0.2 else rng.choice([10, 15, 20, 30])
                qs.append(choice_question(refer(field, rng.choice([
                    "What does the user want the assistant to do?", "Which intent does this request express?",
                    "Route this request to the matching intent.", "Classify the user's request."]), rng),
                    intents, intent, rng, max_options=max_opts, p_none=0.1))
            elif r < 0.85:
                qs.append(choice_question(refer(field, rng.choice([
                    "Which area of the assistant does this request belong to?", "Which domain handles this request?"]), rng),
                    scenarios, scen, rng, descs=SCENARIO_DESC, p_none=0.05))
            if rng.random() < 0.45:
                cand = scen if rng.random() < 0.5 else rng.choice(scenarios)
                qs.append(about_noul(SCENARIO_DESC[cand], cand == scen, rng, subject="request"))
            if rng.random() < 0.15:
                qs.append(noul_question(f"Is the request written in English?", loc == "en-US"))
            ex = _req("massive", state, qs, rng)
            if ex:
                out.append(ex)
    rng.shuffle(out)
    return out[:n]


# ------------------------------------------------------------------------------ SIB-200

SIB_DESC = {"entertainment": "films, music, arts, celebrities", "geography": "places, landscapes, countries",
            "health": "medicine, illness, fitness", "politics": "government, elections, policy",
            "science/technology": "science, research, technology", "sports": "sport and competitions",
            "travel": "travel, tourism, transport"}


def sib200(rng: random.Random, n: int, split: str, n_langs: int = 120) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files("Davlan/sib200", repo_type="dataset")
    langs = sorted({f.split("/")[1] for f in files if f.startswith("data/") and f.endswith("/train.tsv")})
    langs = rng.sample(langs, min(n_langs, len(langs)))
    per = max(1, n // len(langs) + 1)
    names = sorted(SIB_DESC)
    out: list[dict[str, Any]] = []
    for lang in langs:
        path = hf_hub_download("Davlan/sib200", f"data/{lang}/{'train' if split == 'train' else 'dev'}.tsv", repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        for row in rng.sample(rows, min(per, len(rows))):
            gold = row["category"]
            state, field = wrap_state(row["text"], rng, ("text", "sentence", "paragraph"))
            qs: list[dict[str, Any] | None] = []
            if rng.random() < 0.8:
                qs.append(choice_question(refer(field, rng.choice([
                    "What is this text about?", "Which topic does this sentence belong to?", "Pick the topic of the text."]), rng),
                    names, gold, rng, descs=SIB_DESC, p_none=0.05))
            if rng.random() < 0.5:
                cand = gold if rng.random() < 0.5 else rng.choice(names)
                qs.append(about_noul(humanize(cand), cand == gold, rng))
            ex = _req("sib200", state, qs, rng)
            if ex:
                out.append(ex)
    rng.shuffle(out)
    return out[:n]


# ------------------------------------------------------------------------------ multilingual sentiment

SENT_LANGS = ["arabic", "chinese", "english", "french", "german", "hindi", "indonesian", "italian", "japanese",
              "malay", "portuguese", "spanish"]
TWEETY = re.compile(r"(^|\s)[@#]\w|https?://|\bRT\b")


def sentiments(rng: random.Random, n: int, split: str) -> list[dict[str, Any]]:
    per = max(1, n // len(SENT_LANGS) + 1)
    out: list[dict[str, Any]] = []
    for lang in SENT_LANGS:
        ds = _parquet("tyqiangz/multilingual-sentiments", f"{lang}/{'train' if split == 'train' else 'validation'}/0000.parquet")
        names = ds.features["label"].names  # positive, neutral, negative
        keep = [i for i, (t, s) in enumerate(zip(ds["text"], ds["source"]))
                if not TWEETY.search(t) and "twitter" not in s.lower() and "tweet" not in s.lower() and 10 <= len(t) <= 1500]
        for i in rng.sample(keep, min(per, len(keep))):
            row = ds[i]
            lab = names[row["label"]]
            state, field = wrap_state(row["text"], rng, ("review", "text", "comment"))
            r = rng.random()
            qs: list[dict[str, Any] | None] = []
            if r < 0.4:
                keys = rng.choice([("positive", "neutral", "negative"), ("good", "mixed or neutral", "bad")])
                qs.append({"type": "choice", "instructions": refer(field, "What is the sentiment of this text?", rng),
                           "criteria": {k: None for k in keys}, "label": keys[names.index(lab)]})
            elif r < 0.7:
                qs.append(score_question(refer(field, "How positive is this text?", rng),
                                         ["negative", "neutral or mixed", "positive"], {"negative": 0, "neutral": 1, "positive": 2}[lab]))
            else:
                if lab == "neutral":
                    qs.append(noul_question(refer(field, "Is the writer clearly positive or clearly negative?", rng), False))
                else:
                    qs.append(noul_question(refer(field, rng.choice(["Is this text positive?", "The writer is satisfied."]), rng),
                                            lab == "positive"))
            ex = _req("ml_sentiment", state, qs, rng)
            if ex:
                out.append(ex)
    rng.shuffle(out)
    return out[:n]


# name -> (generator(rng, n, split), n_train, n_dev)
BULK = {"massive": (massive, 12000, 150), "sib200": (sib200, 7000, 100), "ml_sentiment": (sentiments, 5000, 80)}
