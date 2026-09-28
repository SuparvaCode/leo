"""Check which Hugging Face models/datasets Leo depends on exist, are gated, and what licence they carry.

Uses only the public Hub metadata API (no token needed). A gated repo needs you to accept its terms on
huggingface.co and then `hf auth login` (or HF_TOKEN) before download.

    python scripts/check_hf_access.py            # prints a table, writes results/hf_access.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    # attention-only bases (packed block-causal mode works; Windows-friendly)
    "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-8B-Base",
    # hybrid Gated-DeltaNet bases (rows mode; need flash-linear-attention for speed)
    "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.5-35B-A3B-Base",
    # encoders
    "answerdotai/ModernBERT-large", "jhu-clsp/mmBERT-base",
    # other open decision-model checkpoints (access check only)
    "jaredpalmer/kev-0.6b", "jaredpalmer/kev-4b", "Mapika/decider-2b", "Mapika/decider-4b",
    # examples of gated families
    "google/gemma-3-1b-pt", "meta-llama/Llama-3.2-1B",
]

DATASETS = {
    "train": [
        "fancyzhx/ag_news", "fancyzhx/dbpedia_14", "community-datasets/yahoo_answers_topics",
        "legacy-datasets/banking77", "clinc/clinc_oos", "CogComp/trec", "AmazonScience/massive",
        "stanfordnlp/snli", "nyu-mll/multi_nli", "facebook/anli", "google/boolq", "nyu-mll/glue",
        "stanfordnlp/imdb", "Yelp/yelp_review_full", "ucirvine/sms_spam", "deepset/prompt-injections",
        "jackhhao/jailbreak-classification", "google/civil_comments", "nvidia/HelpSteer2",
        "allenai/ai2_arc", "allenai/openbookqa", "tau/commonsense_qa", "Rowan/hellaswag",
        "allenai/winogrande", "PKU-Alignment/BeaverTails", "lmsys/toxic-chat", "allenai/wildguardmix",
    ],
    "heldout_eval": [
        "dair-ai/emotion", "cardiffnlp/tweet_topic_single", "zeroshot/twitter-financial-news-topic",
        "OpenRL/daily_dialog", "SetFit/sst5", "cais/mmlu", "allenai/sciq",
        "google-research-datasets/paws", "cardiffnlp/tweet_eval",
    ],
}


def _info(kind: str, repo: str, client: httpx.Client) -> dict:
    url = f"https://huggingface.co/api/{kind}/{repo}"
    try:
        r = client.get(url)
    except httpx.HTTPError as e:  # network problem: report, don't crash the whole table
        return {"repo": repo, "kind": kind, "status": "error", "detail": str(e)}
    if r.status_code == 404:
        return {"repo": repo, "kind": kind, "status": "missing"}
    if r.status_code in (401, 403):
        return {"repo": repo, "kind": kind, "status": "private_or_auth"}
    if not r.is_success:
        return {"repo": repo, "kind": kind, "status": f"http_{r.status_code}"}
    j = r.json()
    card = j.get("cardData") or {}
    lic = card.get("license") or next((t.split(":", 1)[1] for t in j.get("tags", []) if t.startswith("license:")), None)
    gated = j.get("gated", False)
    return {
        "repo": repo,
        "kind": kind,
        "status": "ok",
        "gated": gated if gated else False,
        "license": lic,
        "downloads": j.get("downloads"),
        "last_modified": j.get("lastModified"),
    }


def main() -> int:
    jobs = [("models", m, "model") for m in MODELS]
    for group, repos in DATASETS.items():
        jobs += [("datasets", d, group) for d in repos]
    with httpx.Client(timeout=20.0, follow_redirects=True) as client, ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: {**_info(j[0], j[1], client), "group": j[2]}, jobs))

    needs_auth = [r for r in results if r.get("gated") or r["status"] == "private_or_auth"]
    missing = [r for r in results if r["status"] not in ("ok",) and r not in needs_auth]
    width = max(len(r["repo"]) for r in results)
    for r in results:
        flag = "GATED(" + str(r["gated"]) + ")" if r.get("gated") else r["status"]
        print(f"{r['group']:<13} {r['repo']:<{width}}  {flag:<14} licence={r.get('license')}")
    print(f"\n{len(needs_auth)} need a Hugging Face login/terms acceptance; {len(missing)} missing or errored.")

    out = ROOT / "results" / "hf_access.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
