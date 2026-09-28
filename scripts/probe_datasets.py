"""Print split names, features and one row for each training source (streams the first row only)."""
from __future__ import annotations

import sys

from datasets import get_dataset_split_names, load_dataset

SOURCES = [
    ("fancyzhx/ag_news", None), ("fancyzhx/dbpedia_14", None), ("community-datasets/yahoo_answers_topics", None),
    ("legacy-datasets/banking77", None), ("clinc/clinc_oos", "plus"), ("CogComp/trec", None),
    ("stanfordnlp/snli", None), ("nyu-mll/multi_nli", None), ("google/boolq", None),
    ("allenai/ai2_arc", "ARC-Easy"), ("tau/commonsense_qa", None), ("stanfordnlp/imdb", None),
    ("Yelp/yelp_review_full", None), ("ucirvine/sms_spam", None), ("deepset/prompt-injections", None),
    ("jackhhao/jailbreak-classification", None), ("nvidia/HelpSteer2", None), ("nyu-mll/glue", "mrpc"),
]

for repo, cfg in SOURCES if len(sys.argv) < 2 else [s for s in SOURCES if s[0] in sys.argv[1:]]:
    try:
        splits = get_dataset_split_names(repo, cfg)
        ds = load_dataset(repo, cfg, split=splits[0], streaming=True)
        row = next(iter(ds))
        feats = {k: (getattr(v, "names", None)[:8] if getattr(v, "names", None) else type(v).__name__) for k, v in (ds.features or {}).items()}
        short = {k: (str(v)[:80]) for k, v in row.items()}
        print(f"OK   {repo} [{cfg}] splits={splits}\n     features={feats}\n     row={short}", flush=True)
    except Exception as e:  # report and continue
        print(f"FAIL {repo} [{cfg}]: {type(e).__name__}: {str(e)[:200]}", flush=True)
