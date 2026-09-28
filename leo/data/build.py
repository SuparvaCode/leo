"""Build data/processed/{train,dev}.jsonl and a manifest from the public sources and synthetic families.

    python -m leo.data.build --out data/processed --scale 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from leo.data import example_specs
from leo.data.augment import RECIPES, set_recipe
from leo.data.sources import SOURCES, Source
from leo.data.synthetic import FAMILIES


def _names(ds: Any, field: str | None) -> list[str]:
    if field is None:
        return []
    feat = ds.features[field]
    return list(feat.names) if hasattr(feat, "names") else []


def _convert(src: Source, ds: Any, n: int, rng: random.Random) -> list[dict[str, Any]]:
    names = _names(ds, src.label_field)
    out: list[dict[str, Any]] = []
    for row in ds:
        if len(out) >= n:
            break
        ex = src.convert(row, names, rng)
        if ex is None:
            continue
        example_specs(ex)  # validates schema and labels; raises on a converter bug
        out.append(ex)
    return out


def build_source(src: Source, scale: float, seed: int) -> tuple[list[dict], list[dict], dict[str, Any]]:
    from datasets import load_dataset

    rng = random.Random(f"{seed}:{src.name}")
    n_train = max(1, int(src.n_train * scale))
    train_ds = load_dataset(src.repo, src.config, split=src.train_split).shuffle(seed=seed)
    if src.dev_split is None:
        dev_ds = train_ds.select(range(min(len(train_ds), src.n_dev * 2)))
        train_ds = train_ds.select(range(min(len(train_ds), src.n_dev * 2), len(train_ds)))
    else:
        dev_ds = load_dataset(src.repo, src.config, split=src.dev_split).shuffle(seed=seed)
        dev_ds = dev_ds.select(range(min(len(dev_ds), src.n_dev * 2)))
    train_ds = train_ds.select(range(min(len(train_ds), int(n_train * 1.5) + 50)))
    train = _convert(src, train_ds, n_train, rng)
    dev = _convert(src, dev_ds, src.n_dev, rng)
    info = {"repo": src.repo, "config": src.config, "train_split": src.train_split, "dev_split": src.dev_split or "carved from train",
            "fingerprint": getattr(train_ds, "_fingerprint", None), "n_train": len(train), "n_dev": len(dev)}
    return train, dev, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--only", help="comma list of source names (default: all)")
    ap.add_argument("--recipe", default="v0", choices=sorted(RECIPES), help="augmentation recipe (see leo.data.augment.RECIPES)")
    args = ap.parse_args()
    try:  # HF_TOKEN from the git-ignored .env: higher Hub rate limits for the many-file multilingual sources
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    set_recipe(args.recipe)

    train: list[dict] = []
    dev: list[dict] = []
    manifest: dict[str, Any] = {"seed": args.seed, "scale": args.scale, "recipe": args.recipe,
                                "recipe_params": dict(RECIPES[args.recipe]), "sources": {}, "failed": {}}
    for src in SOURCES:
        if only and src.name not in only:
            continue
        t0 = time.time()
        try:
            tr, dv, info = build_source(src, args.scale, args.seed)
        except Exception as e:  # keep going; report at the end
            manifest["failed"][src.name] = f"{type(e).__name__}: {e}"
            print(f"FAIL {src.name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        train += tr
        dev += dv
        manifest["sources"][src.name] = info
        print(f"ok   {src.name:<18} train={len(tr):<5} dev={len(dv):<4} ({time.time() - t0:.0f}s)", flush=True)

    recipe = RECIPES[args.recipe]
    families: dict[str, Any] = dict(FAMILIES)
    bulk: dict[str, Any] = {}  # name -> (fn(rng, n, split), n_train, n_dev): sources that load many rows at once
    if recipe.get("extra_families"):
        from leo.data import browser, reasoning

        families |= reasoning.FAMILIES
        bulk[browser.SOURCE] = (lambda rng, n, split: browser.examples(rng, n), recipe.get("browser_n", 6000), 200)
    if recipe.get("multilingual"):
        from leo.data import multilingual

        bulk |= multilingual.BULK
    if recipe.get("mind2web"):
        from leo.data import mind2web

        if not mind2web.CACHE.exists():
            raise SystemExit(f"{mind2web.CACHE} is missing; run `python -m leo.data.mind2web` first")
        bulk[mind2web.SOURCE] = (mind2web.load, 20000, 150)  # all converted train-task steps

    for name, (fn, n, n_dev) in bulk.items():
        if only and name not in only:
            continue
        t0 = time.time()
        tr = fn(random.Random(f"{args.seed}:{name}"), max(1, int(n * args.scale)), "train")
        dv = fn(random.Random(f"{args.seed}:{name}:dev"), n_dev, "dev")
        for ex in tr + dv:
            example_specs(ex)
        train += tr
        dev += dv
        manifest["sources"][name] = {"generator": f"{fn.__module__}.{getattr(fn, '__name__', name)}",
                                     "n_train": len(tr), "n_dev": len(dv)}
        print(f"ok   {name:<18} train={len(tr):<5} dev={len(dv):<4} ({time.time() - t0:.0f}s)", flush=True)

    for name, (fn, n) in families.items():
        if only and name not in only:
            continue
        rng = random.Random(f"{args.seed}:{name}")
        n_tr = max(1, int(n * args.scale))
        tr = [fn(rng) for _ in range(n_tr)]
        dv = [fn(rng) for _ in range(60)]
        for ex in tr + dv:
            example_specs(ex)
        train += tr
        dev += dv
        manifest["sources"][name] = {"generator": f"{fn.__module__}.{fn.__name__}", "n_train": len(tr), "n_dev": len(dv)}
        print(f"ok   {name:<18} train={len(tr):<5} dev={len(dv):<4}", flush=True)

    random.Random(args.seed).shuffle(train)
    for split, rows in (("train", train), ("dev", dev)):
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        qtypes = Counter(q["type"] for ex in rows for q in ex["questions"].values())
        manifest[split] = {"requests": len(rows), "questions": sum(qtypes.values()), "by_type": dict(qtypes), "sha256": digest}
        print(f"{split}: {len(rows)} requests, {dict(qtypes)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
