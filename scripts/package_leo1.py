"""Package a Leo checkpoint as a self-contained, offline folder: weights, bundled base model, code, results.

    python scripts/package_leo1.py --src checkpoints/leo-4b-soup3 --config <calibrated leo_config.json> \
        --name leo-1 --results leo-4b-soup3 --out D:/Leo-1
    python scripts/package_leo1.py --src checkpoints/leo-1.7b-v5/best --name leo-1-lite --results leo-1.7b-v5 --out D:/Leo-1-lite

Layout:  adapter/  leo_head.safetensors  leo_config.json  base/ (the exact Qwen3 revision it was trained on)
         leo/ (inference + server code)  results/  README.md  LICENSE  requirements.txt  SHA256SUMS.txt
`Leo.load(folder)` uses base/ when it exists, so nothing is downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["__init__.py", "schema.py", "render.py", "encode.py", "model.py", "infer.py", "calibrate.py",
        "checkpoint.py", "serve.py", "client.py"]
BASE_FILES = ["config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", help="calibrated leo_config.json to use instead of the one in --src")
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", required=True, help="result name under results/ to copy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = ROOT / a.src, Path(a.out)
    cfg = json.loads(Path(a.config or src / "leo_config.json").read_text(encoding="utf-8"))
    if not cfg.get("completed") or not cfg.get("temperatures"):
        raise SystemExit("config is not calibrated (no temperatures); refusing to package")
    cfg["name"] = a.name
    cfg["packaged_from"] = {"checkpoint": a.src, "results": a.results}

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("adapter", "leo", "results", "base"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    shutil.copytree(src / "adapter", out / "adapter", ignore=shutil.ignore_patterns("README.md"))
    shutil.copy2(src / "leo_head.safetensors", out / "leo_head.safetensors")
    (out / "leo_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "leo").mkdir()
    for f in CODE:
        shutil.copy2(ROOT / "leo" / f, out / "leo" / f)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    (out / "requirements.txt").write_text("torch>=2.6\ntransformers==5.17.0\npeft==0.21.0\nsafetensors\nnumpy\n"
                                          "pydantic>=2\nfastapi\nuvicorn\nhttpx\n", encoding="utf-8")

    # bundled base model: the exact revision the adapter was trained on
    repo = cfg["base_model"].replace("/", "--")
    snap = ROOT / "hf_cache" / "hub" / f"models--{repo}" / "snapshots" / cfg["base_revision"]
    (out / "base").mkdir()
    weights = sorted(p.name for p in snap.glob("*.safetensors")) + (["model.safetensors.index.json"]
                                                                    if (snap / "model.safetensors.index.json").exists() else [])
    for f in BASE_FILES + weights:
        shutil.copy2(snap / f, out / "base" / f)
    (out / "base" / "SOURCE.txt").write_text(f"{cfg['base_model']} at revision {cfg['base_revision']} (Apache-2.0)\n",
                                             encoding="utf-8")

    res = out / "results"
    res.mkdir()
    n = a.results
    for rel in [f"short/{n}.json", f"short/{n}-canon.json", f"jevbench/{n}.json", f"jevbench/{n}.md",
                f"bench/{n}/summary.json", f"multilingual/{n}.json", f"probes/{n}.md", f"probes/{n}.json",
                f"probes/done/{n}.json", "compare_v5.md"]:
        p = ROOT / "results" / rel
        if p.exists():
            dst = res / rel.replace("/", "__")
            shutil.copy2(p, dst)
    for rel in [f"browser/final-{n}/report.md"]:
        p = ROOT / "results" / rel
        if p.exists():
            shutil.copy2(p, res / "browser_report.md")

    sums = [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS.txt" and "__pycache__" not in p.parts]
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{a.name}: {len(sums)} files, {total / 2**30:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
