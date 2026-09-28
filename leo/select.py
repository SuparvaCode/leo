"""Pick a run's released checkpoint on dev data only, after calibration.

The trainer keeps ``best/``, the checkpoint with the lowest *uncalibrated* dev loss. Uncalibrated loss
also punishes over-confidence that the temperature step removes, so it can prefer an earlier, less
accurate checkpoint. This step exports the final-step weights (``final_state.pt``) to ``final/``,
calibrates it on the same dev set, and records whichever candidate has the lower *calibrated* dev loss
in ``<run>/selected.json``. Loading the run directory then returns that model. Held-out benchmarks are
not touched here.

    python -m leo.select --run checkpoints/leo-0.6b-v0 --data data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from leo.calibrate import fit_temperatures
from leo.checkpoint import (
    BEST_DIR, CONFIG_FILE, SELECTED_FILE, atomic_export_dir, atomic_write_text, load_trainable, read_resume,
)
from leo.encode import Encoder
from leo.infer import make_tokenize
from leo.model import LeoModel, supports_block_mask
from leo.train import build_rows, calibrated_summary, evaluate, load_jsonl, prepare

FINAL_DIR = "final"


def dev_scores(model_dir: Path, dev_items, device, dtype, tokenize, packable: bool, train_cfg: dict[str, Any]) -> dict[str, Any]:
    model, cfg = LeoModel.load(model_dir, device=device, dtype=dtype, merge=False)
    enc = Encoder(tokenize, max_state_tokens=train_cfg["train_state_tokens"])
    rows = build_rows(dev_items, enc, None, train_cfg["max_row_tokens"] if packable else 0)
    raw, records = evaluate(model, rows, dev_items, device, dtype, 8192, temps={})
    temps = fit_temperatures([(q, k, z, y) for q, k, z, y, _ in records])
    cal = calibrated_summary(records, temps)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"cfg": cfg, "raw": raw, "calibrated": cal, "temperatures": temps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--state", default="final_state.pt", help="final-step resume state inside --run")
    args = ap.parse_args()
    run, data = Path(args.run), Path(args.data)
    best_cfg = json.loads((run / BEST_DIR / CONFIG_FILE).read_text(encoding="utf-8"))
    if not best_cfg.get("completed"):
        raise SystemExit(f"{run / BEST_DIR} has not been calibrated yet; let training finish first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    from transformers import AutoConfig, AutoTokenizer

    base, rev, tcfg = best_cfg["base_model"], best_cfg.get("base_revision"), best_cfg["train"]
    tokenize = make_tokenize(AutoTokenizer.from_pretrained(base, revision=rev))
    packable = supports_block_mask(AutoConfig.from_pretrained(base, revision=rev))
    dev_items = prepare(load_jsonl(data / "dev.jsonl"))

    candidates: dict[str, dict[str, Any]] = {
        BEST_DIR: {"step": best_cfg.get("best_step"), "raw": best_cfg["dev"]["uncalibrated"], "calibrated": best_cfg["dev"]["calibrated"]}
    }
    state_path = run / args.state
    if state_path.exists():
        state = read_resume(state_path)
        step = state["meta"]["step"]
        if step != best_cfg.get("best_step"):
            model = LeoModel.from_pretrained_base(
                base, tokenize, revision=rev, dtype=dtype, lora_r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"],
                head_dim=tcfg["head_dim"], gradient_checkpointing=False,
            )
            load_trainable(state, model)
            cfg = {**best_cfg, "best_step": None, "best_dev_loss": None, "step": step, "temperatures": {}, "completed": False}
            atomic_export_dir(run / FINAL_DIR, lambda d: model.save(d, cfg))
            del model, state
            res = dev_scores(run / FINAL_DIR, dev_items, device, dtype, tokenize, packable, tcfg)
            cfg.update(temperatures=res["temperatures"], dev={"uncalibrated": res["raw"], "calibrated": res["calibrated"]}, completed=True)
            atomic_write_text(run / FINAL_DIR / CONFIG_FILE, json.dumps(cfg, indent=2))
            candidates[FINAL_DIR] = {"step": step, "raw": res["raw"], "calibrated": res["calibrated"]}
    else:
        print(f"no {state_path}; only {BEST_DIR}/ is a candidate", flush=True)

    chosen = min(candidates, key=lambda k: candidates[k]["calibrated"]["loss"])
    summary = {
        "selected": chosen,
        "rule": "lowest calibrated dev loss (log loss after per-bucket temperature scaling), dev set only",
        "candidates": {
            k: {"step": v["step"], "dev_loss_raw": v["raw"]["loss"], "dev_loss_calibrated": v["calibrated"]["loss"],
                "dev_ece_calibrated": v["calibrated"]["ece_top"],
                **{f"dev_{m}": v["raw"][m] for m in ("acc/choice", "acc/noul", "acc/score") if m in v["raw"]}}
            for k, v in candidates.items()
        },
    }
    atomic_write_text(run / SELECTED_FILE, json.dumps(summary, indent=2))
    for k, v in summary["candidates"].items():
        print(f"{k:>6}: step {v['step']}, dev loss raw {v['dev_loss_raw']:.4f} -> calibrated {v['dev_loss_calibrated']:.4f}, "
              f"ECE {v['dev_ece_calibrated']:.4f}, acc choice {v.get('dev_acc/choice', 0):.3f} noul {v.get('dev_acc/noul', 0):.3f} "
              f"score {v.get('dev_acc/score', 0):.3f}", flush=True)
    print(f"selected: {chosen} -> {run / chosen}", flush=True)


if __name__ == "__main__":
    main()
