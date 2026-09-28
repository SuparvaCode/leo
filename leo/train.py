"""Stage 1 training (proper-scoring-rule loss) plus Stage 3 calibration. Resumable after a crash.

Loss per question: log loss against the (possibly soft) target distribution, plus a ranked probability
score term on Score questions so that near-misses on an ordinal scale cost less than far misses. Both
are strictly proper, so the expected loss is minimised only by honest probabilities; this is the exact
gradient of the reward that RLCD-style policy gradients estimate by sampling.

    python -m leo.train --data data/processed --base Qwen/Qwen3-0.6B-Base --out checkpoints/leo-0.6b
    python -m leo.train ...same arguments... --resume      # continue after a crash or power cut

Every ``--save_every`` optimizer steps the trainer writes ``<out>/resume.pt`` (trainable weights,
optimizer, counters, RNG state). Each epoch's data order is derived from ``(seed, epoch)``, so a resumed
run replays exactly the batches it has not seen yet. The best model by dev loss lives in ``<out>/best``.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leo.calibrate import fit_temperatures, temperature_for
from leo.checkpoint import (
    BEST_DIR, CONFIG_FILE, RESUME_FILE, apply_resume, atomic_export_dir, atomic_write_text, read_resume, recover_dir,
    save_resume, truncate_log,
)
from leo.data import example_specs
from leo.encode import Batch, Encoder, Row, batches_by_tokens, collate, pack_rows, training_perm
from leo.infer import make_tokenize
from leo.model import LeoModel, supports_block_mask

TYPE_NAMES = ("choice", "score", "noul")
# Arguments that change the data order, the schedule or parameter shapes. A resumed run must match them.
RUN_KEYS = ("base", "epochs", "lr", "head_lr", "warmup", "lora_r", "lora_alpha", "head_dim", "rps_weight",
            "max_batch_tokens", "accum", "max_row_tokens", "train_state_tokens", "max_steps", "seed")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(examples: list[dict[str, Any]]) -> list[tuple[Any, list, str]]:
    return [(ex["state"], example_specs(ex), ex.get("source", "?")) for ex in examples]


def build_rows(items, encoder: Encoder, rng: random.Random | None, max_row_tokens: int) -> list[Row]:
    rows: list[Row] = []
    for ei, (state, specs, _) in enumerate(items):
        es = encoder.encode_state(state)
        qs = [encoder.encode_question(s, key=(ei, s.qid), perm=training_perm(s, rng) if rng else None) for s in specs]
        rows += pack_rows(es, qs, max_row_tokens)
    return rows


def epoch_plan(items, encoder: Encoder, seed: int, epoch: int, max_row_tokens: int, max_batch_tokens: int):
    """Option orders and batch order for one epoch, derived only from (seed, epoch)."""
    rng = random.Random(f"{seed}:epoch:{epoch}")
    rows = build_rows(items, encoder, rng, max_row_tokens)
    groups = batches_by_tokens(rows, max_batch_tokens)
    rng.shuffle(groups)
    return rows, groups


def decision_loss(logits: torch.Tensor, batch: Batch, rps_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(logits)
    logp = torch.log_softmax(logits, dim=-1).masked_fill(~valid, 0.0)
    target = batch.targets
    loss = -(target * logp).sum(-1)
    is_score = batch.q_type == 1
    if rps_weight > 0 and bool(is_score.any()):
        p = logp.exp() * valid
        p_o = torch.zeros_like(p).scatter(1, batch.q_perm, p)       # back to ordinal order
        t_o = torch.zeros_like(target).scatter(1, batch.q_perm, target)
        rps = ((p_o.cumsum(-1) - t_o.cumsum(-1)) ** 2).sum(-1) / (batch.q_nopt - 1).clamp(min=1)
        loss = loss + rps_weight * rps * is_score
    return loss, logp


@torch.no_grad()
def evaluate(model: LeoModel, rows: list[Row], items, device, dtype, max_tokens: int, temps: dict[str, float] | None = None):
    """Dev loss/accuracy per type and per source; also returns raw logits for calibration."""
    was_training = model.training
    model.eval()
    stats: dict[str, list[float]] = defaultdict(list)
    records = []
    for idx in batches_by_tokens(rows, max_tokens):
        batch = collate([rows[i] for i in idx]).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(batch)
        loss, _ = decision_loss(logits, batch, rps_weight=0.0)
        tgt = batch.targets
        for qi, (key, perm) in enumerate(zip(batch.keys, batch.perms)):
            k = len(perm)
            z = logits[qi, :k].float().cpu().numpy()
            y = tgt[qi, :k].cpu().numpy()
            qtype = TYPE_NAMES[int(batch.q_type[qi])]
            src = items[key[0]][2]
            records.append((qtype, k, z, y, src))
            if y.max() > 0.99:
                correct = float(z.argmax() == y.argmax())
                stats[f"acc/{qtype}"].append(correct)
                stats[f"acc_src/{src}"].append(correct)
            stats[f"loss/{qtype}"].append(float(loss[qi]))
    model.train(was_training)
    summary = {k: float(np.mean(v)) for k, v in sorted(stats.items())}
    summary["loss"] = float(np.mean([r for k, v in stats.items() if k.startswith("loss/") for r in v]))
    if temps is not None:
        summary["ece_top"] = _top_ece(records, temps)
    return summary, records


def _softmax_t(z: np.ndarray, t: float) -> np.ndarray:
    zz = z / t
    p = np.exp(zz - zz.max())
    return p / p.sum()


def _top_ece(records, temps: dict[str, float], bins: int = 15) -> float:
    conf, corr = [], []
    for qtype, k, z, y, _ in records:
        if y.max() < 0.99:
            continue
        p = _softmax_t(z, temperature_for(temps, qtype, k))
        conf.append(p.max())
        corr.append(float(p.argmax() == y.argmax()))
    conf_a, corr_a = np.asarray(conf), np.asarray(corr)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf_a > edges[i]) & (conf_a <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(corr_a[m].mean() - conf_a[m].mean())
    return float(total)


def calibrated_summary(records, temps: dict[str, float]) -> dict[str, float]:
    """Dev log loss per type and top-label ECE after temperature scaling (accuracy is unchanged)."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for qtype, k, z, y, _ in records:
        p = _softmax_t(z, temperature_for(temps, qtype, k))
        by_type[qtype].append(float(-(y * np.log(np.clip(p, 1e-12, 1.0))).sum()))
    out = {f"loss/{t}": float(np.mean(v)) for t, v in sorted(by_type.items())}
    out["loss"] = float(np.mean([x for v in by_type.values() for x in v]))
    out["ece_top"] = _top_ece(records, temps)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up to 1, then cosine decay to 0.1. ``step`` counts from 1."""
    if step <= warmup:
        return step / max(1, warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    """bf16 on Ampere or newer; fp16 (with loss scaling) on older GPUs such as the T4, where bf16 is emulated
    and about 10x slower even though torch reports it as supported."""
    if device.type != "cuda":
        return torch.float32
    if name == "auto":
        name = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def sync_grads(params: list[torch.nn.Parameter], world: int) -> None:
    """Average gradients across data-parallel ranks in one flat all-reduce (LoRA + head are ~20M floats)."""
    import torch.distributed as dist

    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        v = flat[offset:offset + n].view_as(g).to(g.dtype)
        if p.grad is None:
            p.grad = v.clone()
        else:
            p.grad.copy_(v)
        offset += n


def base_revision(base: str) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(base).sha
    except Exception as e:  # offline, e.g. network not back yet after a restart
        print(f"could not resolve the {base} revision online ({type(e).__name__}); using the cached copy", flush=True)
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="checkpoints/leo-0.6b")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--head_dim", type=int, default=512)
    ap.add_argument("--rps_weight", type=float, default=0.5)
    ap.add_argument("--max_batch_tokens", type=int, default=6144)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_row_tokens", type=int, default=1536)
    ap.add_argument("--train_state_tokens", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=50, help="optimizer steps between resume checkpoints")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue the run in --out from its last resume checkpoint")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto: bf16 on Ampere+, fp16 with loss scaling on older GPUs")
    ap.add_argument("--time_budget_min", type=float, default=0.0,
                    help="save resume state and stop cleanly after this many minutes (0 = no limit)")
    ap.add_argument("--init", default=None,
                    help="warm start: exported Leo model dir (adapter/ + leo_head.safetensors) whose weights start "
                         "this run; base, LoRA shape and head size must match. Optimizer and schedule start fresh.")
    return ap.parse_args(argv)


def init_from_export(model: LeoModel, path: Path) -> None:
    """Copy every trainable tensor (LoRA, markers, head) from an exported model into a fresh training model."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from leo.checkpoint import resolve_checkpoint

    path = resolve_checkpoint(path)
    adapter = load_file(str(path / "adapter" / "adapter_model.safetensors"))
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if p.requires_grad}
    res = set_peft_model_state_dict(model.backbone, adapter)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"--init adapter has keys this model lacks: {unexpected[:3]}")
    changed = sum(not torch.equal(before[n], p.detach()) for n, p in model.backbone.named_parameters() if p.requires_grad)
    if changed != len(before):  # lora_B starts at zero, so every tensor must have been overwritten by a trained one
        raise RuntimeError(f"--init set only {changed} of {len(before)} LoRA tensors")
    state = load_file(str(path / "leo_head.safetensors"))
    model.markers.load_state_dict({k[len("markers."):]: v for k, v in state.items() if k.startswith("markers.")})
    model.head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
    print(f"warm start from {path}: {changed} LoRA tensors, markers and head", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out, data = Path(args.out), Path(args.data)
    # Data parallel when launched by torchrun: every rank trains on its own share of each epoch's batches and
    # gradients are averaged before each optimizer step. Rank 0 alone writes logs, checkpoints and evals.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    main_rank = rank == 0
    out.mkdir(parents=True, exist_ok=True)
    resume_path, best_dir, log_path = out / RESUME_FILE, out / BEST_DIR, out / "train_log.jsonl"

    # Undo anything a crash left half-written before deciding what to do.
    if main_rank:
        recover_dir(best_dir)
        resume_path.with_name(resume_path.name + ".tmp").unlink(missing_ok=True)
    if world > 1:
        dist.barrier()

    data_sha = sha256_file(data / "train.jsonl")
    state = None
    if resume_path.exists():
        if not args.resume:
            raise SystemExit(f"{out} holds an unfinished run. Add --resume to continue it, or choose another --out.")
        state = read_resume(resume_path)
        saved = state["meta"]
        diff = {k: (saved["args"].get(k), getattr(args, k)) for k in RUN_KEYS if saved["args"].get(k) != getattr(args, k)}
        if saved["data_sha256"] != data_sha:
            diff["train.jsonl sha256"] = (saved["data_sha256"][:12], data_sha[:12])
        if saved.get("world", 1) != world:  # batches are sharded by rank, so the GPU count fixes the order
            diff["world size"] = (saved.get("world", 1), world)
        if diff:
            raise SystemExit(f"cannot resume, settings differ from the saved run (saved, now): {diff}")
    elif (best_dir / CONFIG_FILE).exists():
        finished = json.loads((best_dir / CONFIG_FILE).read_text(encoding="utf-8")).get("completed")
        raise SystemExit(f"{out} already holds a {'finished' if finished else 'partial'} run without resume state; "
                         "choose another --out or delete it.")
    elif args.resume:
        print(f"no resume state in {out}; starting a fresh run", flush=True)

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(args.precision, device)
    if main_rank:
        print(f"precision {dtype}, world size {world}", flush=True)

    from transformers import AutoConfig, AutoTokenizer

    revision = state["meta"]["base_revision"] if state else base_revision(args.base)
    tokenize = make_tokenize(AutoTokenizer.from_pretrained(args.base, revision=revision))
    packable = supports_block_mask(AutoConfig.from_pretrained(args.base, revision=revision))
    row_tokens = args.max_row_tokens if packable else 0  # 0 = one question per row
    if not packable:
        print(f"{args.base} has non-attention layers that ignore masks: one question per row", flush=True)
    train_enc = Encoder(tokenize, max_state_tokens=args.train_state_tokens)
    train_items = prepare(load_jsonl(data / "train.jsonl"))
    dev_items = prepare(load_jsonl(data / "dev.jsonl"))
    dev_rows = build_rows(dev_items, train_enc, None, row_tokens)
    print(f"train {len(train_items)} requests, dev {len(dev_items)} requests ({len(dev_rows)} rows)", flush=True)

    model = LeoModel.from_pretrained_base(
        args.base, tokenize, revision=revision, dtype=dtype, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        head_dim=args.head_dim, gradient_checkpointing=True,
    ).to(device)
    bb, head = model.trainable_parameter_groups()
    for p in bb + head:  # master copies of trained weights stay fp32; the frozen base runs in `dtype`
        p.data = p.data.float()
    if args.init and state is None:  # a resumed run gets its weights from resume.pt instead
        init_from_export(model, Path(args.init))
    if main_rank:
        print(f"trainable {sum(p.numel() for p in bb + head) / 1e6:.2f}M params", flush=True)
    base_lrs = (args.lr, args.head_lr)
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=0.0, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    step = micro = start_epoch = start_batch = 0
    best, tokens_total, seconds_before = float("inf"), 0, 0.0
    if state is not None:
        meta = apply_resume(state, model, opt)
        step, micro = meta["step"], meta["micro"]
        start_epoch, start_batch = meta["epoch"], meta["batch_in_epoch"]
        best, tokens_total, seconds_before = meta["best"], meta["tokens_total"], meta["train_seconds"]
        if meta.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(meta["scaler"])
        if (best_dir / CONFIG_FILE).exists():  # the best export can be newer than the last resume point
            saved_best = json.loads((best_dir / CONFIG_FILE).read_text(encoding="utf-8")).get("best_dev_loss")
            if saved_best is not None:
                best = min(best, saved_best)
        if main_rank:
            truncate_log(log_path, step)
            print(f"resumed at step {step} (epoch {start_epoch}, batch {start_batch}), best dev loss {best:.4f}", flush=True)
        del state

    def shard(plan_: tuple[list[Row], list[list[int]]]) -> tuple[list[Row], list[list[int]]]:
        """This rank's share of an epoch; every rank gets the same number of batches."""
        rows_, groups_ = plan_
        per_rank = len(groups_) // world
        return rows_, groups_[rank::world][:per_rank]

    plan = {start_epoch: shard(epoch_plan(train_items, train_enc, args.seed, start_epoch, row_tokens, args.max_batch_tokens))}
    n_batches = len(plan[start_epoch][1])
    total_steps = args.max_steps or max(1, int(math.ceil(n_batches * args.epochs / args.accum)))
    run_args = {k: v for k, v in vars(args).items() if k != "resume"}
    cfg: dict[str, Any] = {
        "name": args.name or out.name, "base_model": args.base, "base_revision": revision, "head_dim": args.head_dim,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "encoder": {"max_state_tokens": max(1024, args.train_state_tokens)},
        "train": run_args, "train_sha256": data_sha,
        "data_manifest": json.loads((data / "manifest.json").read_text()) if (data / "manifest.json").exists() else None,
        "temperatures": {}, "best_dev_loss": None, "best_step": None, "completed": False,
    }
    if main_rank:
        print(f"{n_batches} micro-batches/epoch per rank, {total_steps} optimizer steps", flush=True)

    log = open(log_path, "a", encoding="utf-8") if main_rank else None
    model.train()
    t0 = time.perf_counter()
    tokens_session = 0
    window: dict[str, list[float]] = defaultdict(list)
    epoch = start_epoch
    out_of_time = False

    def save_state(batch_in_epoch: int) -> None:
        if not main_rank:
            return
        save_resume(resume_path, model, opt, {
            "step": step, "micro": micro, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "best": best,
            "tokens_total": tokens_total, "train_seconds": seconds_before + time.perf_counter() - t0,
            "total_steps": total_steps, "args": run_args, "base_revision": revision, "data_sha256": data_sha,
            "world": world, "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        })

    def time_is_up() -> bool:
        """Same answer on every rank (rank 0 decides), so no rank is left waiting in an all-reduce."""
        up = args.time_budget_min > 0 and (time.perf_counter() - t0) / 60 >= args.time_budget_min
        if world > 1:
            flag = torch.tensor([float(up)], device=device)
            dist.broadcast(flag, src=0)
            up = bool(flag.item())
        return up

    done = step >= total_steps
    while not done:
        rows, groups = plan.pop(epoch, None) or shard(
            epoch_plan(train_items, train_enc, args.seed, epoch, row_tokens, args.max_batch_tokens))
        first = start_batch if epoch == start_epoch else 0
        for bi in range(first, len(groups)):
            batch = collate([rows[i] for i in groups[bi]]).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(batch)
            loss_q, _ = decision_loss(logits, batch, args.rps_weight)
            loss = loss_q.mean()
            scaler.scale(loss / args.accum).backward()
            micro += 1
            tokens_total += batch.n_tokens
            tokens_session += batch.n_tokens
            window["loss"].append(loss.item())
            hard = batch.targets.max(-1).values > 0.99
            if bool(hard.any()):
                pred = logits.detach().argmax(-1)  # padded slots are -inf here
                window["acc"].append((pred == batch.targets.argmax(-1))[hard].float().mean().item())
            if micro % args.accum:
                continue
            if world > 1:
                sync_grads(bb + head, world)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(bb + head, 1.0)
            f = lr_factor(step + 1, args.warmup, total_steps)
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            scaler.step(opt)  # skipped (and the scale lowered) if fp16 gradients overflowed
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if main_rank and step % args.log_every == 0:
                dt = time.perf_counter() - t0
                rec = {"step": step, "epoch": epoch, "loss": float(np.mean(window["loss"])), "acc": float(np.mean(window["acc"] or [0])),
                       "lr": opt.param_groups[0]["lr"], "tok_per_s": round(tokens_session / dt),
                       "elapsed_min": round((seconds_before + dt) / 60, 1),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0}
                if scaler.is_enabled():
                    rec["loss_scale"] = scaler.get_scale()
                rec["tok_per_s"] *= world
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                window.clear()
            evaluated = False
            if main_rank and (step % args.eval_every == 0 or step >= total_steps):
                summary, _ = evaluate(model, dev_rows, dev_items, device, dtype, 8192)
                summary["step"] = step
                print("DEV " + json.dumps({k: round(v, 4) for k, v in summary.items() if not k.startswith("acc_src")}), flush=True)
                log.write(json.dumps({"dev": summary}) + "\n")
                log.flush()
                if summary["loss"] < best:
                    best = summary["loss"]
                    cfg.update(best_dev_loss=best, best_step=step)
                    atomic_export_dir(best_dir, lambda d: model.save(d, cfg))
                    print(f"saved best model at step {step} (dev loss {best:.4f})", flush=True)
                evaluated = True
            out_of_time = step < total_steps and time_is_up()
            if evaluated or out_of_time or step % args.save_every == 0 or step >= total_steps:
                save_state(bi + 1)
                if main_rank:
                    print(f"resume state saved at step {step}", flush=True)
            if step >= total_steps or out_of_time:
                done = True
                break
        epoch += 1
    if log:
        log.close()
    if world > 1:
        dist.barrier()
    if out_of_time:
        if main_rank:
            print(f"TIME BUDGET reached at step {step} of {total_steps}; rerun with --resume to continue", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    if not main_rank:  # calibration runs on rank 0 only
        dist.destroy_process_group()
        return

    # ---- Stage 3: calibration on dev, using the best checkpoint's weights
    del model, opt, bb, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_model, best_cfg = LeoModel.load(best_dir, device=device, dtype=dtype, merge=False)
    raw, records = evaluate(best_model, dev_rows, dev_items, device, dtype, 8192, temps={})
    temps = fit_temperatures([(q, k, z, y) for q, k, z, y, _ in records])
    cal = calibrated_summary(records, temps)
    best_cfg.update(temperatures=temps, dev={"uncalibrated": raw, "calibrated": cal}, completed=True)
    atomic_write_text(best_dir / CONFIG_FILE, json.dumps(best_cfg, indent=2))
    # Keep the last-step state under a name --resume ignores: leo.select compares it with best/.
    if resume_path.exists():
        os.replace(resume_path, out / "final_state.pt")
    print("TEMPERATURES " + json.dumps({k: round(v, 3) for k, v in temps.items()}), flush=True)
    print(f"dev top-label ECE {raw['ece_top']:.4f} -> {cal['ece_top']:.4f}; dev loss {raw['loss']:.4f} -> {cal['loss']:.4f}", flush=True)
    print("DEV per source: " + json.dumps({k[8:]: round(v, 3) for k, v in raw.items() if k.startswith("acc_src/")}), flush=True)
    print(f"done: model in {best_dir}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
