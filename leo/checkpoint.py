"""Crash-safe checkpoint files for training runs.

A power cut can stop the process in the middle of a write. Everything here is written to a temporary
name, flushed to disk, and only then renamed over the target, so a reader always sees either the old
version or the new one, never a torn file.

Run directory layout::

    <run>/train_log.jsonl   one JSON record per log line / dev eval
    <run>/resume.pt         trainable weights, optimizer, counters and RNG state (continue a run exactly)
    <run>/best/             exported model with the lowest dev loss (what serving and benchmarks load)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch

RESUME_FILE = "resume.pt"
BEST_DIR = "best"
CONFIG_FILE = "leo_config.json"


# ------------------------------------------------------------------------------ primitives

def _replace(src: Path, dst: Path, attempts: int = 10) -> None:
    """os.replace with retries: on Windows a virus scanner or indexer can briefly hold a new file open."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.2 * (i + 1))


def _fsync_file(path: Path) -> None:
    # FlushFileBuffers on Windows needs a handle with write access.
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            _fsync_file(p)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


# ------------------------------------------------------------------------------ directories

def _siblings(target: Path) -> tuple[Path, Path]:
    return target.with_name(target.name + ".tmp"), target.with_name(target.name + ".old")


def recover_dir(target: Path) -> None:
    """Undo an interrupted directory swap: restore ``.old`` if the target is missing, drop half-written ``.tmp``."""
    target = Path(target)
    tmp, old = _siblings(target)
    if not target.exists() and old.exists():
        _replace(old, target)
    for leftover in (tmp, old):
        if leftover.exists():
            shutil.rmtree(leftover, ignore_errors=True)


def atomic_export_dir(target: Path, write: Callable[[Path], None]) -> None:
    """Write a whole directory with ``write(tmp_dir)``, then swap it in place of ``target``."""
    target = Path(target)
    tmp, old = _siblings(target)
    recover_dir(target)
    write(tmp)
    fsync_tree(tmp)
    if target.exists():
        _replace(target, old)
    _replace(tmp, target)
    shutil.rmtree(old, ignore_errors=True)


SELECTED_FILE = "selected.json"


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a model directory or a run directory.

    In a run directory, use the model named in ``selected.json`` (written by ``leo.select``), else ``best/``,
    else ``best.old`` (left by a power cut in the middle of swapping in a new best).
    """
    path = Path(path)
    if (path / CONFIG_FILE).exists():
        return path
    if (path / SELECTED_FILE).exists():
        chosen = path / json.loads((path / SELECTED_FILE).read_text(encoding="utf-8"))["selected"]
        if (chosen / CONFIG_FILE).exists():
            return chosen
    for cand in (path / BEST_DIR, path / (BEST_DIR + ".old")):
        if (cand / CONFIG_FILE).exists():
            return cand
    raise FileNotFoundError(f"no Leo checkpoint in {path} (expected {CONFIG_FILE} or {BEST_DIR}/{CONFIG_FILE})")


# ------------------------------------------------------------------------------ resume state

def save_resume(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, meta: dict[str, Any]) -> None:
    state = {
        "format": 1,
        "trainable": {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "meta": meta,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    atomic_torch_save(state, path)


def read_resume(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != 1:
        raise RuntimeError(f"{path}: unknown resume format {state.get('format')!r}")
    return state


def load_trainable(state: dict[str, Any], model: torch.nn.Module) -> None:
    """Copy saved trainable tensors into ``model``; names and set must match exactly."""
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    got = set(state["trainable"])
    if want != got:
        raise RuntimeError(
            f"resume state does not match the model: missing {sorted(want - got)[:3]}, unexpected {sorted(got - want)[:3]}"
        )
    with torch.no_grad():
        for name, tensor in state["trainable"].items():
            params[name].copy_(tensor)


def apply_resume(state: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    load_trainable(state, model)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_torch"])
    if state["rng_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["meta"]


def truncate_log(path: Path, step: int) -> None:
    """Keep log records up to ``step`` (the resume point); drop later ones and any line torn by a crash."""
    path = Path(path)
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = rec.get("step", rec.get("dev", {}).get("step") if isinstance(rec.get("dev"), dict) else None)
        if s is None or s <= step:
            keep.append(line)
    atomic_write_text(path, "".join(f"{line}\n" for line in keep))
