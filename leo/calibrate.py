"""Post-hoc temperature scaling, one temperature per (question type, option-count) bucket.

Every open decision model measured so far ships over-confident before this step. The fit minimises
NLL on an in-distribution dev set and never touches held-out benchmark data. A temperature does not
change which option wins, only how peaked the distribution is.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

T_MIN, T_MAX = 0.25, 10.0
MIN_BUCKET = 40


def bucket(qtype: str, n_options: int) -> str:
    if qtype != "choice":
        return qtype
    if n_options <= 5:
        return "choice:2-5"
    if n_options <= 10:
        return "choice:6-10"
    if n_options <= 30:
        return "choice:11-30"
    return "choice:31+"


def _nll(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray], t: float) -> float:
    total = 0.0
    for z, y in zip(logits, targets):
        z = z / t
        z = z - z.max()
        logp = z - math.log(np.exp(z).sum())
        total -= float((y * logp).sum())
    return total / max(1, len(logits))


def fit_temperature(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> float:
    """1-D search on log T: coarse grid, then golden-section refinement."""
    grid = np.linspace(math.log(T_MIN), math.log(T_MAX), 41)
    losses = [_nll(logits, targets, math.exp(g)) for g in grid]
    i = int(np.argmin(losses))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(logits, targets, math.exp(c)), _nll(logits, targets, math.exp(d))
    for _ in range(30):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(logits, targets, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(logits, targets, math.exp(d))
    return float(math.exp((a + b) / 2))


def fit_temperatures(records: Iterable[tuple[str, int, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """records: (qtype, n_options, logits, target distribution). Small buckets fall back to their type."""
    by_bucket: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    by_type: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for qtype, k, z, y in records:
        by_bucket[bucket(qtype, k)].append((z, y))
        by_type[qtype].append((z, y))
    temps: dict[str, float] = {}
    for qtype, items in by_type.items():
        temps[qtype] = fit_temperature([z for z, _ in items], [y for _, y in items]) if len(items) >= MIN_BUCKET else 1.0
    for b, items in by_bucket.items():
        if b in temps:
            continue
        qtype = b.split(":")[0]
        temps[b] = fit_temperature([z for z, _ in items], [y for _, y in items]) if len(items) >= MIN_BUCKET else temps.get(qtype, 1.0)
    return temps


def temperature_for(temps: dict[str, float], qtype: str, n_options: int) -> float:
    return float(temps.get(bucket(qtype, n_options), temps.get(qtype, 1.0)))
