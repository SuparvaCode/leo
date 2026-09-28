"""Metrics for categorical predictions, matching the elcronos/jev-vs-open-decision-models protocol.

``probs`` is an ``[N, K]`` array of distributions, ``gold`` an ``[N]`` array of class ids.
ECE is top-label, 15 equal-width bins; Brier is the multi-class sum; NLL clips at 1e-6.
"""
from __future__ import annotations

import numpy as np


def accuracy(probs: np.ndarray, gold: np.ndarray) -> float:
    return float((probs.argmax(1) == gold).mean())


def macro_f1(probs: np.ndarray, gold: np.ndarray, n_classes: int | None = None) -> float:
    k = n_classes or probs.shape[1]
    pred = probs.argmax(1)
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def balanced_accuracy(probs: np.ndarray, gold: np.ndarray) -> float:
    pred = probs.argmax(1)
    recalls = [float((pred[gold == c] == c).mean()) for c in np.unique(gold)]
    return float(np.mean(recalls))


def brier(probs: np.ndarray, gold: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[gold]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs: np.ndarray, gold: np.ndarray, eps: float = 1e-6) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(gold)), gold], eps, 1.0)).mean())


def ece(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == gold).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def accuracy_at_coverage(probs: np.ndarray, gold: np.ndarray, coverage: float) -> float:
    """Accuracy on the ``coverage`` fraction of rows with the highest top probability."""
    n = max(1, int(round(coverage * len(gold))))
    order = np.argsort(-probs.max(1), kind="stable")[:n]
    return float((probs[order].argmax(1) == gold[order]).mean())


def summarize(probs: np.ndarray, gold: np.ndarray, n_classes: int | None = None) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(1, keepdims=True)
    gold = np.asarray(gold, dtype=np.int64)
    counts = np.bincount(gold, minlength=probs.shape[1])
    return {
        "n": int(len(gold)),
        "accuracy": accuracy(probs, gold),
        "macro_f1": macro_f1(probs, gold, n_classes),
        "balanced_accuracy": balanced_accuracy(probs, gold),
        "brier": brier(probs, gold),
        "ece": ece(probs, gold),
        "nll": nll(probs, gold),
        "mean_confidence": float(probs.max(1).mean()),
        "acc_at_50": accuracy_at_coverage(probs, gold, 0.5),
        "acc_at_80": accuracy_at_coverage(probs, gold, 0.8),
        "majority_class_accuracy": float(counts.max() / counts.sum()),
    }
