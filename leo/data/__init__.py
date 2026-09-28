"""Training data: public human-labelled datasets plus code-generated families with verifiable labels.

Every example is one request with labels on its questions::

    {"source": "ag_news", "state": ..., "questions": {"q0": {"type": "choice", "instructions": ...,
     "criteria": {...}, "label": "sports"}}}

Labels: choice -> option key (or {key: prob}); noul -> bool or P(yes); score -> level index (or a
distribution over levels).
"""
from __future__ import annotations

from typing import Any

from leo.schema import QuestionSpec, to_spec


def label_distribution(spec: QuestionSpec, label: Any) -> list[float]:
    k = spec.n_options
    if spec.type == "noul":
        p = float(label)
        return [1.0 - p, p]
    if spec.type == "score":
        if isinstance(label, (list, tuple)):
            dist = [float(x) for x in label]
        else:
            dist = [0.0] * k
            dist[int(label)] = 1.0
    elif isinstance(label, dict):
        dist = [float(label.get(key, 0.0)) for key in spec.keys]
    else:
        dist = [0.0] * k
        dist[spec.keys.index(str(label))] = 1.0
    if len(dist) != k:
        raise ValueError(f"label has {len(dist)} entries for {k} options")
    s = sum(dist)
    if s <= 0:
        raise ValueError("empty label distribution")
    return [x / s for x in dist]


def example_specs(example: dict[str, Any]) -> list[QuestionSpec]:
    specs = []
    for qid, q in example["questions"].items():
        body = {k: v for k, v in q.items() if k != "label"}
        spec = to_spec(qid, body)
        spec.target = label_distribution(spec, q["label"])
        spec.meta["source"] = example.get("source", "?")
        specs.append(spec)
    return specs
