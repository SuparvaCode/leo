"""Request validation and answer construction for the /v1/systemone wire format.

Mirrors TypeSafe's public API reference (https://docs.typesafe.ai/api):

* ``state``: string, JSON object or JSON array.
* ``questions``: map of caller-chosen ids to typed questions. The id is never shown to the model.
* ``choice``: ``criteria`` maps option key -> description (string, object, array or null), 1..255 options.
* ``score``: ``criteria`` is an ordered list of 2..10 level descriptions (low to high); no nulls.
* ``noul``: optional ``criteria`` with only the keys ``true`` / ``false``.

Confidence formulas follow TypeSafe's reference adapter (system-one-adapter): for a Choice with K>1
options ``(K * p_max - 1) / (K - 1)``; for a Score ``max(0, 1 - sum_i p_i * |i - mode| / D)`` where
``D`` is the mean distance of the levels from the middle of the scale. Noul answers carry no confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
QuestionType = Literal["choice", "score", "noul"]


def _check_entry(value: Any, where: str) -> None:
    """An EntryType is a string, object, array or null, recursively JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_entry(v, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{where}: object keys must be strings")
            _check_entry(v, f"{where}.{k}")
        return
    raise ValueError(f"{where}: unsupported value of type {type(value).__name__}")


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instructions: Any = None

    @field_validator("instructions")
    @classmethod
    def _instructions_json(cls, v: Any) -> Any:
        _check_entry(v, "instructions")
        return v


class ChoiceQuestion(_QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, Any]

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not 1 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"a choice needs 1 to {MAX_CHOICE_OPTIONS} options, got {len(v)}")
        for key, desc in v.items():
            if not key.strip():
                raise ValueError("option keys must be non-empty strings")
            _check_entry(desc, f"criteria.{key}")
        return v


class ScoreQuestion(_QuestionBase):
    type: Literal["score"]
    criteria: list[Any]

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: list[Any]) -> list[Any]:
        if not MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(f"a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(v)}")
        for i, level in enumerate(v):
            if level is None or (isinstance(level, str) and not level.strip()):
                raise ValueError(f"criteria[{i}]: every score level needs a description")
            _check_entry(level, f"criteria[{i}]")
        return v


class NoulQuestion(_QuestionBase):
    type: Literal["noul"]
    criteria: dict[str, Any] | None = None

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        bad = set(v) - {"true", "false"}
        if bad:
            raise ValueError(f"noul criteria only accepts the keys 'true' and 'false', got {sorted(bad)}")
        for k, desc in v.items():
            _check_entry(desc, f"criteria.{k}")
        return v


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    state: str | dict[str, Any] | list[Any]
    model: str = "leo-latest"
    questions: dict[str, Question] = Field(default_factory=dict)

    @field_validator("questions", mode="before")
    @classmethod
    def _dispatch(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError("questions must be an object mapping question ids to questions")
        for qid, q in v.items():
            if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
                raise ValueError(f"questions.{qid}: 'type' must be one of choice, score, noul")
        return v

    @model_validator(mode="after")
    def _state_json(self) -> "SystemOneRequest":
        _check_entry(self.state, "state")
        return self


# ---------------------------------------------------------------------------------------------
# Internal, model-facing question description
# ---------------------------------------------------------------------------------------------

NOUL_KEYS = ("false", "true")  # slot 0 = no, slot 1 = yes; noul value = P(slot 1)


@dataclass
class QuestionSpec:
    """A question normalised to "one distribution over ``keys``".

    ``keys`` are the answer keys returned to the caller (option names for a choice, "0".."n-1" for a
    score, ("false", "true") for a noul). ``descriptions`` holds what the model reads for each key.
    """

    qid: str
    type: QuestionType
    instructions: Any
    keys: list[str]
    descriptions: list[Any]
    target: list[float] | None = None  # training only: distribution over keys
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.keys)


def to_spec(qid: str, q: ChoiceQuestion | ScoreQuestion | NoulQuestion | dict[str, Any]) -> QuestionSpec:
    if isinstance(q, dict):
        q = SystemOneRequest.model_validate({"state": "", "questions": {qid: q}}).questions[qid]
    if isinstance(q, ChoiceQuestion):
        keys = list(q.criteria)
        return QuestionSpec(qid, "choice", q.instructions, keys, [q.criteria[k] for k in keys])
    if isinstance(q, ScoreQuestion):
        return QuestionSpec(qid, "score", q.instructions, [str(i) for i in range(len(q.criteria))], list(q.criteria))
    crit = q.criteria or {}
    return QuestionSpec(qid, "noul", q.instructions, list(NOUL_KEYS), [crit.get("false"), crit.get("true")])


# ---------------------------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------------------------

def choice_confidence(probs: Sequence[float]) -> float:
    k = len(probs)
    if k <= 1:
        return 1.0
    return max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    mode = max(range(n), key=lambda i: probs[i])
    mid = (n - 1) / 2.0
    d = sum(abs(i - mid) for i in range(n)) / n
    spread = sum(p * abs(i - mode) for i, p in enumerate(probs))
    return max(0.0, min(1.0, 1.0 - spread / d))


def _r(x: float, decimals: int | None) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite probability")
    return round(x, decimals) if decimals is not None else x


def round_simplex(probs: Sequence[float], decimals: int | None) -> list[float]:
    """Round a distribution to ``decimals`` places so it still sums to exactly 1 (largest remainder).

    Plain per-value rounding can drift by several hundredths with many options, and clients such as
    jev-ultrafast reject distributions that do not sum to 1 within 0.02. Largest remainder never reorders
    two options, so the argmax is kept.
    """
    if decimals is None:
        return [float(p) for p in probs]
    if any(not math.isfinite(float(p)) for p in probs):
        raise ValueError("non-finite probability")
    unit = 10 ** decimals
    total = sum(probs) or 1.0
    scaled = [float(p) / total * unit for p in probs]
    floors = [math.floor(x) for x in scaled]
    short = unit - sum(floors)
    order = sorted(range(len(scaled)), key=lambda i: (floors[i] - scaled[i], -scaled[i]))  # largest remainder first
    for i in order[:short]:
        floors[i] += 1
    return [round(f / unit, decimals) for f in floors]


def build_answer(spec: QuestionSpec, probs: Sequence[float], decimals: int | None = 4) -> dict[str, Any]:
    probs = [float(p) for p in probs]
    if spec.type == "noul":
        return {"type": "noul", "noul": _r(probs[1], decimals)}
    shown = round_simplex(probs, decimals)
    if spec.type == "choice":
        best = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "type": "choice",
            "choice": spec.keys[best],
            "probabilities": dict(zip(spec.keys, shown)),
            "confidence": _r(choice_confidence(probs), decimals),
        }
    score = sum(i * p for i, p in enumerate(probs))
    return {
        "type": "score",
        "score": _r(score, decimals),
        "legend": {str(i): d for i, d in enumerate(spec.descriptions)},
        "probabilities": {str(i): p for i, p in enumerate(shown)},
        "confidence": _r(score_confidence(probs), decimals),
    }
