"""Shared building blocks for turning labelled rows into varied decision requests.

The benchmark-style prompt (raw text state, bare label names) must be common in training, but so must
the other shapes TypeSafe users send: descriptions, opaque keys, JSON state, "none of the above",
negated and statement-form yes/no questions, and several questions about one state.
"""
from __future__ import annotations

import random
import re
import string
from typing import Any, Sequence

NONE_KEYS = ("other", "none of the above", "none", "something else")
NONE_DESC = ("none of the other options apply", "use this when no other option fits", None)

# Data recipe knobs, set by leo.data.build --recipe. v0 reproduces data/processed of leo-0.6b-v0 exactly.
RECIPES: dict[str, dict[str, Any]] = {
    "v0": {"none_distractor_p": 0.0, "none_keys": NONE_KEYS},
    # v1: a "none"/"other"/"unknown" option also appears when a real option is correct. In v0 such an option
    # was the answer every time it appeared (1,078 of 1,078 training questions), which teaches a shortcut.
    "v1": {"none_distractor_p": 0.15, "none_keys": NONE_KEYS + ("unknown", "not stated", "no match")},
    # v2: v1 plus the browser-agent family (leo.data.browser) and the reasoning families (leo.data.reasoning).
    # The v1 knobs are unchanged, so any benchmark change comes from the added data, not from re-tuned knobs.
    "v2": {"none_distractor_p": 0.15, "none_keys": NONE_KEYS + ("unknown", "not stated", "no match"),
           "extra_families": True},
    # v3: v2 plus multilingual human-labelled sets (leo.data.multilingual) and real-website browser steps
    # from Mind2Web (leo.data.mind2web), with the simulator doubled.
    "v3": {"none_distractor_p": 0.15, "none_keys": NONE_KEYS + ("unknown", "not stated", "no match"),
           "extra_families": True, "multilingual": True, "mind2web": True, "browser_n": 12000},
}
RECIPE: dict[str, Any] = dict(RECIPES["v0"])


def set_recipe(name: str) -> None:
    RECIPE.clear()
    RECIPE.update(RECIPES[name])


def humanize(label: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def snake(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", humanize(label)).strip("_")


def title(label: str) -> str:
    return " ".join(w.capitalize() for w in humanize(label).split())


def letter_keys(n: int, rng: random.Random) -> list[str]:
    style = rng.choice(["upper", "opt", "num"])
    if style == "upper" and n <= 26:
        return list(string.ascii_uppercase[:n])
    if style == "opt":
        return [f"opt_{i + 1}" for i in range(n)]
    return [str(i + 1) for i in range(n)]


def style_options(
    names: Sequence[str],
    descs: dict[str, str] | None,
    rng: random.Random,
    allow_letters: bool = True,
) -> list[tuple[str, Any]]:
    """Return (key, description) per option, in the order of ``names``."""
    weights = {"bare": 0.35, "described": 0.30, "snake": 0.10, "title": 0.10, "letters": 0.15 if allow_letters else 0.0}
    style = rng.choices(list(weights), weights=list(weights.values()))[0]
    d = lambda n: (descs or {}).get(n)  # noqa: E731
    empty = rng.choice(["", None])
    if style == "bare":
        out = [(humanize(n) if rng.random() < 0.5 else n, empty) for n in names]
    elif style == "described":
        out = [(humanize(n), d(n) or empty) for n in names]
    elif style == "snake":
        out = [(snake(n), d(n) or humanize(n)) for n in names]
    elif style == "title":
        out = [(title(n), d(n) or empty) for n in names]
    else:
        keys = letter_keys(len(names), rng)
        out = [(k, humanize(n) + (f": {d(n)}" if d(n) and rng.random() < 0.5 else "")) for k, n in zip(keys, names)]
    if len({k for k, _ in out}) != len(out):  # collisions after normalisation: fall back to raw names
        out = [(n, d(n)) for n in names]
    return out


def choice_question(
    instructions: Any,
    names: Sequence[str],
    gold: str,
    rng: random.Random,
    descs: dict[str, str] | None = None,
    max_options: int | None = None,
    min_options: int = 4,
    p_none: float = 0.08,
    allow_letters: bool = True,
    extra_none_option: bool = False,
    p_none_distractor: float | None = None,
) -> dict[str, Any]:
    """One choice question over ``names`` with ``gold`` correct.

    With probability ``p_none`` the gold option is removed and a "none of the above" option is correct.
    With probability ``p_none_distractor`` (default: the active recipe) a "none" option is added while gold
    stays correct, so a "none"/"other" option is not a shortcut to the answer. TypeSafe's docs recommend
    adding such an option to open-ended lists, where it is usually not the answer.
    """
    if p_none_distractor is None:
        p_none_distractor = RECIPE["none_distractor_p"]
    names = list(dict.fromkeys(names))
    if max_options and len(names) > max_options:
        k = rng.randint(min(min_options, len(names)), max_options)
        others = [n for n in names if n != gold]
        names = [gold] + rng.sample(others, k - 1)
        rng.shuffle(names)
    none_is_gold = len(names) >= 3 and rng.random() < p_none
    if none_is_gold:
        names = [n for n in names if n != gold]
    # Draw only when the knob is on, so recipe v0 consumes exactly the random numbers it always did.
    add_none = none_is_gold or extra_none_option or (p_none_distractor > 0 and rng.random() < p_none_distractor)
    opts = style_options(names, descs, rng, allow_letters)
    by_name = dict(zip(names, (k for k, _ in opts)))
    criteria = {k: v for k, v in opts}
    if add_none:
        nk = rng.choice([k for k in RECIPE["none_keys"] if k not in criteria] or ["other"])
        criteria[nk] = rng.choice(NONE_DESC)
        label = nk if none_is_gold else by_name[gold]
    else:
        label = by_name[gold]
    return {"type": "choice", "instructions": instructions, "criteria": criteria, "label": label}


def noul_question(instructions: Any, label: bool | float, criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions, "label": float(label)}
    if criteria:
        q["criteria"] = criteria
    return q


def score_question(instructions: Any, levels: Sequence[Any], label: int | Sequence[float]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels), "label": label}


def about_noul(topic_phrase: str, is_true: bool, rng: random.Random, subject: str = "text") -> dict[str, Any]:
    """Yes/no about one label, sometimes negated or as a statement (literal reading matters)."""
    r = rng.random()
    if r < 0.12:
        return noul_question(f"Is this {subject} NOT about {topic_phrase}?", not is_true)
    if r < 0.45:
        return noul_question(f"This {subject} is about {topic_phrase}.", is_true)
    return noul_question(f"Is this {subject} about {topic_phrase}?", is_true)


def wrap_state(text: str, rng: random.Random, fields: Sequence[str] = ("text", "message", "content", "body"),
               p_raw: float = 0.65) -> tuple[Any, str | None]:
    """Raw string most of the time (matches the benchmark), otherwise a small JSON object."""
    if rng.random() < p_raw:
        return text, None
    field = rng.choice(list(fields))
    state: dict[str, Any] = {field: text}
    if rng.random() < 0.4:
        state = {"id": f"{rng.choice('ABCDEFGH')}-{rng.randint(100, 9999)}", **state}
    if rng.random() < 0.2:
        state["received_at"] = f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"
    return state, field


def refer(field: str | None, instruction: str, rng: random.Random) -> str:
    """Sometimes point the instruction at the state field by path, as TypeSafe docs recommend."""
    if field and rng.random() < 0.4:
        return f"{instruction} (see `{field}`)"
    return instruction


def qids(n: int, rng: random.Random) -> list[str]:
    pool = ["q", "answer", "decision", "label", "check", "x", "result"]
    base = rng.choice(pool)
    return [f"{base}_{i}" for i in range(n)]


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " ..."
