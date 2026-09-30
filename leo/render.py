"""Turn JSON state, instructions and option descriptions into text the model reads.

Objects and arrays are flattened into ``path: value`` lines using the same dot-and-index paths
TypeSafe questions use to point at parts of the state (``ticket.messages[0].text``), so a question
that names `ticket.messages[0].text` finds that exact string next to its value.
"""
from __future__ import annotations

import json
from typing import Any


def _leaf(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False)


def _walk(value: Any, path: str, out: list[str]) -> None:
    if isinstance(value, dict):
        if not value:
            out.append(f"{path}: {{}}" if path else "{}")
        for k, v in value.items():
            _walk(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list):
        if not value:
            out.append(f"{path}: []" if path else "[]")
        for i, v in enumerate(value):
            _walk(v, f"{path}[{i}]", out)
    else:
        out.append(f"{path}: {_leaf(value)}" if path else _leaf(value))


def render(value: Any) -> str:
    """Render any EntryType. Strings pass through unchanged; JSON is flattened to path lines."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    lines: list[str] = []
    _walk(value, "", lines)
    return "\n".join(lines)


# ------------------------------------------------------------------------------ yes/no conditions

_QUESTION_WORDS = {"is", "are", "was", "were", "am", "do", "does", "did", "can", "could", "will", "would", "should",
                   "shall", "has", "have", "had", "may", "might", "must", "which", "what", "who", "whom", "whose",
                   "when", "where", "why", "how"}
_SUBJECTS = {"the", "this", "that", "these", "those", "it", "its", "they", "he", "she", "i", "we", "you", "my", "our",
             "their", "his", "her", "a", "an", "there", "someone", "somebody", "everyone", "nobody", "user", "customer"}
CONDITION_TEMPLATES = {
    "input": "Is the input {c}?",
    "describe": "Does this describe the input: {c}?",
    "true_of": "Is this true of the input: {c}?",
}


def condition_form(text: str) -> str:
    """Classify a yes/no instruction: "question", "statement" or "fragment" ("angry", "asking for a refund",
    "Contains personal data"). Only fragments are rewritten by ``canonical_condition``."""
    s = text.strip()
    if not s:
        return "fragment"
    words = s.split()
    first = words[0].lower().strip("\"'([")
    if s.endswith("?") or first in _QUESTION_WORDS:
        return "question"
    if len(words) >= 3 and (first in _SUBJECTS or s.endswith(".")) and len(words) <= 60:
        return "statement"
    # Long rule-like fragments ("Contains hate speech, slurs, ...") read fine as sent; rewriting them did not help
    # on scripts/condition_probe.py, so only short conditions count as fragments.
    return "fragment" if len(words) <= 8 else "statement"


def canonical_condition(instructions: Any, template: str = "input") -> Any:
    """Rewrite a bare yes/no condition into a full question, keeping the caller's words verbatim.

    A question without its "?" gets one; statements and structured instructions pass through unchanged.
    """
    if not isinstance(instructions, str):
        return instructions
    s = instructions.strip()
    form = condition_form(s)
    if form == "question":
        return s if s.endswith("?") else s.rstrip(".") + "?"
    if form == "statement":
        return instructions
    c = s.rstrip(".?!").strip()
    if c[:1].isupper() and not c[:2].isupper():
        c = c[0].lower() + c[1:]
    return CONDITION_TEMPLATES[template].format(c=c)


def option_text(qtype: str, key: str, description: Any) -> str:
    """What the model reads inside one option block.

    Choice shows ``key: description`` (or the bare key). Score levels show only their description:
    TypeSafe states that levels are judged without their number. Noul shows yes/no plus the optional
    criteria text.
    """
    desc = render(description).strip()
    if qtype == "score":
        return desc
    if qtype == "noul":
        word = "yes" if key == "true" else "no"
        return f"{word}: {desc}" if desc else word
    return f"{key}: {desc}" if desc else key
