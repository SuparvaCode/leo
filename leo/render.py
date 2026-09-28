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
