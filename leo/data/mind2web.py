"""Real-website browser decisions from Mind2Web (osunlp/Mind2Web, CC BY 4.0), in jev-ultrafast's request format.

Each Mind2Web step is a real page snapshot (cleaned HTML), a natural-language task, the action history and
the human-chosen action (CLICK / TYPE / SELECT on one element). A step becomes one jev-ultrafast decision:

* the page's text and a set of candidate elements (the human's target plus up to ``max_elements - 1``
  interactive distractors, in document order) become the observation that ``snapshot.js`` would produce;
* ``jev_ultrafast.model.choose`` builds the request (network call intercepted), so field names,
  instructions and option layout are byte-identical to what the agent sends at run time;
* the human action is the label for the ``operation`` head and for the matching ``*_target`` head.

Mind2Web has no final-page snapshots, so DONE/BLOCKED come only from the simulator (leo.data.browser).

    python -m leo.data.mind2web          # converts every train file once -> data/cache/mind2web_train.jsonl
"""
from __future__ import annotations

import html
import json
import random
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "mind2web_train.jsonl"
SOURCE = "mind2web"
VOID = {"input", "img", "br", "hr", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "summary", "option", "label"}
ROLES = {"button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemradio", "option", "gridcell",
         "combobox", "textbox", "searchbox", "spinbutton"}
TEXT_INPUTS = {"", "text", "search", "email", "url", "tel", "number", "password", "date"}
MAX_SELECT_OPTIONS = 20   # options kept per dropdown
MAX_SELECT_TOTAL = 120    # options across all dropdowns on one page (a single select_target question)


class _Tree(HTMLParser):
    """Text content, tag, attributes and <select> options for every element carrying a backend_node_id."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.buf: list[str] = []
        self.pos = 0
        self.stack: list[tuple[str, str | None, int]] = []
        self.text: dict[str, str] = {}
        self.tag: dict[str, str] = {}
        self.attrs: dict[str, dict[str, str]] = {}
        self.order: dict[str, int] = {}
        self.options: dict[str, list[str]] = {}
        self.select: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        bid = a.get("backend_node_id")
        if bid:
            self.tag[bid] = tag
            self.attrs[bid] = a
            self.order.setdefault(bid, len(self.order))
        if tag == "select" and bid:
            self.select.append(bid)
            self.options[bid] = []
        if tag in VOID:
            if bid:
                self.text[bid] = ""
            return
        self.stack.append((tag, bid, self.pos))

    def handle_endtag(self, tag: str) -> None:
        # pop to the matching tag (cleaned_html is generated and well formed; this also tolerates strays)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                for t, bid, start in self.stack[i:][::-1]:
                    if bid:
                        self.text[bid] = self._slice(start)
                    if t == "select" and self.select:
                        self.select.pop()
                    if t == "option" and self.select and bid:
                        self.options[self.select[-1]].append(self.text.get(bid, ""))
                del self.stack[i:]
                return

    def _slice(self, start: int) -> str:
        return " ".join("".join(self.buf[start:]).split())

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.buf.append(" " + data)
            self.pos += 1


def _attr_json(c: dict[str, Any]) -> dict[str, str]:
    try:
        return json.loads(c.get("attributes") or "{}")
    except json.JSONDecodeError:
        return {}


def _role(tag: str, a: dict[str, str]) -> str:
    r = a.get("role", "")
    if r in ROLES:
        return r
    t = a.get("type", "").lower()
    if tag == "a":
        return "link"
    if tag in ("button", "summary") or (tag == "input" and t in ("button", "submit", "reset", "image")):
        return "button"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        if t in ("checkbox", "radio"):
            return t
        if t == "search":
            return "searchbox"
        if t == "number":
            return "spinbutton"
        return "textbox"
    return "button"  # a clickable div/span/li the page wired up with a handler


def _label(tag: str, a: dict[str, str], text: str) -> str:
    for k in ("aria_label", "aria-label"):
        if a.get(k):
            return a[k]
    if text and tag not in ("input", "select", "textarea"):
        return text
    for k in ("placeholder", "title", "alt", "name", "value"):
        if a.get(k) and not (k == "value" and tag in ("input", "textarea") and a.get("type", "text") in TEXT_INPUTS):
            return a[k]
    return text or tag


def _interactive(tag: str, a: dict[str, str]) -> bool:
    return tag in INTERACTIVE_TAGS or a.get("role") in ROLES or a.get("is_clickable") == "true"


def _history(reprs: list[str]) -> list[dict[str, Any]]:
    out = []
    for r in reprs:
        m = re.match(r"\[(.*?)\]\s+(.*?)\s*->\s*(CLICK|TYPE|SELECT|HOVER|ENTER)(?::\s*(.*))?$", r.strip())
        if not m:
            continue
        _, label, op, value = m.groups()
        kind = {"TYPE": "fill", "SELECT": "select"}.get(op, "click")
        out.append({"action": (label or "").strip()[:120] or "element", "kind": kind,
                    "text": value.strip() if op == "TYPE" and value else None, "page_changed": True})
    return out


def convert_step(task: dict[str, Any], i: int, rng: random.Random, max_elements: int = 25,
                 max_text_chars: int = 1200) -> dict[str, Any] | None:
    """Caps keep the rendered state under the 2,048-token training window; the encoder cuts from the end,
    which would drop elements and history rather than page text."""
    from leo.data.browser import build_body, target_key

    act = task["actions"][i]
    op = act["operation"]["op"]
    if op not in ("CLICK", "TYPE", "SELECT") or not act["pos_candidates"]:
        return None
    tree = _Tree()
    try:
        tree.feed(act["cleaned_html"])
        tree.close()
    except Exception:
        return None
    pos = act["pos_candidates"][0]
    gold = pos["backend_node_id"]
    if gold not in tree.tag:
        return None
    negs = [c for c in act["neg_candidates"]
            if c["backend_node_id"] in tree.tag and _interactive(c["tag"], _attr_json(c))
            and (tree.text.get(c["backend_node_id"]) or _attr_json(c).get("aria_label") or _attr_json(c).get("placeholder"))]
    negs = rng.sample(negs, min(len(negs), rng.randint(max(3, max_elements // 3), max_elements - 1)))
    cands = sorted([pos] + negs, key=lambda c: tree.order.get(c["backend_node_id"], 1 << 30))

    actions: list[dict[str, Any]] = []
    gold_id = None
    seen_labels: set[str] = set()
    for c in cands:
        bid = c["backend_node_id"]
        tag = tree.tag.get(bid, c["tag"])
        a = {**tree.attrs.get(bid, {}), **_attr_json(c)}
        text = html.unescape(tree.text.get(bid, ""))[:150]
        label = html.unescape(_label(tag, a, text)).strip()[:150]
        is_gold = bid == gold
        if not is_gold and label.lower() in seen_labels:  # duplicate-looking distractors add nothing
            continue
        seen_labels.add(label.lower())
        base = {"node": int(bid) if bid.isdigit() else hash(bid) & 0xFFFFFF, "role": _role(tag, a), "label": label or tag}
        if a.get("type") in ("checkbox", "radio"):
            base["checked"] = "true" if "checked" in a else "false"
        if a.get("aria_expanded"):
            base["expanded"] = a["aria_expanded"]
        if tag == "select" or (is_gold and op == "SELECT"):
            opts = list(dict.fromkeys(o for o in tree.options.get(bid, []) if o))
            if not opts:
                if is_gold:
                    return None
                continue
            current = a.get("input_value") or opts[0]
            if len(opts) > MAX_SELECT_OPTIONS:  # e.g. country or year lists: keep the chosen and current option plus
                want = act["operation"]["value"].strip().lower() if is_gold else None  # a sample, in page order
                must = {o for o in opts if (want and o.strip().lower() == want) or o == current}
                rest = [o for o in opts if o not in must]
                keep = must | set(rng.sample(rest, max(0, MAX_SELECT_OPTIONS - len(must))))
                opts = [o for o in opts if o in keep]
            n_select = sum(1 for x in actions if x["kind"] == "select")
            if not is_gold and n_select + len(opts) > MAX_SELECT_TOTAL:
                continue  # one select_target question must stay well under TypeSafe's 255-option limit
            for o in dict.fromkeys(opts):
                if o == current and not (is_gold and o == act["operation"]["value"]):
                    continue
                actions.append({**base, "kind": "select", "value": o, "current_value": current,
                                "label": f"{base['label']} → {o}"})
                if is_gold and o.strip().lower() == act["operation"]["value"].strip().lower():
                    gold_id = len(actions) - 1
            if is_gold and gold_id is None:
                return None
            continue
        editable = tag == "textarea" or (tag == "input" and a.get("type", "").lower() in TEXT_INPUTS) \
            or base["role"] in ("textbox", "searchbox", "combobox") and tag in ("input", "textarea")
        value = a.get("input_value", a.get("value", "")) if tag in ("input", "textarea") else ""
        if is_gold and op == "TYPE":
            editable = True
        if editable:
            actions.append({**base, "kind": "fill", "value": value})
            if is_gold and op == "TYPE":
                gold_id = len(actions) - 1
            actions.append({**base, "kind": "click", "value": value, "label": "Open " + base["label"]})
            if is_gold and op == "CLICK":
                gold_id = len(actions) - 1
        else:
            if is_gold and op == "TYPE":
                return None
            actions.append({**base, "kind": "click", "value": value})
            if is_gold:
                gold_id = len(actions) - 1
    if gold_id is None:
        return None
    for k, a in enumerate(actions):
        a["id"] = f"e{k + 1}"
    if rng.random() < 0.7:
        actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560})
    if rng.random() < 0.3:
        actions.append({"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560})
    actions.append({"id": "wait", "kind": "wait", "label": "Wait for the page to update"})

    page_text = "\n".join(" ".join(t.split()) for t in tree.buf)[:max_text_chars]  # one line per text node
    site = task["website"]
    page = {"url": f"https://www.{site}.com/" if "." not in site else f"https://{site}.com/",
            "title": site.replace(".", " ").title(), "text": page_text, "actions": actions}
    history = _history(task["action_reprs"][:i])
    body = build_body(page, task["confirmed_task"], history)
    operation, tgt = target_key(page, actions[gold_id]["id"])
    qs = body["questions"]
    out = {"operation": {**qs["operation"], "label": operation}}
    qid = operation.lower() + "_target"
    if tgt and qid in qs:
        out[qid] = {**qs[qid], "label": tgt}
    return {"source": SOURCE, "state": body["state"], "questions": out,
            "meta": {"annotation_id": task["annotation_id"], "step": i, "website": site}}


def convert_all(files: list[Path], seed: int = 13) -> int:
    from leo.data import example_specs

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(CACHE, "w", encoding="utf-8") as fh:
        for f in files:
            tasks = json.load(open(f, encoding="utf-8"))
            kept = invalid = 0
            for t in tasks:
                rng = random.Random(f"{seed}:{t['annotation_id']}")
                for i in range(len(t["actions"])):
                    ex = convert_step(t, i, rng)
                    if not ex:
                        continue
                    try:
                        example_specs(ex)  # same validation the data build applies
                    except ValueError:
                        invalid += 1
                        continue
                    fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    kept += 1
            n += kept
            print(f"{f.name}: {len(tasks)} tasks -> {kept} steps, {invalid} invalid skipped (total {n})", flush=True)
            del tasks
    return n


def load(rng: random.Random, n: int, split: str, dev_frac: float = 0.03) -> list[dict[str, Any]]:
    """Examples from the converted cache, split by task (annotation id) so dev never shares a task with train."""
    import hashlib

    rows = []
    with open(CACHE, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            h = int(hashlib.sha1(ex["meta"]["annotation_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if (h < dev_frac) == (split == "dev"):
                ex.pop("meta")
                rows.append(ex)
    rng.shuffle(rows)
    return rows[:n]


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download("osunlp/Mind2Web", repo_type="dataset", allow_patterns=["data/train/*"]))
    files = sorted((snap / "data" / "train").glob("train_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    print(convert_all(files), "steps")
