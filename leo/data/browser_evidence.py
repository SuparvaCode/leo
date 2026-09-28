"""Browser steps where DONE needs visible evidence, not a finished-looking action history.

leo-1.7b-v3 learned a shortcut from the catalog simulator (leo.data.browser): once the action history covers
every part of the goal, it answers DONE, whatever the page shows. On Google Flights it chose DONE with
P = 0.985 while a seat-class listbox was the only thing on screen, and with P = 0.999 on a blank page
between two renders. In the v3 simulator results were always on screen, overlays never covered the page and
DONE was the label on 15% of steps, so the history alone predicted DONE well.

This family breaks that link. Every episode is a search form whose history can look complete while the page
does not prove it:

* custom dropdowns that open an overlay listbox hiding the rest of the page (pick the value to close it);
* a counter popover (guests, tickets, seats) closed by its own "Done" / "Apply" button, so a click on a
  button called Done shows up in the history without meaning the task is done;
* deferred results: before the search is submitted the page shows unrelated "Popular" items, and after a
  change the results still describe the previous search until it is submitted again;
* loading and blank transitional renders, where the right move is WAIT;
* a requested value the site does not offer, where the honest answer is BLOCKED.

DONE is the label only when the page itself shows results (or the opened item) for exactly what the goal
asks. Requests are built by jev-ultrafast's own request builder, as in leo.data.browser.

Hold-out rule (as leo.data.browser): no flight search, no encyclopedia, none of the Forma fixture's names.
Two themes are held out of training entirely and used only by scripts/done_probe.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from leo.data.browser import CITIES, THEMES, Theme, _example

SOURCE = "syn_browser_evidence"
PROBE_THEMES = ("Earshot", "Learnwell")  # never in training data; scripts/done_probe.py only
TRAIN_THEMES = tuple(t for t in THEMES if t.brand not in PROBE_THEMES)
HELD_OUT_THEMES = tuple(t for t in THEMES if t.brand in PROBE_THEMES)

COUNTERS = [("Guests", "guest"), ("Tickets", "ticket"), ("People", "person"), ("Seats", "seat"), ("Attendees", "attendee")]
PLURAL = {"person": "people"}
COMMIT = ["Done", "Done", "Done", "Apply", "OK", "Confirm"]


def _units(unit: str, n: int) -> str:
    return f"{n} {unit if n == 1 else PLURAL.get(unit, unit + 's')}"


@dataclass
class Site:
    theme: Theme
    items: list[dict[str, Any]]
    facets: list[tuple[str, str, list[str], str]]  # (label, key, offered values, all label)
    counter: tuple[str, str] | None
    live: bool                  # closing an overlay re-runs an already submitted search
    autocomplete: bool
    button: str
    commit: str
    facet_style: int
    close_button: bool
    url: str


@dataclass
class Goal:
    text: str
    query: str | None
    selects: dict[str, str]
    count: int | None
    target: int | None          # item to open; None = stop at the results
    missing: bool = False       # a requested value is not offered


@dataclass
class State:
    field_value: str = ""
    chosen: bool = True
    suggestions: bool = False
    selects: dict[str, str] = field(default_factory=dict)
    count: int = 1
    draft: int = 1
    overlay: str | None = None  # facet key, "counter" or None
    applied: tuple | None = None  # (query, selects, count) of the last search that ran
    loading: int = 0             # renders still to wait for
    blank: bool = False
    detail: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def make_site(rng: random.Random, themes: tuple[Theme, ...]) -> Site:
    th = rng.choice(themes)
    pool = rng.sample(list(CITIES) if th.search_attr == "city" else list(th.search_words), 3)
    items, names = [], set()
    while len(items) < rng.randint(4, 8):
        name = f"{rng.choice(th.name_a)} {rng.choice(th.name_b)}"
        if name in names:
            continue
        names.add(name)
        items.append({"name": name, "search": rng.choice(pool), "attrs": {f.key: rng.choice(f.values) for f in th.facets}})
    facets = []
    for f in rng.sample(list(th.facets), rng.randint(1, len(th.facets))):
        offered = list(f.values)
        facets.append((f.label, f.key, offered, f.all_label))
    counter = rng.choice(COUNTERS) if rng.random() < 0.6 else None
    host = th.brand.lower() + rng.choice([".example", ".test"])
    return Site(th, items, facets, counter, rng.random() < 0.3, th.search_attr == "city" and rng.random() < 0.5,
                rng.choice(th.button), rng.choice(COMMIT), rng.randrange(3), rng.random() < 0.3,
                f"https://{host}/{th.plural}")


def make_goal(site: Site, rng: random.Random) -> Goal:
    th = site.theme
    t = rng.randrange(len(site.items))
    it = site.items[t]
    query = it["search"]
    selects = {key: it["attrs"][key] for _, key, _, _ in site.facets if rng.random() < 0.8}
    count = rng.choice([1, 2, 2, 3, 4]) if site.counter and rng.random() < 0.85 else None
    open_it = rng.random() < 0.45
    desc = " ".join(selects.values())
    where = f" in {query}" if th.search_attr == "city" else f" for \"{query}\""
    who = f" for {_units(site.counter[1], count)}" if count else ""
    style = rng.randrange(3)
    if open_it:
        text = [f"Find {('a ' + desc + ' ') if desc else 'the '}{th.noun}{where}{who} and open {it['name']}.",
                f"Search {th.plural}{where}{who}{', ' + desc if desc else ''}, then open {it['name']}.",
                f"Open {it['name']}: it is a {desc + ' ' if desc else ''}{th.noun}{where}{who}."][style]
    else:
        text = [f"Find {desc + ' ' if desc else ''}{th.plural}{where}{who}. Stop when matching {th.plural} are visible. "
                f"Do not open any {th.noun}.",
                f"Show {desc + ' ' if desc else ''}{th.plural}{where}{who}. Stop when matching results are visible; "
                f"do not open or book anything.",
                f"Search for {desc + ' ' if desc else ''}{th.plural}{where}{who} and stop once the matching results are shown."][style]
    return Goal(text.replace("  ", " "), query, selects, count, t if open_it else None)


def _wanted(site: Site, g: Goal) -> tuple:
    return (g.query.lower() if g.query else None, tuple(sorted(g.selects.items())), g.count or 1)


def _current(site: Site, s: State) -> tuple:
    return (s.field_value.lower() or None, tuple(sorted((k, v) for k, v in s.selects.items() if v)), s.count)


def _matches(site: Site, s: State) -> list[int]:
    if s.applied is None:
        return []
    q, sel, _ = s.applied
    sel = dict(sel)
    return [i for i, it in enumerate(site.items)
            if (not q or q in it["search"].lower()) and all(it["attrs"][k] == v for k, v in sel.items())]


def _facet_label(site: Site, label: str, cur: str) -> str:
    return [f"{label}: {cur}", f"Change {label.lower()}. {cur}", f"{label} {cur}"][site.facet_style]


def observe(site: Site, s: State, rng: random.Random) -> tuple[dict[str, Any], str]:
    """(page in snapshot.js shape, kind of screen) for the current state."""
    th = site.theme
    texts: list[str] = []
    acts: list[dict[str, Any]] = []
    node = [300]

    def add(text: str | None, **a: Any) -> dict[str, Any] | None:
        if text:
            texts.append(text)
        if not a:
            return None
        node[0] += 1
        a.setdefault("value", "")
        acts.append({"node": node[0], **a})
        return acts[-1]

    title = f"{th.brand} · {th.plural.capitalize()}"
    url = site.url + (f"?q={s.applied[0].replace(' ', '+')}" if s.applied and s.applied[0] else "")
    kind = "form"
    if s.blank:
        kind = "blank"
    elif s.loading:
        kind = "loading"
        add(rng.choice(["Loading…", f"Searching {th.plural}…", "Updating results…"]))
    elif s.overlay and s.overlay != "counter":
        kind = "overlay"
        label, key, offered, all_label = next(f for f in site.facets if f[1] == s.overlay)
        if rng.random() < 0.4:
            add(label)
        cur = s.selects.get(key, "")
        for v in [all_label] + offered:
            sel = (v == all_label and not cur) or v == cur
            add(v, role="option", label=v, kind="click", selected=str(sel).lower(), _pick=("" if v == all_label else v))
        if site.close_button:
            add(None, role="button", label="Close", kind="click", _close=True)
    elif s.overlay == "counter":
        kind = "popover"
        name, unit = site.counter
        add(name)
        add(unit.capitalize() + ("s" if unit != "person" else ""))
        add(str(s.draft))
        add(None, role="button", label=f"Remove {unit}", kind="click", _delta=-1)
        add(None, role="button", label=f"Add {unit}", kind="click", _delta=1)
        if rng.random() < 0.5:
            add("Cancel", role="button", label="Cancel", kind="click", _cancel=True)
        add(site.commit, role="button", label=site.commit, kind="click", _commit=True)
    elif s.detail is not None:
        kind = "detail"
        it = site.items[s.detail]
        title = f"{it['name']} · {th.brand}"
        url = site.url + "/" + it["name"].lower().replace(" ", "-")
        add(f"← Back to {th.plural}", role="button", label=f"← Back to {th.plural}", kind="click", _back=True)
        add(it["name"])
        add(" · ".join([it["search"]] + list(it["attrs"].values())))
        add(f"Save this {th.noun}", role="button", label=f"Save this {th.noun}", kind="click")
    else:
        add(th.brand.lower() + ".", role="link", label=th.brand.lower() + ".", kind="click")
        for n in th.nav[:2]:
            add(n, role="link", label=n, kind="click")
        add(th.tagline)
        role = "combobox" if site.autocomplete else "textbox"
        fld = add(None, role=role, label=th.search_label, kind="fill", value=s.field_value)
        if role == "combobox":
            fld["expanded"] = str(s.suggestions).lower()
        if s.suggestions:
            opts = [s.field_value] + [c for c in rng.sample(CITIES, 3) if c.lower() != s.field_value.lower()][:2]
            for o in opts:
                sub = rng.choice(["City", "Region", "Area"])
                add(f"{o}\n{sub}", role="option", label=f"{o}, {sub}", kind="click", _suggest=o)
        for label, key, _, all_label in site.facets:
            cur = s.selects.get(key) or all_label
            add(label if site.facet_style != 0 else None, role=rng.choice(["combobox", "button"]),
                label=_facet_label(site, label, cur), kind="click", value=cur, expanded="false", _open=key)
        if site.counter:
            add(None, role="button", label=_units(site.counter[1], s.count), kind="click", expanded="false", _open="counter")
        add(site.button, role="button", label=site.button, kind="click", _submit=True)
        if s.applied is None:
            kind = "unsubmitted"
            add(rng.choice([f"Popular {th.plural}", f"Trending {th.plural}", f"Recently viewed {th.plural}"]))
            if rng.random() < 0.5:
                add(f"Search to see matching {th.plural}.")
            shown = rng.sample(range(len(site.items)), min(3, len(site.items)))
        else:
            q, sel, cnt = s.applied
            shown = _matches(site, s)
            bits = [q.title() if q else f"All {th.plural}"] + [v for _, v in sel]
            if site.counter:
                bits.append(_units(site.counter[1], cnt))
            head = f"{len(shown)} {th.plural if len(shown) != 1 else th.noun} · " + " · ".join(bits)
            add(rng.choice([head, "Results: " + head, "Showing " + head]))
            if not shown:
                add(f"No {th.plural} match this search.")
        for i in shown:
            it = site.items[i]
            add(it["name"], role="link", label=it["name"], kind="click", _item=i)
            add(" · ".join([it["search"]] + list(it["attrs"].values())))
    add(None, role="link", label="Terms", kind="click") if kind in ("form", "unsubmitted", "detail") else None
    for i, a in enumerate(acts):
        a["id"] = f"e{i + 1}"
    acts.append({"id": "wait", "kind": "wait", "label": "Wait for the page to update"})
    if kind == "form" and s.applied is not None and _current(site, s) != s.applied:
        kind = "stale"
    return {"url": url, "title": title, "text": "\n".join(dict.fromkeys(texts))[:6000], "actions": acts}, kind


def oracle(site: Site, g: Goal, s: State, page: dict[str, Any]) -> str:
    acts = page["actions"]
    by = lambda pred: next((a["id"] for a in acts if pred(a)), None)  # noqa: E731
    if s.blank or s.loading:
        return "wait"
    if s.overlay == "counter":
        want = g.count or s.count
        if s.draft != want:
            return by(lambda a: a.get("_delta") == (1 if want > s.draft else -1))
        return by(lambda a: a.get("_commit"))
    if s.overlay:
        want = g.selects.get(s.overlay)
        if want is None:
            want = s.selects.get(s.overlay, "")
            return by(lambda a: a.get("_close")) or by(lambda a: a.get("_pick") == want)
        return by(lambda a: a.get("_pick") == want) or "BLOCKED"  # the requested value is not offered
    if s.detail is not None:
        if s.detail == g.target and s.applied == _wanted(site, g):
            return "DONE"
        return by(lambda a: a.get("_back"))
    if g.query:
        if s.suggestions and s.field_value.lower() == g.query.lower():
            return by(lambda a: a.get("_suggest", "").lower() == g.query.lower())
        if s.field_value.lower() != g.query.lower() or not s.chosen:
            return by(lambda a: a["kind"] == "fill")
    for _, key, _, _ in site.facets:
        if key in g.selects and s.selects.get(key, "") != g.selects[key]:
            return by(lambda a: a.get("_open") == key)
    if g.count and s.count != g.count:
        return by(lambda a: a.get("_open") == "counter")
    if s.applied != _wanted(site, g):
        return by(lambda a: a.get("_submit"))
    if g.target is None:
        return "DONE"
    return by(lambda a: a.get("_item") == g.target) or "BLOCKED"


def apply(site: Site, s: State, page: dict[str, Any], aid: str, rng: random.Random) -> None:
    rec = {"action": "Wait for the page to update", "kind": "wait", "text": None, "page_changed": True}
    if aid == "wait":
        rec["page_changed"] = bool(s.blank or s.loading)
        if s.blank:
            s.blank = False
        elif s.loading:
            s.loading -= 1
        s.history.append(rec)
        return
    a = next(x for x in page["actions"] if x["id"] == aid)
    rec.update(action=a["label"], kind="fill" if a["kind"] == "fill" else "click")

    def rerun() -> None:
        if site.live and s.applied is not None:
            s.applied = _current(site, s)
            s.loading = rng.choice([0, 0, 1])

    if a["kind"] == "fill":
        s.field_value = s.goal_query  # type: ignore[attr-defined]
        rec["text"] = s.field_value
        s.chosen = not site.autocomplete
        s.suggestions = site.autocomplete
    elif "_suggest" in a:
        s.field_value, s.chosen, s.suggestions = a["_suggest"], True, False
    elif "_open" in a:
        s.overlay = a["_open"]
        s.draft = s.count
    elif "_pick" in a:
        s.selects[s.overlay] = a["_pick"]
        s.overlay = None
        rerun()
    elif a.get("_close"):
        s.overlay = None
    elif "_delta" in a:
        s.draft = max(1, s.draft + a["_delta"])
    elif a.get("_commit"):
        s.count, s.overlay = s.draft, None
        rerun()
    elif a.get("_cancel"):
        s.overlay = None
    elif a.get("_submit"):
        s.applied = _current(site, s)
        s.suggestions = False
        r = rng.random()
        s.loading, s.blank = (1, False) if r < 0.3 else (0, True) if r < 0.45 else (0, False)
    elif "_item" in a:
        s.detail = a["_item"]
        s.blank = rng.random() < 0.15
    elif a.get("_back"):
        s.detail = None
    else:
        rec["page_changed"] = False
    s.history.append(rec)


def _perturb(site: Site, g: Goal, s: State, rng: random.Random) -> None:
    r = rng.random()
    if r < 0.2:  # an earlier search for something else is still on screen
        other = rng.choice([it["search"] for it in site.items])
        s.field_value, s.applied = other, (other.lower(), (), 1)
        s.history.append({"action": site.theme.search_label, "kind": "fill", "text": other, "page_changed": True})
        s.history.append({"action": site.button, "kind": "click", "text": None, "page_changed": True})
    elif r < 0.3:  # a dropdown was left open
        s.overlay = rng.choice(site.facets)[1]
        s.history.append({"action": _facet_label(site, *next((f[0], f[3]) for f in site.facets if f[1] == s.overlay)),
                          "kind": "click", "text": None, "page_changed": True})
    elif r < 0.35:
        s.blank = True


def episode(rng: random.Random, themes: tuple[Theme, ...] = TRAIN_THEMES, max_steps: int = 24):
    """[(page, action id, history, goal text, screen kind)] for one simulated episode."""
    site = make_site(rng, themes)
    g = make_goal(site, rng)
    if rng.random() < 0.07 and g.selects:  # a requested value this site does not offer
        key = rng.choice(list(g.selects))
        i = next(i for i, f in enumerate(site.facets) if f[1] == key)
        label, _, offered, all_label = site.facets[i]
        site.facets[i] = (label, key, [v for v in offered if v != g.selects[key]], all_label)
        g.missing = True
    s = State()
    s.goal_query = g.query or ""  # type: ignore[attr-defined]
    _perturb(site, g, s, rng)
    out = []
    detours = 0
    for _ in range(max_steps):
        page, kind = observe(site, s, rng)
        aid = oracle(site, g, s, page)
        if aid is None:
            raise AssertionError(f"oracle found no action on a {kind} screen")
        # Late detours: the history already covers the whole goal when a control is reopened or the page
        # re-renders blank. This is the exact situation v3 answered DONE on (seat-class listbox, blank page).
        if kind in ("form", "stale") and detours < 2 and rng.random() < 0.35:
            detours += 1
            r = rng.random()
            if r < 0.55:
                opener = next(a for a in page["actions"] if a.get("_open") in {k for _, k, _, _ in site.facets})
                if rng.random() < 0.5:
                    opener = next((a for a in page["actions"] if a.get("_open") == "counter"), opener)
                s.history.append({"action": opener["label"], "kind": "click", "text": None, "page_changed": True})
                s.overlay, s.draft = opener["_open"], s.count
            elif r < 0.8:
                s.blank = True
            else:
                s.loading = 1
            continue
        out.append((page, aid, list(s.history[-10:]), g.text, kind))
        if aid in ("DONE", "BLOCKED"):
            break
        apply(site, s, page, aid, rng)
    return out


def examples(rng: random.Random, n: int, themes: tuple[Theme, ...] = TRAIN_THEMES, with_kind: bool = False,
             keep: float = 0.7) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while len(out) < n:
        for page, aid, history, goal, kind in episode(rng, themes):
            if rng.random() < (1.0 if aid in ("DONE", "BLOCKED") or kind != "form" else keep):
                ex = _example(page, aid, history, goal)
                if with_kind:
                    ex["kind"] = kind
                out.append(ex)
    rng.shuffle(out)
    return out[:n]
