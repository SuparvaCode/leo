"""Data added for Phase 1: browser simulator, Mind2Web converter, reasoning families. No network, no GPU."""
import json
import random

import pytest

from leo.data import example_specs
from leo.data.browser import _requirements_met, catalog_episode, examples
from leo.data.reasoning import FAMILIES


def test_browser_simulator_ends_every_episode_correctly():
    rng = random.Random(0)
    for _ in range(400):
        steps = catalog_episode(rng, max_steps=30)
        page, aid, s, cat, g = steps[-1]
        impossible = g.target < 0 or bool(g.missing)
        assert aid in ("DONE", "BLOCKED"), "every episode must end with a terminal decision"
        if aid == "DONE":
            assert not impossible and _requirements_met(cat, g, s)
            assert (s.page == "detail" and s.detail == g.target) if g.open_target else True
        else:
            assert impossible


def test_browser_examples_use_jev_ultrafast_request_shape():
    for ex in examples(random.Random(1), 300):
        specs = example_specs(ex)  # validates schema and labels
        qs = ex["questions"]
        assert "operation" in qs
        assert set(ex["state"]) == {"page", "elements", "recent_actions"}
        instr = qs["operation"]["instructions"]
        assert set(instr) == {"goal", "rules"} and instr["goal"]
        op = qs["operation"]["label"]
        assert op in qs["operation"]["criteria"]
        target = op.lower() + "_target"
        if op in ("CLICK", "TYPE_TEXT", "SELECT"):
            assert target in qs and qs[target]["label"] in qs[target]["criteria"]
        assert all(len(s.keys) <= 255 for s in specs)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_reasoning_families_validate_and_are_seeded(name):
    fn, _ = FAMILIES[name]
    a = [fn(random.Random(f"x{i}")) for i in range(150)]
    b = [fn(random.Random(f"x{i}")) for i in range(150)]
    assert json.dumps(a) == json.dumps(b), "same seed must give the same examples"
    for ex in a:
        example_specs(ex)


def _fake_task(op: str, value: str = "") -> dict:
    """A tiny Mind2Web-shaped task: a search box, a submit button, a dropdown and some links."""
    html = (
        '<html backend_node_id="1"><body backend_node_id="2">'
        '<a backend_node_id="10">Home</a><a backend_node_id="11">Deals</a>'
        '<input backend_node_id="20" type="text" placeholder="Where to?"/>'
        '<button backend_node_id="21"><text backend_node_id="22">Search</text></button>'
        '<select backend_node_id="30" aria_label="Guests">'
        '<option backend_node_id="31">1 guest</option><option backend_node_id="32">2 guests</option>'
        '<option backend_node_id="33">3 guests</option></select>'
        '<p backend_node_id="40">Find a place to stay.</p></body></html>'
    )
    gold = {"TYPE": "20", "CLICK": "21", "SELECT": "30"}[op]
    cand = lambda bid, tag, extra="": {  # noqa: E731
        "tag": tag, "backend_node_id": bid,
        "attributes": json.dumps({"backend_node_id": bid, "is_clickable": "true", **({"placeholder": extra} if extra else {})})}
    cands = {"10": cand("10", "a"), "11": cand("11", "a"), "20": cand("20", "input", "Where to?"),
             "21": cand("21", "button"), "30": cand("30", "select")}
    action = {"operation": {"op": op, "original_op": op, "value": value}, "cleaned_html": html,
              "pos_candidates": [cands[gold]], "neg_candidates": [c for k, c in cands.items() if k != gold]}
    return {"website": "stays", "annotation_id": "t1", "confirmed_task": "Find a stay in Oslo for 2 guests",
            "action_reprs": [], "actions": [action]}


@pytest.mark.parametrize("op,value,want_op,want_label", [
    ("TYPE", "Oslo", "TYPE_TEXT", "Where to?"),
    ("CLICK", "", "CLICK", "Search"),
    ("SELECT", "2 guests", "SELECT", "2 guests"),
])
def test_mind2web_step_labels_the_human_action(op, value, want_op, want_label):
    pytest.importorskip("jev_ultrafast")
    from leo.data.mind2web import convert_step

    ex = convert_step(_fake_task(op, value), 0, random.Random(0))
    assert ex is not None
    example_specs(ex)
    qs = ex["questions"]
    assert qs["operation"]["label"] == want_op
    head = qs[want_op.lower() + "_target"]
    chosen = head["criteria"][head["label"]]
    assert want_label in chosen["element"]
