import numpy as np
import pytest
from pydantic import ValidationError

from leo.encode import Encoder, block_causal_mask, collate, pack_rows
from leo.render import render
from leo.schema import SystemOneRequest, build_answer, choice_confidence, score_confidence, to_spec

from conftest import char_tokenize

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
    "urgent": {"type": "noul", "instructions": "The message conveys urgency"},
}
STATE = {"ticket": {"messages": [{"from": "customer", "text": "Charged twice, fix ASAP"}]}}


# ------------------------------------------------------------------------------ schema / math

def test_confidence_formulas_match_typesafe_examples():
    # Score doc example: 0.57 / 0.43 on levels 1 and 2 -> 0.35; 0.95 / 0.05 -> 0.92
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.355, abs=1e-3)
    assert score_confidence([0.0, 0.95, 0.05]) == pytest.approx(0.925, abs=1e-3)
    assert score_confidence([1 / 3] * 3) == 0.0
    # Choice doc example: 0.88 / 0.12 / 0.0 -> 0.81 (docs round probabilities first)
    assert choice_confidence([0.88, 0.12, 0.0]) == pytest.approx(0.82, abs=0.01)
    assert choice_confidence([0.25] * 4) == 0.0
    assert choice_confidence([1.0]) == 1.0


def test_score_answer_shape():
    spec = to_spec("s", QUESTIONS["frustration"])
    a = build_answer(spec, [0.0, 0.57, 0.43])
    assert a["score"] == pytest.approx(1.43)
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(a["probabilities"]) == {"0", "1", "2"}


@pytest.mark.parametrize("bad", [
    {"type": "choice", "instructions": "x", "criteria": {}},
    {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(256)}},
    {"type": "score", "instructions": "x", "criteria": ["only one"]},
    {"type": "score", "instructions": "x", "criteria": ["a", None]},
    {"type": "noul", "instructions": "x", "criteria": {"yes": "a"}},
    {"type": "dunno", "instructions": "x"},
])
def test_invalid_questions_rejected(bad):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {"q": bad}})


def test_render_uses_typesafe_paths():
    text = render(STATE)
    assert "ticket.messages[0].text: Charged twice, fix ASAP" in text
    assert render("plain") == "plain"


# ------------------------------------------------------------------------------ layout / mask

def test_mask_isolates_questions():
    enc = Encoder(char_tokenize)
    specs = [to_spec(k, v) for k, v in QUESTIONS.items()]
    rows = pack_rows(enc.encode_state(STATE), [enc.encode_question(s, key=s.qid) for s in specs], 10_000)
    assert len(rows) == 1
    b = collate(rows)
    m = block_causal_mask(b.seg_ids)[0, 0]
    seg = b.seg_ids[0]
    q1 = (seg == 1).nonzero().flatten()
    q2 = (seg == 2).nonzero().flatten()
    st = (seg == 0).nonzero().flatten()
    assert not m[q2][:, q1].any(), "question 2 must not see question 1"
    assert m[q2][:, st].all(), "questions see the whole state"
    assert not m[st][:, q1].any(), "state never sees questions"
    # positions restart after the state for every question
    assert b.position_ids[0, q1[0]] == b.position_ids[0, q2[0]] == len(st)


def test_question_too_long():
    from leo.encode import QuestionTooLong
    enc = Encoder(char_tokenize, max_question_tokens=40)
    spec = to_spec("q", {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(30)}})
    with pytest.raises(QuestionTooLong):
        enc.encode_question(spec, key="q")


# ------------------------------------------------------------------------------ model invariants

def _probs(leo, questions, state=STATE):
    specs = [to_spec(k, v) for k, v in questions.items()]
    return leo.probabilities([(state, specs)])[0]


def test_packed_equals_separate(tiny_leo):
    together = _probs(tiny_leo, QUESTIONS)
    for qid, q in QUESTIONS.items():
        alone = _probs(tiny_leo, {qid: q})
        np.testing.assert_allclose(together[qid], alone[qid], atol=1e-5)


def test_adding_a_question_changes_nothing_else(tiny_leo):
    base = _probs(tiny_leo, QUESTIONS)
    extra = dict(QUESTIONS, spam={"type": "noul", "instructions": "Is this spam?"})
    more = _probs(tiny_leo, extra)
    for qid in QUESTIONS:
        np.testing.assert_allclose(base[qid], more[qid], atol=1e-5)


def test_padding_and_batching_invariance(tiny_leo):
    specs = [to_spec(k, v) for k, v in QUESTIONS.items()]
    solo = tiny_leo.probabilities([(STATE, specs)])[0]
    long_state = "x" * 700
    batched = tiny_leo.probabilities([(long_state, specs), (STATE, specs), ("short", specs)])[1]
    for qid in QUESTIONS:
        np.testing.assert_allclose(solo[qid], batched[qid], atol=1e-5)


def test_rows_split_when_budget_is_small(tiny_leo):
    full = _probs(tiny_leo, QUESTIONS)
    tiny_leo.max_row_tokens = 64  # forces one question per row, state repeated
    split = _probs(tiny_leo, QUESTIONS)
    for qid in QUESTIONS:
        np.testing.assert_allclose(full[qid], split[qid], atol=1e-5)


def test_permutation_layout_and_targets():
    """Slot j shows original option perm[j], and the training target is permuted the same way."""
    enc = Encoder(char_tokenize)
    spec = to_spec("department", QUESTIONS["department"])
    spec.target = [0.0, 1.0, 0.0]  # "technical"
    perm = [2, 0, 1]
    q = enc.encode_question(spec, key="d", perm=perm)
    starts = [i for i, m in enumerate(q.markers) if m == 4]  # <opt>
    shown = ["".join(chr((t - 5) % 290) for t in q.tokens[s + 1 : e]) for s, e in zip(starts, q.opt_end)]
    assert shown == ["sales", "billing: Payments", "technical: Bugs"]
    assert q.target == [0.0, 0.0, 1.0]


def test_non_packable_backbone_gets_one_question_per_row(tiny_leo):
    packed = _probs(tiny_leo, QUESTIONS)
    tiny_leo.model.packable = False  # what a Gated-DeltaNet / linear-attention base gets
    rows = []
    enc = tiny_leo.encoder
    specs = [to_spec(k, v) for k, v in QUESTIONS.items()]
    rows = pack_rows(enc.encode_state(STATE), [enc.encode_question(s, key=s.qid) for s in specs], 0)
    assert [len(r.questions) for r in rows] == [1, 1, 1]
    single = _probs(tiny_leo, QUESTIONS)
    for qid in QUESTIONS:
        np.testing.assert_allclose(packed[qid], single[qid], atol=1e-5)


def test_supports_block_mask():
    from types import SimpleNamespace

    from leo.model import supports_block_mask

    assert supports_block_mask(SimpleNamespace(layer_types=["full_attention"] * 4))
    assert supports_block_mask(SimpleNamespace(layer_types=["sliding_attention", "full_attention"]))
    assert not supports_block_mask(SimpleNamespace(layer_types=["linear_attention", "full_attention"]))
    assert supports_block_mask(SimpleNamespace())


def test_system_one_response_shape(tiny_leo):
    r = tiny_leo.system_one(STATE, QUESTIONS)
    a = r["answers"]
    assert a["department"]["choice"] in {"billing", "technical", "sales"}
    assert sum(a["department"]["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert 0 <= a["urgent"]["noul"] <= 1 and "confidence" not in a["urgent"]
    assert 0 <= a["frustration"]["score"] <= 2
    assert r["usage"]["input_tokens"] > 0
    assert tiny_leo.system_one("x", {})["answers"] == {}
