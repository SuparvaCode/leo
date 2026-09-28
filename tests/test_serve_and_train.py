import pytest
import torch
from fastapi.testclient import TestClient

from leo.client import SystemOneClient
from leo.data import example_specs, label_distribution
from leo.encode import collate, pack_rows
from leo.schema import to_spec
from leo.serve import create_app
from leo.train import decision_loss

REQ = {"state": "I was charged twice, refund please", "model": "leo-latest",
       "questions": {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": "bugs"}},
                     "angry": {"type": "noul", "instructions": "Is the customer angry?"}}}


@pytest.fixture()
def client(tiny_leo):
    return TestClient(create_app(tiny_leo, api_key="secret"))


def test_auth_required(client):
    assert client.post("/v1/systemone", json=REQ).status_code == 401
    assert client.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_systemone_ok(client):
    r = client.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["answers"]["dept"]["choice"] in ("billing", "tech")
    assert 0 <= body["answers"]["angry"]["noul"] <= 1
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}


def test_validation_is_422(client):
    bad = {"state": "x", "questions": {"s": {"type": "score", "instructions": "x", "criteria": ["one"]}}}
    r = client.post("/v1/systemone", json=bad, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request"
    bad = {"state": "x", "questions": {"n": {"type": "noul", "instructions": "x", "criteria": {"yes": "a"}}}}
    r = client.post("/v1/systemone", json=bad, headers={"Authorization": "Bearer secret"})
    msg = r.json()["error"]["details"][0]["msg"]
    assert r.status_code == 422 and "true" in msg and "false" in msg, msg


def test_body_limit(tiny_leo):
    c = TestClient(create_app(tiny_leo, api_key=None, max_body_bytes=100))
    assert c.post("/v1/systemone", json=REQ).status_code == 413


def test_client_talks_to_leo_server(tiny_leo):
    """The same client drives Leo and Jev; check it against an in-process Leo server."""
    import httpx

    app = create_app(tiny_leo, api_key="secret")
    client = SystemOneClient("http://testserver", api_key="secret")
    client._http = TestClient(app, headers=dict(client._http.headers))  # TestClient is an httpx.Client
    out = client.system_one(REQ["state"], REQ["questions"])
    assert out["answers"]["dept"]["choice"] in ("billing", "tech") and "client_latency_ms" in out


def test_jev_cache_key_changes_with_body():
    from leo.bench.jev import request_key

    a = request_key("1", "jev-latest", {"state": "x", "questions": {"q": {"type": "noul", "instructions": "a"}}})
    b = request_key("1", "jev-latest", {"state": "x", "questions": {"q": {"type": "noul", "instructions": "b"}}})
    c = request_key("1", "jev-1.13.0", {"state": "x", "questions": {"q": {"type": "noul", "instructions": "a"}}})
    assert len({a, b, c}) == 3 and a.startswith("1:")


def test_labels_and_loss():
    ex = {"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None, "c": None}, "label": "b"},
        "n": {"type": "noul", "instructions": "i", "label": 0.8},
        "s": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"], "label": 2}}}
    specs = example_specs(ex)
    assert [s.target for s in specs] == [[0.0, 1.0, 0.0], [pytest.approx(0.2), 0.8], [0.0, 0.0, 1.0]]
    assert label_distribution(to_spec("u", {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}),
                              {"a": 1, "b": 1}) == [0.5, 0.5]


def test_rps_uses_ordinal_order_under_permutation(tiny_leo):
    """A near miss on a shuffled score scale must cost less than a far miss."""
    enc = tiny_leo.encoder
    spec = to_spec("s", {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]})
    spec.target = [0.0, 0.0, 1.0]
    q = enc.encode_question(spec, key=(0, "s"), perm=[2, 0, 1])  # slots show hi, lo, mid
    b = collate(pack_rows(enc.encode_state("x"), [q], 10_000))
    near = torch.tensor([[0.0, -9.0, 5.0]])  # mass on "mid" (slot 2)
    far = torch.tensor([[0.0, 5.0, -9.0]])   # mass on "lo" (slot 1)
    l_near, _ = decision_loss(near, b, rps_weight=1.0)
    l_far, _ = decision_loss(far, b, rps_weight=1.0)
    ce_near, _ = decision_loss(near, b, rps_weight=0.0)
    ce_far, _ = decision_loss(far, b, rps_weight=0.0)
    assert (l_near - ce_near) < (l_far - ce_far)
