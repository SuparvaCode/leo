"""Jev-exact response shape: 2-decimal probabilities that sum to 1 and keep the argmax."""
import random

from leo.schema import build_answer, round_simplex, to_spec


def test_round_simplex_sums_to_one_and_keeps_argmax():
    rng = random.Random(0)
    for _ in range(2000):
        k = rng.randint(2, 60)
        raw = [rng.random() ** rng.choice([1, 3, 8]) for _ in range(k)]
        s = sum(raw)
        p = [x / s for x in raw]
        r = round_simplex(p, 2)
        assert abs(sum(r) - 1.0) < 1e-9
        best = max(range(k), key=lambda i: p[i])
        assert r[best] >= max(r) - 1e-12  # the true argmax is still (one of) the largest shown values
        assert all(abs(a - b) <= 0.01 + 1e-12 for a, b in zip(p, r))


def test_jev_exact_choice_answer_passes_jev_ultrafast_validation():
    from jev_ultrafast.model import validate_choice

    spec = to_spec("q", {"type": "choice", "criteria": {str(i): None for i in range(40)}})
    probs = [1 / 40] * 40
    probs[7] += 0.001
    probs[3] -= 0.001
    ans = build_answer(spec, probs, decimals=2)
    assert validate_choice(ans, set(spec.keys)) is ans
    assert ans["choice"] == "7"


def test_order_views_are_distinct_permutations():
    from leo.infer import Leo

    leo = Leo.__new__(Leo)
    for views in (1, 2, 3, 4):
        leo.order_views = views
        for n in (2, 3, 6, 20):
            spec = to_spec("q", {"type": "choice", "criteria": {str(i): None for i in range(n)}})
            perms = leo.view_perms(spec)
            assert perms[0] is None  # the caller's own order is always scored
            full = [list(range(n))] + [p for p in perms[1:]]
            assert all(sorted(p) == list(range(n)) for p in full)
            assert len({tuple(p) for p in full}) == len(full) == min(views, 2 if n == 2 else views)
        noul = to_spec("q", {"type": "noul"})
        assert len(leo.view_perms(noul)) == min(views, 2)
        score = to_spec("q", {"type": "score", "criteria": ["low", "mid", "high"]})
        assert leo.view_perms(score) == [None]  # ordinal levels are never reordered
