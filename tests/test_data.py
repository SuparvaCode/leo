import random

import pytest

from leo.data import example_specs
from leo.data.augment import RECIPE, choice_question, set_recipe
from leo.data.synthetic import FAMILIES


@pytest.fixture(autouse=True)
def restore_recipe():
    yield
    set_recipe("v0")


def _none_stats(recipe: str, n: int = 4000) -> tuple[float, float]:
    set_recipe(recipe)
    rng = random.Random(0)
    present = right = 0
    for _ in range(n):
        q = choice_question("Which topic?", ["world", "sports", "business", "science"], "sports", rng)
        none = [k for k in q["criteria"] if k in RECIPE["none_keys"]]
        if none:
            present += 1
            right += q["label"] in none
    return present / n, right / max(1, present)


def test_v0_none_option_is_always_the_answer():
    present, right = _none_stats("v0")
    assert 0.0 < present < 0.15 and right == 1.0


def test_v1_none_option_is_often_wrong():
    present, right = _none_stats("v1")
    assert present > 0.15 and 0.2 < right < 0.6


def test_v0_draws_the_same_random_numbers_as_before():
    """Recipe v0 must not consume extra randomness, or data/processed would not rebuild identically."""
    set_recipe("v0")
    a, b = random.Random(7), random.Random(7)
    choice_question("Which topic?", ["a", "b", "c", "d"], "b", a)
    choice_question("Which topic?", ["a", "b", "c", "d"], "b", b, p_none_distractor=0.0)
    assert a.random() == b.random()


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_synthetic_families_produce_valid_requests(name):
    fn, _ = FAMILIES[name]
    rng = random.Random(0)
    for _ in range(50):
        specs = example_specs(fn(rng))
        assert specs and all(abs(sum(s.target) - 1) < 1e-6 for s in specs)
