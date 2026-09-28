"""v4: warm start from an exported model, and the evidence-required browser family."""
import random
from collections import Counter

import torch

from leo.data import browser_evidence, example_specs
from leo.model import LeoModel
from leo.train import init_from_export
from tests.conftest import char_tokenize


def test_init_from_export_copies_every_trainable_tensor(tmp_path, tiny_backbone_factory):
    src = LeoModel.from_backbone(tiny_backbone_factory(0), char_tokenize, lora_r=4, head_dim=32)
    torch.manual_seed(3)
    for n, p in src.named_parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.05)
    src.save(tmp_path / "m", {"name": "src", "base_model": "tiny", "head_dim": 32})
    dst = LeoModel.from_backbone(tiny_backbone_factory(0), char_tokenize, lora_r=4, head_dim=32)
    init_from_export(dst, tmp_path / "m")
    a = {n: p for n, p in src.named_parameters() if p.requires_grad}
    b = {n: p for n, p in dst.named_parameters() if p.requires_grad}
    assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)


def test_evidence_family_labels_follow_the_page():
    exs = browser_evidence.examples(random.Random(0), 1500, with_kind=True)
    by = Counter((e["kind"], e["questions"]["operation"]["label"]) for e in exs)
    # DONE only on a settled page that shows the results or the opened item
    assert {k for k, lab in by if lab == "DONE"} <= {"form", "detail"}
    # screens where the history can look finished but the page proves nothing
    for kind in ("overlay", "popover", "blank", "loading", "stale", "unsubmitted"):
        assert sum(v for (k, _), v in by.items() if k == kind) > 0, kind
        assert by[(kind, "DONE")] == 0
    assert by[("blank", "WAIT")] > 0 and by[("overlay", "BLOCKED")] > 0
    for e in exs[:200]:
        e.pop("kind")
        example_specs(e)


def test_probe_themes_never_in_training():
    exs = browser_evidence.examples(random.Random(1), 400)
    for brand in browser_evidence.PROBE_THEMES:
        assert not any(brand.lower() in str(e["state"]).lower() for e in exs)
