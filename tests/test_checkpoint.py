import json
import os

import pytest
import torch

from leo.checkpoint import (
    apply_resume, atomic_export_dir, read_resume, recover_dir, resolve_checkpoint, save_resume, truncate_log,
)
from leo.encode import Encoder, collate, pack_rows
from leo.model import LeoModel
from leo.schema import to_spec
from leo.train import decision_loss, epoch_plan, lr_factor

from conftest import char_tokenize

QUESTIONS = {
    "c": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "sales": None}},
    "s": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    "n": {"type": "noul", "instructions": "Refund requested?"},
}


def _batch():
    enc = Encoder(char_tokenize)
    specs = [to_spec(k, v) for k, v in QUESTIONS.items()]
    for s in specs:
        s.target = [0.0] * (s.n_options - 1) + [1.0]
    qs = [enc.encode_question(s, key=(0, s.qid), perm=list(reversed(range(s.n_options)))) for s in specs]
    return collate(pack_rows(enc.encode_state("charged twice, fix it"), qs, 10_000))


def _model_and_opt(factory, seed):
    torch.manual_seed(seed)
    m = LeoModel.from_backbone(factory(0), char_tokenize, lora_r=4, lora_dropout=0.0, head_dim=32)
    bb, head = m.trainable_parameter_groups()
    return m, torch.optim.AdamW([{"params": bb, "lr": 1e-2}, {"params": head, "lr": 1e-2}])


def _step(m, opt, batch):
    loss, _ = decision_loss(m(batch), batch, 0.5)
    loss.mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_resume_continues_exactly(tmp_path, tiny_backbone_factory):
    batch = _batch()
    a, opt_a = _model_and_opt(tiny_backbone_factory, seed=0)
    for _ in range(3):
        _step(a, opt_a, batch)
    save_resume(tmp_path / "resume.pt", a, opt_a, {"step": 3, "best": float("inf")})
    for _ in range(2):
        _step(a, opt_a, batch)

    b, opt_b = _model_and_opt(tiny_backbone_factory, seed=999)  # different head / LoRA init on purpose
    meta = apply_resume(read_resume(tmp_path / "resume.pt"), b, opt_b)
    assert meta["step"] == 3 and meta["best"] == float("inf")
    for _ in range(2):
        _step(b, opt_b, batch)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=1e-6, msg=name)


def test_resume_rejects_a_different_model(tmp_path, tiny_backbone_factory):
    a, opt_a = _model_and_opt(tiny_backbone_factory, seed=0)
    save_resume(tmp_path / "resume.pt", a, opt_a, {"step": 1})
    torch.manual_seed(0)
    other = LeoModel.from_backbone(tiny_backbone_factory(0), char_tokenize, lora_r=8, head_dim=32)
    opt = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad])
    with pytest.raises(RuntimeError):
        apply_resume(read_resume(tmp_path / "resume.pt"), other, opt)


def _writer(version):
    def write(d):
        d.mkdir(parents=True)
        (d / "leo_config.json").write_text(json.dumps({"v": version}))
    return write


def test_atomic_export_and_crash_recovery(tmp_path):
    best = tmp_path / "best"
    atomic_export_dir(best, _writer(1))
    atomic_export_dir(best, _writer(2))
    assert json.loads((best / "leo_config.json").read_text())["v"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best"]

    # Power cut after the old copy was moved aside but before the new one landed.
    os.replace(best, tmp_path / "best.old")
    _writer(3)(tmp_path / "best.tmp")
    assert resolve_checkpoint(tmp_path) == tmp_path / "best.old"  # still loadable before recovery
    recover_dir(best)
    assert json.loads((best / "leo_config.json").read_text())["v"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best"]
    assert resolve_checkpoint(tmp_path) == best and resolve_checkpoint(best) == best

    # A selection record points the run directory at another export.
    _writer(4)(tmp_path / "final")
    (tmp_path / "selected.json").write_text(json.dumps({"selected": "final"}))
    assert resolve_checkpoint(tmp_path) == tmp_path / "final"
    (tmp_path / "selected.json").write_text(json.dumps({"selected": "missing"}))
    assert resolve_checkpoint(tmp_path) == best


def test_truncate_log_drops_replayed_and_torn_lines(tmp_path):
    p = tmp_path / "train_log.jsonl"
    p.write_text('{"step": 25}\n{"step": 50}\n{"dev": {"step": 50, "loss": 1.0}}\n{"step": 75}\n{"step": 1', encoding="utf-8")
    truncate_log(p, 50)
    assert [json.loads(line) for line in p.read_text().splitlines()] == [
        {"step": 25}, {"step": 50}, {"dev": {"step": 50, "loss": 1.0}}]


def test_epoch_plan_is_deterministic():
    items = [("state %d" % i, [to_spec("c", QUESTIONS["c"])], "src") for i in range(40)]
    for _, specs, _ in items:
        specs[0].target = [1.0, 0.0, 0.0]
    enc = Encoder(char_tokenize)
    r1, g1 = epoch_plan(items, enc, 0, 1, 1536, 256)
    r2, g2 = epoch_plan(items, enc, 0, 1, 1536, 256)
    _, g3 = epoch_plan(items, enc, 0, 2, 1536, 256)
    assert g1 == g2 and [q.perm for r in r1 for q in r.questions] == [q.perm for r in r2 for q in r.questions]
    assert g1 != g3


def test_lr_schedule():
    assert lr_factor(1, 60, 1000) == pytest.approx(1 / 60)
    assert lr_factor(60, 60, 1000) == pytest.approx(1.0)
    assert lr_factor(1000, 60, 1000) == pytest.approx(0.1)
