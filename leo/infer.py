"""Inference: ``Leo.system_one(state, questions)`` returns a TypeSafe-shaped response.

Each request's state is encoded once; its questions are packed behind it into as few rows as fit,
and rows from many requests are batched by token budget. The model returns logits per option; a
per-bucket temperature and a softmax turn them into the returned probabilities.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from leo.calibrate import temperature_for
from leo.encode import Encoder, Row, batches_by_tokens, collate, pack_rows
from leo.model import LeoModel
from leo.schema import QuestionSpec, SystemOneRequest, build_answer, to_spec


def make_tokenize(tokenizer: Any) -> Callable[[str], list[int]]:
    """Tokenize user text with special-token parsing off, so strings like ``<|im_end|>`` stay plain text."""

    def tokenize(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])

    probe = "<|im_start|>"
    special = tokenizer.convert_tokens_to_ids(probe)
    ids = tokenize(probe)
    if isinstance(special, int) and special != getattr(tokenizer, "unk_token_id", None) and ids == [special]:
        raise RuntimeError("tokenizer still parses special tokens inside user text")
    return tokenize


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"dtype must be auto, bf16 or fp32, got {name!r}")
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Leo:
    def __init__(
        self,
        model: LeoModel,
        encoder: Encoder,
        config: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_row_tokens: int = 2048,
        max_batch_tokens: int = 8192,
        decimals: int | None = 4,
        jev_exact: bool = False,
        order_views: int = 1,
    ) -> None:
        """``jev_exact``: responses shaped exactly like TypeSafe's (2-decimal probabilities that sum to 1,
        only ``model``/``answers``/``usage``), for clients that compare byte for byte.

        ``order_views``: score every choice/noul question under this many option orders (the given order,
        then shifted and reversed copies) and average the logits in the original order. It lowers option-order
        sensitivity at the cost of that many times the question tokens; the state is still read once."""
        self.model = model
        self.encoder = encoder
        self.config = config
        self.device = device
        self.dtype = dtype
        self.max_row_tokens = max_row_tokens
        self.max_batch_tokens = max_batch_tokens
        self.jev_exact = jev_exact
        self.decimals = 2 if jev_exact else decimals
        self.order_views = max(1, int(order_views))
        self.temperatures: dict[str, float] = dict(config.get("temperatures", {}))
        self.name = config.get("name", "leo")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = None,
        merge: bool = True,
        max_state_tokens: int | None = None,
        dtype: str = "auto",
        encoder_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Leo":
        """Load a checkpoint (a model directory or a run directory with ``best/``).

        ``max_state_tokens`` overrides the checkpoint's state budget, e.g. for long policy documents.
        ``dtype``: "auto" (bf16 on a bf16-capable GPU, else fp32), "bf16" or "fp32". bf16 is faster; fp32 gives
        answers that do not depend on which other questions share the request (bf16 rounding moves
        probabilities by up to about 0.01 with sequence shape).
        """
        from transformers import AutoTokenizer

        dev = pick_device(device)
        dtype = resolve_dtype(dtype, dev)
        from leo.checkpoint import resolve_checkpoint
        from leo.model import base_source

        model, cfg = LeoModel.load(path, device=dev, dtype=dtype, merge=merge)
        base, rev = base_source(resolve_checkpoint(path), cfg)
        tok = AutoTokenizer.from_pretrained(base, revision=rev)
        enc_cfg = dict(cfg.get("encoder", {}))
        if max_state_tokens:
            enc_cfg["max_state_tokens"] = max_state_tokens
        enc_cfg.update(encoder_overrides or {})
        enc = Encoder(make_tokenize(tok), **enc_cfg)
        return cls(model, enc, cfg, dev, dtype, **kwargs)

    # ------------------------------------------------------------------------------ core

    def view_perms(self, spec: QuestionSpec) -> list[list[int] | None]:
        """Option orders scored for one question: the given order, then shifted / reversed copies."""
        n = spec.n_options
        if self.order_views <= 1 or n < 2 or spec.type == "score":  # score levels keep their ordinal order
            return [None]
        idx = list(range(n))
        cands = [idx[::-1]]  # reversed first: it moves every option, including the first and last
        for j in range(1, self.order_views):
            s = (j * n) // self.order_views
            shifted = idx[s:] + idx[:s]
            cands += [shifted, shifted[::-1]]
        perms: list[list[int] | None] = [None]
        for p in cands:
            if p != idx and p not in perms:
                perms.append(p)
            if len(perms) == self.order_views:
                break
        return perms

    @torch.no_grad()
    def raw_logits(self, requests: Sequence[tuple[Any, Sequence[QuestionSpec]]]) -> list[dict[str, np.ndarray]]:
        """Logits in each question's original option order, before temperature (averaged over order views)."""
        rows: list[Row] = []
        row_budget = self.max_row_tokens if self.model.packable else 0  # 0 = one question per row
        for ri, (state, specs) in enumerate(requests):
            es = self.encoder.encode_state(state)
            qs = [self.encoder.encode_question(s, key=(ri, s.qid, vi), perm=p)
                  for s in specs for vi, p in enumerate(self.view_perms(s))]
            rows += pack_rows(es, qs, row_budget)
        sums: list[dict[str, np.ndarray]] = [{} for _ in requests]
        counts: list[dict[str, int]] = [{} for _ in requests]
        for idx in batches_by_tokens(rows, self.max_batch_tokens):
            batch = collate([rows[i] for i in idx]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(batch).float().cpu().numpy()
            for qi, ((ri, qid, _), perm) in enumerate(zip(batch.keys, batch.perms)):
                z = np.empty(len(perm), dtype=np.float64)
                z[np.asarray(perm)] = logits[qi, : len(perm)]
                if qid in sums[ri]:
                    sums[ri][qid] += z
                    counts[ri][qid] += 1
                else:
                    sums[ri][qid], counts[ri][qid] = z, 1
        return [{qid: z / counts[ri][qid] for qid, z in req.items()} for ri, req in enumerate(sums)]

    def probabilities(self, requests: Sequence[tuple[Any, Sequence[QuestionSpec]]]) -> list[dict[str, np.ndarray]]:
        raw = self.raw_logits(requests)
        result: list[dict[str, np.ndarray]] = []
        for (_, specs), logits in zip(requests, raw):
            probs = {}
            for s in specs:
                z = logits[s.qid] / temperature_for(self.temperatures, s.type, s.n_options)
                z = np.exp(z - z.max())
                probs[s.qid] = z / z.sum()
            result.append(probs)
        return result

    # ------------------------------------------------------------------------------ API surface

    def predict_many(self, payloads: Iterable[dict[str, Any] | SystemOneRequest]) -> list[dict[str, Any]]:
        reqs = [p if isinstance(p, SystemOneRequest) else SystemOneRequest.model_validate(p) for p in payloads]
        specs = [[to_spec(qid, q) for qid, q in r.questions.items()] for r in reqs]
        t0 = time.perf_counter()
        probs = self.probabilities([(r.state, s) for r, s in zip(reqs, specs)]) if reqs else []
        elapsed = (time.perf_counter() - t0) * 1000 / max(1, len(reqs))
        responses = []
        for r, ss, pr in zip(reqs, specs, probs):
            answers = {s.qid: build_answer(s, pr[s.qid], self.decimals) for s in ss}
            resp = {"model": self.name, "answers": answers, "usage": self._usage(r.state, ss, answers)}
            if not self.jev_exact:  # TypeSafe's response has exactly model, answers and usage
                resp["latency_ms"] = round(elapsed, 2)
            responses.append(resp)
        return responses

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([{"state": state, "questions": questions}])[0]

    def _usage(self, state: Any, specs: Sequence[QuestionSpec], answers: dict[str, Any]) -> dict[str, int]:
        if not specs:
            return {"input_tokens": 0, "output_tokens": 0}
        n_in = len(self.encoder.encode_state(state).tokens)
        n_in += sum(len(self.encoder.encode_question(s, key=None).tokens) for s in specs)
        n_out = len(self.encoder.tok(json.dumps(answers, separators=(",", ":"))))
        usage: dict[str, Any] = {"input_tokens": n_in, "output_tokens": n_out}
        if self.encoder.state_tokens(state) > self.encoder.max_state_tokens:  # only reachable when not rejecting
            usage["truncated"] = True
        return usage
