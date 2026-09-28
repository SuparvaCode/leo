"""Token layout, row packing and batch collation.

One request becomes one or more *rows*. A row holds the state once, followed by several questions:

    <state> state tokens
    <q_type> instruction tokens  <opt> option 1 </opt> ... <opt> option K </opt> <decide>
    <q_type> ...

Markers are not vocabulary tokens. They are looked up in a separate trainable embedding table, so no
user string can forge an option boundary. Every question restarts its position ids right after the
state and may attend only to the state and to earlier tokens of its own question (block-causal), so
answers are independent of which other questions share the row.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import torch

from leo.render import option_text, render
from leo.schema import QuestionSpec

MARKERS = ("state", "q_choice", "q_score", "q_noul", "opt", "opt_end", "decide")
MID = {name: i for i, name in enumerate(MARKERS)}
Q_MARKER = {"choice": MID["q_choice"], "score": MID["q_score"], "noul": MID["q_noul"]}
TYPE_ID = {"choice": 0, "score": 1, "noul": 2}


class QuestionTooLong(ValueError):
    """The question cannot fit the token budget even after trimming option descriptions."""


@dataclass
class EncodedState:
    tokens: list[int]
    markers: list[int]


@dataclass
class EncodedQuestion:
    key: Any                  # caller-side identity, e.g. (request_index, qid)
    qtype: str
    tokens: list[int]
    markers: list[int]
    opt_end: list[int]        # positions (relative to question start) of each </opt>, in slot order
    decide: int               # position of <decide>
    perm: list[int]           # slot j shows original option perm[j]
    target: list[float] | None = None  # in slot order, training only

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class Row:
    state: EncodedState
    questions: list[EncodedQuestion] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.state.tokens) + sum(len(q) for q in self.questions)


class Encoder:
    def __init__(
        self,
        tokenize: Callable[[str], list[int]],
        max_state_tokens: int = 1024,
        max_instruction_tokens: int = 512,
        max_question_tokens: int = 2048,
        min_option_tokens: int = 4,
    ) -> None:
        self._tok = lru_cache(maxsize=200_000)(lambda s: tuple(tokenize(s)))
        self.max_state_tokens = max_state_tokens
        self.max_instruction_tokens = max_instruction_tokens
        self.max_question_tokens = max_question_tokens
        self.min_option_tokens = min_option_tokens

    def tok(self, text: str) -> list[int]:
        return list(self._tok(text)) if text else []

    def encode_state(self, state: Any) -> EncodedState:
        ids = self.tok(render(state))[: self.max_state_tokens]
        return EncodedState([0] + ids, [MID["state"]] + [-1] * len(ids))

    def encode_question(
        self,
        spec: QuestionSpec,
        key: Any,
        perm: Sequence[int] | None = None,
    ) -> EncodedQuestion:
        k = spec.n_options
        perm = list(range(k)) if perm is None else list(perm)
        instr = self.tok(render(spec.instructions).strip())[: self.max_instruction_tokens]
        opts = [self.tok(option_text(spec.type, spec.keys[i], spec.descriptions[i])) for i in perm]

        # Fixed overhead: q marker + decide + (opt, /opt) per option. Trim long descriptions evenly.
        overhead = 2 + 2 * k + len(instr)
        budget = self.max_question_tokens - overhead
        if budget < k * 1:
            raise QuestionTooLong(f"question {spec.qid!r}: {k} options do not fit {self.max_question_tokens} tokens")
        if sum(len(o) for o in opts) > budget:
            cap = max(self.min_option_tokens, budget // k)
            opts = [o[:cap] for o in opts]
            if sum(len(o) for o in opts) > budget:
                cap = max(1, budget // k)
                opts = [o[:cap] for o in opts]

        tokens = [0] + instr
        markers = [Q_MARKER[spec.type]] + [-1] * len(instr)
        opt_end: list[int] = []
        for o in opts:
            tokens += [0] + o + [0]
            markers += [MID["opt"]] + [-1] * len(o) + [MID["opt_end"]]
            opt_end.append(len(tokens) - 1)
        tokens.append(0)
        markers.append(MID["decide"])
        target = [spec.target[i] for i in perm] if spec.target is not None else None
        return EncodedQuestion(key, spec.type, tokens, markers, opt_end, len(tokens) - 1, perm, target)


def training_perm(spec: QuestionSpec, rng: random.Random, p_shuffle_score: float = 0.3) -> list[int]:
    """Random option order used during training (choice always, noul 50/50, score sometimes)."""
    idx = list(range(spec.n_options))
    if spec.type == "choice" or spec.type == "noul":
        rng.shuffle(idx)
    elif spec.type == "score" and rng.random() < p_shuffle_score:
        rng.shuffle(idx)
    return idx


def pack_rows(state: EncodedState, questions: Sequence[EncodedQuestion], max_row_tokens: int) -> list[Row]:
    """Greedily pack questions behind one copy of the state; spill into more rows when full."""
    rows: list[Row] = []
    cur = Row(state)
    for q in questions:
        if cur.questions and len(cur) + len(q) > max_row_tokens:
            rows.append(cur)
            cur = Row(state)
        cur.questions.append(q)
    if cur.questions:
        rows.append(cur)
    return rows


@dataclass
class Batch:
    input_ids: torch.Tensor      # [B, L]
    marker_ids: torch.Tensor     # [B, L], -1 for ordinary tokens
    position_ids: torch.Tensor   # [B, L]
    seg_ids: torch.Tensor        # [B, L], -1 pad, 0 state, 1.. question
    q_row: torch.Tensor          # [Q]
    q_decide: torch.Tensor       # [Q]
    q_type: torch.Tensor         # [Q]
    q_nopt: torch.Tensor         # [Q]
    o_row: torch.Tensor          # [O]
    o_pos: torch.Tensor          # [O]
    o_q: torch.Tensor            # [O] question index
    o_slot: torch.Tensor         # [O] slot index inside its question
    q_perm: torch.Tensor         # [Q, Kmax] original option index shown in each slot (pads: slot index)
    keys: list[Any]
    perms: list[list[int]]
    targets: torch.Tensor | None  # [Q, Kmax] in slot order, zero padded
    n_tokens: int

    def to(self, device: torch.device | str) -> "Batch":
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device, non_blocking=True))
        return self


def collate(rows: Sequence[Row], pad_id: int = 0, multiple: int = 16) -> Batch:
    length = max(len(r) for r in rows)
    length = -(-length // multiple) * multiple
    b = len(rows)
    input_ids = torch.full((b, length), pad_id, dtype=torch.long)
    marker_ids = torch.full((b, length), -1, dtype=torch.long)
    position_ids = torch.zeros((b, length), dtype=torch.long)
    seg_ids = torch.full((b, length), -1, dtype=torch.long)

    q_row, q_dec, q_type, q_nopt = [], [], [], []
    o_row, o_pos, o_q, o_slot = [], [], [], []
    keys, perms, tgts = [], [], []
    for ri, row in enumerate(rows):
        s = len(row.state.tokens)
        input_ids[ri, :s] = torch.tensor(row.state.tokens)
        marker_ids[ri, :s] = torch.tensor(row.state.markers)
        position_ids[ri, :s] = torch.arange(s)
        seg_ids[ri, :s] = 0
        off = s
        for qi, q in enumerate(row.questions, start=1):
            n = len(q)
            input_ids[ri, off : off + n] = torch.tensor(q.tokens)
            marker_ids[ri, off : off + n] = torch.tensor(q.markers)
            position_ids[ri, off : off + n] = torch.arange(s, s + n)
            seg_ids[ri, off : off + n] = qi
            gq = len(q_row)
            q_row.append(ri)
            q_dec.append(off + q.decide)
            q_type.append(TYPE_ID[q.qtype])
            q_nopt.append(len(q.opt_end))
            for slot, p in enumerate(q.opt_end):
                o_row.append(ri)
                o_pos.append(off + p)
                o_q.append(gq)
                o_slot.append(slot)
            keys.append(q.key)
            perms.append(q.perm)
            tgts.append(q.target)
            off += n

    kmax = max(q_nopt) if q_nopt else 1
    q_perm = torch.arange(kmax).repeat(len(q_nopt), 1)
    for i, p in enumerate(perms):
        q_perm[i, : len(p)] = torch.tensor(p, dtype=torch.long)
    targets = None
    if tgts and all(t is not None for t in tgts):
        targets = torch.zeros((len(tgts), kmax), dtype=torch.float32)
        for i, t in enumerate(tgts):
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.float32)

    as_long = lambda xs: torch.tensor(xs, dtype=torch.long)  # noqa: E731
    return Batch(
        input_ids, marker_ids, position_ids, seg_ids,
        as_long(q_row), as_long(q_dec), as_long(q_type), as_long(q_nopt),
        as_long(o_row), as_long(o_pos), as_long(o_q), as_long(o_slot),
        q_perm, keys, perms, targets, n_tokens=sum(len(r) for r in rows),
    )


def block_causal_mask(seg_ids: torch.Tensor) -> torch.Tensor:
    """[B, L] segment ids -> [B, 1, L, L] boolean mask (True = may attend).

    A token attends to earlier-or-equal positions that are state (segment 0) or in its own segment.
    Padding attends only to itself so no attention row is empty.
    """
    b, length = seg_ids.shape
    idx = torch.arange(length, device=seg_ids.device)
    causal = idx[None, :] <= idx[:, None]
    sq = seg_ids[:, :, None]
    sk = seg_ids[:, None, :]
    allowed = causal[None] & ((sk == sq) | (sk == 0)) & (sq >= 0) & (sk >= 0)
    pad_self = (sq < 0) & torch.eye(length, dtype=torch.bool, device=seg_ids.device)[None]
    return (allowed | pad_self)[:, None]


def batches_by_tokens(rows: Sequence[Row], max_tokens: int, max_rows: int = 64, multiple: int = 16) -> list[list[int]]:
    """Group row indices (sorted by length) so that padded rows * length stays under ``max_tokens``."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        n = -(-len(rows[i]) // multiple) * multiple
        new_max = max(cur_max, n)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, new_max = [], n
        cur.append(i)
        cur_max = new_max
    if cur:
        out.append(cur)
    return out
