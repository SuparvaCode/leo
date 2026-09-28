"""Load a real base model, check the tokenizer, and measure train/infer throughput on this GPU.

    python scripts/smoke_gpu.py --base Qwen/Qwen3-0.6B-Base
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from leo.encode import Encoder, batches_by_tokens, collate, pack_rows, training_perm
from leo.infer import make_tokenize
from leo.model import LeoModel
from leo.schema import to_spec


def fake_rows(enc: Encoder, n: int, state_words: int, rng: random.Random):
    rows = []
    labels = {f"label_{i}": f"description of category {i}" for i in range(8)}
    for i in range(n):
        state = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "order", "refund"]) for _ in range(state_words))
        specs = [
            to_spec("c", {"type": "choice", "instructions": "Which category fits?", "criteria": labels}),
            to_spec("n", {"type": "noul", "instructions": "Is this about a refund?"}),
        ]
        for s in specs:
            s.target = [1.0] + [0.0] * (s.n_options - 1)
        es = enc.encode_state(state)
        qs = [enc.encode_question(s, key=(i, s.qid), perm=training_perm(s, rng)) for s in specs]
        rows += pack_rows(es, qs, 2048)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--max_batch_tokens", type=int, default=4096)
    ap.add_argument("--state_words", type=int, default=150)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    tokenize = make_tokenize(tok)
    print("special split check ok:", tokenize("<|im_end|> hi")[:6])
    dev = torch.device("cuda")
    model = LeoModel.from_pretrained_base(args.base, tokenize, gradient_checkpointing=True).to(dev)
    bb, head = model.trainable_parameter_groups()
    n_train = sum(p.numel() for p in bb + head)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e6:.2f}M of {n_all/1e6:.1f}M")
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-4}, {"params": head, "lr": 1e-3}])

    enc = Encoder(tokenize, max_state_tokens=512)
    rng = random.Random(0)
    rows = fake_rows(enc, 200, args.state_words, rng)
    groups = batches_by_tokens(rows, args.max_batch_tokens)
    print(f"{len(rows)} rows, mean len {sum(len(r) for r in rows)/len(rows):.0f}, {len(groups)} batches")

    model.train()
    torch.cuda.reset_peak_memory_stats()
    toks, t0 = 0, None
    for step, idx in enumerate(groups[: args.steps]):
        if step == 2:
            torch.cuda.synchronize()
            t0, toks = time.perf_counter(), 0
        b = collate([rows[i] for i in idx]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b)
        logp = torch.log_softmax(logits, -1).masked_fill(torch.isinf(logits), 0)
        loss = -(b.targets * logp).sum(-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        toks += b.n_tokens
        print(f"step {step} loss {loss.item():.3f} rows {len(idx)} len {b.input_ids.shape[1]}")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"TRAIN {toks/dt:.0f} tokens/s, peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for warm in range(2):
            b = collate([rows[i] for i in groups[warm]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(b)
        torch.cuda.synchronize()
        t0, toks = time.perf_counter(), 0
        for idx in batches_by_tokens(rows, 8192):
            b = collate([rows[i] for i in idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(b)
            toks += b.n_tokens
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"INFER {toks/dt:.0f} tokens/s, peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
