"""Weight-space merge ("model soup", WiSE-FT) of two Leo checkpoints on the same base, e.g. leo-4b-v5 and its
continuation leo-4b-v5.1.

The LoRA part is merged *exactly*: each adapted weight W + s*B*A is replaced by
    W + s * ((1 - t) * B0 A0 + t * B1 A1)
which is itself a LoRA of rank r0 + r1 with  B = [(1 - t) B0 | t B1]  and  A = [A0 ; A1].
The scale s = alpha / r is kept by doubling alpha with the rank. The small head and marker embeddings are
averaged linearly with the same t. Temperatures must be re-fitted afterwards (scripts/modal: calibrate).

    python scripts/soup.py --a checkpoints/leo-4b-v5/best --b checkpoints/leo-4b-v5.1/best --t 0.5 --out checkpoints/leo-4b-soup50
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first model dir (weight 1 - t)")
    ap.add_argument("--b", required=True, help="second model dir (weight t)")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    A, B, out, t = Path(a.a), Path(a.b), Path(a.out), a.t
    ca, cb = (json.loads((p / "adapter" / "adapter_config.json").read_text()) for p in (A, B))
    ga, gb = (json.loads((p / "leo_config.json").read_text(encoding="utf-8")) for p in (A, B))
    assert ga["base_model"] == gb["base_model"] and ga["base_revision"] == gb["base_revision"], "different bases"
    assert ca["r"] == cb["r"] and ca["lora_alpha"] == cb["lora_alpha"], "different LoRA shapes"
    wa = load_file(str(A / "adapter" / "adapter_model.safetensors"))
    wb = load_file(str(B / "adapter" / "adapter_model.safetensors"))
    assert wa.keys() == wb.keys()
    merged = {}
    for k in wa:
        if k.endswith("lora_A.weight"):
            merged[k] = torch.cat([wa[k], wb[k]], dim=0).contiguous()           # [r0 + r1, in]
        elif k.endswith("lora_B.weight"):
            merged[k] = torch.cat([(1 - t) * wa[k], t * wb[k]], dim=1).contiguous()  # [out, r0 + r1]
        else:
            raise SystemExit(f"unexpected adapter tensor {k}")
    # exactness check on one module
    k = next(x for x in wa if x.endswith("lora_A.weight"))
    kb = k.replace("lora_A", "lora_B")
    want = (1 - t) * (wa[kb].double() @ wa[k].double()) + t * (wb[kb].double() @ wb[k].double())
    got = merged[kb].double() @ merged[k].double()
    err = (want - got).abs().max().item()
    assert err < 1e-5, err

    if out.exists():
        shutil.rmtree(out)
    (out / "adapter").mkdir(parents=True)
    save_file(merged, str(out / "adapter" / "adapter_model.safetensors"))
    cfg = dict(ca, r=ca["r"] * 2, lora_alpha=ca["lora_alpha"] * 2)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    ha, hb = load_file(str(A / "leo_head.safetensors")), load_file(str(B / "leo_head.safetensors"))
    save_file({k: ((1 - t) * ha[k] + t * hb[k]).contiguous() for k in ha}, str(out / "leo_head.safetensors"))

    name = a.name or out.name
    g = dict(gb, name=name, lora={"r": cfg["r"], "alpha": cfg["lora_alpha"]}, temperatures={}, completed=False,
             soup={"a": str(A), "b": str(B), "t": t, "method": "exact LoRA concatenation + linear head average"})
    g.pop("dev", None)
    (out / "leo_config.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    print(f"{name}: t={t}, max |dW error| {err:.2e}, rank {cfg['r']}")


if __name__ == "__main__":
    main()
