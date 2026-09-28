"""Leo decision model: a pretrained causal LM run prefill-only, with a listwise pointer readout.

* Backbone: any Hugging Face causal decoder with attention-only layers (Qwen3 family by default),
  loaded without its LM head, adapted with LoRA.
* Markers: a separate trainable embedding table (see ``leo.encode.MARKERS``) swapped into the input
  embeddings, so boundaries cannot be forged from text.
* Readout: for every option, score its ``</opt>`` hidden state against the question's ``<decide>``
  hidden state. ``<decide>`` comes after the whole option list, so the readout is listwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from leo.encode import MARKERS, Batch, block_causal_mask

DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MARKER_INIT_TEXT = {
    "state": "Document:",
    "q_choice": "Question (pick one option):",
    "q_score": "Question (place it on the scale):",
    "q_noul": "Question (yes or no):",
    "opt": "Option:",
    "opt_end": ";",
    "decide": "Answer:",
}


class PointerHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 512) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_o = nn.LayerNorm(hidden)
        self.proj_q = nn.Linear(hidden, dim)
        self.proj_o = nn.Linear(hidden, dim)
        self.type_emb = nn.Embedding(3, dim)
        self.mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h_q: torch.Tensor, h_o: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
        q = self.proj_q(self.norm_q(h_q)) + self.type_emb(qtype)
        o = self.proj_o(self.norm_o(h_o))
        return self.mlp(torch.cat([q, o, q * o], dim=-1)).squeeze(-1)


def supports_block_mask(config: Any) -> bool:
    """True when every layer is softmax attention, so a 4D block mask isolates questions packed in one row.

    Recurrent or linear-attention layers (e.g. Gated DeltaNet in Qwen3.5 / Qwen3-Next) ignore attention masks,
    so on those backbones each question must get its own row or questions would see each other.
    """
    kinds = set(getattr(config, "layer_types", None) or ["full_attention"])
    return kinds <= {"full_attention", "sliding_attention"}


class LeoModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 512) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.packable = supports_block_mask(backbone.config)
        self.markers = nn.Embedding(len(MARKERS), hidden)
        self.head = PointerHead(hidden, head_dim)

    # ----------------------------------------------------------------------------- construction

    @torch.no_grad()
    def init_markers(self, tokenize: Callable[[str], list[int]]) -> None:
        """Start each marker at the mean embedding of a short, related phrase."""
        emb = self.backbone.get_input_embeddings().weight
        for i, name in enumerate(MARKERS):
            ids = tokenize(MARKER_INIT_TEXT[name]) or [0]
            self.markers.weight[i] = emb[torch.tensor(ids, device=emb.device)].float().mean(0).to(self.markers.weight.device)

    @classmethod
    def from_pretrained_base(
        cls,
        base: str,
        tokenize: Callable[[str], list[int]],
        revision: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        head_dim: int = 512,
        gradient_checkpointing: bool = True,
        attn_implementation: str = "sdpa",
    ) -> "LeoModel":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(base, revision=revision, dtype=dtype, attn_implementation=attn_implementation)
        return cls.from_backbone(backbone, tokenize, lora_r, lora_alpha, lora_dropout, lora_targets, head_dim, gradient_checkpointing)

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        tokenize: Callable[[str], list[int]] | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        head_dim: int = 512,
        gradient_checkpointing: bool = False,
    ) -> "LeoModel":
        backbone.config.use_cache = False
        backbone.requires_grad_(False)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model

            backbone = get_peft_model(
                backbone,
                LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=list(lora_targets)),
            )
        model = cls(backbone, head_dim)
        if tokenize is not None:
            model.init_markers(tokenize)
        return model

    # ----------------------------------------------------------------------------- forward

    def embed(self, input_ids: torch.Tensor, marker_ids: torch.Tensor) -> torch.Tensor:
        emb = self.backbone.get_input_embeddings()(input_ids)
        is_marker = (marker_ids >= 0)[..., None]
        m = self.markers(marker_ids.clamp(min=0)).to(emb.dtype)
        return torch.where(is_marker, m, emb)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Returns option logits ``[Q, Kmax]`` in slot order (fp32), ``-inf`` beyond each question's options."""
        emb = self.embed(batch.input_ids, batch.marker_ids)
        mask = block_causal_mask(batch.seg_ids)
        out = self.backbone(inputs_embeds=emb, attention_mask=mask, position_ids=batch.position_ids, use_cache=False)
        h = out.last_hidden_state
        with torch.autocast(device_type=h.device.type, enabled=False):
            h_q = h[batch.q_row, batch.q_decide].float()
            h_o = h[batch.o_row, batch.o_pos].float()
            scores = self.head(h_q[batch.o_q], h_o, batch.q_type[batch.o_q])
            kmax = int(batch.q_nopt.max().item())
            logits = torch.full((h_q.shape[0], kmax), float("-inf"), device=h.device)
            logits = logits.index_put((batch.o_q, batch.o_slot), scores)
        return logits

    # ----------------------------------------------------------------------------- persistence

    def trainable_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        head = list(self.markers.parameters()) + list(self.head.parameters())
        return backbone, head

    def save(self, path: str | Path, config: dict[str, Any]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "peft_config"):
            self.backbone.save_pretrained(path / "adapter")
        tensors = {f"markers.{k}": v.detach().float().cpu().contiguous() for k, v in self.markers.state_dict().items()}
        tensors |= {f"head.{k}": v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()}
        save_file(tensors, str(path / "leo_head.safetensors"))
        (path / "leo_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        merge: bool = True,
    ) -> tuple["LeoModel", dict[str, Any]]:
        from transformers import AutoModel

        from leo.checkpoint import resolve_checkpoint

        path = resolve_checkpoint(path)  # a run directory resolves to its best/ export
        cfg = json.loads((path / "leo_config.json").read_text(encoding="utf-8"))
        backbone = AutoModel.from_pretrained(
            cfg["base_model"], revision=cfg.get("base_revision"), dtype=dtype, attn_implementation="sdpa"
        )
        backbone.config.use_cache = False
        if (path / "adapter").exists():
            from peft import PeftModel

            backbone = PeftModel.from_pretrained(backbone, str(path / "adapter"))
            if merge:
                backbone = backbone.merge_and_unload()
        model = cls(backbone, cfg.get("head_dim", 512))
        state = load_file(str(path / "leo_head.safetensors"))
        model.markers.load_state_dict({k[len("markers."):]: v for k, v in state.items() if k.startswith("markers.")})
        model.head.load_state_dict({k[len("head."):]: v for k, v in state.items() if k.startswith("head.")})
        model.requires_grad_(False)
        return model.to(device).eval(), cfg
