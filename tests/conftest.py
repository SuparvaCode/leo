import pytest
import torch


def char_tokenize(text: str) -> list[int]:
    return [5 + (ord(c) % 290) for c in text]


@pytest.fixture(scope="session")
def tiny_backbone_factory():
    from transformers import AutoModel, Qwen3Config

    def make(seed: int = 0):
        torch.manual_seed(seed)
        cfg = Qwen3Config(
            vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
        )
        cfg._attn_implementation = "sdpa"
        return AutoModel.from_config(cfg).float().eval()

    return make


@pytest.fixture()
def tiny_leo(tiny_backbone_factory):
    from leo.encode import Encoder
    from leo.infer import Leo
    from leo.model import LeoModel

    model = LeoModel.from_backbone(tiny_backbone_factory(), char_tokenize, lora_r=4, head_dim=32).eval()
    torch.manual_seed(1)
    # LoRA B starts at zero; perturb it so the adapter path is exercised too.
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    return Leo(model, Encoder(char_tokenize), {"name": "leo-test"}, torch.device("cpu"), torch.float32, max_row_tokens=4096)
