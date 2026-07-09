"""Extractor for GPT-2 family (fused c_attn Conv1D)."""

from typing import Iterator, Tuple

import torch

from .base import WeightExtractor

# GPT-2 safetensors key patterns:
# h.{i}.attn.c_attn.weight   -> fused Q/K/V, shape (768, 2304) Conv1D format
# h.{i}.attn.c_proj.weight   -> O projection, shape (768, 768)
# h.{i}.mlp.c_fc.weight      -> FFN up, shape (768, 3072)
# h.{i}.mlp.c_proj.weight    -> FFN down, shape (3072, 768)


class GPT2Extractor(WeightExtractor):
    """Supports GPT-2 Small/Medium/Large/XL."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.hidden_size = config.get("n_embd", 768)

    def iter_weight_names(self) -> Iterator[Tuple[str, str]]:
        for i in range(self.num_layers):
            # c_attn is fused Q/K/V. We yield it once and split later.
            yield f"layer_{i}.q_proj", f"h.{i}.attn.c_attn.weight"
            yield f"layer_{i}.k_proj", f"h.{i}.attn.c_attn.weight"
            yield f"layer_{i}.v_proj", f"h.{i}.attn.c_attn.weight"
            yield f"layer_{i}.o_proj", f"h.{i}.attn.c_proj.weight"
            yield f"layer_{i}.up_proj", f"h.{i}.mlp.c_fc.weight"
            yield f"layer_{i}.down_proj", f"h.{i}.mlp.c_proj.weight"

    def needs_split(self, safetensors_key: str) -> bool:
        return safetensors_key.endswith("attn.c_attn.weight")

    def split_fused(self, safetensors_key: str, tensor: torch.Tensor) -> list:
        """Split fused c_attn (in, 3*in) into Q, K, V each (in, in).

        GPT-2 Conv1D stores weights as (in_features, out_features).
        c_attn.weight shape: (768, 2304) where 2304 = 3 * 768.
        Split along dim=1 into three (768, 768) matrices.
        """
        h = self.hidden_size
        # Extract layer index from key for canonical naming
        parts = safetensors_key.split(".")
        layer_idx = int(parts[1])  # h.{i}.attn.c_attn.weight
        q, k, v = torch.split(tensor, h, dim=1)
        return [
            (f"layer_{layer_idx}.q_proj", q),
            (f"layer_{layer_idx}.k_proj", k),
            (f"layer_{layer_idx}.v_proj", v),
        ]
