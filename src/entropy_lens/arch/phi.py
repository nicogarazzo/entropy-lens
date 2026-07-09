"""Extractor for Phi-2/3 family."""

from typing import Iterator, Tuple

from .base import WeightExtractor

# Phi-2 safetensors key patterns:
# model.layers.{i}.self_attn.q_proj.weight
# model.layers.{i}.self_attn.k_proj.weight
# model.layers.{i}.self_attn.v_proj.weight
# model.layers.{i}.self_attn.dense.weight  (output projection)
# model.layers.{i}.mlp.fc1.weight          (up projection)
# model.layers.{i}.mlp.fc2.weight          (down projection)


class PhiExtractor(WeightExtractor):
    """Supports Phi-2 and Phi-3."""

    def iter_weight_names(self) -> Iterator[Tuple[str, str]]:
        for i in range(self.num_layers):
            yield f"layer_{i}.q_proj", f"model.layers.{i}.self_attn.q_proj.weight"
            yield f"layer_{i}.k_proj", f"model.layers.{i}.self_attn.k_proj.weight"
            yield f"layer_{i}.v_proj", f"model.layers.{i}.self_attn.v_proj.weight"
            yield f"layer_{i}.o_proj", f"model.layers.{i}.self_attn.dense.weight"
            yield f"layer_{i}.up_proj", f"model.layers.{i}.mlp.fc1.weight"
            yield f"layer_{i}.down_proj", f"model.layers.{i}.mlp.fc2.weight"

    def needs_split(self, safetensors_key: str) -> bool:
        return False

    def split_fused(self, safetensors_key: str, tensor) -> list:
        raise NotImplementedError("Phi family has no fused weights")
