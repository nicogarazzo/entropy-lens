"""Extractor for LLaMA/Mistral/Qwen family (GQA, SwiGLU)."""

from typing import Iterator, Tuple

from .base import WeightExtractor

# Standard safetensors key pattern for this family:
# model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
# model.layers.{i}.mlp.{gate,up,down}_proj.weight

_ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


class LlamaExtractor(WeightExtractor):
    """Supports LLaMA-2/3, Mistral, Qwen2 and similar architectures."""

    def iter_weight_names(self) -> Iterator[Tuple[str, str]]:
        for i in range(self.num_layers):
            for proj in _ATTN_PROJS:
                canonical = f"layer_{i}.{proj}"
                st_key = f"model.layers.{i}.self_attn.{proj}.weight"
                yield canonical, st_key
            for proj in _MLP_PROJS:
                canonical = f"layer_{i}.{proj}"
                st_key = f"model.layers.{i}.mlp.{proj}.weight"
                yield canonical, st_key

    def needs_split(self, safetensors_key: str) -> bool:
        return False  # LLaMA family has separate Q/K/V, no fused weights

    def split_fused(self, safetensors_key: str, tensor) -> list:
        raise NotImplementedError("LLaMA family has no fused weights")
