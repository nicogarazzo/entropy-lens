"""Abstract base class for architecture-specific weight extractors."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class WeightExtractor(ABC):
    """Maps safetensors keys to canonical layer names for a given architecture.

    Each subclass knows the naming convention of one model family and yields
    (canonical_name, safetensors_key) pairs. canonical_name format:
        layer_{i}.{proj_type}
    where proj_type is one of: q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj.
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))

    @abstractmethod
    def iter_weight_names(self) -> Iterator[Tuple[str, str]]:
        """Yield (canonical_name, safetensors_key) pairs for all weight matrices."""

    @abstractmethod
    def needs_split(self, safetensors_key: str) -> bool:
        """Return True if this key is a fused weight that needs splitting."""

    @abstractmethod
    def split_fused(self, safetensors_key: str, tensor) -> list:
        """Split a fused tensor into individual projections.

        Returns list of (canonical_name, tensor) tuples.
        """
