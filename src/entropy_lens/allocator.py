"""Entropy-guided rank allocation for SVD compression.

Given a budget (fraction of total parameters to keep) and per-layer spectral
statistics (S1, shapes), assigns a truncation rank D_i to each layer.

Three strategies:
  - uniform:      same compression ratio for every layer.
  - proportional: D_i proportional to min(m_i, n_i), so each layer keeps
                   the same fraction of its full rank.
  - entropy:      layers with higher S1 (more spread spectrum) get more rank.
                   Uses bisection to find an error threshold epsilon such that
                   sum D_i(epsilon) * (m_i + n_i) = budget.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Strategy = Literal["uniform", "proportional", "entropy"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    """Everything the allocator needs to know about one weight matrix."""

    name: str
    shape_m: int  # rows
    shape_n: int  # cols
    s1: float  # von Neumann entropy
    rank: int = 0  # min(shape_m, shape_n), set automatically

    def __post_init__(self):
        self.rank = min(self.shape_m, self.shape_n)

    @property
    def original_params(self) -> int:
        return self.shape_m * self.shape_n

    def compressed_params(self, d: int) -> int:
        return d * (self.shape_m + self.shape_n)


@dataclass
class Allocation:
    """Result of rank allocation: per-layer ranks + summary stats."""

    ranks: dict[str, int]
    strategy: Strategy
    budget_ratio: float
    total_original_params: int
    total_compressed_params: int
    layers: list[LayerSpec] = field(default_factory=list)

    @property
    def actual_ratio(self) -> float:
        if self.total_original_params == 0:
            return 0.0
        return self.total_compressed_params / self.total_original_params

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "budget_ratio": self.budget_ratio,
            "actual_ratio": round(self.actual_ratio, 6),
            "total_original_params": self.total_original_params,
            "total_compressed_params": self.total_compressed_params,
            "n_layers": len(self.ranks),
            "ranks": self.ranks,
        }


# ---------------------------------------------------------------------------
# Shape inference from architecture config
# ---------------------------------------------------------------------------

# Mapping from proj_type to (rows, cols) as functions of config params.
# Conventions: weight matrices are stored as (out_features, in_features)
# in PyTorch, but SVD doesn't care about transpose — only shapes matter.

_LLAMA_SHAPE_MAP = {
    "q_proj": lambda c: (c["num_attention_heads"] * c["head_dim"], c["hidden_size"]),
    "k_proj": lambda c: (c["num_key_value_heads"] * c["head_dim"], c["hidden_size"]),
    "v_proj": lambda c: (c["num_key_value_heads"] * c["head_dim"], c["hidden_size"]),
    "o_proj": lambda c: (c["hidden_size"], c["num_attention_heads"] * c["head_dim"]),
    "gate_proj": lambda c: (c["intermediate_size"], c["hidden_size"]),
    "up_proj": lambda c: (c["intermediate_size"], c["hidden_size"]),
    "down_proj": lambda c: (c["hidden_size"], c["intermediate_size"]),
}


def _resolve_head_dim(config: dict) -> dict:
    """Ensure head_dim is present in config dict."""
    c = dict(config)
    if "head_dim" not in c:
        c["head_dim"] = c["hidden_size"] // c["num_attention_heads"]
    if "num_key_value_heads" not in c:
        c["num_key_value_heads"] = c["num_attention_heads"]
    return c


# GPT-2 shape map: Conv1D stores (in_features, out_features).
# c_attn splits into Q/K/V each (n_embd, n_embd).
# c_fc is (n_embd, 4*n_embd), c_proj is (4*n_embd, n_embd) in Conv1D layout.
_GPT2_SHAPE_MAP = {
    "q_proj": lambda c: (c["hidden_size"], c["hidden_size"]),
    "k_proj": lambda c: (c["hidden_size"], c["hidden_size"]),
    "v_proj": lambda c: (c["hidden_size"], c["hidden_size"]),
    "o_proj": lambda c: (c["hidden_size"], c["hidden_size"]),
    "up_proj": lambda c: (c["hidden_size"], c["intermediate_size"]),
    "down_proj": lambda c: (c["intermediate_size"], c["hidden_size"]),
}


def _normalize_config(raw: dict) -> dict:
    """Normalize architecture-specific config keys to canonical names."""
    c = dict(raw)
    # GPT-2: n_embd -> hidden_size, n_head -> num_attention_heads
    if "n_embd" in c and "hidden_size" not in c:
        c["hidden_size"] = c["n_embd"]
    if "n_head" in c and "num_attention_heads" not in c:
        c["num_attention_heads"] = c["n_head"]
    if "n_inner" in c and "intermediate_size" not in c:
        c["intermediate_size"] = c["n_inner"]
    # GPT-2 default: intermediate_size = 4 * hidden_size
    if "intermediate_size" not in c and "hidden_size" in c:
        c["intermediate_size"] = 4 * c["hidden_size"]
    return c


def infer_shapes_from_config(config_path: str | Path) -> dict[str, tuple[int, int]]:
    """Read a HuggingFace config.json and return {proj_type: (m, n)} shapes.

    Works for LLaMA/Mistral/Qwen families and GPT-2.
    """
    with open(config_path) as f:
        raw = json.load(f)

    normalized = _normalize_config(raw)
    model_type = raw.get("model_type", "").lower()

    # Pick the right shape map
    if model_type == "gpt2":
        shape_map = _GPT2_SHAPE_MAP
        config = normalized
    else:
        config = _resolve_head_dim(normalized)
        shape_map = _LLAMA_SHAPE_MAP

    shapes = {}
    for proj_type, shape_fn in shape_map.items():
        try:
            shapes[proj_type] = shape_fn(config)
        except KeyError:
            pass
    return shapes


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_layers_from_csv(
    csv_path: str | Path,
    config_path: str | Path | None = None,
    shapes: dict[str, tuple[int, int]] | None = None,
) -> list[LayerSpec]:
    """Load layer specs from an entropy-lens results CSV.

    The CSV must have at minimum: name, s1, and either:
      - shape_m + shape_n columns (both dimensions), or
      - shape_m or rank column (min dimension only), plus a way to infer
        the other dimension (via config_path or shapes dict).

    Args:
        csv_path: path to results CSV.
        config_path: optional path to HuggingFace config.json for shape inference.
        shapes: optional dict mapping proj_type -> (m, n). Overrides config_path.

    Returns:
        List of LayerSpec, one per row in the CSV.
    """
    if shapes is None and config_path is not None:
        shapes = infer_shapes_from_config(config_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    has_both_dims = "shape_m" in fieldnames and "shape_n" in fieldnames
    has_proj_type = "proj_type" in fieldnames

    # Determine which column has the min-dimension
    rank_col = None
    if "shape_m" in fieldnames and "shape_n" not in fieldnames:
        rank_col = "shape_m"
    elif "rank" in fieldnames:
        rank_col = "rank"

    layers = []
    for row in rows:
        name = row["name"]
        s1 = float(row["s1"])

        if has_both_dims:
            m = int(row["shape_m"])
            n = int(row["shape_n"])
        elif has_proj_type and shapes is not None:
            proj_type = row["proj_type"]
            if proj_type not in shapes:
                raise ValueError(
                    f"proj_type '{proj_type}' not found in shapes dict. "
                    f"Available: {list(shapes.keys())}"
                )
            m, n = shapes[proj_type]
        elif rank_col is not None:
            # Fallback: assume square matrix (conservative estimate)
            r = int(row[rank_col])
            m = n = r
        else:
            raise ValueError(
                f"Cannot determine matrix dimensions from CSV columns: {fieldnames}. "
                "Provide config_path or shapes dict, or ensure CSV has shape_m + shape_n."
            )

        layers.append(LayerSpec(name=name, shape_m=m, shape_n=n, s1=s1))

    return layers


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------


def _clamp_rank(d: float, rank: int) -> int:
    """Clamp a rank value to [1, rank]."""
    return max(1, min(int(round(d)), rank))


def _total_params(layers: list[LayerSpec], ranks: list[int]) -> int:
    """Total compressed parameters for a given rank assignment."""
    return sum(layer.compressed_params(d) for layer, d in zip(layers, ranks))


def _total_original(layers: list[LayerSpec]) -> int:
    return sum(layer.original_params for layer in layers)


def allocate_uniform(layers: list[LayerSpec], budget_ratio: float) -> list[int]:
    """Same compression ratio for every layer.

    Each layer gets D_i = ratio * min(m_i, n_i), where ratio is chosen so
    that total compressed params = budget_ratio * total original params.

    This means all layers lose the same fraction of their rank, but layers
    with different aspect ratios will have different parameter savings.
    We solve for ratio via bisection.
    """
    total_orig = _total_original(layers)
    target = budget_ratio * total_orig

    def params_at_ratio(ratio: float) -> int:
        ranks = [_clamp_rank(ratio * layer.rank, layer.rank) for layer in layers]
        return _total_params(layers, ranks)

    # Bisection over ratio in [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if params_at_ratio(mid) < target:
            lo = mid
        else:
            hi = mid

    return [_clamp_rank(hi * layer.rank, layer.rank) for layer in layers]


def allocate_proportional(layers: list[LayerSpec], budget_ratio: float) -> list[int]:
    """D_i proportional to min(m_i, n_i).

    This is equivalent to uniform when all aspect ratios are the same.
    When they differ, it accounts for the different cost-per-rank:
    a layer with large (m+n) costs more parameters per rank increment.

    We solve: D_i = alpha * rank_i, where alpha is chosen so
    sum D_i * (m_i + n_i) = budget * sum(m_i * n_i).
    """
    total_orig = _total_original(layers)
    target = budget_ratio * total_orig

    # D_i = alpha * rank_i => cost = alpha * sum(rank_i * (m_i + n_i))
    cost_per_alpha = sum(layer.rank * (layer.shape_m + layer.shape_n) for layer in layers)
    if cost_per_alpha == 0:
        return [1] * len(layers)

    alpha = target / cost_per_alpha
    return [_clamp_rank(alpha * layer.rank, layer.rank) for layer in layers]


def allocate_entropy(layers: list[LayerSpec], budget_ratio: float) -> list[int]:
    """Entropy-guided allocation.

    Layers with higher S1 have more spread spectra and need more ranks to
    achieve the same reconstruction error. We model the required rank as:

        D_i(epsilon) = c * exp(alpha * S1_i)

    where epsilon is a uniform error tolerance, and c, alpha are the fitted
    law parameters (slope ~ 1 from the entropy-compression law).

    We use bisection to find the right scale factor `c` such that
    total compressed params hits the budget. Alpha = 1 (the law says
    slope ~ 1 for the exponential relationship).
    """
    total_orig = _total_original(layers)
    target = budget_ratio * total_orig

    s1_arr = np.array([layer.s1 for layer in layers])

    # Raw weights: exp(S1_i). Layers with higher entropy get exponentially
    # more rank. This is the core insight from the entropy-compression law.
    raw_weights = np.exp(s1_arr)

    def params_at_scale(scale: float) -> int:
        ranks = [
            _clamp_rank(scale * w, layer.rank)
            for w, layer in zip(raw_weights, layers)
        ]
        return _total_params(layers, ranks)

    # Bisection over scale
    # Lower bound: scale that would give D=1 for all (or close)
    # Upper bound: scale that would give full rank for all
    lo = 0.0
    hi = max(layer.rank / max(w, 1e-30) for w, layer in zip(raw_weights, layers))

    for _ in range(64):
        mid = (lo + hi) / 2.0
        if params_at_scale(mid) < target:
            lo = mid
        else:
            hi = mid

    return [
        _clamp_rank(hi * w, layer.rank)
        for w, layer in zip(raw_weights, layers)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "uniform": allocate_uniform,
    "proportional": allocate_proportional,
    "entropy": allocate_entropy,
}


def allocate_ranks(
    csv_path: str | Path,
    budget_ratio: float,
    strategy: Strategy = "entropy",
    config_path: str | Path | None = None,
    shapes: dict[str, tuple[int, int]] | None = None,
) -> Allocation:
    """Allocate truncation ranks to each layer under a parameter budget.

    Args:
        csv_path: path to entropy-lens results CSV.
        budget_ratio: fraction of original parameters to keep (0 < ratio <= 1).
        strategy: "uniform", "proportional", or "entropy".
        config_path: optional path to HuggingFace config.json for shape inference.
        shapes: optional dict mapping proj_type -> (m, n).

    Returns:
        Allocation with per-layer ranks and summary statistics.

    Raises:
        ValueError: if budget_ratio is out of range or strategy is unknown.
    """
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError(f"budget_ratio must be in (0, 1], got {budget_ratio}")

    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGIES.keys())}")

    layers = load_layers_from_csv(csv_path, config_path=config_path, shapes=shapes)
    if not layers:
        raise ValueError(f"No layers loaded from {csv_path}")

    alloc_fn = _STRATEGIES[strategy]
    rank_list = alloc_fn(layers, budget_ratio)

    ranks_dict = {layer.name: d for layer, d in zip(layers, rank_list)}
    total_orig = _total_original(layers)
    total_comp = _total_params(layers, rank_list)

    return Allocation(
        ranks=ranks_dict,
        strategy=strategy,
        budget_ratio=budget_ratio,
        total_original_params=total_orig,
        total_compressed_params=total_comp,
        layers=layers,
    )
