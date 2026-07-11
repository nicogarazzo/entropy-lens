"""Tests for entropy-guided rank allocator."""

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from entropy_lens.allocator import (
    Allocation,
    LayerSpec,
    allocate_entropy,
    allocate_proportional,
    allocate_ranks,
    allocate_uniform,
    load_layers_from_csv,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------


def _make_layers(
    n: int = 20,
    *,
    shape_m: int = 4096,
    shape_n: int = 4096,
    s1_low: float = 3.0,
    s1_high: float = 8.0,
    seed: int = 42,
) -> list[LayerSpec]:
    """Create n synthetic layers with linearly spaced S1 values."""
    rng = np.random.default_rng(seed)
    s1_values = np.linspace(s1_low, s1_high, n)
    layers = []
    for i, s1 in enumerate(s1_values):
        layers.append(
            LayerSpec(
                name=f"layer_{i}.proj",
                shape_m=shape_m,
                shape_n=shape_n,
                s1=float(s1),
            )
        )
    return layers


def _make_mixed_layers() -> list[LayerSpec]:
    """Layers with different shapes, mimicking a real transformer."""
    specs = [
        # Attention: q/o are square, k/v are rectangular
        ("q_proj", 4096, 4096, 3.5),
        ("k_proj", 1024, 4096, 3.8),
        ("v_proj", 1024, 4096, 6.5),
        ("o_proj", 4096, 4096, 7.0),
        # MLP: rectangular
        ("gate_proj", 14336, 4096, 7.5),
        ("up_proj", 14336, 4096, 7.8),
        ("down_proj", 4096, 14336, 7.2),
    ]
    layers = []
    for name, m, n, s1 in specs:
        layers.append(LayerSpec(name=f"layer_0.{name}", shape_m=m, shape_n=n, s1=s1))
    return layers


def _write_csv(layers: list[LayerSpec], path: Path, include_shape_n: bool = True):
    """Write a minimal CSV from LayerSpec list."""
    fieldnames = ["name", "s1", "shape_m"]
    if include_shape_n:
        fieldnames.append("shape_n")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer in layers:
            row = {"name": layer.name, "s1": layer.s1, "shape_m": layer.shape_m}
            if include_shape_n:
                row["shape_n"] = layer.shape_n
            writer.writerow(row)


# ---------------------------------------------------------------------------
# LayerSpec
# ---------------------------------------------------------------------------


class TestLayerSpec:
    def test_rank_is_min(self):
        layer = LayerSpec(name="test", shape_m=1024, shape_n=4096, s1=5.0)
        assert layer.rank == 1024

    def test_original_params(self):
        layer = LayerSpec(name="test", shape_m=1024, shape_n=4096, s1=5.0)
        assert layer.original_params == 1024 * 4096

    def test_compressed_params(self):
        layer = LayerSpec(name="test", shape_m=1024, shape_n=4096, s1=5.0)
        assert layer.compressed_params(100) == 100 * (1024 + 4096)


# ---------------------------------------------------------------------------
# Budget constraint: all strategies must respect it
# ---------------------------------------------------------------------------


class TestBudgetConstraint:
    """Every strategy must produce total_compressed <= budget * total_original."""

    @pytest.mark.parametrize("strategy", ["uniform", "proportional", "entropy"])
    @pytest.mark.parametrize("budget", [0.20, 0.40, 0.60, 0.80, 0.95])
    def test_budget_respected_square(self, strategy, budget):
        layers = _make_layers(20)
        fn = {"uniform": allocate_uniform, "proportional": allocate_proportional, "entropy": allocate_entropy}[strategy]
        ranks = fn(layers, budget)
        total_orig = sum(l.original_params for l in layers)
        total_comp = sum(l.compressed_params(d) for l, d in zip(layers, ranks))
        # Allow 1% tolerance for rounding
        assert total_comp <= budget * total_orig * 1.01, (
            f"{strategy} budget={budget}: compressed={total_comp} > "
            f"budget={budget * total_orig:.0f}"
        )

    @pytest.mark.parametrize("strategy", ["uniform", "proportional", "entropy"])
    def test_budget_respected_mixed_shapes(self, strategy):
        layers = _make_mixed_layers()
        budget = 0.50
        fn = {"uniform": allocate_uniform, "proportional": allocate_proportional, "entropy": allocate_entropy}[strategy]
        ranks = fn(layers, budget)
        total_orig = sum(l.original_params for l in layers)
        total_comp = sum(l.compressed_params(d) for l, d in zip(layers, ranks))
        assert total_comp <= budget * total_orig * 1.01

    @pytest.mark.parametrize("strategy", ["uniform", "proportional", "entropy"])
    def test_ranks_in_valid_range(self, strategy):
        layers = _make_layers(20)
        fn = {"uniform": allocate_uniform, "proportional": allocate_proportional, "entropy": allocate_entropy}[strategy]
        ranks = fn(layers, 0.60)
        for layer, d in zip(layers, ranks):
            assert 1 <= d <= layer.rank, f"{layer.name}: rank {d} out of [1, {layer.rank}]"


# ---------------------------------------------------------------------------
# Entropy-guided: core property
# ---------------------------------------------------------------------------


class TestEntropyGuidedProperties:
    """The entropy strategy must assign more rank to layers with higher S1."""

    def test_higher_s1_gets_more_rank(self):
        """With identical shapes, rank must be monotonically non-decreasing with S1."""
        layers = _make_layers(20, s1_low=2.0, s1_high=8.0)
        ranks = allocate_entropy(layers, 0.50)
        # Check monotonicity: rank[i] <= rank[i+1] since S1 is sorted ascending
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1], (
                f"layer {i} (S1={layers[i].s1:.2f}) got rank {ranks[i]} "
                f"> layer {i+1} (S1={layers[i+1].s1:.2f}) rank {ranks[i+1]}"
            )

    def test_low_s1_gets_less_than_high_s1(self):
        """Layer with S1=2 should get strictly fewer ranks than layer with S1=7."""
        layers = [
            LayerSpec(name="low_entropy", shape_m=4096, shape_n=4096, s1=2.0),
            LayerSpec(name="high_entropy", shape_m=4096, shape_n=4096, s1=7.0),
        ]
        ranks = allocate_entropy(layers, 0.50)
        assert ranks[0] < ranks[1], f"low_s1 rank={ranks[0]} >= high_s1 rank={ranks[1]}"

    def test_extreme_s1_difference(self):
        """Very low S1 layer should get near-minimum rank."""
        layers = [
            LayerSpec(name="trivial", shape_m=4096, shape_n=4096, s1=0.5),
            LayerSpec(name="complex", shape_m=4096, shape_n=4096, s1=8.0),
        ]
        ranks = allocate_entropy(layers, 0.50)
        # exp(0.5) / exp(8.0) ~ 0.0006, so trivial should get very little
        assert ranks[0] < ranks[1] * 0.1, (
            f"trivial rank={ranks[0]} is too close to complex rank={ranks[1]}"
        )


# ---------------------------------------------------------------------------
# Uniform strategy
# ---------------------------------------------------------------------------


class TestUniformStrategy:
    def test_same_shapes_same_rank(self):
        """All square layers with same shape should get identical ranks."""
        layers = _make_layers(10, s1_low=3.0, s1_high=8.0)
        ranks = allocate_uniform(layers, 0.50)
        assert len(set(ranks)) == 1, f"Expected uniform ranks, got {set(ranks)}"


# ---------------------------------------------------------------------------
# Proportional strategy
# ---------------------------------------------------------------------------


class TestProportionalStrategy:
    def test_same_shapes_same_rank(self):
        """Like uniform, same shapes => same rank."""
        layers = _make_layers(10)
        ranks = allocate_proportional(layers, 0.50)
        assert len(set(ranks)) == 1

    def test_larger_matrix_gets_more_rank(self):
        """A matrix with larger rank capacity should get more absolute rank."""
        layers = [
            LayerSpec(name="small", shape_m=512, shape_n=512, s1=5.0),
            LayerSpec(name="big", shape_m=4096, shape_n=4096, s1=5.0),
        ]
        ranks = allocate_proportional(layers, 0.50)
        assert ranks[1] > ranks[0]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class TestCSVLoading:
    def test_load_with_both_dims(self, tmp_path):
        layers = _make_mixed_layers()
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path, include_shape_n=True)
        loaded = load_layers_from_csv(csv_path)
        assert len(loaded) == len(layers)
        for orig, loaded_l in zip(layers, loaded):
            assert loaded_l.name == orig.name
            assert loaded_l.shape_m == orig.shape_m
            assert loaded_l.shape_n == orig.shape_n
            assert abs(loaded_l.s1 - orig.s1) < 1e-6

    def test_load_with_shape_m_only_assumes_square(self, tmp_path):
        """Without shape_n or config, falls back to square assumption."""
        layers = [LayerSpec(name="test", shape_m=1024, shape_n=4096, s1=5.0)]
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path, include_shape_n=False)
        loaded = load_layers_from_csv(csv_path)
        # Falls back to square: both dims = shape_m
        assert loaded[0].shape_m == 1024
        assert loaded[0].shape_n == 1024

    def test_load_with_config(self, tmp_path):
        """With a config.json, correctly infers both dimensions."""
        config = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        csv_path = tmp_path / "results.csv"
        fieldnames = ["name", "s1", "shape_m", "proj_type"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"name": "layer_0.k_proj", "s1": "3.5", "shape_m": "1024", "proj_type": "k_proj"})
            writer.writerow({"name": "layer_0.gate_proj", "s1": "7.5", "shape_m": "4096", "proj_type": "gate_proj"})

        loaded = load_layers_from_csv(csv_path, config_path=config_path)
        assert len(loaded) == 2
        # k_proj: (num_kv_heads * head_dim, hidden_size) = (8*128, 4096) = (1024, 4096)
        assert loaded[0].shape_m == 1024
        assert loaded[0].shape_n == 4096
        # gate_proj: (14336, 4096)
        assert loaded[1].shape_m == 14336
        assert loaded[1].shape_n == 4096


# ---------------------------------------------------------------------------
# Integration: allocate_ranks top-level API
# ---------------------------------------------------------------------------


class TestAllocateRanks:
    def test_full_pipeline(self, tmp_path):
        layers = _make_mixed_layers()
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        result = allocate_ranks(csv_path, budget_ratio=0.60, strategy="entropy")
        assert isinstance(result, Allocation)
        assert len(result.ranks) == len(layers)
        assert result.actual_ratio <= 0.60 * 1.01
        assert result.strategy == "entropy"

    def test_invalid_budget(self, tmp_path):
        layers = _make_layers(5)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        with pytest.raises(ValueError, match="budget_ratio"):
            allocate_ranks(csv_path, budget_ratio=0.0)
        with pytest.raises(ValueError, match="budget_ratio"):
            allocate_ranks(csv_path, budget_ratio=1.5)

    def test_invalid_strategy(self, tmp_path):
        layers = _make_layers(5)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        with pytest.raises(ValueError, match="Unknown strategy"):
            allocate_ranks(csv_path, budget_ratio=0.5, strategy="bogus")

    def test_to_dict(self, tmp_path):
        layers = _make_layers(5)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        result = allocate_ranks(csv_path, budget_ratio=0.50, strategy="uniform")
        d = result.to_dict()
        assert "ranks" in d
        assert "strategy" in d
        assert d["strategy"] == "uniform"
        assert "actual_ratio" in d
        assert d["actual_ratio"] <= 0.50 * 1.01

    @pytest.mark.parametrize("strategy", ["uniform", "proportional"])
    def test_budget_1_uses_full_budget(self, tmp_path, strategy):
        """Budget=1.0 should use as many params as allowed by the SVD format.

        Note: for square matrices, D*(m+n) = D*2n. To equal m*n = n^2,
        we need D = n/2. So full budget doesn't mean full rank.

        Entropy strategy is excluded because it deliberately concentrates
        budget on high-S1 layers, leaving low-S1 layers with small ranks.
        """
        layers = _make_layers(5, shape_m=64, shape_n=64)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        result = allocate_ranks(csv_path, budget_ratio=1.0, strategy=strategy)
        # All ranks should be at or near n/2 for square matrices
        for layer in layers:
            max_useful_rank = (layer.shape_m * layer.shape_n) // (layer.shape_m + layer.shape_n)
            assert result.ranks[layer.name] >= max_useful_rank - 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_layer(self, tmp_path):
        layers = [LayerSpec(name="only", shape_m=4096, shape_n=4096, s1=5.0)]
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        for strategy in ["uniform", "proportional", "entropy"]:
            result = allocate_ranks(csv_path, budget_ratio=0.50, strategy=strategy)
            assert len(result.ranks) == 1
            assert result.actual_ratio <= 0.50 * 1.01

    def test_very_tight_budget(self, tmp_path):
        layers = _make_layers(10)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        result = allocate_ranks(csv_path, budget_ratio=0.01, strategy="entropy")
        # All ranks should be 1 (minimum)
        for name, d in result.ranks.items():
            assert d >= 1

    def test_all_same_s1(self, tmp_path):
        """When all layers have identical S1, entropy should behave like uniform."""
        layers = _make_layers(10, s1_low=5.0, s1_high=5.0)
        csv_path = tmp_path / "results.csv"
        _write_csv(layers, csv_path)

        result_entropy = allocate_ranks(csv_path, budget_ratio=0.50, strategy="entropy")
        result_uniform = allocate_ranks(csv_path, budget_ratio=0.50, strategy="uniform")

        # Should give same ranks since S1 is constant
        for name in result_entropy.ranks:
            assert result_entropy.ranks[name] == result_uniform.ranks[name]
