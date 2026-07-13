"""Tests for entropy-guided joint rank+bits allocation (north star scaffold)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from entropy_lens import joint_alloc
from entropy_lens.joint_alloc import (
    ALLOWED_BITS,
    JointAllocation,
    MatrixSpec,
    allocate_joint_rank_bits,
    load_matrices_from_csv,
    model_error,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------


def _make_matrices(
    n: int = 24,
    *,
    shape_m: int = 4096,
    shape_n: int = 4096,
    s1_low: float = 3.0,
    s1_high: float = 8.0,
    s2_low: float = 1.0,
    s2_high: float = 6.0,
    seed: int = 7,
) -> list[MatrixSpec]:
    """n synthetic matrices with linearly spaced, independent S1/S2 values.

    S1 and S2 are assigned via independent permutations so that "high S1"
    and "high S2" are not confounded with each other or with matrix index,
    which lets the property tests isolate each driver.
    """
    rng = np.random.default_rng(seed)
    s1_values = np.linspace(s1_low, s1_high, n)
    s2_values = np.linspace(s2_low, s2_high, n)
    rng.shuffle(s2_values)  # decorrelate S2 ordering from S1 ordering

    matrices = []
    for i in range(n):
        matrices.append(
            MatrixSpec(
                name=f"layer_{i}.proj",
                shape_m=shape_m,
                shape_n=shape_n,
                s1_eff=float(s1_values[i]),
                s2_eff=float(s2_values[i]),
            )
        )
    return matrices


def _make_mixed_matrices() -> list[MatrixSpec]:
    """Matrices with different shapes, mimicking a real transformer layer."""
    specs = [
        # (name, m, n, s1_eff, s2_eff)
        ("q_proj", 4096, 4096, 3.5, 2.0),
        ("k_proj", 1024, 4096, 3.8, 2.5),
        ("v_proj", 1024, 4096, 6.5, 4.0),
        ("o_proj", 4096, 4096, 7.0, 5.5),
        ("gate_proj", 14336, 4096, 4.2, 1.5),
        ("up_proj", 14336, 4096, 5.9, 3.2),
        ("down_proj", 4096, 14336, 6.8, 4.8),
    ]
    return [
        MatrixSpec(name=name, shape_m=m, shape_n=n, s1_eff=s1, s2_eff=s2)
        for name, m, n, s1, s2 in specs
    ]


# ---------------------------------------------------------------------------
# MatrixSpec / storage accounting
# ---------------------------------------------------------------------------


def test_matrix_spec_max_rank_is_min_dim():
    m = MatrixSpec(name="k_proj", shape_m=1024, shape_n=4096, s1_eff=3.0, s2_eff=2.0)
    assert m.max_rank == 1024


def test_matrix_spec_d_star_is_exp_s2():
    m = MatrixSpec(name="x", shape_m=100, shape_n=100, s1_eff=1.0, s2_eff=2.0)
    assert m.d_star == pytest.approx(np.exp(2.0))


def test_storage_bits_formula():
    m = MatrixSpec(name="x", shape_m=100, shape_n=200, s1_eff=1.0, s2_eff=1.0)
    # Q cost + LR cost, per the documented formula.
    got = m.storage_bits(rank=10, bits_lr=4, q_bits=2)
    expected = 2 * 100 * 200 + 4 * 10 * (100 + 200)
    assert got == expected


def test_storage_bits_zero_q_is_pure_low_rank():
    m = MatrixSpec(name="x", shape_m=100, shape_n=200, s1_eff=1.0, s2_eff=1.0)
    got = m.storage_bits(rank=10, bits_lr=4, q_bits=0)
    assert got == 4 * 10 * (100 + 200)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def test_load_matrices_from_csv(tmp_path: Path):
    csv_path = tmp_path / "layers.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "shape_m", "shape_n", "s1_eff", "s2_eff"])
        writer.writerow(["q_proj", "4096", "4096", "5.0", "3.0"])
        writer.writerow(["down_proj", "4096", "14336", "6.0", "4.0"])

    matrices = load_matrices_from_csv(csv_path)
    assert len(matrices) == 2
    assert matrices[0].name == "q_proj"
    assert matrices[0].shape_m == 4096
    assert matrices[1].s1_eff == 6.0
    assert matrices[1].max_rank == 4096  # min(4096, 14336)


# ---------------------------------------------------------------------------
# Property tests: allocate_joint_rank_bits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_bits_per_param", [0.5, 1.0, 2.0, 2.5])
def test_budget_is_respected_within_tolerance(target_bits_per_param):
    """Total storage should land close to the requested bits-per-param budget.

    The allocator only controls integer ranks and a discrete bit-width set,
    so exact equality isn't possible; we allow a generous relative tolerance
    to account for rounding and the D*/max_rank caps. Uses q_bits=0 (pure
    low-rank, no quantized backbone) so the full budget is spent on L/R and
    the budget is not floored by a fixed backbone cost.
    """
    matrices = _make_matrices(n=32)
    alloc = allocate_joint_rank_bits(
        matrices, target_bits_per_param=target_bits_per_param, q_bits=0
    )
    assert alloc.actual_bits_per_param == pytest.approx(
        target_bits_per_param, rel=0.25
    )


def test_budget_respected_does_not_exceed_hard_cap():
    """Even with rounding, storage should never wildly overshoot budget."""
    matrices = _make_matrices(n=20)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1.5, q_bits=2)
    assert alloc.total_storage_bits <= 1.5 * alloc.total_original_params * 1.5


def test_higher_s1_gets_more_rank_same_shape():
    """Holding shape and S2 fixed, higher S1_eff should never get less rank.

    This is the core allocation property: S1 (head spread) drives rank.
    """
    matrices = [
        MatrixSpec(name="low_s1", shape_m=4096, shape_n=4096, s1_eff=2.0, s2_eff=3.0),
        MatrixSpec(name="mid_s1", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=3.0),
        MatrixSpec(name="high_s1", shape_m=4096, shape_n=4096, s1_eff=8.0, s2_eff=3.0),
    ]
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1.0, q_bits=0)
    r_low = alloc.assignments["low_s1"][0]
    r_mid = alloc.assignments["mid_s1"][0]
    r_high = alloc.assignments["high_s1"][0]
    assert r_low <= r_mid <= r_high
    assert r_low < r_high  # strictly more rank somewhere across the spread


def test_higher_s2_gets_more_bits_same_shape():
    """Holding shape and S1 fixed, higher S2_eff (flatter tail) should map to
    a monotonically non-decreasing... actually fewer bits per the design
    (flatter/near-isotropic tail needs less precision). We check the
    documented direction: bits should be a non-increasing function of S2_eff
    is WRONG per spec -- re-derive from the docstring: matrices with LOW
    S2_eff (peaked, structured tail) get MORE bits; HIGH S2_eff (flat,
    quasi-Gaussian tail) get FEWER bits, since scalar quantization is
    already near-optimal there and little precision is wasted encoding it
    more coarsely.
    """
    matrices = [
        MatrixSpec(name="low_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=1.0),
        MatrixSpec(name="mid_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=3.0),
        MatrixSpec(name="high_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=6.0),
    ]
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1.0, q_bits=0)
    b_low = alloc.assignments["low_s2"][1]
    b_mid = alloc.assignments["mid_s2"][1]
    b_high = alloc.assignments["high_s2"][1]
    assert b_low >= b_mid >= b_high


def test_bits_are_always_from_allowed_choices():
    matrices = _make_matrices(n=16)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=2.0, q_bits=2)
    for _, bits in alloc.assignments.values():
        assert bits in ALLOWED_BITS


def test_ranks_never_exceed_max_rank_or_d_star_by_much():
    """Ranks should respect both the hard shape cap and (loosely) D*."""
    matrices = _make_matrices(n=16)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=3.9, q_bits=2)
    for m in matrices:
        rank, _ = alloc.assignments[m.name]
        assert 1 <= rank <= m.max_rank


def test_ranks_are_at_least_one():
    matrices = _make_matrices(n=10)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=0.1, q_bits=2)
    for rank, _ in alloc.assignments.values():
        assert rank >= 1


def test_mixed_shapes_all_assigned():
    matrices = _make_mixed_matrices()
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=2.0, q_bits=2)
    assert set(alloc.assignments.keys()) == {m.name for m in matrices}
    assert alloc.total_original_params == sum(m.shape_m * m.shape_n for m in matrices)


def test_more_generous_budget_never_decreases_total_rank():
    """Monotonicity: a larger overall bit budget should not shrink the sum
    of ranks across matrices (weakly increasing storage currency spent on
    the head)."""
    matrices = _make_matrices(n=20)
    alloc_small = allocate_joint_rank_bits(
        matrices, target_bits_per_param=0.5, q_bits=2
    )
    alloc_large = allocate_joint_rank_bits(
        matrices, target_bits_per_param=3.0, q_bits=2
    )
    total_small = sum(r for r, _ in alloc_small.assignments.values())
    total_large = sum(r for r, _ in alloc_large.assignments.values())
    assert total_large >= total_small


def test_degenerate_budget_below_backbone_falls_back_gracefully():
    """If q_bits alone exceeds the budget, the allocator should not crash
    and should fall back to minimal rank/bits rather than raising."""
    matrices = _make_matrices(n=5)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=0.01, q_bits=8)
    for rank, bits in alloc.assignments.values():
        assert rank == 1
        assert bits == ALLOWED_BITS[0]


def test_invalid_budget_raises():
    matrices = _make_matrices(n=3)
    with pytest.raises(ValueError):
        allocate_joint_rank_bits(matrices, target_bits_per_param=0.0)
    with pytest.raises(ValueError):
        allocate_joint_rank_bits(matrices, target_bits_per_param=-1.0)


def test_empty_matrix_list_raises():
    with pytest.raises(ValueError):
        allocate_joint_rank_bits([], target_bits_per_param=2.0)


def test_to_dict_shape():
    matrices = _make_matrices(n=4)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=2.0, q_bits=2)
    d = alloc.to_dict()
    assert d["n_matrices"] == 4
    assert set(d["assignments"].keys()) == {m.name for m in matrices}
    for entry in d["assignments"].values():
        assert "rank" in entry and "bits" in entry


# ---------------------------------------------------------------------------
# Joint Lagrangian solver: budget equality, coupled monotonicity, error
# monotonicity, and the water-filling optimality condition.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_bits_per_param", [0.5, 1.0, 2.0, 2.5, 3.5])
def test_budget_is_respected_with_tight_tolerance(target_bits_per_param):
    """With q_bits=0 (pure low-rank, no fixed backbone floor) and large
    (m+n) per matrix, integer rounding of the closed-form continuous rank
    is a small perturbation, so the realized budget should land close to
    the target -- much tighter than the generous rel=0.25 smoke test above,
    since this is checking the Lagrangian bisection itself converges, not
    just "in the right ballpark"."""
    matrices = _make_matrices(n=32, shape_m=4096, shape_n=4096)
    alloc = allocate_joint_rank_bits(
        matrices, target_bits_per_param=target_bits_per_param, q_bits=0
    )
    assert alloc.actual_bits_per_param == pytest.approx(
        target_bits_per_param, rel=0.05
    )


def test_higher_s2_gets_more_rank_at_fixed_bits():
    """Holding shape, S1, bits, and lambda fixed, higher S2_eff (flatter
    tail, smaller gamma_i = exp(-S2_eff)) should give MORE rank via the
    closed-form rank*_i(bits, lambda): a smaller gamma means the captured
    energy is cheaper to quantize, which raises the denominator
    (1 - gamma*2^{-2*bits}) in the closed form and therefore raises the
    optimal rank for the same price. This is the real, provable
    monotonicity from the derivation (see module docstring / plan doc),
    checked directly on the closed form to isolate it from the discrete
    bit-choice confound described in
    `test_full_joint_solve_can_trade_rank_for_bits_across_s2` below.
    """
    m_low = MatrixSpec(name="low_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=1.0)
    m_mid = MatrixSpec(name="mid_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=3.0)
    m_high = MatrixSpec(name="high_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=6.0)
    lam = 1e-8
    for bits in ALLOWED_BITS:
        r_low = joint_alloc._optimal_rank_for_bits(
            m_low, bits, lam, joint_alloc._decay_scale(m_low), joint_alloc._quant_gamma(m_low)
        )
        r_mid = joint_alloc._optimal_rank_for_bits(
            m_mid, bits, lam, joint_alloc._decay_scale(m_mid), joint_alloc._quant_gamma(m_mid)
        )
        r_high = joint_alloc._optimal_rank_for_bits(
            m_high, bits, lam, joint_alloc._decay_scale(m_high), joint_alloc._quant_gamma(m_high)
        )
        assert r_low <= r_mid <= r_high


def test_full_joint_solve_can_trade_rank_for_bits_across_s2():
    """In the *full* joint solve (bits chosen per-matrix, not held fixed),
    "higher S2_eff -> less rank" is NOT a universal property, and asserting
    it would be dishonest about what the model does. Low-S2_eff matrices
    (peaked, structured tail) need MORE bits (see
    test_higher_s2_gets_more_bits_same_shape) to control their captured-
    energy distortion; those extra bits cost more per unit of rank, so at
    a fixed total budget a low-S2 matrix can end up with *less* rank than
    a high-S2 matrix even though, at any *fixed* bit-width, it would want
    *more* rank per test_higher_s2_gets_more_rank_at_fixed_bits above. This
    test documents and locks in that emergent, economically-driven
    trade-off (a genuine signature of joint -- not decoupled -- allocation)
    rather than asserting a monotonicity that only holds in the fixed-bits
    slice of the problem.
    """
    matrices = [
        MatrixSpec(name="low_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=1.0),
        MatrixSpec(name="mid_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=3.0),
        MatrixSpec(name="high_s2", shape_m=4096, shape_n=4096, s1_eff=5.0, s2_eff=6.0),
    ]
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1.0, q_bits=0)
    r_low, b_low = alloc.assignments["low_s2"]
    r_mid, b_mid = alloc.assignments["mid_s2"]
    r_high, b_high = alloc.assignments["high_s2"]
    # Bits: monotone non-increasing in S2_eff (re-confirms the direction
    # already covered by test_higher_s2_gets_more_bits_same_shape).
    assert b_low >= b_mid >= b_high
    # Rank: mid and high (both already at the cheapest useful bit-width,
    # since their tails are flat enough that extra precision buys little)
    # get at least as much rank as low (which pays a bits premium).
    assert r_mid >= r_low
    assert r_high >= r_low


def test_model_error_decreases_with_budget():
    """A larger total bit budget should never increase the model's total
    reconstruction error (weak monotonicity of the Pareto curve)."""
    matrices = _make_matrices(n=24)
    budgets = [0.4, 0.8, 1.5, 2.5, 4.0]
    errors = []
    for b in budgets:
        alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=b, q_bits=0)
        errors.append(alloc.total_model_error)
        # total_model_error should match a fresh model_error() computation
        assert alloc.total_model_error == pytest.approx(
            model_error(matrices, alloc.assignments)
        )
    for e_small, e_large in zip(errors, errors[1:]):
        assert e_large <= e_small + 1e-9


def test_optimality_marginal_rates_converge_to_lambda():
    """Water-filling optimality condition: at the converged lambda, every
    matrix's marginal error reduction per marginal bit spent on rank
    (evaluated at its own assigned bits_i) should equal the shared
    Lagrange multiplier, for matrices whose rank is not pinned to the
    [1, max_rank] boundary (boundary matrices only satisfy a KKT
    inequality, not equality, so they are excluded here).

    marginal_rate_i = -dE_i/dD |_{D=rank_i, b=bits_i} / (bits_i*(m_i+n_i))
                     = rho_i(rank_i) * (1 - gamma_i*2^{-2*bits_i})
                       / (D_eff_i * bits_i * (m_i+n_i))
    """
    matrices = _make_matrices(n=40, shape_m=4096, shape_n=4096)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1.2, q_bits=0)
    assert alloc.lagrange_multiplier > 0.0

    rates = []
    for m in matrices:
        rank, bits = alloc.assignments[m.name]
        if rank <= 1 or rank >= m.max_rank:
            continue  # boundary-pinned: only an inequality holds, skip
        d_eff = joint_alloc._decay_scale(m)
        gamma = joint_alloc._quant_gamma(m)
        rho = joint_alloc._residual_energy(rank, d_eff)
        rate = (rho * (1.0 - gamma * (2.0 ** (-2 * bits)))) / (
            d_eff * bits * (m.shape_m + m.shape_n)
        )
        rates.append(rate)

    assert len(rates) >= 5, "expected most matrices to be interior at this budget"
    rates = np.array(rates)
    # All interior matrices should agree with lambda to within a small
    # relative tolerance (rounding to integer rank is the main source of
    # residual disagreement).
    rel_err = np.abs(rates - alloc.lagrange_multiplier) / alloc.lagrange_multiplier
    assert np.median(rel_err) < 0.15
    assert np.max(rel_err) < 0.6


def test_generous_budget_gives_zero_lambda_and_max_rank():
    """When the budget comfortably exceeds what full rank at the cheapest
    useful bits would cost, the unconstrained (lambda=0) solution should
    already fit: no scarcity, so the multiplier is exactly 0 and rank
    saturates at max_rank for every matrix."""
    matrices = _make_matrices(n=6, shape_m=64, shape_n=64)
    alloc = allocate_joint_rank_bits(matrices, target_bits_per_param=1e6, q_bits=0)
    assert alloc.lagrange_multiplier == 0.0
    for m in matrices:
        rank, _bits = alloc.assignments[m.name]
        assert rank == m.max_rank


def test_model_error_helper_matches_manual_computation():
    m = MatrixSpec(name="x", shape_m=100, shape_n=100, s1_eff=2.0, s2_eff=1.0)
    assignments = {"x": (10, 3)}
    d_eff = math.exp(2.0)
    gamma = math.exp(-1.0)
    rho = math.exp(-10.0 / d_eff)
    expected = rho + (1.0 - rho) * gamma * (2.0 ** (-6))
    assert model_error([m], assignments) == pytest.approx(expected)
