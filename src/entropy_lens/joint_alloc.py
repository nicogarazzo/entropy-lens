"""Entropy-guided JOINT rank + bit-width allocation (north star, CALDERA fork).

Rationale (mesa conjunta 2026-07-12, see /Users/nicolascalderon/Documents/dev/Qtech
Idea/mesa-conjunta-2026-07-12.md): a weight matrix W decomposes as

    W ~= W_D + R

where W_D is a rank-D low-rank component that captures the anisotropic *head*
of the (whitened) spectrum, and R is a quantized residual covering the flat,
near-isotropic *tail*. The two mechanisms are spectrally complementary:

  - Rank is the efficient currency for the head: it is exact (in the data
    metric) up to the truncation point and each additional dimension buys
    a lot of variance when the spectrum is steep. S1_eff (von Neumann
    entropy of the whitened singular values, see whiten.whitened_svdvals +
    spectral.compute_s1) measures how "spread" that head is: high S1_eff
    means the energy is smeared over many directions, so more rank is
    needed to capture a fixed fraction of it.
  - Bits are the efficient currency for the tail: once the residual
    spectrum is flat and quasi-Gaussian, scalar/lattice quantization is
    close to information-theoretically optimal, and there is little to be
    gained from spending more rank there. S2_eff (Renyi-2 entropy /
    participation ratio, spectral.compute_s2) measures the effective size
    of that tail; the crossover point where the head stops being worth
    more rank is D* ~ exp(S2_eff) (participation ratio of the residual).

This module implements the *reallocation* half of the north star: given
S1/S2 per matrix (already computed by entropy_lens.spectral over the
whitened spectrum, see entropy_lens.whiten) and a joint bit budget, decide
a (rank, bits) pair per matrix. This is the piece meant to replace
CALDERA's uniform CalderaParams(rank=.., Q_bits=.., L_bits=.., R_bits=..)
-- see experiments/caldera_integration_plan.md for the full integration
design, the CALDERA algorithm summary, and the licensing caveat that
blocks copying code from pilancilab/caldera directly.

Budget accounting (see MatrixSpec.storage_bits): for a matrix decomposed
as W ~= Q + L @ R with Q full-size at Q_bits and L (m x rank), R
(rank x n) at bits_lr each,

    storage_bits(rank, bits_lr) = Q_bits * m * n + bits_lr * rank * (m + n)

This mirrors allocator.py's LayerSpec.compressed_params, generalized to
bit-currency instead of parameter-count currency, and separates the two
knobs (rank, bits) that CALDERA today holds uniform across all layers.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Bit-widths CALDERA's lattice quantizer (QuIP# E8P codebooks) supports today.
# A from-scratch quantizer is not restricted to these, but we default to this
# set for compatibility with the CALDERA/QuIP# backbone.
ALLOWED_BITS: tuple[int, ...] = (2, 3, 4)


@dataclass
class MatrixSpec:
    """Everything the joint allocator needs to know about one weight matrix.

    S1 and S2 are expected to be computed in the *data metric* (i.e. on the
    whitened spectrum sigma(W @ L), per whiten.whitened_svdvals +
    spectral.compute_s1/compute_s2), not the raw Frobenius spectrum -- the
    mesa's finding is that raw-metric S1 has ~zero rank correlation with
    compression damage (results/mistralai_Mistral-7B-v0.3/ablation_analysis.md).
    """

    name: str
    shape_m: int  # rows (out_features)
    shape_n: int  # cols (in_features)
    s1_eff: float  # von Neumann entropy of the whitened spectrum (head)
    s2_eff: float  # Renyi-2 entropy of the whitened spectrum (tail / D*)
    max_rank: int = 0  # min(shape_m, shape_n), set automatically

    def __post_init__(self):
        self.max_rank = min(self.shape_m, self.shape_n)

    @property
    def d_star(self) -> float:
        """Head/tail crossover rank, D* ~ exp(S2_eff) (participation ratio).

        Beyond this rank the residual spectrum is close to flat/isotropic,
        so additional rank buys little relative to spending the same
        storage on bits. Used as a soft cap on how much rank a single
        matrix should absorb, independent of the S1-driven head weight.
        """
        return float(np.exp(self.s2_eff))

    def storage_bits(self, rank: int, bits_lr: int, q_bits: int) -> float:
        """Total storage in bits for W ~= Q + L @ R at this configuration.

        Q is the full-size quantized backbone (m*n entries at q_bits each).
        L is (m, rank), R is (rank, n), both quantized at bits_lr.
        Set q_bits=0 to model a pure low-rank (no residual) matrix.
        """
        q_cost = q_bits * self.shape_m * self.shape_n
        lr_cost = bits_lr * rank * (self.shape_m + self.shape_n)
        return q_cost + lr_cost


@dataclass
class JointAllocation:
    """Result of joint rank+bits allocation: per-matrix (rank, bits) + stats."""

    assignments: dict[str, tuple[int, int]]  # name -> (rank, bits_lr)
    q_bits: int
    target_bits_per_param: float
    total_original_params: int
    total_storage_bits: float
    matrices: list[MatrixSpec] = field(default_factory=list)
    # Lagrange multiplier (shadow price of one bit of LR storage) found by
    # the joint solver's outer bisection -- see allocate_joint_rank_bits'
    # docstring for the derivation. At optimality, every matrix whose rank
    # is not pinned to a [1, max_rank] boundary should have (marginal model
    # error reduction per marginal bit spent) ~= this value; see
    # `marginal_rate` and the "optimality" property tests. 0.0 in the
    # degenerate fallback path (budget below the fixed backbone cost),
    # where no Lagrangian search happens.
    lagrange_multiplier: float = 0.0
    # Total model error (sum of per-matrix E_i(rank_i, bits_i), see
    # `model_error`/`_error` below) at the returned assignment. Provided so
    # callers/tests can check error monotonically decreases as the budget
    # grows without recomputing the model by hand.
    total_model_error: float = 0.0

    @property
    def actual_bits_per_param(self) -> float:
        if self.total_original_params == 0:
            return 0.0
        return self.total_storage_bits / self.total_original_params

    def to_dict(self) -> dict:
        return {
            "q_bits": self.q_bits,
            "target_bits_per_param": self.target_bits_per_param,
            "actual_bits_per_param": round(self.actual_bits_per_param, 4),
            "total_original_params": self.total_original_params,
            "total_storage_bits": self.total_storage_bits,
            "lagrange_multiplier": self.lagrange_multiplier,
            "total_model_error": self.total_model_error,
            "n_matrices": len(self.assignments),
            "assignments": {
                name: {"rank": r, "bits": b}
                for name, (r, b) in self.assignments.items()
            },
        }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_matrices_from_csv(csv_path: str | Path) -> list[MatrixSpec]:
    """Load matrix specs from a CSV with columns: name, shape_m, shape_n,
    s1_eff, s2_eff.

    This is the expected output format of a whitening-aware entropy_lens
    run (see whiten.whitened_svdvals -> spectral.compute_s1/compute_s2),
    analogous to allocator.load_layers_from_csv but keyed on the
    data-metric statistics instead of raw S1.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    matrices = []
    for row in rows:
        matrices.append(
            MatrixSpec(
                name=row["name"],
                shape_m=int(row["shape_m"]),
                shape_n=int(row["shape_n"]),
                s1_eff=float(row["s1_eff"]),
                s2_eff=float(row["s2_eff"]),
            )
        )
    return matrices


# ---------------------------------------------------------------------------
# Allocation algorithm
# ---------------------------------------------------------------------------


def _clamp_rank(d: float, max_rank: int) -> int:
    return max(1, min(int(round(d)), max_rank))


def _clamp_bits(b: float) -> int:
    """Snap a continuous bit-width to the nearest allowed lattice codebook."""
    b = max(ALLOWED_BITS[0], min(b, ALLOWED_BITS[-1]))
    return min(ALLOWED_BITS, key=lambda a: abs(a - b))


# ---------------------------------------------------------------------------
# Error model (see module docstring + experiments/caldera_integration_plan.md,
# section "Joint optimization: Lagrangian / water-filling derivation", for
# the full derivation this code implements).
# ---------------------------------------------------------------------------
#
# Per matrix i, normalized residual (Frobenius) energy remaining after
# truncating to rank D is modeled as an exponential decay in D:
#
#     rho_i(D) = exp(-D / D_eff_i),      D_eff_i = exp(S1_eff_i)
#
# consistent with the entropy-compression law (rank needed for a fixed
# relative error scales like exp(S1_eff)): a matrix with a more spread-out
# whitened head (higher S1_eff) has a slower-decaying residual curve, so it
# needs more rank to reach the same energy fraction. rho_i(0) = 1 (no rank
# spent, no energy captured) and rho_i decreases monotonically to 0.
#
# The captured energy 1 - rho_i(D) is what ends up in the low-rank factors
# L, R and is quantized at bits_lr. High-rate scalar quantization theory
# gives a captured-energy distortion contribution proportional to
# gamma_i * 2^{-2*bits_lr}, where gamma_i is an excess-distortion
# coefficient. We tie gamma_i directly to S2_eff / D* = exp(S2_eff): a flat,
# quasi-isotropic tail (high S2_eff, large participation ratio D*) is
# already close to what a scalar/lattice quantizer is efficient at
# representing, so gamma_i is small (little extra distortion per bit spent);
# a peaked, structured tail (low S2_eff) is a worse match for a uniform
# quantizer, so gamma_i is large. Concretely:
#
#     gamma_i = exp(-S2_eff_i) = 1 / D*_i
#
# Combined per-matrix error model (drops the Q-backbone's own contribution,
# a constant at fixed q_bits that does not affect the (D, bits) argmin):
#
#     E_i(D, bits) = rho_i(D) + (1 - rho_i(D)) * gamma_i * 2^{-2*bits}
#
# This is the resolution to the scaffold's open gap: instead of a hard
# min(rank, D*) cap (which can make the budget unreachable), D* enters as
# the *quantization efficiency* of whatever rank is spent, so the rank/bits
# trade-off is always feasible and the "head stops being worth more rank"
# idea shows up as bits becoming the cheaper way to reduce error once the
# captured-energy penalty term dominates, rather than as an infeasible
# constraint.


def _decay_scale(m: MatrixSpec) -> float:
    """D_eff_i, the e-folding rank of the residual-energy curve rho_i(D)."""
    return float(np.exp(m.s1_eff))


def _quant_gamma(m: MatrixSpec) -> float:
    """gamma_i = exp(-S2_eff) = 1 / D*_i, the captured-energy distortion coefficient."""
    return float(np.exp(-m.s2_eff))


def _residual_energy(rank: float, d_eff: float) -> float:
    if d_eff <= 0:
        return 0.0 if rank > 0 else 1.0
    return float(np.exp(-rank / d_eff))


def _error(rank: float, bits: int, d_eff: float, gamma: float) -> float:
    rho = _residual_energy(rank, d_eff)
    return rho + (1.0 - rho) * gamma * (2.0 ** (-2 * bits))


def model_error(matrices: list[MatrixSpec], assignments: dict[str, tuple[int, int]]) -> float:
    """Sum of per-matrix model error E_i(rank_i, bits_i) at a given assignment.

    Public helper mirroring the internal objective the solver minimizes, so
    callers (tests, validation scripts) can compare error across different
    budgets/assignments without duplicating the model.
    """
    total = 0.0
    for m in matrices:
        rank, bits = assignments[m.name]
        total += _error(rank, bits, _decay_scale(m), _quant_gamma(m))
    return total


def _optimal_rank_for_bits(m: MatrixSpec, bits: int, lam: float, d_eff: float, gamma: float) -> float:
    """Closed-form D*_i(bits, lambda) minimizing E_i(D, bits) + lambda * cost_i(D, bits).

    Derived by setting d/dD [E_i(D, bits) + lambda * bits * (m+n) * D] = 0;
    see the module-level derivation comment above and the plan doc for the
    full algebra. Returns a continuous (unclamped-to-integer) rank in
    [0, max_rank]; the caller clamps and rounds.
    """
    price = bits * (m.shape_m + m.shape_n)
    denom = 1.0 - gamma * (2.0 ** (-2 * bits))
    if lam <= 0.0 or denom <= 1e-12 or d_eff <= 0.0:
        return float(m.max_rank)
    arg = lam * price * d_eff / denom
    if arg <= 0.0:
        return float(m.max_rank)
    if arg >= 1.0:
        # ln(arg) >= 0 => D* <= 0: renting any rank costs more (in the
        # Lagrangian) than the error it saves at this price.
        return 0.0
    rank = -d_eff * math.log(arg)
    return float(min(max(rank, 0.0), m.max_rank))


def _solve_layer(m: MatrixSpec, lam: float, bits_choices: tuple[int, ...]) -> tuple[float, int, float]:
    """For fixed lambda, jointly pick (rank, bits) minimizing this matrix's
    Lagrangian contribution E_i(D, b) + lambda * cost_i(D, b).

    Since bits_choices is small (default 3 values), this is an exact
    discrete search over bits with the closed-form optimal rank plugged in
    for each candidate -- i.e. an exact (not approximate) per-layer solve
    of the inner Lagrangian minimization.

    Returns (rank, bits, lagrangian_value).
    """
    d_eff = _decay_scale(m)
    gamma = _quant_gamma(m)
    best: tuple[float, int, float] | None = None
    for b in bits_choices:
        rank = _optimal_rank_for_bits(m, b, lam, d_eff, gamma)
        err = _error(rank, b, d_eff, gamma)
        cost = b * rank * (m.shape_m + m.shape_n)
        lagrangian = err + lam * cost
        if best is None or lagrangian < best[2]:
            best = (rank, b, lagrangian)
    assert best is not None
    return best


def _lr_cost_at_lambda(
    matrices: list[MatrixSpec], lam: float, bits_choices: tuple[int, ...]
) -> tuple[float, list[tuple[float, int]]]:
    total_cost = 0.0
    choices = []
    for m in matrices:
        rank, bits, _ = _solve_layer(m, lam, bits_choices)
        choices.append((rank, bits))
        total_cost += bits * rank * (m.shape_m + m.shape_n)
    return total_cost, choices


def allocate_joint_rank_bits(
    csv_path: str | Path | list[MatrixSpec],
    target_bits_per_param: float,
    q_bits: int = 2,
    bits_choices: tuple[int, ...] = ALLOWED_BITS,
) -> JointAllocation:
    """Jointly allocate rank and low-rank-factor bit-width across all
    matrices under a total storage budget, via Lagrangian relaxation /
    water-filling on a real (S1_eff, S2_eff)-driven error model.

    Problem (see experiments/caldera_integration_plan.md, section "Joint
    optimization" for the full derivation):

        minimize   sum_i E_i(rank_i, bits_i)
        subject to sum_i [q_bits * m_i * n_i + bits_i * rank_i * (m_i + n_i)]
                       <= target_bits_per_param * sum_i m_i * n_i
                   1 <= rank_i <= max_rank_i,  bits_i in bits_choices

    where E_i(rank, bits) = rho_i(rank) + (1 - rho_i(rank)) * gamma_i *
    2^{-2*bits} is a two-term error model: rho_i(rank) = exp(-rank/D_eff_i)
    (D_eff_i = exp(S1_eff_i)) is the residual energy the low-rank factors
    have *not yet* captured (entropy-compression law: e-folding rank scales
    with the whitened head's spread), and gamma_i = exp(-S2_eff_i) = 1/D*_i
    is the excess quantization-distortion coefficient for whatever energy
    *has* been captured (Renyi-2 entropy / participation ratio argument:
    flat, near-isotropic tails are cheap to quantize, peaked/structured
    ones are expensive). See `_decay_scale`/`_quant_gamma`/`_error` above.

    This is a genuinely *joint* optimization: the same rank increment buys
    a different marginal error reduction depending on the bit-width chosen
    for that matrix (and vice versa), so rank and bits cannot be decided
    independently -- unlike the earlier two-phase scaffold this replaces.

    Solver (Lagrangian relaxation / water-filling):

    1. Relax the bit-budget constraint with a multiplier lambda >= 0
       ("price per bit of LR storage"). For fixed lambda the problem is
       separable across matrices:

           min_{rank_i, bits_i} E_i(rank_i, bits_i)
               + lambda * bits_i * rank_i * (m_i + n_i)

    2. For each matrix and each candidate bits in bits_choices (a small,
       fixed set -- {2,3,4} by default), the optimal *continuous* rank has
       a closed form (stationarity of the Lagrangian in rank):

           rank*_i(bits, lambda) = -D_eff_i * ln(
               lambda * bits * (m_i+n_i) * D_eff_i / (1 - gamma_i * 2^{-2*bits})
           )

       clamped to [0, max_rank_i]. Evaluating the Lagrangian at each
       candidate bits (with its optimal rank plugged in) and taking the
       arg-min over bits_choices gives an *exact* per-layer solve of the
       coupled (rank, bits) inner problem -- see `_solve_layer`.
    3. lambda is found by bisection so that total LR storage
       sum_i bits_i * rank_i * (m_i + n_i) matches the LR budget (total
       budget minus the fixed backbone cost q_bits * m_i * n_i, reserved
       up front exactly as in the original scaffold). Cost is monotonically
       non-increasing in lambda (a higher price buys less rank at every
       layer), so bisection converges to the budget-matching lambda.
    4. Final integer ranks are obtained by rounding/clamping the converged
       continuous rank*_i(bits_i, lambda) -- this is the only place
       non-exactness enters versus the continuous relaxation, and its
       effect on the realized budget is small for the (m+n) scales here
       (single-digit ranks' worth of rounding out of hundreds/thousands).

    At the optimum (ignoring integer rounding and matrices pinned to a
    rank boundary), every matrix's marginal error reduction per marginal
    bit of storage spent on rank equals lambda -- the classical
    water-filling optimality condition, generalized to a rank+bits joint
    currency. See JointAllocation.lagrange_multiplier and the "optimality"
    property tests in tests/test_joint_alloc.py.

    Args:
        csv_path: path to a CSV with name/shape_m/shape_n/s1_eff/s2_eff
            columns, or an already-loaded list of MatrixSpec.
        target_bits_per_param: total storage budget, in bits per original
            parameter, averaged over all matrices (this is CALDERA's
            native "bits per parameter" budget knob, e.g. 2.5).
        q_bits: fixed bit-width for the quantized backbone Q. Set to 0 to
            model pure low-rank (no residual) matrices.
        bits_choices: allowed lattice bit-widths for L/R, ascending.

    Returns:
        JointAllocation with a (rank, bits) pair per matrix name, the
        converged Lagrange multiplier, the total model error, and summary
        storage statistics.

    Raises:
        ValueError: if target_bits_per_param is not positive, or no
            matrices are provided.
    """
    if target_bits_per_param <= 0:
        raise ValueError(
            f"target_bits_per_param must be positive, got {target_bits_per_param}"
        )

    if isinstance(csv_path, list):
        matrices = csv_path
    else:
        matrices = load_matrices_from_csv(csv_path)

    if not matrices:
        raise ValueError(f"No matrices loaded from {csv_path!r}")

    total_original_params = sum(m.shape_m * m.shape_n for m in matrices)
    total_bit_budget = target_bits_per_param * total_original_params

    # Reserve the backbone Q's storage first; whatever remains is spent on
    # the low-rank factors L, R (the joint rank+bits solve below).
    q_total_bits = sum(q_bits * m.shape_m * m.shape_n for m in matrices)
    lr_budget = total_bit_budget - q_total_bits
    if lr_budget <= 0:
        # Degenerate case: the backbone alone exceeds the budget. Fall back
        # to rank=1 everywhere and the lowest allowed bit-width; the caller
        # should lower q_bits or raise target_bits_per_param.
        assignments = {m.name: (1, bits_choices[0]) for m in matrices}
        total_storage = sum(
            m.storage_bits(1, bits_choices[0], q_bits) for m in matrices
        )
        return JointAllocation(
            assignments=assignments,
            q_bits=q_bits,
            target_bits_per_param=target_bits_per_param,
            total_original_params=total_original_params,
            total_storage_bits=total_storage,
            matrices=matrices,
            lagrange_multiplier=0.0,
            total_model_error=model_error(matrices, assignments),
        )

    # --- Lagrangian bisection on lambda, the price per bit of LR storage --
    # Cost is monotonically non-increasing in lambda: lambda=0 spends the
    # maximum (every matrix saturates rank at max_rank for whichever bits
    # minimizes pure error); as lambda grows, rank -- and eventually the
    # chosen bits -- shrink toward the rank=0/bits=lowest floor.
    lo, hi = 0.0, 1e-12
    cost_at_zero, _ = _lr_cost_at_lambda(matrices, 0.0, bits_choices)
    if cost_at_zero <= lr_budget:
        # The budget is generous enough that the unconstrained (lambda=0)
        # optimum already fits; no scarcity, skip the search.
        lam = 0.0
        _, choices = cost_at_zero, _lr_cost_at_lambda(matrices, 0.0, bits_choices)[1]
    else:
        while True:
            cost_hi, _ = _lr_cost_at_lambda(matrices, hi, bits_choices)
            if cost_hi <= lr_budget or hi > 1e6:
                break
            hi *= 2.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            cost_mid, _ = _lr_cost_at_lambda(matrices, mid, bits_choices)
            if cost_mid > lr_budget:
                lo = mid
            else:
                hi = mid
        lam = hi
        _, choices = _lr_cost_at_lambda(matrices, lam, bits_choices)

    final_ranks = [_clamp_rank(rank, m.max_rank) for (rank, _bits), m in zip(choices, matrices)]
    bits_lr = [bits for _rank, bits in choices]

    assignments = {
        m.name: (r, b) for m, r, b in zip(matrices, final_ranks, bits_lr)
    }
    total_storage = sum(
        m.storage_bits(r, b, q_bits)
        for m, r, b in zip(matrices, final_ranks, bits_lr)
    )

    return JointAllocation(
        assignments=assignments,
        q_bits=q_bits,
        target_bits_per_param=target_bits_per_param,
        total_original_params=total_original_params,
        total_storage_bits=total_storage,
        matrices=matrices,
        lagrange_multiplier=lam,
        total_model_error=model_error(matrices, assignments),
    )
