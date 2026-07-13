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


def allocate_joint_rank_bits(
    csv_path: str | Path | list[MatrixSpec],
    target_bits_per_param: float,
    q_bits: int = 2,
    bits_choices: tuple[int, ...] = ALLOWED_BITS,
) -> JointAllocation:
    """Jointly allocate rank (per S1_eff) and low-rank-factor bit-width
    (per S2_eff / D*) across all matrices under a total storage budget.

    Design (see experiments/caldera_integration_plan.md, section "Algorithm"):

    1. Rank allocation follows the same exponential-weighting idea as
       allocator.allocate_entropy, but on S1_eff instead of raw S1:
       matrices whose whitened head is more spread (higher S1_eff) get
       exponentially more rank, i.e. raw_rank_weight_i = exp(S1_eff_i),
       scaled by bisection so the *rank* component of storage matches
       its share of the budget.
    2. d_star = exp(S2_eff) (the head/tail crossover) is exposed on every
       MatrixSpec and is the intended long-run ceiling on how much rank a
       matrix should absorb before bits become the more efficient
       currency. THIS SCAFFOLD DOES NOT YET ENFORCE d_star AS A HARD CAP
       during the budget bisection: a hard min(rank, d_star) can make the
       requested bits_per_param budget mathematically unreachable
       whenever d_star is small relative to max_rank (all rank weight
       saturates at the cap and the leftover budget has nowhere to go,
       since bit-width is fixed by S2 in step 3, not re-opened to spend
       the surplus). Enforcing the cap correctly requires redistributing
       any budget a capped matrix cannot absorb back into the bisection
       for the uncapped matrices -- left as follow-up work, tracked in
       experiments/caldera_integration_plan.md ("Known gaps in the
       scaffold"). For now rank is bounded only by [1, max_rank].
    3. Bit-width for the L/R factors is allocated from S2_eff directly:
       matrices with high S2_eff (flatter, more isotropic tail -- close
       to what remains after the rank cap has already removed the head)
       need less precision per residual coefficient to hit a target
       reconstruction error, so they get *fewer* bits; matrices with low
       S2_eff (peaked, still-structured tail) get more bits. This is
       implemented as an inverse-monotonic map onto bits_choices via
       rank ordering (ties broken toward more bits), then snapped to the
       nearest allowed lattice codebook.
    4. The backbone Q is left at a fixed q_bits (CALDERA already treats Q
       uniformly via QuIP#'s E8P lattice codebooks; only per-layer
       skipping of Q via compute_quantized_component=False is exposed
       upstream). Reallocating q_bits per layer is future work, noted in
       the integration plan as a second-order refinement once the
       rank/bits split is validated.
    5. Total budget is enforced by bisecting a single global scale factor
       on the rank weights (as in allocator.allocate_entropy), then
       re-deriving bits from the resulting per-matrix ranks -- so the
       *rank* allocation is the primary budget lever and bits reallocate
       storage within whatever the rank pass leaves for the LR factors.

    Args:
        csv_path: path to a CSV with name/shape_m/shape_n/s1_eff/s2_eff
            columns, or an already-loaded list of MatrixSpec.
        target_bits_per_param: total storage budget, in bits per original
            parameter, averaged over all matrices (this is CALDERA's
            native "bits per parameter" budget knob, e.g. 2.5).
        q_bits: fixed bit-width for the quantized backbone Q (see point 4
            above). Set to 0 to model pure low-rank (no residual) matrices.
        bits_choices: allowed lattice bit-widths for L/R, ascending.

    Returns:
        JointAllocation with a (rank, bits) pair per matrix name and
        summary statistics.

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
    # the low-rank factors L, R (rank allocation + bit allocation below).
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
        )

    # --- Step 1+2: S1-driven rank allocation, capped by S2-driven D* ------
    s1_arr = np.array([m.s1_eff for m in matrices])
    raw_weights = np.exp(s1_arr)

    # A representative bit-width used only to convert the LR-budget (bits)
    # into a rank-scale bisection; final per-matrix bits are re-derived in
    # step 3, so this choice does not bias the final storage accounting.
    mid_bits = bits_choices[len(bits_choices) // 2]

    def _cost_at_scale(scale: float) -> float:
        total = 0.0
        for w, m in zip(raw_weights, matrices):
            rank = _clamp_rank(scale * w, m.max_rank)
            total += mid_bits * rank * (m.shape_m + m.shape_n)
        return total

    lo, hi = 0.0, max(m.max_rank / max(w, 1e-30) for w, m in zip(raw_weights, matrices))
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _cost_at_scale(mid) < lr_budget:
            lo = mid
        else:
            hi = mid

    ranks = [
        _clamp_rank(hi * w, m.max_rank)
        for w, m in zip(raw_weights, matrices)
    ]

    # --- Step 3: S2-driven bit-width for L/R -------------------------------
    # Higher S2_eff (flatter/more isotropic residual) -> fewer bits needed.
    # Map S2_eff to bits_choices via rank ordering (robust to scale/units):
    # the matrix with the highest S2_eff gets the lowest bit-width and vice
    # versa, evenly split across the ordered matrices.
    s2_arr = np.array([m.s2_eff for m in matrices])
    order = np.argsort(s2_arr)  # ascending S2_eff -> descending bit need
    n = len(matrices)
    bits_for_rank_pos = np.array_split(np.arange(n), len(bits_choices))
    bits_of_index = np.empty(n, dtype=int)
    for level, idx_group in enumerate(bits_for_rank_pos):
        # level 0 = lowest S2_eff group -> highest bits (last in bits_choices)
        chosen_bits = bits_choices[len(bits_choices) - 1 - level]
        for pos in idx_group:
            bits_of_index[order[pos]] = chosen_bits

    bits_lr = [int(bits_of_index[i]) for i in range(n)]

    # --- Rescale ranks so actual LR storage matches lr_budget exactly -----
    # The bit re-derivation in step 3 changes the per-matrix bit cost versus
    # the mid_bits placeholder used during the step-1/2 bisection, so do one
    # more bisection pass over a multiplicative rank correction to land on
    # budget, respecting the same D*/max_rank caps.
    def _cost_with_ranks(rank_scale: float) -> float:
        total = 0.0
        for m, r, b in zip(matrices, ranks, bits_lr):
            rr = _clamp_rank(rank_scale * r, m.max_rank)
            total += b * rr * (m.shape_m + m.shape_n)
        return total

    lo, hi = 0.0, 2.0
    # Expand hi until it overshoots the budget or hits a sane ceiling.
    while _cost_with_ranks(hi) < lr_budget and hi < 1e6:
        hi *= 2.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _cost_with_ranks(mid) < lr_budget:
            lo = mid
        else:
            hi = mid

    final_ranks = [
        _clamp_rank(lo * r, m.max_rank)
        for r, m in zip(ranks, matrices)
    ]

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
    )
