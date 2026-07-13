"""Clean-room LPLR (Low-Precision Low-Rank) alternating-minimization solver.

Rationale / provenance (read `experiments/caldera_integration_plan.md` section
2, "Licensing", before touching this file): CALDERA (Saha, Sagan, Srivastava,
Goldsmith, Pilanci; NeurIPS 2024; arXiv:2405.18886) decomposes a weight matrix

    W ~= Q + L @ R

where Q is a full-size low-precision quantized backbone and L (m x rank),
R (rank x n) are low-rank factors, fit by alternating minimization of

    || (W - Q - L @ R) @ Lw ||_F^2

with `Lw` a whitening factor (`Lw @ Lw.T = C`, C the activation covariance --
exactly `entropy_lens.whiten.cholesky_factor`'s output). `pilancilab/caldera`
has no LICENSE file (all-rights-reserved by default), so no code from that
repo is copied, vendored, or paraphrased here. This module is written from
the published paper's description of the algorithm only. The actual
CUDA/E8P-lattice quantizer this project intends to pair with the solver is
QuIP# (`Cornell-RelaxML/quip-sharp`, GPL-3.0 -- the license has been
knowingly accepted for this project, see the plan doc's NOTICE) -- but that
quantizer is NOT reimplemented here either. Instead, `lplr_decompose*` takes
a `quantize_fn: Callable[[Tensor, int], Tensor]` as a dependency: the
alternating-minimization *logic* (this file) is decoupled from *how* a
tensor gets quantized to `bits` precision. On a Mac with no CUDA, QuIP#'s
kernels cannot be built or run, so this module ships a simple fake/round-trip
quantizer (`round_quantize`) for testing the alternation logic in isolation,
clearly documented as NOT representative of E8P lattice quantization
accuracy. On the RunPod GPU box, wire a real QuIP#-backed `quantize_fn` (see
`QUIP_SHARP_WIRING_NOTES` below) before trusting any PPL numbers that come
out of `experiments/runpod_caldera.py`.

Algorithm (`lplr_decompose_raw`, operating directly on whatever matrix `M` is
passed in -- Frobenius metric on `M`):

    Q_0 = 0
    for t in 1..max_iters:
        R_t = M - Q_{t-1}                      # residual after backbone
        L_t, R_mat_t = truncated_svd(R_t, rank) # low-rank fit of residual (unquantized)
        L_t = quantize_fn(L_t, lr_bits)
        R_mat_t = quantize_fn(R_mat_t, lr_bits)
        Q_t = quantize_fn(M - L_t @ R_mat_t, q_bits)
        err_t = || M - Q_t - L_t @ R_mat_t ||_F / || M ||_F
        if |err_{t-1} - err_t| < tol: stop (converged)

This is a direct, from-scratch transcription of the paper's alternating
structure (fix Q, solve L/R by an SVD-style low-rank step; fix L/R, quantize
the residual into Q; repeat), with no CALDERA source consulted.

`lplr_decompose_whitened` is a thin wrapper matching `whiten.whiten_truncate`'s
calling convention: it whitens `W` into `M = W @ L_whiten`, runs the raw
solver on `M`, and maps the result back to the original (out, in) weight
space via the same triangular solve `whiten_truncate` uses. `whiten.py` is
only imported/consumed here, never modified (per the 2026-07-12 mesa
division of labor: another agent is improving `joint_alloc.py`'s internals
in parallel, and `whiten.py` is validated/frozen for this work).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

QuantizeFn = Callable[[torch.Tensor, int], torch.Tensor]


# ---------------------------------------------------------------------------
# Fake / round-trip quantizer -- FOR TESTING THE ALTERNATION LOGIC ONLY.
# ---------------------------------------------------------------------------


def round_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Simple per-tensor uniform affine round-trip "quantizer".

    NOT a lattice quantizer, NOT QuIP#, NOT representative of E8P codebook
    accuracy. This exists solely so `lplr_decompose*`'s alternating-
    minimization logic can be unit-tested on a CPU-only machine (this repo's
    dev environment is a Mac with no CUDA, so QuIP#'s kernels cannot be
    built/run here -- see `experiments/caldera_integration_plan.md`). Any
    reconstruction-error number produced using this function is a sanity
    check on the solver's control flow, not a compression-quality estimate.

    Maps x into `2**bits` uniform levels spanning [min(x), max(x)], per
    tensor (no groupwise/channelwise scaling, unlike real quantizers).

    Args:
        x: tensor to fake-quantize.
        bits: bit-width, e.g. 2, 3, 4. Must be >= 1.

    Returns:
        Dequantized tensor, same shape/dtype as `x`.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    x = x.detach()
    lo = x.min()
    hi = x.max()
    if torch.isclose(hi, lo):
        return x.clone()
    levels = 2**bits - 1
    scale = (hi - lo) / levels
    q = torch.round((x - lo) / scale)
    q = q.clamp(0, levels)
    return (q * scale + lo).to(x.dtype)


# ---------------------------------------------------------------------------
# Notes for wiring a real QuIP#-backed quantize_fn on the RunPod GPU box.
# ---------------------------------------------------------------------------

QUIP_SHARP_WIRING_NOTES = """
This module does NOT include a QuIP#-backed quantize_fn because (a) QuIP#'s
lattice codebooks require CUDA kernels this dev machine (Mac, no CUDA)
cannot build or run, and (b) writing one without ever running the real
package risks silently misrepresenting quip-sharp's GPL-3.0 code (better to
leave an honest gap than a guessed-at reimplementation). On the pod, after
installing quip-sharp (see caldera_integration_plan.md section 6 for the
verified install steps), the real codebooks live at:

  quip-sharp/lib/codebook/latticee8_padded12.py         (E8P12, 2-bit)
  quip-sharp/lib/codebook/latticee8_padded12_rvq3bit.py (E8P12RVQ3B, 3-bit)
  quip-sharp/lib/codebook/latticee8_padded12_rvq4bit.py (E8P12RVQ4B, 4-bit)

Each codebook class exposes a `quantize(...)` and the surrounding
`lib/linear/quantized_linear.py` shows how a matrix gets round-tripped through
one for inference. Adapt (do not copy) that call pattern into a
`quantize_fn(tensor, bits) -> tensor` closure that: (1) picks the codebook
matching `bits` in {2,3,4}, (2) runs the codebook's own incoherence
processing (Hadamard transform via fast-hadamard-transform) if required by
its API, (3) quantizes and immediately dequantizes, returning a dense tensor
of the same shape/dtype as the input so it drops into `lplr_decompose*`
unchanged. This adapter is left as pod-side follow-up work, tracked in
experiments/caldera_integration_plan.md.
"""


@dataclass
class LPLRResult:
    """Output of one `lplr_decompose_raw` / `lplr_decompose_whitened` run."""

    Q: torch.Tensor
    L: torch.Tensor
    R: torch.Tensor
    reconstruction: torch.Tensor
    errors: list = field(default_factory=list)
    converged: bool = False
    iters_run: int = 0

    @property
    def final_error(self) -> float:
        return self.errors[-1] if self.errors else float("nan")


def _truncated_svd_lowrank(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-`rank` factorization matrix ~= L @ R via SVD (Eckart-Young optimal).

    Returns L (m, rank), R (rank, n) such that L @ R is the best rank-`rank`
    approximation of `matrix` in Frobenius norm. Singular values are folded
    into L (L = U_r * S_r) so R is left/right-orthonormal-ish (Vh_r), matching
    the convention used elsewhere in this repo (whiten.whiten_truncate).
    """
    m, n = matrix.shape
    rank = max(1, min(rank, min(m, n)))
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    L = u[:, :rank] * s[:rank]
    R = vh[:rank]
    return L, R


def lplr_decompose_raw(
    matrix: torch.Tensor,
    rank: int,
    q_bits: int,
    lr_bits: int,
    quantize_fn: QuantizeFn = round_quantize,
    max_iters: int = 20,
    tol: float = 1e-6,
) -> LPLRResult:
    """Alternating-minimization LPLR fit of `matrix ~= Q + L @ R`.

    Operates directly on whatever 2D tensor is passed in (Frobenius metric
    on that tensor). Callers that want the data-metric objective from the
    CALDERA paper (`|| (W - Q - LR) @ Lw ||_F`) should whiten first -- see
    `lplr_decompose_whitened` below, which whitens via `entropy_lens.whiten`
    and maps the result back to the original weight space.

    Args:
        matrix: (m, n) tensor to decompose.
        rank: rank of the L, R factors. Clamped to [1, min(m, n)].
        q_bits: bit-width for the quantized backbone Q. Use 0 to disable Q
            entirely (pure low-rank fit, Q stays all-zero) -- useful for
            isolating the low-rank step from the quantization step in tests.
        lr_bits: bit-width for quantizing L and R. Use 0 to skip quantizing
            L/R (keep them full precision) -- isolates the Q-quantization
            step from the LR-quantization step in tests.
        quantize_fn: `(tensor, bits) -> dequantized tensor`. Defaults to the
            fake `round_quantize` (see its docstring: NOT representative of
            real lattice-quantizer accuracy). Pass a QuIP#-backed closure on
            the RunPod box for real numbers.
        max_iters: maximum alternating-minimization rounds.
        tol: stop early once |error_t - error_{t-1}| < tol.

    Returns:
        LPLRResult with Q, L, R, the combined reconstruction, the per-iter
        relative-Frobenius-error trace, and whether it converged before
        max_iters.
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    m, n = matrix.shape
    rank = max(1, min(rank, min(m, n)))
    mat_norm = torch.linalg.norm(matrix)
    if mat_norm == 0:
        zero_lr_m = torch.zeros(m, rank, dtype=matrix.dtype)
        zero_lr_n = torch.zeros(rank, n, dtype=matrix.dtype)
        zero = torch.zeros_like(matrix)
        return LPLRResult(Q=zero, L=zero_lr_m, R=zero_lr_n, reconstruction=zero,
                           errors=[0.0], converged=True, iters_run=0)

    Q = torch.zeros_like(matrix)
    L = torch.zeros(m, rank, dtype=matrix.dtype)
    R = torch.zeros(rank, n, dtype=matrix.dtype)
    errors: list = []
    converged = False
    iters_run = 0

    for t in range(max_iters):
        iters_run = t + 1
        # (a) fix Q, refit L/R on the residual via truncated SVD, then quantize
        residual = matrix - Q
        L, R = _truncated_svd_lowrank(residual, rank)
        if lr_bits > 0:
            L = quantize_fn(L, lr_bits)
            R = quantize_fn(R, lr_bits)

        # (b) fix L/R, quantize the new residual into Q
        lr_term = L @ R
        if q_bits > 0:
            Q = quantize_fn(matrix - lr_term, q_bits)
        else:
            Q = torch.zeros_like(matrix)

        reconstruction = Q + lr_term
        err = float(torch.linalg.norm(matrix - reconstruction) / mat_norm)
        errors.append(err)

        if t > 0 and abs(errors[-2] - errors[-1]) < tol:
            converged = True
            break

    reconstruction = Q + L @ R
    return LPLRResult(Q=Q, L=L, R=R, reconstruction=reconstruction,
                       errors=errors, converged=converged, iters_run=iters_run)


def lplr_decompose_whitened(
    weight: torch.Tensor,
    whiten_L: torch.Tensor,
    rank: int,
    q_bits: int,
    lr_bits: int,
    quantize_fn: QuantizeFn = round_quantize,
    max_iters: int = 20,
    tol: float = 1e-6,
    triangular: bool = True,
) -> LPLRResult:
    """LPLR fit in the data metric, mapped back to original weight space.

    Mirrors `whiten.whiten_truncate`'s calling convention: whitens
    `M = W @ whiten_L`, runs `lplr_decompose_raw` on `M` (so the Frobenius
    objective on `M` equals the data-metric objective on `W`, per
    `whiten.py`'s module docstring), then maps `Q`, `L @ R` back through the
    (triangular) solve against `whiten_L`.

    Args:
        weight: (out, in) original weight matrix.
        whiten_L: (in, in) whitening factor from `whiten.cholesky_factor`.
        rank, q_bits, lr_bits, quantize_fn, max_iters, tol: see
            `lplr_decompose_raw`.
        triangular: whether `whiten_L` is lower-triangular (fast solve) --
            see `whiten.cholesky_factor`'s return value.

    Returns:
        LPLRResult with all tensors mapped back to the original (out, in)
        weight space (Q and the combined `reconstruction` are in W-space;
        `L`/`R` themselves are returned in the *whitened* M-space, since that
        is where the rank/bit budget accounting in `joint_alloc.py` applies).
    """
    dtype = weight.dtype
    w = weight.to(torch.float32)
    lw = whiten_L.to(torch.float32).to(w.device)

    m = w @ lw
    result = lplr_decompose_raw(m, rank, q_bits, lr_bits, quantize_fn, max_iters, tol)

    def _map_back(x: torch.Tensor) -> torch.Tensor:
        if triangular:
            return torch.linalg.solve_triangular(lw, x, upper=False, left=False)
        return x @ torch.linalg.inv(lw)

    reconstruction_w = _map_back(result.reconstruction).to(dtype)
    q_w = _map_back(result.Q).to(dtype)

    return LPLRResult(
        Q=q_w,
        L=result.L,
        R=result.R,
        reconstruction=reconstruction_w,
        errors=result.errors,
        converged=result.converged,
        iters_run=result.iters_run,
    )
