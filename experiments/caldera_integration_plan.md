# CALDERA integration plan: entropy-guided joint rank + bits allocation

Status: scaffolding only. No CALDERA code has been copied into this repo (see
"Licensing" below for why). This document is the design + evaluation plan for
the north star decided at the mesa conjunta of 2026-07-12 (see
`/Users/nicolascalderon/Documents/dev/Qtech Idea/mesa-conjunta-2026-07-12.md`):
**entropy-guided JOINT allocation of rank and bit-width per layer, on a
low-rank + quantization base, using S1 to drive rank and S2 to drive bits.**

## 1. What CALDERA is

Paper: "Compressing Large Language Models using Low Rank and Low Precision
Decomposition" (Saha, Sagan, Srivastava, Goldsmith, Pilanci; NeurIPS 2024;
arXiv:2405.18886; OpenReview `lkx3OpcqSZ`).

Decomposition: for each weight matrix W,

    W ~= Q + L R

- **Q**: full-size, low-precision component (2-4 bit lattice quantization,
  via QuIP#'s E8P codebooks), capturing everything the low-rank part misses.
- **L, R**: low-rank factors (`L` is `m x rank`, `R` is `rank x n`), also
  quantized to low precision, capturing the top singular directions.

The paper's own framing (consistent with our Feynman/Karpathy/Carmack mesa,
independently arrived at): Q handles the "trailing singular values" (the
flat tail) at `B_Q` bits, and `L`,`R` handle the "top singular values" (the
head) at `B_L`, `B_R` bits — this is exactly the head/tail split our
Entropy-Compression Law reasons about, just without an entropy-driven
per-layer allocation rule on top of it.

**Algorithm (LPLR / alternating minimization).** CALDERA solves

    min_{Q,L,R} || (W - Q - LR) H^{1/2} ||_F^2

where `H = E[xx^T]` is the (Hessian-like) activation covariance — i.e. the
same whitening idea as our `whiten.py`, but CALDERA calls it "activation-aware"
rather than data-metric whitening. It alternates: fix `Q`, solve for `L,R` via
a quantized/truncated SVD-style update (LPLR step); fix `L,R`, quantize the
residual into `Q` (reusing QuIP#'s Hessian-aware lattice quantizer); repeat for
`iters` rounds (default 20 in the repo, with 5 inner `lplr_iters`). Optional:
LoRA fine-tuning of a slice of `L,R` post-hoc for task recovery.

**Confirmed from the repo (`src/caldera/decomposition/dataclasses.py`,
`CalderaParams`): rank and bit-widths are single scalars for the whole run,
not per layer.** Concretely:

```python
Q_bits: int = 2      # 2, 3, or 4
L_bits: int = 2
R_bits: int = 2
rank: int = 64
```

`ActivationAwareWeightCompressor` / `ActivationAwareLayerQuant`
(`weight_compression.py`, `layer_quantization.py`) apply these same scalars to
every transformer sublayer it touches (q/k/v/o/gate/up/down). There is no
S1/S2-driven (or any spectral-entropy-driven) per-layer schedule anywhere in
the codebase — this directly confirms the mesa's claim ("CALDERA composes
Q+LR but with uniform structure") and is exactly the gap our allocator is
meant to fill.

## 2. Licensing — CRITICAL, read before writing any code that touches CALDERA

**`github.com/pilancilab/caldera` has no LICENSE file.** Verified via
`gh repo view pilancilab/caldera --json licenseInfo` (`licenseInfo: null`) and
`gh api repos/pilancilab/caldera/license` (404 Not Found), and by listing the
repo root — there is no `LICENSE`, `LICENSE.md`, `COPYING`, or equivalent
anywhere in the tree. **No license means default copyright: all rights
reserved.** GitHub's own ToS is explicit that the absence of a license file
means visitors may view and fork the repo, but have **no legal permission to
copy, modify, or redistribute** the code, even in an academic/research
context, without asking the authors directly.

Additionally, CALDERA's quantizer backbone is the `quip-sharp` submodule
(`Cornell-RelaxML/quip-sharp`), which **is** licensed, under **GPL-3.0**. That
is copyleft: any code that statically/dynamically links against it and is
distributed must also be GPL-3.0-compatible. `entropy-lens` is MIT-licensed
(`pyproject.toml`); pulling in GPL-3.0 code would force a relicense of
anything that depends on it, which is a separate decision from the CALDERA
licensing gap and should not be made implicitly.

**Recommendation (what this scaffold does, and what to do next):**

1. **Do not `git submodule` or vendor `pilancilab/caldera` source into this
   repo.** Do not copy functions, docstrings, or file structure from it
   verbatim. This scaffold does not.
2. **Do a clean-room reimplementation** of the `W ~= Q + LR` idea, informed
   only by the published paper (arXiv:2405.18886 / OpenReview `lkx3OpcqSZ`)
   and by architecture facts obtainable from public API surfaces (parameter
   names, shapes) rather than copied implementation code. This is the
   standard path when an academic repo's algorithm is public (via the paper)
   but its code is unlicensed.
3. **Before any real fork/PR/derivative work**, email the CALDERA authors
   (Rajarshi Saha / Mert Pilanci, Stanford) asking them to add a license (MIT
   or Apache-2.0 would be ideal and match our stack) or grant explicit
   permission. This is a 5-minute ask and de-risks the whole north star; flag
   to the human to send.
4. **Quantizer choice is a separate, human-decidable question**: either (a)
   accept GPL-3.0 scope for a quantization-only submodule if the paper /
   release is kept GPL-compatible or quantization is siloed into a separable
   process, or (b) implement a simpler in-house scalar/NF4-style quantizer
   (as `whiten.py`'s comments already gesture toward with `bitsandbytes`-style
   language) and forgo QuIP#'s E8 lattice codebooks, at some cost to
   compression ratio (lattice quantizers systematically beat scalar
   quantizers at the same bit-width, ~0.1-0.3 bits/param equivalent per public
   QuIP#/AQLM benchmarks — not independently reverified here).

This scaffold takes path (2): `joint_alloc.py` is written from the paper's
public description plus first-principles reasoning, not from CALDERA's
source. It also does not implement the LPLR alternating-minimization solver
itself or a lattice quantizer — those remain open engineering work, discussed
in "Feasibility" below.

## 3. Integration points

| CALDERA concept | Where it lives today | Our replacement |
|---|---|---|
| Activation covariance `H = E[xx^T]` for the data-metric objective | `quip-sharp/quantize_llama/hessian_offline_llama.py`, wired through `AccumulatorArgs`/`ActivationAwareWeightCompressor` | Already have this: `whiten.collect_covariances` + `whiten.cholesky_factor`, including the channel-prescale fix for the "massive activations" ill-conditioning found empirically on Mistral 7B (2026-07-12). This is a **direct match**, arguably more robust (documented eigh fallback + escalating ridge) than what's visible in CALDERA's Hessian code. |
| Per-run scalar `rank`, `Q_bits`, `L_bits`, `R_bits` (`CalderaParams`) | `src/caldera/decomposition/dataclasses.py` | **This is the piece we replace.** `joint_alloc.allocate_joint_rank_bits` produces a `{name: (rank, bits)}` map from per-matrix S1_eff/S2_eff instead of one global scalar pair. |
| Where rank/bits get applied per sublayer | `ActivationAwareLayerQuant.compress_sublayer` (`layer_quantization.py`), called once per layer with the *same* `CalderaParams` object | Would need a thin adapter that looks up `(rank, bits)` per matrix name from a `JointAllocation` and constructs a per-call `CalderaParams`-equivalent (or, in the clean-room build, our own per-matrix decomposition config) before compressing that specific sublayer. Not yet built (see "Known gaps"). |
| Low-rank truncation given a rank | `caldera()` alternating step (`alg.py`, `maybe_update_LR`) | We already have `whiten.whiten_truncate` for the pure-low-rank half (`M = W @ L`, SVD-truncate, map back). The alternating min against a *quantized* residual (not just truncation) is new work; `whiten_truncate` alone does not model the `Q` interaction term. |
| Bit-width per component | `Q_bits`/`L_bits`/`R_bits`, uniform | `joint_alloc.allocate_joint_rank_bits`'s `bits_lr` output, one value per matrix from `bits_choices` (mirrors CALDERA's `{2,3,4}`-bit lattice codebook constraint, `ALLOWED_BITS` in `joint_alloc.py`). `q_bits` for the backbone `Q` is still a single global scalar in this scaffold (see gaps). |

### What's reused vs. new

**Reused from `allocator.py` (the design pattern, generalized):**
- The `LayerSpec` -> `Allocation` dataclass shape becomes `MatrixSpec` ->
  `JointAllocation`, same idea (per-matrix stats in, per-matrix decision +
  summary stats out).
- The bisection-over-a-scale-factor technique from `allocate_entropy`
  (`raw_weights = exp(S1)`, bisect a global scale to hit a total-parameter
  budget) is reused directly for the rank half of the joint allocation, just
  swapping "parameter budget" for "bit budget" and `S1` for `S1_eff`.
- CSV loading conventions (`load_layers_from_csv` -> `load_matrices_from_csv`)
  keep the same column-based contract so a single whitening-aware pipeline run
  can emit one CSV consumed by either allocator.

**New, in `joint_alloc.py`:**
- `MatrixSpec.d_star` (`exp(S2_eff)`), the head/tail crossover concept from
  Feynman's mesa argument. Exposed as a property today; not yet enforced as a
  hard constraint (see gaps below).
- The S2-driven bit-width assignment (`bits_lr`): matrices with a flatter,
  more isotropic residual (`high S2_eff`) get *fewer* bits per LR coefficient;
  peaked/structured residuals get more. This has no analog in `allocator.py`
  since that module only ever decided rank.
- Joint budget accounting in bits-per-parameter (CALDERA's native currency)
  rather than parameter-count (entropy-lens's `allocator.py` currency).

## 4. Algorithm design: `allocate_joint_rank_bits`

Implemented in `src/entropy_lens/joint_alloc.py`. Signature:

```python
def allocate_joint_rank_bits(
    csv_path: str | Path | list[MatrixSpec],
    target_bits_per_param: float,
    q_bits: int = 2,
    bits_choices: tuple[int, ...] = (2, 3, 4),
) -> JointAllocation
```

- **Input CSV** (or pre-loaded `list[MatrixSpec]`): `name, shape_m, shape_n,
  s1_eff, s2_eff` — i.e. per-matrix stats computed **in the data metric**
  (whitened spectrum via `whiten.whitened_svdvals` -> `spectral.compute_s1`
  / `compute_s2`), not the raw Frobenius spectrum. This is a hard requirement:
  the ablation study (`results/mistralai_Mistral-7B-v0.3/ablation_analysis.md`)
  found raw-metric S1 has ~zero rank correlation with actual compression
  damage (Spearman +0.008), while the data-metric gap correlates at
  Spearman +0.438 (p=0.012). Feeding raw S1/S2 into this allocator would
  reproduce that failure.
- **Output**: `{matrix_name: (rank, bits)}` plus summary stats
  (`actual_bits_per_param`, `total_storage_bits`, etc.), mirroring
  `Allocation` in `allocator.py`.

**Reparto algorithm** (see full docstring in `joint_alloc.py`):

1. Reserve storage for the quantized backbone `Q` at a fixed `q_bits` (global
   scalar, matching CALDERA's current uniform treatment — see gaps). What's
   left (`lr_budget`) is spent on `L`,`R`.
2. **Rank ~ S1_eff**: `raw_weight_i = exp(S1_eff_i)`, same functional form as
   `allocator.allocate_entropy`'s `exp(S1)` weighting (the mesa's physical
   picture: higher entropy in the whitened head means the energy is smeared
   over more directions, so more rank is needed to capture the same
   fraction of it). Bisect a single global scale factor so total LR storage
   (using a placeholder mid-range bit-width) matches `lr_budget`.
3. **Bits ~ S2_eff**: sort matrices by `S2_eff` and split into
   `len(bits_choices)` groups; the group with the *lowest* `S2_eff` (most
   peaked, least isotropic residual — needs the most precision to represent
   faithfully) gets the *highest* bit-width, and vice versa. This is the
   concrete mechanism for "S2 decides bits."
4. **Re-derive final ranks**: because step 3's real per-matrix bit-widths
   differ from step 2's placeholder, do a second bisection over a
   multiplicative rank correction so the *actual* combined (rank, bits)
   storage matches `lr_budget` as closely as integer rounding allows.
5. `MatrixSpec.d_star = exp(S2_eff)` (the crossover rank per the Feynman
   argument) is computed and exposed, but **not yet enforced as a hard rank
   cap** in this scaffold — see "Known gaps" immediately below, this is the
   most important open design question.

### Known gaps in the scaffold (be honest about what's unfinished)

1. **D* is not a hard constraint yet.** The original design intent (S2 caps
   how much rank a matrix may absorb before bits become more efficient) was
   implemented and then reverted during testing: a hard `min(rank, d_star)`
   makes `target_bits_per_param` unreachable whenever `d_star` is small
   relative to `max_rank` for many matrices simultaneously, because the
   current algorithm doesn't reopen the bit-width choice to spend the
   resulting surplus (bits are fixed by S2 in step 3, independently of
   whether rank saturated its cap in step 2). The correct fix is a proper
   joint (not two-phase) optimization — e.g. Lagrangian relaxation with
   coupled multipliers on rank and bits simultaneously (this is what D-Rank,
   arXiv:2509.25622, does for rank alone, per the mesa notes) — or a
   redistribution pass that reroutes budget from capped matrices to
   uncapped ones. Left as the first real research task, not scaffolding.
2. **`q_bits` (the backbone) is a single global scalar**, matching CALDERA's
   current behavior exactly, but leaving unused the fact that entropy-lens
   could in principle also vary the backbone's precision per-layer (a further
   refinement past the initial rank+bits split, deliberately deferred to keep
   the first experiment interpretable: change one axis of uniformity at a
   time).
3. **No solver.** This module decides `(rank, bits)`, not how to *compute* a
   matrix satisfying them — i.e. it does not implement CALDERA's alternating
   minimization or a lattice quantizer. It is meant to sit upstream of either
   a clean-room LPLR solver or (short term) a simpler baseline: rank-`rank`
   whitened truncation (`whiten.whiten_truncate`, already implemented) plus a
   naive uniform/NF4 quantization of the residual at `bits` per weight, as a
   first Pareto point before investing in lattice quantization.
4. **Bit-width granularity**: `bits_choices` defaults to `{2,3,4}` to mirror
   QuIP#'s lattice codebooks (`E8P12`, `E8P12RVQ3B`, `E8P12RVQ4B`), but if we
   build a from-scratch quantizer this constraint can be relaxed (e.g.
   fractional/mixed formats), which would change the S2-driven grouping logic
   in step 3.

## 5. Experiment plan: joint allocation vs. CALDERA uniform

**Goal.** Show that entropy-guided joint allocation beats CALDERA's uniform
`(rank, Q_bits, L_bits, R_bits)` at matched average bits/param, on the same
whitening backbone, isolating the allocation rule as the only variable.

**Design.**
1. Compute whitened S1_eff/S2_eff per matrix on Mistral-7B-v0.3 (extending
   the in-progress whitening work referenced in `whiten.py`'s module
   docstring: covariances per input group, chunked over layers for GPU
   memory).
2. Run CALDERA (once we can legally run its code — see licensing) at 2-3
   fixed uniform configs spanning ~2-2.5 bits/param, record WikiText-2 PPL.
3. Run our joint allocator at matched average bits/param (using either a
   clean-room LPLR-lite solver, or the interim naive-truncation +
   uniform-quantization baseline from gap #3 above, clearly labeled as a
   lower bound on what a real solver would achieve) and record PPL at the
   same budget.
4. Report Pareto curves (PPL vs bits/param) for: CALDERA uniform, our joint
   allocation, and — as an honest sanity floor — uniform rank/bits with our
   own solver (isolates "does entropy guidance help" from "is our solver
   competitive with CALDERA's").
5. Ablate: joint allocation with rank-only guidance (S1, bits uniform) vs
   bits-only guidance (S2, rank uniform) vs both, to attribute gains.

**The bar to beat.** The mesa's notes cite two numbers, one verified and one
not:
- **D-Rank, Llama-2-7B at 20% low-rank (no quantization), WikiText-2: PPL
  7.51** — this is a *rank-only* allocation result (arXiv:2509.25622), not
  CALDERA's, and is **not independently re-verified here**; take it from the
  mesa notes with that caveat.
- **CALDERA's own headline table (Llama-2-7B, various bits/param) was
  explicitly flagged as unverified in the mesa notes** ("validar ... la tabla
  completa de CALDERA antes de fijar el bar definitivo"). I did not locate a
  copy of the full results table in this session (would require pulling
  the PDF/HTML of arXiv:2405.18886 and transcribing table values, which I did
  not do — flagging honestly rather than guessing numbers). **Before setting
  a definitive numeric bar for the paper, fetch and transcribe the CALDERA
  paper's Table 2/3 PPL-vs-bits-per-parameter numbers for Llama-2-7B
  directly.** Until then, treat "beat CALDERA uniform at matched bits/param,
  measured on our own harness" as the real bar, not any single quoted number.

## 6. Feasibility on RunPod L4 24GB, Mistral-7B-v0.3 fp16

- **Whitening / Hessian computation**: already demonstrated feasible in this
  repo (chunked covariance accumulation, `whiten.collect_covariances`); this
  is the easy part and mostly a rerun of existing infra.
- **CALDERA's own solver, as published**: uses QuIP#'s `quiptools` CUDA
  kernels for lattice dequantization and the E8P codebooks. These are
  reasonably lightweight (2-4 bit codebooks, not full fine-tuning), and the
  paper's own reference hardware for the base algorithm is consumer/prosumer
  GPUs — a single L4 24GB is very plausibly sufficient for **quantizing**
  Mistral 7B (not training it), since Hessian computation + alternating
  minimization on a 7B model's weight matrices one at a time is far lighter
  than full-model forward/backward passes. This is a reasonable expectation,
  not independently benchmarked by me in this session (no CALDERA code was
  run, per the licensing constraint above) — **flag as a real unknown until
  someone runs the actual `quip-sharp` build on the L4 and reports wall
  time.** `fast-hadamard-transform` and `quiptools` require CUDA
  extension builds; matching CUDA/PyTorch/Python versions to the pinned wheel
  availability (repo README calls out Python 3.10/3.11, CUDA 12.1/12.2,
  PyTorch 2.2) is itself a day-scale integration task, independent of the
  license question.
- **A clean-room LPLR-lite solver** (no QuIP# dependency, naive/NF4
  quantization of the residual): strictly lighter than the above, definitely
  fits an L4, and is the pragmatic first Pareto point given the licensing
  blocker on CALDERA's own quantizer.
- **Estimated effort** (explicitly uncertain, treat as an order-of-magnitude
  planning number, not a commitment):
  - Whitening rerun to produce S1_eff/S2_eff CSV for all 224 Mistral-7B
    matrices: **~1 day** (mostly already built).
  - Fixing the D* hard-cap gap in `joint_alloc.py` (gap #1 above) into a real
    joint rank+bits optimizer: **~2-3 days**, research-flavored (needs a
    correct Lagrangian or redistribution scheme, not just engineering).
  - Clean-room LPLR-lite solver (whitened truncation + naive/NF4 quantized
    residual, no lattice codebook): **~2-3 days**, building on
    `whiten.whiten_truncate` and `compress.py`.
  - Standing up CALDERA itself as a baseline (env build, QuIP# submodule,
    CUDA kernels, license resolution): **~2-4 days**, dominated by
    uncertainty (license reply time from the authors is out of our control;
    CUDA/wheel matching can eat a day by itself) rather than compute.
  - **Total: roughly 1.5-2.5 weeks of engineering + research time**, before
    counting evaluation/PPL-sweep GPU time (which is cheap, single-digit
    GPU-hours per config on an L4 for a 7B model's forward-pass eval).
- **GPU cost**: at typical L4 RunPod pricing (roughly $0.3-0.5/hr as of past
  sessions in this project, not reverified today), the compute itself is
  cheap — tens of dollars, not hundreds — for the whitening + eval sweeps.
  The dominant cost is engineering time, not GPU-hours.

## 7. Risks

- **License risk (highest priority, blocking)**: CALDERA has no license. Any
  path that copies its code is legally unsound; the recommended path
  (clean-room reimplementation from the paper) avoids this but forgoes
  reusing tested, working code, which increases engineering risk elsewhere
  (a from-scratch LPLR solver + quantizer is nontrivial to get numerically
  right).
- **GPL-3.0 exposure via QuIP#** if we do choose to depend on its lattice
  quantizer for a stronger baseline — a licensing decision for the human,
  not something to default into silently.
- **D* hard-cap is unsolved** (gap #1): the current scaffold's rank/bits
  split is a reasonable v1 but not yet the "true" joint optimization the
  mesa's physics argues for. Treat `joint_alloc.py` as a testable strawman,
  not the final allocator.
- **Scoop risk continues**: per the mesa notes, monitor arXiv every 2 weeks;
  D-Rank/Swift-SVD/SigmaScale/UniRank already occupy rank-only allocation.
  The *joint* rank+bits space is the claimed opening, but that claim is
  based on a single reading session (this mesa), not an exhaustive search of
  quantization-composition papers — recommend one more explicit search pass
  (arXiv cs.LG, "joint rank quantization allocation", "layer-wise mixed
  precision low-rank") before committing significant engineering time.
- **Numeric bar unverified**: the CALDERA paper's own PPL table and the exact
  D-Rank number at 20-50% were both flagged as unverified in the mesa notes
  and remain unverified after this session (I did not transcribe the paper's
  tables). Do not quote either number in a paper draft without pulling the
  primary source table directly.
