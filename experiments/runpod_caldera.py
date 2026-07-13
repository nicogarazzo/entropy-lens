"""RunPod experiment: CALDERA-uniform vs entropy-guided joint allocation vs baseline.

Why this exists (north star, mesa conjunta 2026-07-12, see
experiments/caldera_integration_plan.md for the full design + licensing
discussion): CALDERA (arXiv:2405.18886) decomposes W ~= Q + L@R with a
SINGLE global (rank, Q_bits, L_bits, R_bits) for every projection in the
model. Our claim is that entropy-guided per-matrix allocation
(entropy_lens.joint_alloc.allocate_joint_rank_bits, driven by S1_eff/S2_eff
of the WHITENED spectrum -- see whiten.py and results_v5b_whitened.csv) beats
CALDERA's uniform allocation at matched total bits/param. This script runs
three conditions at (approximately) equal average bits/param and reports
WikiText-2 perplexity for each:

  (A) uniform  -- same rank & bits everywhere, CALDERA-style. The rank is
                  solved for by bisection so this condition lands at
                  target_bits_per_param exactly.
  (B) joint    -- entropy_lens.joint_alloc.allocate_joint_rank_bits on the
                  v5b whitened CSV; rank driven by S1_eff, LR bit-width
                  driven by S2_eff, at the same target_bits_per_param.
  (C) baseline -- uncompressed fp16 model, no decomposition at all (upper
                  bound / sanity floor).

For (A) and (B), each compressible projection (q/k/v/o/gate/up/down) is
replaced by entropy_lens.lplr.lplr_decompose_whitened's reconstruction at its
assigned (rank, bits). Whitening factors are recomputed layer-by-layer on
this run (same calibration + covariance pipeline as experiments/runpod_v5.py)
rather than reloaded from the CSV, since the CSV stores only the derived
S1_eff/S2_eff/dmin_eff summary stats, not the whitening matrices themselves.

*** HONESTY NOTICE -- READ BEFORE TRUSTING ANY NUMBER THIS SCRIPT PRINTS ***
This script's LPLR quantization step defaults to `entropy_lens.lplr.round_quantize`,
a simple per-tensor uniform round-trip "quantizer" -- NOT QuIP#'s E8P lattice
codebook. QuIP# requires CUDA kernels (fast-hadamard-transform, quiptools)
that were never built or run in the session that wrote this script (dev
machine: Mac, no CUDA -- see experiments/caldera_integration_plan.md section
6 for the verified QuIP# install steps and the risks). Concretely:
  - This script has NEVER been executed end-to-end on a GPU as of this
    commit. It is "ready to run", not "has been run".
  - Any PPL numbers it produces with the default quantizer are a real,
    honest comparison of ALLOCATION STRATEGIES under a shared (crude)
    quantizer -- i.e. condition (B) vs (A) is a fair test of "does
    entropy-guided allocation beat uniform allocation", holding the
    quantizer fixed. That comparison is meaningful even with the fake
    quantizer.
  - It is NOT a fair comparison to CALDERA's own published numbers, which
    use the real QuIP# E8P lattice quantizer (systematically better than
    per-tensor uniform rounding at the same bit-width). Do not quote this
    script's absolute PPL against CALDERA's paper table without first
    swapping in a real QuIP#-backed quantize_fn (see `--quantizer quip`
    below and entropy_lens.lplr.QUIP_SHARP_WIRING_NOTES).
  - The CALDERA paper's own PPL table was flagged as unverified in the mesa
    notes and remains untranscribed as of this script (see
    caldera_integration_plan.md section 5, "The bar to beat"). Do not
    invent a numeric bar; report this script's three conditions against
    each other and note the quantizer used.

Run (always with nohup on the pod):
  cd /workspace/entropy-lens && git pull && pip install -q -e .
  nohup python experiments/runpod_caldera.py > /workspace/experiment_caldera.log 2>&1 &
  tail -f /workspace/experiment_caldera.log

To wire in the real QuIP# quantizer once quip-sharp is installed and built:
  python experiments/runpod_caldera.py --quantizer quip
(this currently raises NotImplementedError with a pointer to
entropy_lens.lplr.QUIP_SHARP_WIRING_NOTES -- implementing the real adapter is
pod-side follow-up work, see caldera_integration_plan.md.)

Smoke-test the plumbing (no GPU, no downloaded model, seconds not hours):
  PYTHONPATH=src python experiments/runpod_caldera.py --smoke-test
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("runpod_caldera")

import torch

from entropy_lens.joint_alloc import (
    MatrixSpec,
    allocate_joint_rank_bits,
    load_matrices_from_csv,
)
from entropy_lens.lplr import lplr_decompose_whitened, round_quantize
from entropy_lens.whiten import (
    PROJ_TO_GROUP,
    cholesky_factor,
    collect_covariances,
)

MODEL = "mistralai/Mistral-7B-v0.3"
CSV_V5B = "results/mistralai_Mistral-7B-v0.3/results_v5b_whitened.csv"
OUT_JSON = "/workspace/results_caldera.json"
TARGET_BITS_PER_PARAM = 3.0  # ~3 bits/param, matches CALDERA's typical operating point
Q_BITS = 2  # backbone bit-width, matches CALDERA's default / QuIP# E8P12 (2-bit)
UNIFORM_LR_BITS = 4  # fixed LR bit-width for the CALDERA-uniform condition
N_CALIB_SEQS = 256
SEQ_LEN = 512
LAYERS_PER_CHUNK = 4  # bounds GPU memory, matches runpod_v5.py
LPLR_MAX_ITERS = 15
LPLR_TOL = 1e-5
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_quantize_fn(name: str):
    if name == "fake":
        log.warning(
            "*** Using round_quantize (fake per-tensor rounding), NOT QuIP#'s E8P "
            "lattice codebook. See this file's module docstring HONESTY NOTICE "
            "before trusting absolute PPL numbers. ***"
        )
        return round_quantize
    if name == "quip":
        raise NotImplementedError(
            "Real QuIP#-backed quantize_fn is not wired up yet -- this is pod-side "
            "follow-up work. See entropy_lens.lplr.QUIP_SHARP_WIRING_NOTES for the "
            "exact codebook files to adapt (lib/codebook/latticee8_padded12*.py in "
            "Cornell-RelaxML/quip-sharp) and experiments/caldera_integration_plan.md "
            "section 6 for the verified install steps."
        )
    raise ValueError(f"Unknown quantizer {name!r}, expected 'fake' or 'quip'")


# ---------------------------------------------------------------------------
# Uniform (CALDERA-style) allocation: single global rank solved by bisection
# to hit target_bits_per_param exactly, at fixed q_bits/lr_bits.
# ---------------------------------------------------------------------------


def solve_uniform_rank(
    matrices: list[MatrixSpec],
    target_bits_per_param: float,
    q_bits: int,
    lr_bits: int,
) -> int:
    """Bisect a single rank (same for every matrix) so total average storage
    matches target_bits_per_param, mirroring CALDERA's uniform CalderaParams
    (rank, Q_bits, L_bits, R_bits all scalars -- see caldera_integration_plan.md
    section 1, "Confirmed from the repo")."""
    total_params = sum(m.shape_m * m.shape_n for m in matrices)
    target_bits = target_bits_per_param * total_params

    def cost(rank: int) -> float:
        return sum(m.storage_bits(rank, lr_bits, q_bits) for m in matrices)

    max_rank = max(m.max_rank for m in matrices)
    lo, hi = 1, max_rank
    if cost(hi) <= target_bits:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= target_bits:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def uniform_assignments(
    matrices: list[MatrixSpec], rank: int, lr_bits: int
) -> dict[str, tuple[int, int]]:
    return {m.name: (rank, lr_bits) for m in matrices}


# ---------------------------------------------------------------------------
# Model compression: apply (rank, bits) assignments layer-by-layer via LPLR.
# ---------------------------------------------------------------------------


def compress_model_lplr(
    model,
    assignments: dict[str, tuple[int, int]],
    q_bits: int,
    quantize_fn,
    n_layers: int,
    batches: list[torch.Tensor],
    device: str = "cuda",
) -> None:
    """In-place LPLR compression of every assigned projection, chunked over
    layers to bound GPU memory (same chunking strategy as runpod_v5.py's
    whitening pass)."""
    for chunk_start in range(0, n_layers, LAYERS_PER_CHUNK):
        chunk = list(range(chunk_start, min(chunk_start + LAYERS_PER_CHUNK, n_layers)))
        log.info(f"  compressing layers {chunk[0]}-{chunk[-1]}...")
        covs = collect_covariances(model, batches, chunk, device=device)

        for i in chunk:
            block = model.model.layers[i]
            modules = {
                "q_proj": block.self_attn.q_proj, "k_proj": block.self_attn.k_proj,
                "v_proj": block.self_attn.v_proj, "o_proj": block.self_attn.o_proj,
                "gate_proj": block.mlp.gate_proj, "up_proj": block.mlp.up_proj,
                "down_proj": block.mlp.down_proj,
            }
            factors = {}
            for group in set(PROJ_TO_GROUP.values()):
                L, triangular, _ = cholesky_factor(covs[(i, group)])
                factors[group] = (L.to(device), triangular)

            for proj in PROJS:
                name = f"layer_{i}.{proj}"
                if name not in assignments:
                    continue
                rank, lr_bits = assignments[name]
                L, triangular = factors[PROJ_TO_GROUP[proj]]
                w = modules[proj].weight.data
                result = lplr_decompose_whitened(
                    w, L, rank=rank, q_bits=q_bits, lr_bits=lr_bits,
                    quantize_fn=quantize_fn, max_iters=LPLR_MAX_ITERS, tol=LPLR_TOL,
                    triangular=triangular,
                )
                modules[proj].weight.data.copy_(result.reconstruction.to(w.dtype))
            log.info(f"    layer {i} done")

        del covs
        gc.collect()
        torch.cuda.empty_cache()


def eval_ppl(model, tokenizer, device: str = "cuda") -> float:
    from datasets import load_dataset
    model.to(device).eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024 * 100).input_ids.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(0, min(ids.size(1), 1024 * 100), 1024):
            c = ids[:, i:i + 1024]
            if c.size(1) < 2:
                continue
            nlls.append(model(c, labels=c).loss.float().item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()))
    model.to("cpu")
    torch.cuda.empty_cache()
    return ppl


def _append_result(path: str, key: str, value) -> None:
    """Incremental JSON save: read-modify-write so a crash mid-run doesn't
    lose earlier conditions' results."""
    try:
        with open(path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    report[key] = value
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def run_smoke_test() -> None:
    """No GPU, no downloaded model: exercises the allocation + LPLR plumbing
    on tiny synthetic matrices in seconds, to catch wiring bugs before
    burning GPU-hours on the pod. Does NOT validate real PPL behavior."""
    log.info("=== smoke test: synthetic matrices, fake quantizer, CPU only ===")
    torch.manual_seed(0)
    matrices = [
        MatrixSpec(name=f"layer_{i}.{p}", shape_m=64, shape_n=64,
                   s1_eff=float(1 + i), s2_eff=float(0.5 + 0.3 * i))
        for i in range(3) for p in PROJS[:2]
    ]
    rank = solve_uniform_rank(matrices, TARGET_BITS_PER_PARAM, Q_BITS, UNIFORM_LR_BITS)
    log.info(f"uniform rank solved: {rank}")
    uni = uniform_assignments(matrices, rank, UNIFORM_LR_BITS)
    joint = allocate_joint_rank_bits(matrices, TARGET_BITS_PER_PARAM, q_bits=Q_BITS)
    log.info(f"uniform actual bits/param: "
              f"{sum(m.storage_bits(rank, UNIFORM_LR_BITS, Q_BITS) for m in matrices) / sum(m.shape_m * m.shape_n for m in matrices):.3f}")
    log.info(f"joint actual bits/param: {joint.actual_bits_per_param:.3f}")

    qfn = get_quantize_fn("fake")
    w = torch.randn(64, 64)
    lw = torch.eye(64)
    name = matrices[0].name
    r, b = uni[name]
    result = lplr_decompose_whitened(w, lw, rank=r, q_bits=Q_BITS, lr_bits=b, quantize_fn=qfn, max_iters=5)
    assert torch.isfinite(result.reconstruction).all()
    log.info(f"smoke test LPLR reconstruction error: {result.final_error:.4f}")
    log.info("=== smoke test passed ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantizer", choices=["fake", "quip"], default="fake")
    parser.add_argument("--target-bits-per-param", type=float, default=TARGET_BITS_PER_PARAM)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--out-json", default=OUT_JSON)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    quantize_fn = get_quantize_fn(args.quantizer)

    log.info(f"=== runpod_caldera: uniform vs joint vs baseline @ {args.target_bits_per_param} bits/param ===")
    matrices = load_matrices_from_csv(CSV_V5B)
    log.info(f"loaded {len(matrices)} matrix specs from {CSV_V5B}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    txt = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(txt, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"].squeeze(0)
    batches = [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0) for i in range(N_CALIB_SEQS)]
    log.info(f"{len(batches)} calibration sequences x {SEQ_LEN} tokens")

    report = {
        "model": MODEL,
        "target_bits_per_param": args.target_bits_per_param,
        "quantizer": args.quantizer,
        "q_bits": Q_BITS,
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    # --- Condition C: baseline, uncompressed ---
    log.info("\n=== (C) baseline ===")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    ppl_baseline = eval_ppl(model, tokenizer)
    log.info(f"baseline PPL: {ppl_baseline:.3f} ({time.time() - t0:.0f}s)")
    _append_result(args.out_json, "baseline", {"ppl": ppl_baseline})
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Condition A: uniform (CALDERA-style) ---
    log.info("\n=== (A) uniform (CALDERA-style) ===")
    t0 = time.time()
    uniform_rank = solve_uniform_rank(matrices, args.target_bits_per_param, Q_BITS, UNIFORM_LR_BITS)
    uniform_assign = uniform_assignments(matrices, uniform_rank, UNIFORM_LR_BITS)
    uniform_bits_per_param = (
        sum(m.storage_bits(uniform_rank, UNIFORM_LR_BITS, Q_BITS) for m in matrices)
        / sum(m.shape_m * m.shape_n for m in matrices)
    )
    log.info(f"uniform rank={uniform_rank}, lr_bits={UNIFORM_LR_BITS}, "
              f"q_bits={Q_BITS}, actual bits/param={uniform_bits_per_param:.3f}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to("cuda")
    n_layers = model.config.num_hidden_layers
    compress_model_lplr(model, uniform_assign, Q_BITS, quantize_fn, n_layers, batches)
    ppl_uniform = eval_ppl(model, tokenizer)
    log.info(f"uniform PPL: {ppl_uniform:.3f} ({time.time() - t0:.0f}s)")
    _append_result(args.out_json, "uniform", {
        "ppl": ppl_uniform, "rank": uniform_rank, "lr_bits": UNIFORM_LR_BITS,
        "actual_bits_per_param": uniform_bits_per_param,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Condition B: joint entropy-guided allocation ---
    log.info("\n=== (B) joint (entropy-guided) ===")
    t0 = time.time()
    joint = allocate_joint_rank_bits(matrices, args.target_bits_per_param, q_bits=Q_BITS)
    log.info(f"joint actual bits/param={joint.actual_bits_per_param:.3f}, "
              f"n_matrices={len(joint.assignments)}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to("cuda")
    n_layers = model.config.num_hidden_layers
    compress_model_lplr(model, joint.assignments, Q_BITS, quantize_fn, n_layers, batches)
    ppl_joint = eval_ppl(model, tokenizer)
    log.info(f"joint PPL: {ppl_joint:.3f} ({time.time() - t0:.0f}s)")
    _append_result(args.out_json, "joint", {
        "ppl": ppl_joint, "actual_bits_per_param": joint.actual_bits_per_param,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()

    log.info("\n=== FINAL RESULTS ===")
    with open(args.out_json) as f:
        log.info(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
