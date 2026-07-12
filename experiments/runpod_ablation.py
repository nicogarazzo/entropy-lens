"""Per-layer naive-SVD ablation: closes part (2) of Feynman's falsable loop.

Mesa conjunta 2026-07-12. Feynman's prediction had two parts:
  (1) S1_eff << S1_raw  -- CONFIRMED, 224/224 matrices, mean gap 2.77 nats (v5).
  (2) the gap S1_raw - S1_eff predicts how much naive SVD truncation hurts
      each layer -- THIS experiment tests it.

Rationale: if a layer's weights are highly anisotropic *relative to the
data* (large raw-vs-effective entropy gap), then truncating in the
Frobenius metric (naive SVD, which ignores the data) should damage it more
than a layer whose raw and effective spectra already agree. We measure the
intrinsic damage of an identical 50%-rank naive truncation, layer by layer,
and correlate it against four candidate predictors. The honest comparison
(which predictor wins) is a paper figure either way -- per the house rule,
we report whatever comes out, confirm or refute.

Post-v5 context (local diagnostics, in ROADMAP): the data-metric law is
bidimensional (S1+S2). So we test s1_gap AND s1_raw/s1_eff/s2_eff as damage
predictors; if s1_gap loses to s2_eff, that is itself the finding.

Design:
  - Mistral-7B-v0.3 fp16 on GPU, loaded once.
  - Fast PPL eval on the first N_CHUNKS 1024-token chunks of WikiText-2 test
    (~30-40s/eval vs full ~5min). Baseline computed once.
  - For each layer i: CPU-snapshot its 7 matrices, truncate all 7 to 50% of
    their own full rank via GPU SVD (fixed uniform ratio -- we want intrinsic
    per-layer sensitivity to *identical* compression, NOT allocation),
    eval, restore from snapshot. delta_log_ppl = log(ppl_i) - log(ppl_base).
  - Correlate delta_log_ppl (32 layers) vs per-layer means of s1_gap,
    s1_raw, s1_eff, s2_eff. Pearson AND Spearman (damage may be nonlinear).

Run (nohup so it survives disconnects):
  cd /workspace/entropy-lens && git pull && pip install -q -e .
  nohup python experiments/runpod_ablation.py > /workspace/ablation.log 2>&1 &
  tail -f /workspace/ablation.log
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

MODEL = "mistralai/Mistral-7B-v0.3"
RAW_CSV = "results/mistralai_Mistral-7B-v0.3/results.csv"          # s1 (raw)
EFF_CSV = "/workspace/results_v5_whitened.csv"                      # s1_eff, s2_eff
OUT_CSV = "/workspace/ablation_per_layer.csv"
OUT_JSON = "/workspace/ablation_correlations.json"
OUT_PNG = "/workspace/ablation_gap_vs_damage.png"
N_CHUNKS = 25          # fast eval subset
CHUNK = 1024
TRUNC_RATIO = 0.50     # keep 50% of each matrix's own full rank
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------
# Core mechanics (import-testable)
# --------------------------------------------------------------------------

def layer_modules(model, i):
    """The 7 nn.Linear modules of decoder layer i, keyed by proj name."""
    b = model.model.layers[i]
    return {
        "q_proj": b.self_attn.q_proj, "k_proj": b.self_attn.k_proj,
        "v_proj": b.self_attn.v_proj, "o_proj": b.self_attn.o_proj,
        "gate_proj": b.mlp.gate_proj, "up_proj": b.mlp.up_proj,
        "down_proj": b.mlp.down_proj,
    }


def truncate_inplace(module, ratio=TRUNC_RATIO):
    """Replace module.weight with its rank-(ratio*full) SVD reconstruction.

    SVD runs on whatever device the weight is on (GPU in production), in
    float32 for stability, cast back to the weight's dtype. Dense output:
    this measures approximation *quality*, not storage.
    """
    w = module.weight.data
    m, n = w.shape
    rank = max(1, int(round(min(m, n) * ratio)))
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    recon = (u[:, :rank] * s[:rank]) @ vh[:rank]
    module.weight.data = recon.to(w.dtype)
    return rank


def eval_ppl_fast(model, ids, n_chunks=N_CHUNKS, chunk=CHUNK):
    """PPL over the first n_chunks non-overlapping chunks of `ids`."""
    nlls = []
    with torch.no_grad():
        for k in range(n_chunks):
            c = ids[:, k * chunk:(k + 1) * chunk]
            if c.size(1) < 2:
                break
            nlls.append(model(c, labels=c).loss.float().item())
    return float(torch.exp(torch.tensor(nlls).mean()))


def ablate_layer(model, i, ids):
    """Snapshot -> truncate all 7 matrices -> eval -> restore. Returns ppl."""
    mods = layer_modules(model, i)
    snap = {name: mod.weight.data.detach().cpu().clone() for name, mod in mods.items()}
    for mod in mods.values():
        truncate_inplace(mod)
    ppl = eval_ppl_fast(model, ids)
    for name, mod in mods.items():
        mod.weight.data = snap[name].to(mod.weight.device, mod.weight.dtype)
    return ppl


# --------------------------------------------------------------------------
# Predictor aggregation + correlation
# --------------------------------------------------------------------------

def load_layer_predictors(raw_csv, eff_csv):
    """Per-layer mean of s1_raw, s1_eff, s2_eff, and gap = s1_raw - s1_eff."""
    raw = {}
    with open(raw_csv) as f:
        for r in csv.DictReader(f):
            raw[r["name"]] = float(r["s1"])
    s1e, s2e, layer_of = {}, {}, {}
    with open(eff_csv) as f:
        for r in csv.DictReader(f):
            s1e[r["name"]] = float(r["s1_eff"])
            s2e[r["name"]] = float(r["s2_eff"])
            layer_of[r["name"]] = int(r["layer"])

    agg = defaultdict(lambda: defaultdict(list))
    for name, li in layer_of.items():
        if name not in raw:
            continue
        agg[li]["s1_raw"].append(raw[name])
        agg[li]["s1_eff"].append(s1e[name])
        agg[li]["s2_eff"].append(s2e[name])
        agg[li]["gap"].append(raw[name] - s1e[name])
    out = {}
    for li, d in agg.items():
        out[li] = {k: float(np.mean(v)) for k, v in d.items()}
    return out


def correlate(x, y):
    """Pearson + Spearman with p-values, as plain dicts."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("=== Per-layer naive-SVD ablation (falsable loop part 2) ===")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model.to("cuda").eval()
    n_layers = model.config.num_hidden_layers

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=CHUNK * (N_CHUNKS + 5)).input_ids.to("cuda")

    baseline = eval_ppl_fast(model, ids)
    print(f"Baseline PPL (first {N_CHUNKS} chunks): {baseline:.3f}")

    preds = load_layer_predictors(RAW_CSV, EFF_CSV)

    fieldnames = ["layer", "ppl_ablated", "delta_log_ppl",
                  "s1_gap_mean", "s1_raw_mean", "s1_eff_mean", "s2_eff_mean"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    for i in range(n_layers):
        ppl = ablate_layer(model, i, ids)
        p = preds.get(i, {})
        row = {
            "layer": i, "ppl_ablated": ppl,
            "delta_log_ppl": float(np.log(ppl) - np.log(baseline)),
            "s1_gap_mean": p.get("gap", float("nan")),
            "s1_raw_mean": p.get("s1_raw", float("nan")),
            "s1_eff_mean": p.get("s1_eff", float("nan")),
            "s2_eff_mean": p.get("s2_eff", float("nan")),
        }
        rows.append(row)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(f"  layer {i:2d}: ppl={ppl:10.2f}  dlogppl={row['delta_log_ppl']:.3f}"
              f"  gap={row['s1_gap_mean']:.3f}")

    dlog = [r["delta_log_ppl"] for r in rows]
    report = {
        "model": MODEL, "baseline_ppl": baseline,
        "config": {"n_chunks": N_CHUNKS, "chunk": CHUNK, "trunc_ratio": TRUNC_RATIO,
                   "n_layers": n_layers},
        "correlations": {
            "s1_gap": correlate([r["s1_gap_mean"] for r in rows], dlog),
            "s1_raw": correlate([r["s1_raw_mean"] for r in rows], dlog),
            "s1_eff": correlate([r["s1_eff_mean"] for r in rows], dlog),
            "s2_eff": correlate([r["s2_eff_mean"] for r in rows], dlog),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Correlations of delta_log_ppl vs predictor (per layer, n=%d) ===" % n_layers)
    for name, c in report["correlations"].items():
        print(f"  {name:8s}: Pearson r={c['pearson_r']:+.3f} (p={c['pearson_p']:.2e})"
              f"  Spearman r={c['spearman_r']:+.3f} (p={c['spearman_p']:.2e})")

    _scatter(rows, report, OUT_PNG)
    print(f"\nSaved {OUT_CSV}, {OUT_JSON}, {OUT_PNG}")


def _scatter(rows, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = [r["s1_gap_mean"] for r in rows]
    dlog = [r["delta_log_ppl"] for r in rows]
    c = report["correlations"]["s1_gap"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(gap, dlog, c=[r["layer"] for r in rows], cmap="viridis", s=60)
    for r in rows:
        ax.annotate(str(r["layer"]), (r["s1_gap_mean"], r["delta_log_ppl"]),
                    fontsize=7, alpha=0.6)
    ax.set_xlabel("mean s1_gap  (S1_raw - S1_eff)")
    ax.set_ylabel("delta_log_ppl  (naive 50% SVD damage)")
    ax.set_title(f"Feynman falsable loop pt.2  |  Pearson r={c['pearson_r']:+.3f} "
                 f"(p={c['pearson_p']:.1e})")
    fig.colorbar(sc, label="layer index")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
