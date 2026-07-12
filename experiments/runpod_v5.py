"""RunPod experiment v5: refit the Entropy-Compression Law in the data metric.

Por qué existe (mesa conjunta 2026-07-12, ver Qtech Idea/mesa-conjunta-2026-07-12.md):
  v3/v4 demostraron que truncar el espectro raw destruye Mistral 7B a cualquier
  budget factorizado. Diagnóstico de la mesa: la ley mide en la métrica de
  Frobenius, pero la pérdida del modelo vive en la métrica de datos
  ||dW @ L||_F con L L^T = C (covarianza de activaciones de calibración).
  Este experimento calcula el espectro de la matriz blanqueada M = W @ L para
  las 224 matrices de Mistral 7B y refitea la ley:

      D_min_eff(eps) ~ c(eps) * exp(alpha * S1_eff)

  Predicciones falsables (Feynman): (1) S1_eff << S1_raw en todas las capas;
  (2) el gap S1_raw - S1_eff es grande donde el SVD naive destruye mas;
  (3) la ley en la metrica de datos mantiene R2 >= 0.99.

  Este es el prerequisito del north star (allocation conjunta rank+bits
  sobre base Q+LR): sin espectros blanqueados no hay criterio de reparto.

Run (siempre con nohup):
  cd /workspace/entropy-lens && git pull && pip install -q -e .
  nohup python experiments/runpod_v5.py > /workspace/experiment_v5.log 2>&1 &
  tail -f /workspace/experiment_v5.log
"""
import csv, gc, json, logging, time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from entropy_lens.spectral import compute_s1, compute_s2, compute_dmin
from entropy_lens.law import fit_entropy_law
from entropy_lens.whiten import (
    PROJ_TO_GROUP,
    cholesky_factor,
    collect_covariances,
    whitened_svdvals,
)

MODEL = "mistralai/Mistral-7B-v0.3"
RAW_CSV = "results/mistralai_Mistral-7B-v0.3/results.csv"
OUT_CSV = "/workspace/results_v5_whitened.csv"
OUT_JSON = "/workspace/results_v5_fit.json"
N_CALIB_SEQS = 256
SEQ_LEN = 512
LAYERS_PER_CHUNK = 4  # bounds GPU memory: ~1 GB of covariances per layer
EPSILONS = {"5pct": 0.05, "10pct": 0.10, "20pct": 0.20, "50pct": 0.50}
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

print("=== v5: whitened spectra + law refit ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.to("cuda").eval()
n_layers = model.config.num_hidden_layers

print("Preparing calibration batches (wikitext-2 train)...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
txt = "\n\n".join([t for t in ds["text"] if t.strip()])
ids = tokenizer(txt, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"].squeeze(0)
batches = [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0) for i in range(N_CALIB_SEQS)]
print(f"{len(batches)} sequences x {SEQ_LEN} tokens")

fieldnames = ["name", "layer", "proj_type", "s1_eff", "s2_eff", "chol_triangular",
              "chol_damp"] + [f"dmin_eff_{lbl}" for lbl in EPSILONS]
with open(OUT_CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

t_start = time.time()
for chunk_start in range(0, n_layers, LAYERS_PER_CHUNK):
    chunk = list(range(chunk_start, min(chunk_start + LAYERS_PER_CHUNK, n_layers)))
    print(f"\n--- Layers {chunk[0]}-{chunk[-1]}: calibrating ---")
    covs = collect_covariances(model, batches, chunk, device="cuda")

    for i in chunk:
        block = model.model.layers[i]
        modules = {
            "q_proj": block.self_attn.q_proj, "k_proj": block.self_attn.k_proj,
            "v_proj": block.self_attn.v_proj, "o_proj": block.self_attn.o_proj,
            "gate_proj": block.mlp.gate_proj, "up_proj": block.mlp.up_proj,
            "down_proj": block.mlp.down_proj,
        }
        # Factor each covariance group once, reuse across projections
        factors = {}
        for group in set(PROJ_TO_GROUP.values()):
            L, triangular, lam = cholesky_factor(covs[(i, group)])
            factors[group] = (L, triangular, lam)

        for proj in PROJS:
            L, triangular, lam = factors[PROJ_TO_GROUP[proj]]
            w = modules[proj].weight.data
            sv = whitened_svdvals(w, L, device="cuda")
            row = {
                "name": f"layer_{i}.{proj}", "layer": i, "proj_type": proj,
                "s1_eff": compute_s1(sv), "s2_eff": compute_s2(sv),
                "chol_triangular": triangular, "chol_damp": lam,
            }
            for lbl, eps in EPSILONS.items():
                row[f"dmin_eff_{lbl}"] = compute_dmin(sv, eps)
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(f"  layer {i} done ({time.time() - t_start:.0f}s elapsed)")

    del covs
    gc.collect()
    torch.cuda.empty_cache()

print("\n=== Refitting the law ===")
raw = {}
with open(RAW_CSV) as f:
    for r in csv.DictReader(f):
        raw[r["name"]] = r
eff = []
with open(OUT_CSV) as f:
    eff = list(csv.DictReader(f))

report = {"model": MODEL, "n_matrices": len(eff), "fits": {}}
for lbl in EPSILONS:
    s1_eff = np.array([float(r["s1_eff"]) for r in eff])
    dmin_eff = np.array([float(r[f"dmin_eff_{lbl}"]) for r in eff])
    fit_eff = fit_entropy_law(s1_eff, dmin_eff)

    matched = [r for r in eff if r["name"] in raw]
    s1_raw = np.array([float(raw[r["name"]]["s1"]) for r in matched])
    dmin_raw = np.array([float(raw[r["name"]][f"dmin_{lbl}"]) for r in matched])
    fit_raw = fit_entropy_law(s1_raw, dmin_raw)

    report["fits"][lbl] = {
        "eff": {"r2": fit_eff.r_squared, "slope": fit_eff.slope, "c": fit_eff.c_constrained},
        "raw": {"r2": fit_raw.r_squared, "slope": fit_raw.slope, "c": fit_raw.c_constrained},
    }
    print(f"eps={lbl}: R2_eff={fit_eff.r_squared:.4f} slope_eff={fit_eff.slope:.3f}"
          f" | R2_raw={fit_raw.r_squared:.4f} slope_raw={fit_raw.slope:.3f}")

gaps = [float(raw[r["name"]]["s1"]) - float(r["s1_eff"]) for r in eff if r["name"] in raw]
report["s1_gap"] = {"mean": float(np.mean(gaps)), "min": float(np.min(gaps)),
                    "max": float(np.max(gaps)),
                    "all_positive": bool(np.all(np.array(gaps) > 0))}
print(f"\nGap S1_raw - S1_eff: mean={report['s1_gap']['mean']:.3f} "
      f"min={report['s1_gap']['min']:.3f} max={report['s1_gap']['max']:.3f} "
      f"(all positive: {report['s1_gap']['all_positive']})")

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {OUT_CSV} and {OUT_JSON}")
print(json.dumps(report["fits"], indent=2))
