#!/usr/bin/env python3
"""
The experiment: validate the Entropy-Compression Law on 7B+ models.

Usage:
    python experiments/validate_7b.py [MODEL_IDS...]

    # Sanity check with GPT-2:
    python experiments/validate_7b.py openai-community/gpt2

    # Full validation:
    python experiments/validate_7b.py openai-community/gpt2 mistralai/Mistral-7B-v0.3

Results are saved to results/<model_name>/ with CSV, JSON report, and plots.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from entropy_lens.extract import extract_svdvals_streaming
from entropy_lens.law import evaluate_go_nogo, fit_entropy_law
from entropy_lens.report import save_report
from entropy_lens.spectral import (
    compute_alpha_hill,
    compute_dmin,
    compute_participation_ratio,
    compute_s1,
    compute_s2,
)


EPSILONS = [0.05, 0.10, 0.20, 0.50]

PROJ_COLORS = {
    "q_proj": "#2166ac",
    "k_proj": "#4393c3",
    "v_proj": "#92c5de",
    "o_proj": "#d1e5f0",
    "gate_proj": "#b2182b",
    "up_proj": "#d6604d",
    "down_proj": "#f4a582",
}

PROJ_MARKERS = {
    "q_proj": "o",
    "k_proj": "s",
    "v_proj": "^",
    "o_proj": "D",
    "gate_proj": "P",
    "up_proj": "X",
    "down_proj": "v",
}


def analyze_model(model_id: str, output_base: str = "results") -> dict:
    """Run full analysis on a single model."""
    safe_name = model_id.replace("/", "_")
    out_dir = Path(output_base) / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ANALYZING: {model_id}")
    print(f"{'='*70}\n")

    records = []
    t_start = time.perf_counter()

    for name, sv in extract_svdvals_streaming(model_id):
        s1 = compute_s1(sv)
        s2 = compute_s2(sv)
        pr = compute_participation_ratio(sv)
        alpha = compute_alpha_hill(sv)

        # Extract proj_type from canonical name: layer_0.q_proj -> q_proj
        proj_type = name.split(".")[-1]
        layer_idx = int(name.split(".")[0].split("_")[1])

        row = {
            "name": name,
            "layer_idx": layer_idx,
            "proj_type": proj_type,
            "rank": len(sv),
            "s1": float(s1),
            "s2": float(s2),
            "pr": float(pr),
            "alpha_hill": float(alpha) if not np.isnan(alpha) else None,
        }
        for eps in EPSILONS:
            row[f"dmin_{int(eps*100)}pct"] = compute_dmin(sv, eps)

        records.append(row)

        dmin_str = " ".join(
            f"D({int(e*100)}%)={row[f'dmin_{int(e*100)}pct']}" for e in EPSILONS
        )
        print(f"  {name}: S1={s1:.3f} {dmin_str}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Matrices: {len(records)}")

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  CSV: {csv_path}")

    # Fit law
    s1_arr = np.array([r["s1"] for r in records])
    fit_results = {}

    print(f"\n  {'='*50}")
    print(f"  ENTROPY-COMPRESSION LAW FIT: {model_id}")
    print(f"  {'='*50}")

    for eps in EPSILONS:
        col = f"dmin_{int(eps*100)}pct"
        dmin_arr = np.array([r[col] for r in records], dtype=float)
        try:
            fit = fit_entropy_law(s1_arr, dmin_arr)
            fit_results[col] = fit
            print(
                f"    eps={int(eps*100)}%: R2={fit.r_squared:.4f} "
                f"slope={fit.slope:.4f} c={fit.c_constrained:.3f} (n={fit.n})"
            )
        except ValueError as e:
            print(f"    eps={int(eps*100)}%: FAILED ({e})")

    verdict = evaluate_go_nogo(fit_results)
    print(f"\n  VERDICT: {verdict}")

    # Summary
    s1_mean = float(np.mean(s1_arr))
    s1_std = float(np.std(s1_arr))
    print(f"  S1 mean: {s1_mean:.4f} +/- {s1_std:.4f} nats")

    # Save report
    report_path = save_report(model_id, records, fit_results, verdict, str(out_dir))

    # Generate plots
    _plot_scatter(records, fit_results, model_id, str(out_dir))
    _plot_by_layer(records, model_id, str(out_dir))

    return {
        "model": model_id,
        "n_matrices": len(records),
        "s1_mean": s1_mean,
        "s1_std": s1_std,
        "fits": fit_results,
        "verdict": verdict,
        "elapsed_s": elapsed,
    }


def _plot_scatter(records, fit_results, model_id, out_dir):
    """S1 vs D_min scatter plot with regression line, one per epsilon."""
    s1_arr = np.array([r["s1"] for r in records])

    for eps in EPSILONS:
        col = f"dmin_{int(eps*100)}pct"
        dmin_arr = np.array([r[col] for r in records], dtype=float)
        fit = fit_results.get(col)
        if fit is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for proj_type in sorted(set(r["proj_type"] for r in records)):
            mask = [r["proj_type"] == proj_type for r in records]
            s1_sub = s1_arr[mask]
            dmin_sub = dmin_arr[mask]
            color = PROJ_COLORS.get(proj_type, "#888888")
            marker = PROJ_MARKERS.get(proj_type, "o")
            ax.scatter(
                np.exp(s1_sub), dmin_sub,
                c=color, marker=marker, s=50, alpha=0.8,
                edgecolors="k", linewidths=0.3, label=proj_type, zorder=3,
            )

        # Regression line
        s_range = np.linspace(s1_arr.min() * 0.9, s1_arr.max() * 1.1, 200)
        exp_range = np.exp(s_range)
        y_ols = np.exp(fit.intercept + fit.slope * s_range)
        ax.plot(exp_range, y_ols, "k-", lw=2, alpha=0.7,
                label=f"OLS (slope={fit.slope:.2f})")
        y_h5 = fit.c_constrained * exp_range
        ax.plot(exp_range, y_h5, "r--", lw=1.5, alpha=0.8,
                label=f"slope=1 (c={fit.c_constrained:.2f})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$e^{S_1}$ (Von Neumann)")
        ax.set_ylabel(f"$D_{{\\min}}$ (Frobenius error <= {int(eps*100)}%)")
        ax.set_title(
            f"{model_id}\n"
            f"R2={fit.r_squared:.3f} | slope={fit.slope:.2f} | c={fit.c_constrained:.2f}"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            Path(out_dir) / f"scatter_s1_vs_dmin_{int(eps*100)}pct.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)


def _plot_by_layer(records, model_id, out_dir):
    """S1 by layer index, colored by projection type."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for proj_type in sorted(set(r["proj_type"] for r in records)):
        subset = [r for r in records if r["proj_type"] == proj_type]
        layers = [r["layer_idx"] for r in subset]
        s1_vals = [r["s1"] for r in subset]
        color = PROJ_COLORS.get(proj_type, "#888888")
        ax.plot(layers, s1_vals, "o-", color=color, label=proj_type,
                markersize=4, alpha=0.8)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("S1 (Von Neumann entropy, nats)")
    ax.set_title(f"{model_id}: S1 by layer")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        Path(out_dir) / "s1_by_layer.png", dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def print_cross_model_summary(results: list):
    """Print comparison table across models."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'N':>4} {'S1 mean':>8} {'R2(10%)':>8} {'slope':>7} {'Verdict'}")
    print(f"  {'-'*35} {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*15}")

    for r in results:
        fit10 = r["fits"].get("dmin_10pct")
        r2_str = f"{fit10.r_squared:.4f}" if fit10 else "N/A"
        slope_str = f"{fit10.slope:.4f}" if fit10 else "N/A"
        verdict_short = r["verdict"].split("(")[0].strip()
        print(
            f"  {r['model']:<35} {r['n_matrices']:>4} "
            f"{r['s1_mean']:>8.4f} {r2_str:>8} {slope_str:>7} {verdict_short}"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate Entropy-Compression Law")
    parser.add_argument(
        "models", nargs="*",
        default=["openai-community/gpt2"],
        help="HuggingFace model IDs to analyze",
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Base output directory",
    )
    args = parser.parse_args()

    results = []
    for model_id in args.models:
        try:
            result = analyze_model(model_id, args.output)
            results.append(result)
        except Exception as e:
            print(f"\nERROR analyzing {model_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if len(results) > 1:
        print_cross_model_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
