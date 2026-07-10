#!/usr/bin/env python3
"""
Compare S1 (Von Neumann entropy) vs alpha_hill (PL_Alpha_Hill from AlphaPruning)
as predictors of D_min across all weight matrices in a model.

Reads results.csv from a model's output directory and produces:
  - Comparative R² table for each epsilon
  - Scatter: S1 vs alpha_hill (metric correlation)
  - Scatter: alpha_hill vs D_min with regression (for each epsilon)
  - Summary JSON

Usage:
    python experiments/compare_alpha_pruning.py results/mistralai_Mistral-7B-v0.3/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_csv(csv_path: Path) -> list[dict]:
    """Load results CSV into list of dicts with proper types."""
    import csv
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {}
            for k, v in row.items():
                if v == "" or v == "None":
                    rec[k] = None
                else:
                    try:
                        rec[k] = int(v)
                    except ValueError:
                        try:
                            rec[k] = float(v)
                        except ValueError:
                            rec[k] = v
            records.append(rec)
    return records


def fit_log_predictor(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit log(y) = a + b*x via OLS. Return slope, intercept, R², p-value."""
    log_y = np.log(y.astype(float))
    result = stats.linregress(x, log_y)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "n": len(x),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare S1 vs alpha_hill as D_min predictors"
    )
    parser.add_argument("results_dir", help="Directory with results.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    records = load_csv(csv_path)
    print(f"Loaded {len(records)} matrices from {csv_path}")

    # Filter records with valid alpha_hill
    valid = [r for r in records if r.get("alpha_hill") is not None]
    print(f"Records with valid alpha_hill: {len(valid)}/{len(records)}")

    if len(valid) < 10:
        print("ERROR: Not enough valid records for comparison", file=sys.stderr)
        sys.exit(1)

    s1_arr = np.array([r["s1"] for r in valid])
    alpha_arr = np.array([r["alpha_hill"] for r in valid])

    # Detect epsilon columns
    eps_cols = sorted([k for k in valid[0].keys() if k.startswith("dmin_")])
    print(f"Epsilon columns: {eps_cols}")

    # ---------------------------------------------------------------
    # 1. Comparative R² table
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("COMPARISON: S1 vs alpha_hill as D_min predictors")
    print(f"{'='*70}")
    print(f"  {'epsilon':<12} {'R²(S1)':>10} {'R²(alpha)':>10} {'Winner':>10} {'Delta':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    comparison = {}
    for col in eps_cols:
        dmin_arr = np.array([r[col] for r in valid], dtype=float)

        # Skip if all D_min are the same (degenerate)
        if np.std(dmin_arr) == 0:
            continue

        fit_s1 = fit_log_predictor(s1_arr, dmin_arr)
        fit_alpha = fit_log_predictor(alpha_arr, dmin_arr)

        r2_s1 = fit_s1["r_squared"]
        r2_alpha = fit_alpha["r_squared"]
        delta = r2_s1 - r2_alpha
        winner = "S1" if delta > 0 else "alpha_hill"

        eps_pct = col.replace("dmin_", "").replace("pct", "%")
        print(f"  {eps_pct:<12} {r2_s1:>10.4f} {r2_alpha:>10.4f} {winner:>10} {delta:>+10.4f}")

        comparison[col] = {
            "fit_s1": fit_s1,
            "fit_alpha": fit_alpha,
            "r2_s1": r2_s1,
            "r2_alpha": r2_alpha,
            "winner": winner,
            "delta_r2": delta,
        }

    # ---------------------------------------------------------------
    # 2. Correlation between S1 and alpha_hill
    # ---------------------------------------------------------------
    corr = stats.pearsonr(s1_arr, alpha_arr)
    spearman = stats.spearmanr(s1_arr, alpha_arr)
    print(f"\n  S1 vs alpha_hill correlation:")
    print(f"    Pearson r  = {corr[0]:.4f} (p={corr[1]:.2e})")
    print(f"    Spearman rho = {spearman.correlation:.4f} (p={spearman.pvalue:.2e})")

    # ---------------------------------------------------------------
    # 3. Plots
    # ---------------------------------------------------------------

    # 3a. S1 vs alpha_hill scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(s1_arr, alpha_arr, c="#2166ac", s=30, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("S1 (Von Neumann entropy, nats)")
    ax.set_ylabel("alpha_hill (PL_Alpha_Hill)")
    ax.set_title(
        f"S1 vs alpha_hill ({len(valid)} matrices)\n"
        f"Pearson r={corr[0]:.3f}, Spearman rho={spearman.correlation:.3f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "s1_vs_alpha_hill.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {results_dir / 's1_vs_alpha_hill.png'}")

    # 3b. alpha_hill vs D_min scatter (one per epsilon)
    for col in eps_cols:
        dmin_arr = np.array([r[col] for r in valid], dtype=float)
        if np.std(dmin_arr) == 0:
            continue

        comp = comparison[col]
        fit_a = comp["fit_alpha"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(alpha_arr, dmin_arr, c="#b2182b", s=30, alpha=0.6,
                   edgecolors="k", linewidths=0.3)

        # Regression line
        x_range = np.linspace(alpha_arr.min() * 0.95, alpha_arr.max() * 1.05, 200)
        y_pred = np.exp(fit_a["intercept"] + fit_a["slope"] * x_range)
        ax.plot(x_range, y_pred, "k-", lw=2, alpha=0.7,
                label=f"OLS (R²={fit_a['r_squared']:.3f})")

        ax.set_yscale("log")
        eps_pct = col.replace("dmin_", "").replace("pct", "%")
        ax.set_xlabel("alpha_hill (PL_Alpha_Hill)")
        ax.set_ylabel(f"D_min (eps={eps_pct})")
        ax.set_title(
            f"alpha_hill vs D_min (eps={eps_pct})\n"
            f"R²(alpha)={comp['r2_alpha']:.4f} vs R²(S1)={comp['r2_s1']:.4f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fname = f"alpha_vs_dmin_{col.replace('dmin_', '')}.png"
        fig.savefig(results_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {results_dir / fname}")

    # 3c. Side-by-side R² bar chart
    eps_labels = []
    r2_s1_vals = []
    r2_alpha_vals = []
    for col in eps_cols:
        if col in comparison:
            eps_labels.append(col.replace("dmin_", "").replace("pct", "%"))
            r2_s1_vals.append(comparison[col]["r2_s1"])
            r2_alpha_vals.append(comparison[col]["r2_alpha"])

    x = np.arange(len(eps_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, r2_s1_vals, width, label="S1 (Von Neumann)", color="#2166ac")
    bars2 = ax.bar(x + width/2, r2_alpha_vals, width, label="alpha_hill (AlphaPruning)", color="#b2182b")

    ax.set_ylabel("R²")
    ax.set_title("S1 vs alpha_hill: Predictive Power for D_min")
    ax.set_xticks(x)
    ax.set_xticklabels([f"eps={e}" for e in eps_labels])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(results_dir / "r2_comparison_s1_vs_alpha.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {results_dir / 'r2_comparison_s1_vs_alpha.png'}")

    # ---------------------------------------------------------------
    # 4. Save summary JSON
    # ---------------------------------------------------------------
    summary = {
        "n_matrices": len(valid),
        "n_total": len(records),
        "correlation_s1_alpha": {
            "pearson_r": corr[0],
            "pearson_p": corr[1],
            "spearman_rho": spearman.correlation,
            "spearman_p": spearman.pvalue,
        },
        "comparison_by_epsilon": {},
    }
    for col, comp in comparison.items():
        summary["comparison_by_epsilon"][col] = {
            "r2_s1": comp["r2_s1"],
            "r2_alpha": comp["r2_alpha"],
            "winner": comp["winner"],
            "delta_r2": comp["delta_r2"],
        }

    # Overall verdict
    s1_wins = sum(1 for c in comparison.values() if c["winner"] == "S1")
    alpha_wins = len(comparison) - s1_wins
    avg_delta = np.mean([c["delta_r2"] for c in comparison.values()])

    if s1_wins == len(comparison):
        verdict = f"S1 DOMINATES: wins all {len(comparison)} epsilon thresholds (avg delta R²={avg_delta:+.4f})"
    elif s1_wins > alpha_wins:
        verdict = f"S1 SUPERIOR: wins {s1_wins}/{len(comparison)} thresholds (avg delta R²={avg_delta:+.4f})"
    elif alpha_wins > s1_wins:
        verdict = f"ALPHA SUPERIOR: wins {alpha_wins}/{len(comparison)} thresholds (avg delta R²={avg_delta:+.4f})"
    else:
        verdict = f"TIE: each wins {s1_wins}/{len(comparison)} thresholds (avg delta R²={avg_delta:+.4f})"

    summary["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    json_path = results_dir / "comparison_s1_vs_alpha.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
