"""CLI entry point for entropy-lens."""

import csv
import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from .extract import extract_svdvals_streaming
from .law import FitResult, evaluate_go_nogo, fit_entropy_law
from .report import save_report
from .spectral import (
    compute_alpha_hill,
    compute_dmin,
    compute_participation_ratio,
    compute_s1,
    compute_s2,
)


@click.group()
def main():
    """entropy-lens: Validate the Entropy-Compression Law across LLM architectures."""


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", default="results", help="Output directory.")
@click.option(
    "--epsilons",
    default="0.05,0.10,0.20,0.50",
    help="Comma-separated Frobenius error thresholds.",
)
@click.option("--dtype", default="float32", help="Compute dtype (float32 or float64).")
def analyze(model_path: str, output: str, epsilons: str, dtype: str):
    """Analyze a model: extract svdvals, compute S1/D_min, fit law, report."""
    eps_list = [float(e) for e in epsilons.split(",")]

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Analyzing: {model_path}")
    click.echo(f"Epsilons: {eps_list}")
    click.echo(f"Output: {out_dir}")
    click.echo()

    records = []
    for name, sv in tqdm(
        extract_svdvals_streaming(model_path, dtype=dtype),
        desc="Processing layers",
    ):
        s1 = compute_s1(sv)
        s2 = compute_s2(sv)
        pr = compute_participation_ratio(sv)
        alpha = compute_alpha_hill(sv)

        row = {
            "name": name,
            "shape": f"{len(sv)}",
            "rank": len(sv),
            "s1": round(s1, 6),
            "s2": round(s2, 6),
            "pr": round(pr, 2),
            "alpha_hill": round(alpha, 4) if not np.isnan(alpha) else None,
        }

        for eps in eps_list:
            d = compute_dmin(sv, eps)
            row[f"dmin_{int(eps * 100)}pct"] = d

        records.append(row)
        tqdm.write(
            f"  {name}: S1={s1:.3f} S2={s2:.3f} "
            + " ".join(f"D({int(e*100)}%)={compute_dmin(sv, e)}" for e in eps_list)
        )

    if not records:
        click.echo("ERROR: No matrices extracted. Check model path.", err=True)
        sys.exit(1)

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    click.echo(f"\nCSV saved: {csv_path}")

    # Fit law for each epsilon
    s1_arr = np.array([r["s1"] for r in records])
    fit_results = {}

    click.echo("\n" + "=" * 60)
    click.echo("ENTROPY-COMPRESSION LAW FIT")
    click.echo("=" * 60)

    for eps in eps_list:
        col = f"dmin_{int(eps * 100)}pct"
        dmin_arr = np.array([r[col] for r in records], dtype=float)
        try:
            fit = fit_entropy_law(s1_arr, dmin_arr)
            fit_results[col] = fit
            click.echo(
                f"\n  eps={int(eps*100)}%: R2={fit.r_squared:.4f} "
                f"slope={fit.slope:.4f} c={fit.c_constrained:.3f} "
                f"RMSE_log={fit.rmse_log:.4f} (n={fit.n})"
            )
        except ValueError as e:
            click.echo(f"\n  eps={int(eps*100)}%: FAILED ({e})")

    # Go/no-go
    verdict = evaluate_go_nogo(fit_results)
    click.echo(f"\n{'=' * 60}")
    click.echo(f"VERDICT: {verdict}")
    click.echo(f"{'=' * 60}")

    # Summary stats
    s1_mean = float(np.mean(s1_arr))
    s1_std = float(np.std(s1_arr))
    click.echo(f"\nS1 mean: {s1_mean:.4f} +/- {s1_std:.4f} nats")
    click.echo(f"Matrices analyzed: {len(records)}")

    # Save JSON report
    report_path = save_report(model_path, records, fit_results, verdict, str(out_dir))
    click.echo(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
