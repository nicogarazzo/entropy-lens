"""CLI entry point for entropy-lens."""

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from .allocator import allocate_ranks
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


@main.command()
@click.argument("csv_path")
@click.option("--budget", "-b", required=True, type=float, help="Parameter budget ratio (0-1].")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["uniform", "proportional", "entropy"]),
    default="entropy",
    help="Allocation strategy.",
)
@click.option("--config", "-c", default=None, help="HuggingFace config.json for shape inference.")
@click.option("--output", "-o", default=None, help="Output JSON file (default: stdout).")
def allocate(csv_path: str, budget: float, strategy: str, config: str, output: str):
    """Allocate SVD truncation ranks under a parameter budget.

    Reads an entropy-lens results CSV and assigns a rank D_i to each layer,
    optimized according to the chosen strategy.
    """
    try:
        result = allocate_ranks(
            csv_path=csv_path,
            budget_ratio=budget,
            strategy=strategy,
            config_path=config,
        )
    except (ValueError, FileNotFoundError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    data = result.to_dict()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"Allocation saved: {out_path}")
    else:
        click.echo(json.dumps(data, indent=2))

    click.echo(
        f"\nStrategy: {strategy} | Budget: {budget:.1%} | "
        f"Actual: {result.actual_ratio:.1%} | Layers: {len(result.ranks)}",
        err=True,
    )


@main.command()
@click.argument("model_path")
@click.option("--ranks", "-r", required=True, help="Path to allocation JSON (from 'allocate' command).")
@click.option("--output", "-o", required=True, help="Output directory for compressed model.")
@click.option("--dtype", default="float16", help="Model dtype (float16, bfloat16, float32).")
@click.option("--no-verify", is_flag=True, help="Skip per-layer error verification.")
def compress(model_path: str, ranks: str, output: str, dtype: str, no_verify: bool):
    """Compress a model using SVD at the ranks from an allocation JSON.

    Loads the model, applies truncated SVD to each layer at the specified rank,
    and saves the result as a standard HuggingFace model.
    """
    import logging

    from .compress import compress_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load ranks from JSON
    ranks_path = Path(ranks)
    if not ranks_path.exists():
        click.echo(f"ERROR: Ranks file not found: {ranks_path}", err=True)
        sys.exit(1)

    with open(ranks_path) as f:
        alloc_data = json.load(f)

    # The allocate command saves {"ranks": {...}, "strategy": ..., ...}
    # Accept both the full allocation JSON and a plain {name: rank} dict.
    if "ranks" in alloc_data and isinstance(alloc_data["ranks"], dict):
        rank_dict = alloc_data["ranks"]
    else:
        rank_dict = alloc_data

    click.echo(f"Model: {model_path}")
    click.echo(f"Ranks: {len(rank_dict)} layers from {ranks_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Dtype: {dtype}")
    click.echo()

    errors = compress_model(
        model_path=model_path,
        ranks=rank_dict,
        output_path=output,
        dtype=dtype,
        verify=not no_verify,
    )

    if errors:
        mean_err = sum(errors.values()) / len(errors)
        max_err = max(errors.values())
        click.echo(f"\nReconstruction errors: mean={mean_err:.4f}, max={max_err:.4f}")


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", required=True, help="Output directory for healed model.")
@click.option("--dataset", "-d", default="wikitext", help="Training dataset (wikitext, slim_pajama, alpaca).")
@click.option("--steps", default=500, type=int, help="Number of training steps.")
@click.option("--lr", default=2e-4, type=float, help="Learning rate.")
@click.option("--lora-rank", default=16, type=int, help="LoRA adapter rank.")
@click.option("--lora-alpha", default=32, type=int, help="LoRA alpha scaling.")
@click.option("--batch-size", default=4, type=int, help="Micro-batch size.")
@click.option("--max-seq-len", default=512, type=int, help="Sequence length for training.")
@click.option("--dtype", default="float32", help="Model dtype (float16, bfloat16, float32).")
def heal(
    model_path: str,
    output: str,
    dataset: str,
    steps: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
):
    """Heal a compressed model with LoRA fine-tuning.

    Takes a compressed model and does a short fine-tune to recover accuracy
    lost during SVD truncation. Uses next-token prediction with LoRA adapters.
    """
    import logging

    from .heal import heal_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    click.echo(f"Model: {model_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Steps: {steps} | LR: {lr} | LoRA rank: {lora_rank}")
    click.echo()

    heal_model(
        model_path=model_path,
        output_path=output,
        dataset=dataset,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        num_steps=steps,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )

    click.echo("\nHealing complete.")


if __name__ == "__main__":
    main()
