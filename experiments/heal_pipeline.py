"""End-to-end pipeline: allocate -> compress -> eval -> heal -> eval.

Tests the full healing workflow on GPT-2 Small.
"""

import gc
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_ID = "openai-community/gpt2"
BUDGET = 0.80
STRATEGY = "entropy"
BASE_DIR = Path("/Users/nicolascalderon/Documents/dev/entropy-lens")
RESULTS_DIR = BASE_DIR / "results" / "gpt2"
COMPRESSED_DIR = BASE_DIR / "compressed" / f"gpt2-{int(BUDGET*100)}pct-{STRATEGY}"
HEALED_DIR = BASE_DIR / "healed" / f"gpt2-{int(BUDGET*100)}pct-{STRATEGY}-healed"


def eval_ppl(model_path: str, label: str) -> float:
    """Evaluate perplexity on wikitext-2 test set."""
    from datasets import load_dataset

    logger.info("=== Evaluating PPL: %s ===", label)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
    )
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all text and tokenize
    text = "\n".join(t for t in ds["text"] if t.strip())
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]

    seq_len = input_ids.size(1)
    max_len = 512
    nlls = []

    for i in range(0, seq_len - max_len, max_len):
        chunk = input_ids[:, i : i + max_len]
        with torch.no_grad():
            outputs = model(input_ids=chunk, labels=chunk)
            nlls.append(outputs.loss.item())

    avg_nll = sum(nlls) / len(nlls)
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    elapsed = time.time() - t0
    logger.info("  %s PPL = %.2f (avg NLL = %.4f, %.1fs)", label, ppl, avg_nll, elapsed)

    del model
    gc.collect()
    return ppl


def main():
    t_total = time.time()

    # ── Step 1: Analyze (extract S1 values) ──────────────────────────
    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        logger.info("=== Step 1: Analyze (extract spectral stats) ===")
        from entropy_lens.cli import main as cli_main
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli_main, [
            "analyze", MODEL_ID,
            "-o", str(RESULTS_DIR),
            "--dtype", "float32",
        ])
        if result.exit_code != 0:
            logger.error("Analyze failed:\n%s", result.output)
            raise RuntimeError("Analyze failed")
        logger.info("Analyze done.")
    else:
        logger.info("=== Step 1: Skipping analyze (CSV exists) ===")

    # ── Step 2: Allocate ranks ───────────────────────────────────────
    # Need config.json for shape inference
    from entropy_lens.extract import _resolve_model_path
    model_dir = _resolve_model_path(MODEL_ID)
    config_path = Path(model_dir) / "config.json"

    alloc_path = RESULTS_DIR / f"alloc_{STRATEGY}_{int(BUDGET*100)}.json"
    logger.info("=== Step 2: Allocate (strategy=%s, budget=%.0f%%) ===", STRATEGY, BUDGET * 100)

    from entropy_lens.allocator import allocate_ranks
    alloc_result = allocate_ranks(
        csv_path=str(csv_path),
        budget_ratio=BUDGET,
        strategy=STRATEGY,
        config_path=str(config_path),
    )
    alloc_data = alloc_result.to_dict()
    with open(alloc_path, "w") as f:
        json.dump(alloc_data, f, indent=2)

    logger.info(
        "  Allocated %d layers, actual ratio=%.1f%%",
        len(alloc_result.ranks), alloc_result.actual_ratio * 100,
    )

    # ── Step 3: Compress ─────────────────────────────────────────────
    logger.info("=== Step 3: Compress ===")
    from entropy_lens.compress import compress_model

    errors = compress_model(
        model_path=MODEL_ID,
        ranks=alloc_result.ranks,
        output_path=str(COMPRESSED_DIR),
        dtype="float32",
        verify=True,
    )
    mean_err = sum(errors.values()) / len(errors) if errors else 0
    logger.info("  Compression done. Mean Frobenius error: %.4f", mean_err)

    # ── Step 4: Eval PPL (pre-healing) ───────────────────────────────
    ppl_baseline = eval_ppl(model_dir, "Baseline (uncompressed)")
    ppl_compressed = eval_ppl(str(COMPRESSED_DIR), "Compressed (no healing)")

    # ── Step 5: Heal ─────────────────────────────────────────────────
    logger.info("=== Step 5: Heal (LoRA fine-tune) ===")
    from entropy_lens.heal import heal_model

    heal_model(
        model_path=str(COMPRESSED_DIR),
        output_path=str(HEALED_DIR),
        dataset="wikitext",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_steps=500,
        batch_size=4,
        max_seq_len=512,
        dtype="float32",
    )

    # ── Step 6: Eval PPL (post-healing) ──────────────────────────────
    ppl_healed = eval_ppl(str(HEALED_DIR), "Healed")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("HEALING PIPELINE RESULTS (GPT-2 Small, budget=%.0f%%, strategy=%s)", BUDGET * 100, STRATEGY)
    logger.info("=" * 60)
    logger.info("  Baseline PPL:    %.2f", ppl_baseline)
    logger.info("  Compressed PPL:  %.2f  (%.1fx baseline)", ppl_compressed, ppl_compressed / ppl_baseline)
    logger.info("  Healed PPL:      %.2f  (%.1fx baseline)", ppl_healed, ppl_healed / ppl_baseline)
    logger.info("  Recovery:        %.1f%% of gap closed",
                100 * (ppl_compressed - ppl_healed) / (ppl_compressed - ppl_baseline)
                if ppl_compressed > ppl_baseline else 0)
    logger.info("  Total time:      %.0fs", elapsed)
    logger.info("=" * 60)

    # Save summary
    summary = {
        "model": MODEL_ID,
        "budget": BUDGET,
        "strategy": STRATEGY,
        "ppl_baseline": round(ppl_baseline, 2),
        "ppl_compressed": round(ppl_compressed, 2),
        "ppl_healed": round(ppl_healed, 2),
        "recovery_pct": round(
            100 * (ppl_compressed - ppl_healed) / (ppl_compressed - ppl_baseline)
            if ppl_compressed > ppl_baseline else 0, 1
        ),
        "heal_steps": 500,
        "lora_rank": 16,
        "total_time_s": round(elapsed, 1),
    }
    summary_path = HEALED_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
