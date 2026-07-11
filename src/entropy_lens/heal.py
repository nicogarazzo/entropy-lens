"""Healing fine-tune for SVD-compressed models via LoRA.

After truncated SVD compression, the model's weight matrices are low-rank
approximations of the originals. The norms, embeddings, and lm_head still
expect the original weight distributions. A short fine-tune ("healing")
lets these untouched parameters adapt to the new truncated weights.

We use LoRA (Low-Rank Adaptation) instead of full fine-tuning because:
  - Much lower memory: only adapter parameters are trained.
  - Faster convergence for this use case.
  - After training, adapters merge back into the base weights, producing
    a standard model with no inference overhead.

References:
  - CompactifAI (arXiv:2407.14117): "healing" fine-tune post-compression.
  - Rethinking TD (arXiv:2606.03465): confirms healing is obligatory.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ── Target modules per architecture family ───────────────────────────
# GPT-2 uses Conv1D (peft handles it), but the module names are different.
_TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense",
            "fc1", "fc2"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"],
}


def _detect_target_modules(config) -> list[str]:
    """Pick LoRA target modules from the model's config.model_type."""
    model_type = getattr(config, "model_type", "").lower()
    for family, modules in _TARGET_MODULES.items():
        if family in model_type:
            return modules
    # Fallback: LLaMA-style names are the most common
    logger.warning(
        "Unknown model_type '%s', falling back to LLaMA-style target modules.",
        model_type,
    )
    return _TARGET_MODULES["llama"]


def _load_dataset(dataset_name: str, tokenizer, max_seq_len: int):
    """Load and tokenize a dataset for next-token prediction.

    Returns a list of input_ids tensors, each of length max_seq_len.
    We concatenate all text and chunk into fixed-length sequences.
    This avoids padding waste and is standard for causal LM training.
    """
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    elif dataset_name == "slim_pajama":
        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
        )
        # Take a small subset for healing
        ds = ds.take(5000)
        text_column = "text"
    elif dataset_name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Concatenate all text
    logger.info("Tokenizing dataset '%s'...", dataset_name)
    all_text = []
    for example in ds:
        t = example[text_column]
        if t and t.strip():
            all_text.append(t)

    full_text = "\n".join(all_text)

    # Tokenize the whole thing at once (wikitext-2 is small enough)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    all_ids = encoded["input_ids"].squeeze(0)
    logger.info("Total tokens: %d", len(all_ids))

    # Chunk into sequences of max_seq_len
    chunks = []
    for i in range(0, len(all_ids) - max_seq_len, max_seq_len):
        chunks.append(all_ids[i : i + max_seq_len])

    logger.info("Created %d chunks of %d tokens each.", len(chunks), max_seq_len)
    return chunks


def heal_model(
    model_path: str,
    output_path: str,
    dataset: str = "wikitext",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_steps: int = 500,
    batch_size: int = 4,
    max_seq_len: int = 512,
    dtype: str = "float32",
    log_every: int = 50,
) -> AutoModelForCausalLM:
    """Heal a compressed model with LoRA fine-tuning.

    Args:
        model_path: path to the compressed model directory.
        output_path: where to save the healed model.
        dataset: training dataset name ("wikitext", "slim_pajama", "alpaca").
        lora_rank: rank of LoRA adapters.
        lora_alpha: LoRA alpha scaling factor.
        learning_rate: AdamW learning rate.
        num_steps: number of training steps.
        batch_size: micro-batch size.
        max_seq_len: sequence length for training chunks.
        dtype: model loading dtype.
        log_every: log loss every N steps.

    Returns:
        The healed model (also saved to output_path).
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load compressed model ─────────────────────────────────────
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float32)

    logger.info("Loading compressed model from %s (dtype=%s)", model_path, dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # ── 2. Apply LoRA ────────────────────────────────────────────────
    target_modules = _detect_target_modules(model.config)
    logger.info("LoRA target modules: %s", target_modules)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,  # No dropout for healing (short training)
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: %d trainable / %d total (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    # ── 3. Load dataset ──────────────────────────────────────────────
    chunks = _load_dataset(dataset, tokenizer, max_seq_len)
    if not chunks:
        raise RuntimeError("Dataset produced zero training chunks.")

    # ── 4. Training loop ─────────────────────────────────────────────
    device = "cpu"  # M1 MPS has issues with some ops; CPU is safe
    model.to(device)
    model.train()

    # AdamW with no weight decay on LoRA params (they're small enough)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    # Simple linear warmup for 10% of steps
    warmup_steps = max(1, num_steps // 10)

    logger.info(
        "Starting healing: %d steps, batch_size=%d, lr=%.1e, warmup=%d",
        num_steps, batch_size, learning_rate, warmup_steps,
    )

    total_loss = 0.0
    step = 0
    epoch = 0
    chunk_idx = 0
    t_start = time.time()

    while step < num_steps:
        epoch += 1
        # Shuffle chunks at start of each epoch
        import random
        indices = list(range(len(chunks)))
        random.shuffle(indices)

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            if step >= num_steps:
                break

            # Build batch
            batch_indices = indices[i : i + batch_size]
            input_ids = torch.stack([chunks[j] for j in batch_indices]).to(device)

            # Linear warmup schedule
            if step < warmup_steps:
                lr_scale = (step + 1) / warmup_steps
            else:
                lr_scale = 1.0
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate * lr_scale

            # Forward pass: next-token prediction
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            if step % log_every == 0 or step == 1:
                avg_loss = total_loss / step
                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                logger.info(
                    "  step %d/%d | loss=%.4f | avg_loss=%.4f | "
                    "%.1f steps/s | ETA %.0fs",
                    step, num_steps, loss.item(), avg_loss,
                    steps_per_sec, eta,
                )

        chunk_idx = 0  # Reset for next epoch
        logger.info("  Epoch %d completed (step %d/%d)", epoch, step, num_steps)

    elapsed = time.time() - t_start
    final_avg_loss = total_loss / max(step, 1)
    logger.info(
        "Healing done: %d steps in %.1fs (%.2f steps/s), final avg loss=%.4f",
        step, elapsed, step / elapsed, final_avg_loss,
    )

    # ── 5. Merge LoRA back into base model ───────────────────────────
    logger.info("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()

    # ── 6. Save healed model ─────────────────────────────────────────
    logger.info("Saving healed model to %s", output_path)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save healing metadata
    metadata = {
        "source_model": model_path,
        "dataset": dataset,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "dtype": dtype,
        "final_avg_loss": round(final_avg_loss, 6),
        "training_time_seconds": round(elapsed, 1),
        "trainable_params": trainable,
        "total_params": total,
    }
    meta_path = out_dir / "healing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Healing metadata saved to %s", meta_path)

    gc.collect()
    return model
