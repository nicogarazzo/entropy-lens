#!/bin/bash
# Step 2: Compress + Heal + Eval on Mistral 7B
# Run this after the initial script completed Step 1 successfully.
set -e

pip install -q datasets==3.6.0

cd /workspace/entropy-lens

cat > /tmp/step2.py << 'PYEOF'
import sys, time, torch, json, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from entropy_lens.allocator import allocate_ranks
from entropy_lens.compress import compress_model
from entropy_lens.heal import heal_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

def eval_ppl(model_path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024*100)
    ids = enc.input_ids.to(device)
    nlls = []
    for i in range(0, min(ids.size(1), 1024*100), 1024):
        chunk = ids[:, i:i+1024]
        if chunk.size(1) < 2:
            continue
        with torch.no_grad():
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.float().item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()))
    del model
    torch.cuda.empty_cache()
    return ppl

print("Baseline PPL...")
ppl_base = eval_ppl("mistralai/Mistral-7B-v0.3")
print(f"Baseline PPL: {ppl_base:.2f}")

all_results = {"baseline": ppl_base}

for budget in [0.90, 0.80, 0.70]:
    print(f"\n=== Budget {int(budget*100)}% ===")
    for strat in ["uniform", "entropy"]:
        tag = f"{int(budget*100)}pct-{strat}"
        print(f"  [{strat}] Allocating + Compressing...")
        result = allocate_ranks("results/mistralai_Mistral-7B-v0.3/results.csv", budget_ratio=budget, strategy=strat)
        comp_path = f"compressed/mistral-{tag}/"
        compress_model("mistralai/Mistral-7B-v0.3", result.ranks, comp_path, dtype="float16", verify=False)
        ppl_comp = eval_ppl(comp_path)
        print(f"  [{strat}] Compressed PPL: {ppl_comp:.2f}")

        print(f"  [{strat}] Healing (500 steps)...")
        heal_path = f"healed/mistral-{tag}/"
        heal_model(comp_path, heal_path, dataset="wikitext", num_steps=500, lora_rank=32, batch_size=4, max_seq_len=512, dtype="float16")
        ppl_heal = eval_ppl(heal_path)
        print(f"  [{strat}] Healed PPL: {ppl_heal:.2f}")
        all_results[tag] = {"compressed": ppl_comp, "healed": ppl_heal}

print("\n=== FINAL RESULTS ===")
print(json.dumps(all_results, indent=2))
PYEOF

python /tmp/step2.py
