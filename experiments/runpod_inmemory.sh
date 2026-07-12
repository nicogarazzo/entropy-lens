#!/bin/bash
set -e
cd /workspace/entropy-lens && git pull

cat > /tmp/run_inmemory.py << 'PYEOF'
import sys, time, torch, json, gc, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from entropy_lens.allocator import allocate_ranks
from entropy_lens.compress import _svd_truncate, _build_canonical_to_statedict_map, _resolve_state_dict_key
from entropy_lens.extract import _resolve_model_path
from entropy_lens.arch.auto import detect_extractor
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "mistralai/Mistral-7B-v0.3"
CSV = "results/mistralai_Mistral-7B-v0.3/results.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def eval_ppl(model, device="cuda"):
    model.eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024*100)
    ids = enc.input_ids.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(0, min(ids.size(1), 1024*100), 1024):
            chunk = ids[:, i:i+1024]
            if chunk.size(1) < 2:
                continue
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.float().item())
    return float(torch.exp(torch.tensor(nlls).mean()))

def compress_inplace(model, ranks):
    model_dir = _resolve_model_path(MODEL)
    extractor = detect_extractor(model_dir)
    canonical_map = _build_canonical_to_statedict_map(extractor)
    sd = model.state_dict()
    sd_keys = set(sd.keys())
    count = 0
    for canonical, st_key in canonical_map.items():
        if canonical not in ranks:
            continue
        rank = ranks[canonical]
        actual_key = _resolve_state_dict_key(st_key, sd_keys)
        if actual_key is None:
            continue
        w = sd[actual_key]
        if rank < min(w.shape):
            sd[actual_key] = _svd_truncate(w, rank)
            count += 1
    model.load_state_dict(sd, strict=True)
    del sd
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    Compressed {count} layers in-place")
    return model

def heal_inplace(model, num_steps=500, lr=2e-4, lora_rank=32):
    from peft import LoraConfig, TaskType, get_peft_model
    target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=64, lora_dropout=0.0, target_modules=target, bias="none")
    model = get_peft_model(model, config)
    model.train()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    all_ids = enc["input_ids"].squeeze(0)
    seq_len = 512
    chunks = [all_ids[i:i+seq_len] for i in range(0, len(all_ids)-seq_len, seq_len)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    import random
    device = next(model.parameters()).device

    for step in range(num_steps):
        idx = random.sample(range(len(chunks)), min(4, len(chunks)))
        batch = torch.stack([chunks[i] for i in idx]).to(device)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (step+1) % 100 == 0:
            print(f"    step {step+1}/{num_steps} loss={out.loss.item():.4f}")

    model = model.merge_and_unload()
    gc.collect()
    torch.cuda.empty_cache()
    return model

# === MAIN ===
print("Loading baseline model...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
ppl_base = eval_ppl(model)
print(f"Baseline PPL: {ppl_base:.2f}")
del model; gc.collect(); torch.cuda.empty_cache()

all_results = {"baseline": ppl_base}

for budget in [0.90, 0.80, 0.70]:
    print(f"\n=== Budget {int(budget*100)}% ===")
    for strat in ["uniform", "entropy"]:
        tag = f"{int(budget*100)}pct-{strat}"
        print(f"  [{strat}]")

        result = allocate_ranks(CSV, budget_ratio=budget, strategy=strat)

        # Load fresh model, compress in memory, eval
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
        model = compress_inplace(model, result.ranks)
        ppl_comp = eval_ppl(model)
        print(f"    Compressed PPL: {ppl_comp:.2f}")

        # Heal in place, eval
        model = heal_inplace(model, num_steps=500)
        ppl_heal = eval_ppl(model)
        print(f"    Healed PPL: {ppl_heal:.2f}")

        all_results[tag] = {"compressed": ppl_comp, "healed": ppl_heal}
        del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== FINAL RESULTS ===")
print(json.dumps(all_results, indent=2))
PYEOF

python /tmp/run_inmemory.py
