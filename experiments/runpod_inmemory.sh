#!/bin/bash
set -e
pip install -q peft datasets
cd /workspace/entropy-lens && git pull && pip install -q -e .

cat > /tmp/run_v3.py << 'PYEOF'
import sys, time, torch, json, gc, logging, random
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

def eval_ppl(model):
    model.to("cuda").eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024*100).input_ids.to("cuda")
    nlls = []
    with torch.no_grad():
        for i in range(0, min(ids.size(1), 1024*100), 1024):
            c = ids[:, i:i+1024]
            if c.size(1) < 2: continue
            nlls.append(model(c, labels=c).loss.float().item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()))
    model.to("cpu"); torch.cuda.empty_cache()
    return ppl

def compress_cpu(ranks):
    print("    Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model_dir = _resolve_model_path(MODEL)
    ext = detect_extractor(model_dir)
    cmap = _build_canonical_to_statedict_map(ext)
    sd = model.state_dict()
    sdk = set(sd.keys())
    n = 0
    for can, stk in cmap.items():
        if can not in ranks: continue
        ak = _resolve_state_dict_key(stk, sdk)
        if ak is None: continue
        r = ranks[can]
        if r < min(sd[ak].shape):
            sd[ak] = _svd_truncate(sd[ak], r)
            n += 1
    model.load_state_dict(sd); del sd; gc.collect()
    print(f"    Compressed {n} layers on CPU")
    return model

def heal(model, steps=500):
    from peft import LoraConfig, TaskType, get_peft_model
    model.to("cuda")
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, cfg); model.train()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    txt = "\n\n".join([t for t in ds["text"] if t.strip()])
    aids = tokenizer(txt, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"].squeeze(0)
    chunks = [aids[i:i+512] for i in range(0, len(aids)-512, 512)]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for s in range(steps):
        b = torch.stack(random.sample(chunks, 4)).to("cuda")
        loss = model(input_ids=b, labels=b).loss
        loss.backward(); opt.step(); opt.zero_grad()
        if (s+1) % 100 == 0: print(f"    step {s+1}/{steps} loss={loss.item():.3f}")
    model = model.merge_and_unload()
    model.to("cpu"); torch.cuda.empty_cache(); gc.collect()
    return model

print("=== Baseline ===")
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
ppl_base = eval_ppl(m); del m; gc.collect(); torch.cuda.empty_cache()
print(f"Baseline PPL: {ppl_base:.2f}")
R = {"baseline": ppl_base}

for bud in [0.80]:
    print(f"\n=== Budget {int(bud*100)}% ===")
    for st in ["uniform", "entropy"]:
        print(f"  [{st}]")
        res = allocate_ranks(CSV, budget_ratio=bud, strategy=st)
        m = compress_cpu(res.ranks)
        pc = eval_ppl(m); print(f"    Compressed PPL: {pc:.2f}")
        m = heal(m, steps=500)
        ph = eval_ppl(m); print(f"    Healed PPL: {ph:.2f}")
        R[f"{int(bud*100)}pct-{st}"] = {"compressed": pc, "healed": ph}
        del m; gc.collect(); torch.cuda.empty_cache()

print("\n=== FINAL RESULTS ===")
print(json.dumps(R, indent=2))
PYEOF

python /tmp/run_v3.py
