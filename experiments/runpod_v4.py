"""RunPod experiment v4: Mistral 7B compress + heal + eval en régimen funcional.

Por qué existe este script (contexto de la corrida v3 del 2026-07-12):
  v3 corrió budget 80% con SVD naive. Resultado: entropy le ganó a uniform
  (compressed 45,250 vs 284,100; healed 6,626 vs 14,733), pero las PPLs
  absolutas son de modelo destruido (baseline 6.07). Dos causas observadas:
    1. Healing divergió/estancó: loss subió de 7.5 a 9.7 (uniform) con
       lr=2e-4. Un modelo sano en wikitext ronda loss ~2.
    2. Budget 80% con SVD naive es mucho más agresivo en Mistral 7B que
       en GPT-2 (45K-284K de PPL comprimida vs ~4K en GPT-2).
  v3 queda como dato del "régimen agresivo" (la ventaja de entropy se
  amplifica), pero el paper necesita la curva Pareto en régimen funcional.

Cambios vs v3:
  - Budgets suaves: 95% y 90% (buscar PPL healed < 10-15, modelo usable).
  - lr 2e-4 -> 1e-4 (evitar divergencia observada en v3).
  - steps 500 -> 1000 (la loss de v3 no había convergido).
  - Log de loss cada 50 steps (detectar divergencia temprano).
  - Resultados parciales a /workspace/results_v4.json tras cada config
    (resiliencia a crashes/OOM: no se pierde lo ya corrido).
  - Silencia el ruido HTTP de huggingface en el log.

Run (siempre con nohup para sobrevivir al cierre de la terminal web):
  cd /workspace/entropy-lens && git pull
  nohup python experiments/runpod_v4.py > /workspace/experiment_v4.log 2>&1 &
  tail -f /workspace/experiment_v4.log
"""
import time, torch, json, gc, logging, random
logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
from entropy_lens.allocator import allocate_ranks
from entropy_lens.compress import _svd_truncate, _build_canonical_to_statedict_map, _resolve_state_dict_key
from entropy_lens.extract import _resolve_model_path
from entropy_lens.arch.auto import detect_extractor
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "mistralai/Mistral-7B-v0.3"
CSV = "results/mistralai_Mistral-7B-v0.3/results.csv"
BUDGETS = [0.95, 0.90]
HEAL_STEPS = 1000
HEAL_LR = 1e-4
RESULTS_PATH = "/workspace/results_v4.json"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def save_partial(R):
    with open(RESULTS_PATH, "w") as f:
        json.dump(R, f, indent=2)
    print(f"    [saved partial results -> {RESULTS_PATH}]")


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


def heal(model, steps=HEAL_STEPS, accum=4, lr=HEAL_LR):
    from peft import LoraConfig, TaskType, get_peft_model
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, cfg); model.train()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    txt = "\n\n".join([t for t in ds["text"] if t.strip()])
    aids = tokenizer(txt, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"].squeeze(0)
    chunks = [aids[i:i+512] for i in range(0, len(aids)-512, 512)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(steps):
        total_loss = 0.0
        for _ in range(accum):
            b = chunks[random.randrange(len(chunks))].unsqueeze(0).to("cuda")
            loss = model(input_ids=b, labels=b).loss / accum
            loss.backward()
            total_loss += loss.item()
        opt.step(); opt.zero_grad()
        if (s+1) % 50 == 0: print(f"    step {s+1}/{steps} loss={total_loss:.3f}")
    model = model.merge_and_unload()
    model.gradient_checkpointing_disable()
    model.to("cpu"); torch.cuda.empty_cache(); gc.collect()
    return model


print("=== Baseline ===")
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
ppl_base = eval_ppl(m); del m; gc.collect(); torch.cuda.empty_cache()
print(f"Baseline PPL: {ppl_base:.2f}")
R = {"baseline": ppl_base,
     "config": {"heal_steps": HEAL_STEPS, "heal_lr": HEAL_LR, "budgets": BUDGETS}}
save_partial(R)

for bud in BUDGETS:
    print(f"\n=== Budget {int(bud*100)}% ===")
    for st in ["uniform", "entropy"]:
        t0 = time.time()
        print(f"  [{st}]")
        res = allocate_ranks(CSV, budget_ratio=bud, strategy=st)
        m = compress_cpu(res.ranks)
        pc = eval_ppl(m); print(f"    Compressed PPL: {pc:.2f}")
        m = heal(m)
        ph = eval_ppl(m); print(f"    Healed PPL: {ph:.2f}")
        R[f"{int(bud*100)}pct-{st}"] = {"compressed": pc, "healed": ph, "minutes": round((time.time()-t0)/60, 1)}
        save_partial(R)
        del m; gc.collect(); torch.cuda.empty_cache()

print("\n=== FINAL RESULTS ===")
print(json.dumps(R, indent=2))
