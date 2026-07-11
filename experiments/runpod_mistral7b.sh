#!/bin/bash
# =============================================================================
# Run this script on a RunPod L4 (24GB VRAM, $0.39/hr)
# Estimated time: ~2-3 hours. Estimated cost: ~$1-2.
#
# What it does:
#   1. Installs entropy-lens from GitHub
#   2. Analyzes Mistral 7B (S₁ per layer)
#   3. Compresses at 3 budgets × 2 strategies
#   4. Heals with LoRA (500 steps each)
#   5. Evaluates perplexity on WikiText-2
#   6. Saves all results to /workspace/results/
#
# To run: paste this entire script into the RunPod terminal.
# =============================================================================

set -e

echo "=== Setting up environment ==="
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers safetensors datasets peft scipy matplotlib click tqdm huggingface-hub
pip install -q lm-eval

echo "=== Cloning entropy-lens ==="
cd /workspace
git clone https://github.com/nicogarazzo/entropy-lens.git
cd entropy-lens
pip install -e .

echo "=== Step 1: Analyze Mistral 7B ==="
python experiments/validate_7b.py mistralai/Mistral-7B-v0.3

echo "=== Step 2: Compress + Heal + Eval ==="
python -u -c "
import sys, time, logging, torch, json
logging.basicConfig(level=logging.INFO, format='%(message)s')

from entropy_lens.allocator import allocate_ranks
from entropy_lens.compress import compress_model
from entropy_lens.heal import heal_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')

def eval_ppl(model_path, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024*100)
    ids = enc.input_ids.to(device)
    nlls = []
    for i in range(0, min(ids.size(1), 1024*100), 1024):
        chunk = ids[:, i:i+1024]
        if chunk.size(1) < 2: continue
        with torch.no_grad():
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.float().item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()))
    del model; torch.cuda.empty_cache()
    return ppl

print('Baseline PPL...')
ppl_base = eval_ppl('mistralai/Mistral-7B-v0.3')
print(f'Baseline PPL: {ppl_base:.2f}')

all_results = {'baseline': ppl_base}

for budget in [0.90, 0.80, 0.70]:
    print(f'\n=== Budget {int(budget*100)}% ===')
    for strat in ['uniform', 'entropy']:
        tag = f'{int(budget*100)}pct-{strat}'
        print(f'  [{strat}] Allocating...')
        result = allocate_ranks('results/mistralai_Mistral-7B-v0.3/results.csv', budget_ratio=budget, strategy=strat)

        print(f'  [{strat}] Compressing...')
        comp_path = f'compressed/mistral-{tag}/'
        compress_model('mistralai/Mistral-7B-v0.3', result.ranks, comp_path, dtype='float16', verify=False)
        ppl_comp = eval_ppl(comp_path)
        print(f'  [{strat}] Compressed PPL: {ppl_comp:.2f}')

        print(f'  [{strat}] Healing (500 steps LoRA)...')
        heal_path = f'healed/mistral-{tag}/'
        heal_model(comp_path, heal_path, dataset='wikitext', num_steps=500,
                   lora_rank=32, batch_size=4, max_seq_len=512, dtype='float16')
        ppl_heal = eval_ppl(heal_path)
        print(f'  [{strat}] Healed PPL: {ppl_heal:.2f}')

        all_results[tag] = {'compressed': ppl_comp, 'healed': ppl_heal}

# Save results
with open('/workspace/results/mistral7b_full_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print('\n=== FINAL RESULTS ===')
print(f'Baseline: {ppl_base:.2f}')
for budget in [90, 80, 70]:
    u_h = all_results[f'{budget}pct-uniform']['healed']
    e_h = all_results[f'{budget}pct-entropy']['healed']
    winner = 'ENTROPY' if e_h < u_h else 'UNIFORM'
    margin = abs(u_h - e_h)
    print(f'  {budget}%: Uniform={u_h:.2f} Entropy={e_h:.2f} → {winner} wins by {margin:.2f}')
"

echo "=== Done! Results in /workspace/results/ ==="
ls -la /workspace/entropy-lens/results/
cat /workspace/results/mistral7b_full_results.json
