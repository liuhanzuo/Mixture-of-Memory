#!/usr/bin/env python3
"""
verify_kvcache.py — bit-identical check + speedup for KV-cache generation.

Usage:
  CUDA_VISIBLE_DEVICES=<id> .venv/bin/python scripts/verify_kvcache.py

Expects:
  outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt
  models/Meta-Llama-3-8B
"""
from __future__ import annotations
import os, sys, time, importlib.util, copy, json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ────────────────────────────── Config ──────────────────────────────────────
CKPT = os.path.join(PROJECT_ROOT,
    "outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt")
BASE_MODEL = os.path.join(PROJECT_ROOT, "models/Meta-Llama-3-8B")
DEVICE_STR = "cuda:0"
MAX_NEW_TOKENS = 20
CHUNK_SIZE = 512

# ─────────────────── Load the two script modules ────────────────────────────
def _load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

orig_mod  = _load_mod(os.path.join(PROJECT_ROOT, "scripts/run_babilong_mem_space.py"),
                      "run_orig")
cache_mod = _load_mod(os.path.join(PROJECT_ROOT, "scripts/run_babilong_mem_space_kvcache.py"),
                      "run_cache")

# ───────────────────── Build one model (shared weights) ─────────────────────
from src.memory.mem_space import MemorySpaceConfig

DEVICE = torch.device(DEVICE_STR)
print(f"[verify] device={DEVICE_STR}", flush=True)

# Read adapter_config.json sitting next to the checkpoint
CKPT_DIR = os.path.dirname(CKPT)
adapter_json = os.path.join(CKPT_DIR, "adapter_config.json")
if os.path.exists(adapter_json):
    with open(adapter_json) as f:
        adapter_cfg = json.load(f)
    print(f"[verify] adapter_config: {adapter_cfg}", flush=True)
    mem_config = orig_mod.build_mem_space_config(adapter_cfg)
else:
    # Fallback: construct from known defaults of this run
    mem_config = MemorySpaceConfig(
        num_slots=25,
        slot_dim=512,
        top_k=4,
        shared_memory_bank=True,
        fifo_select=True,
        supervised_select=False,
    )
    print(f"[verify] No adapter_config.json found; using defaults: {mem_config}", flush=True)

print(f"[verify] Loading base model from {BASE_MODEL} …", flush=True)
model_no  = orig_mod.load_mem_space_model(BASE_MODEL, CKPT, mem_config, DEVICE)
model_no.eval()
print("[verify] Model loaded.", flush=True)

# Deep-copy weights into a second model instance for the cache version
# (same weights, but separate state so bank state doesn't bleed across)
model_kv = copy.deepcopy(model_no)
model_kv.eval()

# ─────────────── Tokenizer ──────────────────────────────────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ─────────────── Build synthetic test sequences ─────────────────────────────
# We need sequences >= 2 chunks so memory streaming actually happens.
# Build: N×chunk_size tokens = N chunks, last chunk contains the "question".
torch.manual_seed(42)

def make_input(total_tokens: int) -> torch.Tensor:
    """Random token ids as a [1, total_tokens] tensor, on CPU."""
    vocab = tokenizer.vocab_size
    ids = torch.randint(10, vocab - 10, (1, total_tokens), dtype=torch.long)
    return ids

SAMPLES = [
    make_input(CHUNK_SIZE * 4),    # 4 chunks (~2k tokens)
    make_input(CHUNK_SIZE * 8),    # 8 chunks (~4k tokens)
    make_input(CHUNK_SIZE * 16),   # 16 chunks (~8k tokens)
    make_input(CHUNK_SIZE * 32),   # 32 chunks (~16k tokens)
    make_input(CHUNK_SIZE * 4),    # another 4-chunk (different seed via different position)
]
# Override sample[4] to be different
torch.manual_seed(99)
SAMPLES[4] = make_input(CHUNK_SIZE * 6)

GEN_FN_ORIG  = orig_mod.generate_with_mem_space
GEN_FN_CACHE = cache_mod.generate_with_mem_space

# ─────────────────────────────── Run verification ───────────────────────────
results = []
all_pass = True

for idx, input_ids in enumerate(SAMPLES):
    input_ids_dev = input_ids.to(DEVICE)
    n_toks = input_ids.shape[1]
    n_chunks = (n_toks + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"\n[verify] Sample {idx+1}/{len(SAMPLES)}: {n_toks} tokens ({n_chunks} chunks)", flush=True)

    # ── No-cache baseline ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        out_no = GEN_FN_ORIG(
            model=model_no,
            input_ids=input_ids_dev,
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )
    t_no = time.perf_counter() - t0

    # ── KV-cache version ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        out_kv = GEN_FN_CACHE(
            model=model_kv,
            input_ids=input_ids_dev,
            tokenizer=tokenizer,
            chunk_size=CHUNK_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )
    t_kv = time.perf_counter() - t0

    match = (out_no == out_kv)
    speedup = t_no / t_kv if t_kv > 0 else float('inf')

    status = "PASS" if match else "FAIL"
    all_pass = all_pass and match

    print(f"  no-cache : {t_no:.3f}s  |  output: {repr(out_no[:80])}")
    print(f"  kv-cache : {t_kv:.3f}s  |  output: {repr(out_kv[:80])}")
    print(f"  match={match}  speedup={speedup:.2f}×  [{status}]", flush=True)

    if not match:
        # Show first differing character
        for ci, (a, b) in enumerate(zip(out_no, out_kv)):
            if a != b:
                print(f"  DIFF at char {ci}: no-cache={repr(a)} kv-cache={repr(b)}")
                break
        if len(out_no) != len(out_kv):
            print(f"  DIFF length: no-cache={len(out_no)} kv-cache={len(out_kv)}")

    results.append(dict(sample=idx+1, tokens=n_toks, t_no=t_no, t_kv=t_kv,
                        speedup=speedup, match=match))

# ─────────────────────────────── Summary ───────────────────────────────────
print("\n" + "="*60)
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({sum(r['match'] for r in results)}/{len(results)} identical)")
avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"Average speedup: {avg_speedup:.2f}×")
print("Per-sample summary:")
for r in results:
    print(f"  sample {r['sample']}: tokens={r['tokens']}, no-cache={r['t_no']:.3f}s, "
          f"kv-cache={r['t_kv']:.3f}s, speedup={r['speedup']:.2f}×, match={r['match']}")
