#!/usr/bin/env python
"""Tiny GPU smoke for P1 store builders: confirm the flagship QCMem path actually
answers on my token-space stores (esp. VT without ICL). Builds one small store per
task at 8k, runs iter_bm25 selection + qcmem_generate, prints recall + output + score.
"""
import os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import scripts.eval_p1_scaling as p1
import scripts.eval_qcmem_babilong as qcb

MODEL = "models/Qwen3-8b-local"
LORA = "outputs/qcmem_distill_qwen_j12_r32_4k/final"
DEV = torch.device("cuda:0")
CS = 512

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    trust_remote_code=True, local_files_only=True).to(DEV).eval()
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA).eval()
model = model.base_model.model
qc = qcb.QCMemModel(model, resume_j=12)
print("model loaded", round(time.time() - t0, 1), "s", flush=True)

pool = p1.build_pool(tok, 30000, "/tmp/smoke_pool.npy")

def run(name, st):
    ids = st["input_ids"]
    tokens = torch.tensor(ids, dtype=torch.long)
    chunks = list(tokens.split(CS))
    ctx = chunks[:-1]
    sel = qcb._iter_bm25_indices(ctx, list(st["bare_q_ids"]), topk=12,
                                 iter_rounds=0, iter_hop_topk=4)
    gold = st["gold_chunk_idx"]
    rec = len([g for g in gold if g in set(sel)]) / len(gold) if gold else -1
    stats = {}
    out = qcb.qcmem_generate(
        qc=qc, tokenizer=tok, input_ids=tokens.unsqueeze(0).to(DEV),
        chunk_size=CS, max_new_tokens=64, selector="iter_bm25", topk=12,
        sink_tokens="bos", bare_question_ids=list(st["bare_q_ids"]), stats=stats,
        iter_rounds=0, iter_hop_topk=4)
    score = p1._string_match_all_one(out, st["answers"]) * 100.0
    print(f"\n=== {name} === gold={gold} sel={sorted(sel)} recall={rec:.2f} "
          f"read_len={stats.get('read_len')} score={score:.0f}")
    print("answers:", st["answers"])
    print("output :", repr(out[:220]), flush=True)

rng = random.Random(123)
run("niah_single@8k E=1", p1.build_niah_store(tok, pool, 8192, CS, 1, "uuids", rng))
run("niah_multivalue@8k E=4", p1.build_niah_store(tok, pool, 8192, CS, 4, "numbers", rng))
run("vt@8k hops=4", p1.build_vt_store(tok, pool, 8192, CS, 4, rng))
print("\nSMOKE DONE", flush=True)
