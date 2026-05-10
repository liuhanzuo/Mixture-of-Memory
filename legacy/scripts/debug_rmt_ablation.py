#!/usr/bin/env python3
"""
Debug: Test if RMT model works with ZERO memory (should degrade to base model behavior).
Also test: does the attention mask format actually work with Qwen3?
"""
import sys, os, torch
sys.path.insert(0, "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTModel, RMTMemory

BASE = "../models/Qwen--Qwen3-8b"
DEVICE = "cuda:0"

print("Loading...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.eval()

text = "The secret code is ABC123.\n\nWhat is the secret code mentioned in the document? The secret code is"
ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").to(DEVICE)

# Test 1: Base model (normal token IDs)
print("\n=== Test 1: Base model ===", flush=True)
with torch.no_grad():
    out = model(input_ids=ids)
    top5 = torch.topk(out.logits[0, -1], 5)
    print("Top-5:", [(tok.decode([t]), f"{v:.2f}") for t,v in zip(top5.indices.tolist(), top5.values.tolist())])

# Test 2: Using inputs_embeds (same as base, no memory)
print("\n=== Test 2: Base model via inputs_embeds (no memory) ===", flush=True)
with torch.no_grad():
    embeds = model.get_input_embeddings()(ids)
    out2 = model.model(inputs_embeds=embeds)
    logits2 = model.lm_head(out2.last_hidden_state)
    top5b = torch.topk(logits2[0, -1], 5)
    print("Top-5:", [(tok.decode([t]), f"{v:.2f}") for t,v in zip(top5b.indices.tolist(), top5b.values.tolist())])

# Test 3: With zero memory prepended + full attention mask
print("\n=== Test 3: Zero memory + RMT-style mask ===", flush=True)
n_mem = 64
with torch.no_grad():
    embeds = model.get_input_embeddings()(ids)
    zero_mem = torch.zeros(1, n_mem, embeds.shape[-1], device=DEVICE, dtype=torch.bfloat16)
    full_embeds = torch.cat([zero_mem, embeds], dim=1)
    
    seq_len = ids.shape[1]
    total_len = n_mem + seq_len
    
    # Build RMT-style mask
    causal = torch.tril(torch.ones(total_len, total_len, device=DEVICE)).bool()
    causal[:n_mem, :] = True
    bool_4d = causal.unsqueeze(0).unsqueeze(0)
    attn_mask = torch.zeros_like(bool_4d, dtype=torch.float32).masked_fill(~bool_4d, float('-inf'))
    
    pos_ids = torch.cat([torch.arange(n_mem, device=DEVICE), torch.arange(seq_len, device=DEVICE)]).unsqueeze(0)
    
    out3 = model.model(
        inputs_embeds=full_embeds,
        attention_mask={"full_attention": attn_mask},
        position_ids=pos_ids,
    )
    logits3 = model.lm_head(out3.last_hidden_state)
    top5c = torch.topk(logits3[0, -1], 5)
    print("Top-5:", [(tok.decode([t]), f"{v:.2f}") for t,v in zip(top5c.indices.tolist(), top5c.values.tolist())])

# Test 4: Same but without dict attention mask (let Qwen3 handle it)
print("\n=== Test 4: Zero memory + NO custom mask (default causal) ===", flush=True)
with torch.no_grad():
    out4 = model.model(inputs_embeds=full_embeds)
    logits4 = model.lm_head(out4.last_hidden_state)
    top5d = torch.topk(logits4[0, -1], 5)
    print("Top-5:", [(tok.decode([t]), f"{v:.2f}") for t,v in zip(top5d.indices.tolist(), top5d.values.tolist())])

# Test 5: With RANDOM memory (not trained) + RMT mask
print("\n=== Test 5: Random memory + RMT-style mask ===", flush=True)
with torch.no_grad():
    rand_mem = torch.randn(1, n_mem, embeds.shape[-1], device=DEVICE, dtype=torch.bfloat16) * 0.02
    full_embeds_r = torch.cat([rand_mem, embeds], dim=1)
    
    out5 = model.model(
        inputs_embeds=full_embeds_r,
        attention_mask={"full_attention": attn_mask},
        position_ids=pos_ids,
    )
    logits5 = model.lm_head(out5.last_hidden_state)
    top5e = torch.topk(logits5[0, -1], 5)
    print("Top-5:", [(tok.decode([t]), f"{v:.2f}") for t,v in zip(top5e.indices.tolist(), top5e.values.tolist())])

# Test 6: Check if the full_attention dict format is even being used
print("\n=== Test 6: Check if dict mask format is accepted ===", flush=True)
with torch.no_grad():
    try:
        out6 = model.model(
            inputs_embeds=full_embeds,
            attention_mask={"full_attention": attn_mask},
        )
        print(f"Dict mask accepted. Output shape: {out6.last_hidden_state.shape}", flush=True)
    except Exception as e:
        print(f"Dict mask FAILED: {e}", flush=True)

print("\nDone.", flush=True)
