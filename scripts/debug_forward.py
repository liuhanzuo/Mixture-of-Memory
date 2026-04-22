"""Minimal debug: test if backbone forward hangs."""
import sys, os
sys.path.insert(0, "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")

import torch

print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Simulate RMT segment forward
seq_len = 64
num_mem = 16
total_len = num_mem + seq_len
B = 1

print("Building inputs...", flush=True)
# Fake inputs_embeds
with torch.no_grad():
    input_ids = torch.randint(0, 1000, (B, total_len), device="cuda:0")
    inputs_embeds = model.get_input_embeddings()(input_ids)

# Build attention mask (bool, True=attend)
causal = torch.tril(torch.ones(total_len, total_len, device="cuda:0")).bool()
causal[:num_mem, :] = True  # memory attends to all
attn_mask_4d = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]

position_ids = torch.cat([
    torch.arange(num_mem, device="cuda:0"),
    torch.arange(seq_len, device="cuda:0")
]).unsqueeze(0).expand(B, -1)

print(f"attn_mask shape: {attn_mask_4d.shape}, dtype: {attn_mask_4d.dtype}", flush=True)
print(f"inputs_embeds shape: {inputs_embeds.shape}", flush=True)
print(f"position_ids shape: {position_ids.shape}", flush=True)

print("Calling backbone with dict attention_mask...", flush=True)
try:
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_4d},
            position_ids=position_ids,
        )
    print(f"SUCCESS! Output shape: {outputs.last_hidden_state.shape}", flush=True)
except Exception as e:
    print(f"ERROR with dict mask: {type(e).__name__}: {e}", flush=True)
    
    print("Retrying with plain bool mask (no dict)...", flush=True)
    with torch.no_grad():
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal,  # 2D bool mask
            position_ids=position_ids,
        )
    print(f"SUCCESS with plain mask! Output shape: {outputs.last_hidden_state.shape}", flush=True)

print("Done.", flush=True)
