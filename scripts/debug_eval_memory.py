"""Minimal diagnostic: confirm RMT memory is loaded and injected during eval."""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_v10 import RMTv10Memory, RMTv10Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "outputs/rmt_v10_l0l2_20260420_110159/final"

# 1. Load model
print("=== Step 1: Load base model ===")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
)
model.eval()

# 2. Create RMT wrapper
print("\n=== Step 2: Create RMT wrapper ===")
with open(os.path.join(CHECKPOINT_DIR, "rmt_config.json")) as f:
    cfg = json.load(f)
print(f"Config: {cfg}")

v10_cfg = RMTv10Config()
v10_cfg.num_mem_tokens = cfg.get("num_mem_tokens", 16)
v10_cfg.segment_length = cfg.get("segment_length", 1024)
v10_cfg.max_n_segments = cfg.get("max_segments", 4)
v10_cfg.use_l1 = cfg.get("use_l1", False)
v10_cfg.use_l2 = cfg.get("use_l2", False)

rmt_model = RMTv10Memory(v10_cfg).wrap(model)
rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)

# 3. Check model state_dict keys
print("\n=== Step 3: Model state_dict keys ===")
model_keys = set(rmt_model.state_dict().keys())
print(f"Total keys in rmt_model.state_dict(): {len(model_keys)}")
rmt_keys = [k for k in model_keys if k.startswith("l0.") or k.startswith("l1.") or k.startswith("l2.") or k.startswith("recon_head.")]
print(f"RMT keys ({len(rmt_keys)}): {rmt_keys[:10]}...")

# 4. Load memory weights
print("\n=== Step 4: Load rmt_memory.pt ===")
rmt_path = os.path.join(CHECKPOINT_DIR, "rmt_memory.pt")
state = torch.load(rmt_path, map_location=device)
print(f"Keys in rmt_memory.pt: {list(state.keys())}")
for k, v in state.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}")

# 5. load_state_dict result
print("\n=== Step 5: load_state_dict result ===")
result = rmt_model.load_state_dict(state, strict=False)
print(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:10]}...")
print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}...")

# 6. Verify loaded values
print("\n=== Step 6: Verify memory values after loading ===")
l0_memory_after = rmt_model.state_dict()["l0.memory"]
print(f"l0.memory after load: shape={l0_memory_after.shape}, mean={l0_memory_after.float().mean().item():.6f}, std={l0_memory_after.float().std().item():.6f}")
l0_mem_from_file = state.get("l0.memory")
if l0_mem_from_file is not None:
    print(f"l0.memory from file: shape={l0_mem_from_file.shape}, mean={l0_mem_from_file.float().mean().item():.6f}, std={l0_mem_from_file.float().std().item():.6f}")
    match = torch.allclose(l0_memory_after.cpu().float(), l0_mem_from_file.cpu().float(), atol=1e-3)
    print(f"Memory values match file: {match}")
else:
    print("WARNING: l0.memory not in rmt_memory.pt!")

# 7. Test actual sandwich injection
print("\n=== Step 7: Test sandwich injection ===")
tok = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

test_text = "The secret code is 42. " * 100
test_ids = tok.encode(test_text, return_tensors="pt").to(device)
print(f"Input length: {test_ids.shape[1]}")

B = test_ids.shape[0]
memory_state = rmt_model.l0.get_initial_memory(B).to(dtype=torch.bfloat16)
print(f"Initial memory shape: {memory_state.shape}, mean: {memory_state.float().mean().item():.6f}")

seg_ids = test_ids[:, :v10_cfg.segment_length]
content_embeds = rmt_model.base_model.get_input_embeddings()(seg_ids)
inputs_embeds, attn_mask = rmt_model.l0.build_sandwich_fast(content_embeds, memory_state)
print(f"Sandwich shape: {inputs_embeds.shape} (should be [1, {v10_cfg.segment_length + 2*v10_cfg.num_mem_tokens}, D])")
print(f"Sandwich mask shape: {attn_mask.shape}")
print(f"Memory IS in sandwich: {inputs_embeds.shape[1] > seg_ids.shape[1]}")

attn_mask_4d = rmt_model._make_4d_attn_mask(attn_mask, torch.bfloat16)
position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0).expand(B, -1)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = rmt_model.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            output_hidden_states=True,
        )

new_memory = rmt_model.l0.extract_new_memory(outputs.hidden_states[-1])
memory_updated, importance = rmt_model.l0.apply_importance_routing(memory_state, new_memory)
print(f"Updated memory shape: {memory_updated.shape}")
print(f"Updated memory mean: {memory_updated.float().mean().item():.6f}")
print(f"Updated memory std: {memory_updated.float().std().item():.6f}")
print(f"Memory changed from initial: {not torch.allclose(memory_state, memory_updated, atol=1e-3)}")

# 8. Test generation with memory
print("\n=== Step 8: Test generation with memory ===")
q_ids = tok.encode("What is the secret code?", return_tensors="pt").to(device)
q_embeds = rmt_model.base_model.get_input_embeddings()(q_ids)
gen_embeds = torch.cat([memory_updated, q_embeds], dim=1)

gen_attn_mask = torch.tril(torch.ones(gen_embeds.shape[1], gen_embeds.shape[1], device=device, dtype=torch.bool))
gen_attn_4d = torch.zeros(1, 1, gen_embeds.shape[1], gen_embeds.shape[1], device=device, dtype=torch.bfloat16)
gen_attn_4d[0, 0] = gen_attn_mask.float()
gen_attn_4d.masked_fill_(~gen_attn_mask, float('-inf'))
position_ids = torch.arange(gen_embeds.shape[1], device=device).unsqueeze(0)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = rmt_model.base_model.model(inputs_embeds=gen_embeds, attention_mask=gen_attn_4d, position_ids=position_ids)
    logits = rmt_model.base_model.lm_head(out.last_hidden_state[:, -1:, :])
    top5 = torch.topk(logits[0, 0], 5)
    top5_tokens = [tok.decode([t]) for t in top5.indices.tolist()]
    print(f"Top 5 next tokens: {top5_tokens}")

print("\n=== DIAGNOSTIC COMPLETE ===")
