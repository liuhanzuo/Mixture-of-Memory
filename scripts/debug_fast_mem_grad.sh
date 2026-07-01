#!/bin/bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./:./third_party/babilong-pkg
export WANDB_MODE=disabled

/opt/conda/envs/torch-base/bin/python -c "
import torch, sys, math
from transformers import LlamaForCausalLM
from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model
from src.memory.mem_space.patch import _reset_fast_mem, _detach_fast_mem

# Load model same as training script
dtype = torch.bfloat16
model = LlamaForCausalLM.from_pretrained('models/Meta-Llama-3-8B', torch_dtype=dtype, attn_implementation='sdpa', local_files_only=True).cuda()

cfg = MemorySpaceConfig(
    num_slots=512, top_k=64, selector_dim=128, selector_temperature=20.0,
    slot_value_norm_cap=5.0, slot_init='random', slot_init_noise=0.05,
    shared_memory_bank=True, use_dual_gate=True, forget_bias_init=2.0,
    use_l3_summary=True, use_fast_mem=True, fast_mem_num_heads=4,
    fast_mem_d_state=128, fast_mem_chunk_size=16, fast_mem_fusion_init=-2.0,
    gradient_checkpointing=True, hidden_to_slot_frozen=False,
)
apply_mem_space_to_model(model, cfg)
model.to(device='cuda', dtype=dtype)

# Freeze backbone exactly like training script
for p in model.parameters():
    p.requires_grad = False

layers = model._mem_space_layers
for w in layers:
    for p in w.selector.parameters(): p.requires_grad = True
    w.gate_param.requires_grad = True
    w.slot_output_gate.requires_grad = True
    for p in w.slot_to_hidden.parameters(): p.requires_grad = True
    for p in w.hidden_to_slot.parameters(): p.requires_grad = True
    if w.gate_proj_new:
        for p in w.gate_proj_new.parameters(): p.requires_grad = True
    if w.gate_proj_mem:
        for p in w.gate_proj_mem.parameters(): p.requires_grad = True
    if w.gate_bias is not None: w.gate_bias.requires_grad = True
    fm = getattr(w, 'fast_mem', None)
    if fm:
        for p in fm.parameters(): p.requires_grad = True

# Info
fm0 = layers[0].fast_mem
print(f'fusion_gate requires_grad: {fm0.fusion_gate.requires_grad}')
print(f'fusion_gate dtype: {fm0.fusion_gate.dtype}')
print(f'fusion_gate value: {fm0.fusion_gate.data.mean().item():.6f}')
print(f'sigmoid(fusion_gate): {torch.sigmoid(fm0.fusion_gate).mean().item():.6f}')
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable params: {n_trainable/1e6:.1f}M')

# Simulate training step: reset → context (no_grad) → detach → target (grad)
_reset_fast_mem(model)
model.train()

# Context chunk (no_grad) - simulates n_ctx=1
ctx_ids = torch.randint(0, 1000, (1, 1024)).cuda()
with torch.no_grad():
    model(input_ids=ctx_ids, use_cache=False)
_detach_fast_mem(model)

# Target chunk (with grad) - this is where loss is computed
target_ids = torch.randint(0, 1000, (1, 1024)).cuda()
out = model(input_ids=target_ids, labels=target_ids, use_cache=False)
loss = out.loss
print(f'loss: {loss.item():.4f}')
loss.backward()

# Check gradients
print(f'\\n=== Layer 0 FastMem gradients ===')
print(f'fusion_gate grad: {fm0.fusion_gate.grad}')
if fm0.fusion_gate.grad is not None:
    print(f'  norm: {fm0.fusion_gate.grad.norm().item():.10f}')
    print(f'  mean: {fm0.fusion_gate.grad.mean().item():.10f}')
    print(f'  max:  {fm0.fusion_gate.grad.abs().max().item():.10f}')
else:
    print('  *** GRAD IS NONE! ***')

print(f'W_o grad: {\"norm=\" + str(fm0.W_o.weight.grad.norm().item()) if fm0.W_o.weight.grad is not None else \"NONE\"}')
print(f'W_k grad: {\"norm=\" + str(fm0.W_k.weight.grad.norm().item()) if fm0.W_k.weight.grad is not None else \"NONE\"}')
print(f'W_q grad: {\"norm=\" + str(fm0.W_q.weight.grad.norm().item()) if fm0.W_q.weight.grad is not None else \"NONE\"}')
print(f'W_v grad: {\"norm=\" + str(fm0.W_v.weight.grad.norm().item()) if fm0.W_v.weight.grad is not None else \"NONE\"}')
print(f'W_gate grad: {\"norm=\" + str(fm0.W_gate.weight.grad.norm().item()) if fm0.W_gate.weight.grad is not None else \"NONE\"}')
print(f'W_beta grad: {\"norm=\" + str(fm0.W_beta.weight.grad.norm().item()) if fm0.W_beta.weight.grad is not None else \"NONE\"}')

# Check across layers
print(f'\\n=== fusion_gate grad across layers ===')
for i in [0, 7, 15, 23, 31]:
    fm_i = layers[i].fast_mem
    fg = fm_i.fusion_gate.grad
    if fg is not None:
        print(f'  Layer {i}: norm={fg.norm().item():.10f} mean={fg.mean().item():.10f}')
    else:
        print(f'  Layer {i}: NONE')

# Also check: is the fast_mem output actually nonzero?
print(f'\\n=== FastMem output check ===')
_reset_fast_mem(model)
with torch.no_grad():
    model(input_ids=ctx_ids, use_cache=False)
    _detach_fast_mem(model)
    # Check state
    state = layers[0]._fast_mem_state
    print(f'Layer 0 state after ctx: norm={state.norm().item():.6f} shape={state.shape}')
    state15 = layers[15]._fast_mem_state
    print(f'Layer 15 state after ctx: norm={state15.norm().item():.6f}')
"
