#!/bin/bash
# NIAH v4 — single-GPU Python (NOT torchrun), both with-memory and bypass
#
# Run 1 (with memory, champion_ckpt_v2) → b200-1
# Run 2 (bypass, no memory, control)    → b200-2
#
# Copy this script to both nodes and run it; each node starts both jobs but
# the --output_dir keeps results separated.  Alternatively kick off each
# nohup line manually on the appropriate node.
#
# NOTE: Do NOT prefix with `torchrun`. eval_niah_mem_space.py is single-GPU.
# torchrun would spawn 8 identical workers all hitting cuda:0 → OOM + 8×
# duplicate log lines.  The script now guards against this (Fix E), but
# launching with plain `python` is the correct approach.

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

# Run 1: with memory (champion_ckpt_v2) — intended for b200-1
nohup python scripts/eval_niah_mem_space.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data data/pg19_chunks_llama3.npy \
  --checkpoint outputs/champion_ckpt_v2/mem_space_adapter.pt \
  --num_slots 512 --top_k 64 --shared_memory_bank \
  --unfreeze_hidden_to_slot \
  --slot_init random --slot_init_noise 0.05 \
  --writeback_warmup_steps 1000 \
  --writeback_gate_max 0.3 \
  --load_balance_weight 0.01 \
  --context_lengths 8192,16384,32768 \
  --depths 0.1,0.3,0.5,0.75 \
  --num_samples 5 \
  --output_dir outputs/niah_mem_space_v4 \
  --skip_pg19_chunks 200 \
  > /root/logs/niah_mem_space_v4_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "niah_mem_space_v4 PID: $!"

# Run 2: bypass (no memory, control) — intended for b200-2
nohup python scripts/eval_niah_mem_space.py \
  --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data data/pg19_chunks_llama3.npy \
  --bypass_memory \
  --context_lengths 8192,16384,32768 \
  --depths 0.1,0.3,0.5,0.75 \
  --num_samples 5 \
  --output_dir outputs/niah_bypass_v4 \
  --skip_pg19_chunks 200 \
  > /root/logs/niah_bypass_v4_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "niah_bypass_v4 PID: $!"
