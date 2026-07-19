#!/bin/bash
# HCache (retrieval-free mid-layer recompute, resume_j=12) — RULER task-breadth
# extension. 3 copy-hard tasks x {64k,128k}, 8-shard (1 shard/GPU), n=100/cell,
# chat ON + no-think, max_new_tokens=128, chunk_size=512. Same sample set as
# CoMem/InfLLM/StreamingLLM/KV-Direct (PYTHONHASHSEED=0, seed=42). baseline=hcache
# keeps resume_j=12 (mid-layer), packs all chunks (no retrieval), no LoRA.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
export PYTHONHASHSEED=0
export WANDB_MODE=offline
PYBIN=/opt/conda/envs/torch-base/bin/python
MODEL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
OUT=hcache_8b_taskbreadth
mkdir -p logs "ruler_results/$OUT"
rm -f logs/hc_tb_ALL_DONE
pids=()
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k $PYBIN scripts/eval_ruler_qcmem.py \
    --model_path "$MODEL" \
    --baseline hcache --resume_j 12 \
    --ruler_tasks niah_single_3 niah_multivalue niah_multiquery \
    --lengths 64k 128k \
    --limit 100 --num_shards 8 --shard_index "$k" \
    --max_new_tokens 128 --use_chat_template \
    --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
    --output_name "$OUT" \
    --results_folder "ruler_results/$OUT" \
    > "logs/hc_tb_shard$k.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL_DONE $(date)" > logs/hc_tb_ALL_DONE
