#!/bin/bash
# step2000 (final) self-study SWA eval — fills the missing late-ckpt cell.
set -u
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
PY=.venv/bin/python
OUT=outputs/mem_space_selfstudy_rawkv_chunk512
RES=babilong_results/selfstudy_late_swa_20260621_0742
LOG=logs/selfstudy_step2000_swa_20260621_0948
mkdir -p "$LOG"
g=0
for task in qa1 qa5; do
  for len in 8k 16k 32k; do
    for swa in 1 2; do
      gpu=$((g % 8))
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint "$OUT/full_model.pt" \
        --adapter_config "$OUT/adapter_config.json" \
        --results_folder "$RES" \
        --output_name "selfstudy_step002000_${task}_${len}_swa${swa}" \
        --tasks "$task" --lengths "$len" --limit 100 \
        --chunk_size 512 --batch_size 1 --max_new_tokens 20 \
        --dtype bfloat16 --attn_impl sdpa \
        --use_instruction --use_examples --use_post_prompt \
        --swa_eval_chunks "$swa" \
        > "$LOG/${task}_${len}_swa${swa}.gpu${gpu}.log" 2>&1 &
      g=$((g + 1))
    done
  done
done
wait
echo "step2000-swa-done"
