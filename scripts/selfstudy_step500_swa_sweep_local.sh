#!/usr/bin/env bash
set -u
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT=outputs/mem_space_selfstudy_rawkv_chunk512
ADP=$OUT/adapter_config.json
CK=$OUT/full_model_step000500.pt   # best self-study ckpt (W0: qa1 4k17/8k19/16k12/32k8)
TS=20260621_0705
LOGDIR="$ROOT/logs/selfstudy_step500_swa_sweep_$TS"
RES="$ROOT/babilong_results/selfstudy_step500_swa_$TS"
mkdir -p "$LOGDIR" "$RES"
# eval-time cross-chunk SWA on the best self-study ckpt — only un-falsified lever, never tested on self-study.
# qa1+qa5 (qa5 long-range = discriminating vs P11 W0 ceiling) x {8k,16k,32k} x W{1,2}.
# Free local GPUs (skip GPU5 busy): 0 1 2 3 4 6 7
GPUS=(0 1 2 3 4 6 7)
NG=${#GPUS[@]}
i=0
for W in 1 2; do
  for task in qa1 qa5; do
    for len in 8k 16k 32k; do
      while [ "$(jobs -rp | wc -l)" -ge "$NG" ]; do wait -n; done
      g=${GPUS[$(( i % NG ))]}; i=$((i+1))
      out="selfstudy_step500_${task}_${len}_swa${W}"
      CUDA_VISIBLE_DEVICES=$g "$PY" scripts/run_babilong_mem_space.py \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint "$CK" --adapter_config "$ADP" \
        --results_folder "$RES" --output_name "$out" \
        --tasks "$task" --lengths "$len" --limit 100 \
        --chunk_size 512 --batch_size 1 --max_new_tokens 20 \
        --dtype bfloat16 --attn_impl sdpa \
        --use_instruction --use_examples --use_post_prompt \
        --swa_eval_chunks "$W" \
        > "$LOGDIR/${out}.gpu${g}.log" 2>&1 &
    done
  done
done
wait
date > "$LOGDIR/DONE"
