#!/usr/bin/env bash
set -u
# eval-time cross-chunk SWA on LATE self-study ckpts (step1000 + step2000) — answers the open
# question: does eval-SWA rescue over-trained later ckpts (step500=best W0, step1000/2000 degraded)?
# Non-duplicative: prior sweep only tested step500. Runs on .196 (diskA shared FS, 7 idle GPUs).
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT=outputs/mem_space_selfstudy_rawkv_chunk512
ADP=$OUT/adapter_config.json
TS=$(date +%Y%m%d_%H%M)
LOGDIR="$ROOT/logs/selfstudy_late_swa_sweep_$TS"
RES="$ROOT/babilong_results/selfstudy_late_swa_$TS"
mkdir -p "$LOGDIR" "$RES"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
GPUS=(0 1 2 3 4 5 6 7)
NG=${#GPUS[@]}
i=0
for CKstep in 001000 002000; do
  CK=$OUT/full_model_step${CKstep}.pt
  [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
  for W in 1 2; do
    for task in qa1 qa5; do
      for len in 8k 16k 32k; do
        while [ "$(jobs -rp | wc -l)" -ge "$NG" ]; do wait -n; done
        g=${GPUS[$(( i % NG ))]}; i=$((i+1))
        out="selfstudy_step${CKstep}_${task}_${len}_swa${W}"
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
done
wait
date > "$LOGDIR/DONE"
