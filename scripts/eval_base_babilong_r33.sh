#!/usr/bin/env bash
# R3-3 base anchor: plain frozen Llama-3-8B (NO memory adapter) on BABILong.
# Fair comparison vs R3-1 chunk128 adapter. Base has 8k native ctx -> only 0k-8k fit
# (16k/32k overflow the window, so we cap at 8k; that overflow IS the point our adapter beats).
# Same tasks (qa1 qa2 qa5), limit 100, greedy. One length per free GPU.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
MODEL=models/Meta-Llama-3-8B
RESULTS=babilong_results/base_model_full
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_base_babilong
mkdir -p "$RESULTS" "$LOGDIR"

# free local GPUs at launch time: 0 2 3 4 5 (1,6,7 busy with longbench tail)
LENGTHS=(0k 1k 2k 4k 8k)
GPUS=(0 2 3 4 5)
for i in "${!LENGTHS[@]}"; do
  L="${LENGTHS[$i]}"; G="${GPUS[$i]}"
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G setsid bash -c "$PYBIN scripts/eval_baseline_babilong.py \
    --baseline plain_hf --model_path $MODEL \
    --results_folder $RESULTS --output_name base_${L} \
    --tasks $TASKS --lengths $L --limit 100 \
    --use_instruction --use_examples --use_post_prompt --max_new_tokens 20" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
done
echo "launched ${#LENGTHS[@]} base eval workers"
wait
echo "[$(date)] all base eval workers done"
