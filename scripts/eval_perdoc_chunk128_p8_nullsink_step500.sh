#!/usr/bin/env bash
# Offline BABILong eval for per-doc chunk128 P8-NULLSINK arm, checkpoint step500.
# Uses GPU 4-7 only (GPU 0-3 occupied by the live chunk128 training run).
# 7 lengths round-robin over 4 GPUs (lengths > #gpus double up; chunk128 eval VRAM is modest).
# Goal: confirm the null/sink + trainable memory_xattn fix resolves the old-P8
#   0k-collapse (old P8: qa1_0k=11, qa2_0k=0) and compare vs P7P9 / chunk1024.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_perdoc_chunk128_p8_nullsink
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
RESULTS=babilong_results/perdoc_chunk128_p8_nullsink_step500
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_perdoc_chunk128_p8_nullsink_step500
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(4 5 6 7)
i=0
for L in "${LENGTHS[@]}"; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name perdoc_chunk128_p8_nullsink_step500_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 128 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
  i=$((i+1))
done
echo "launched ${#LENGTHS[@]} eval workers on GPU 4-7"
wait
echo "[$(date)] all eval workers done"
