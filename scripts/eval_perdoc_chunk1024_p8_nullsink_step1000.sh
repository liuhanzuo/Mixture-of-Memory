#!/usr/bin/env bash
# Offline BABILong eval for per-doc chunk1024 P8-NULLSINK arm, checkpoint step1000.
# chunk1024 trained on diskB node .76; ckpts rsynced to local outputs/.
# Last (largest-chunk) arm of the P8 compression ladder. chunk128/256/512 step1000
# already scored. Uses local idle GPU 1,6,7 (GPU1,6,7 draining chunk512 tails).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_perdoc_chunk1024_p8_nullsink_diskB
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step001000.pt
RESULTS=babilong_results/perdoc_chunk1024_p8_nullsink_step1000
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_perdoc_chunk1024_p8_nullsink_step1000
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(1 6 7)
i=0
for L in "${LENGTHS[@]}"; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name perdoc_chunk1024_p8_nullsink_step1000_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 1024 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
  i=$((i+1))
done
echo "launched ${#LENGTHS[@]} eval workers on GPU 1,6,7"
wait
echo "[$(date)] all eval workers done"
