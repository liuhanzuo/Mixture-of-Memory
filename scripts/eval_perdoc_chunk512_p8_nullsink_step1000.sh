#!/usr/bin/env bash
# Offline BABILong eval for per-doc chunk512 P8-NULLSINK arm, checkpoint step1000.
# chunk512 trained on diskB node .249; ckpts rsynced to local outputs/.
# Completes the same-step-500 compression-ladder comparison (chunk128 / chunk256
# step1000 already scored). Uses local idle GPU 0,1,2,3,5 (chunk128 training done;
# GPU 4,6,7 still draining chunk256/chunk128 evals).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_perdoc_chunk512_p8_nullsink_diskB
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step001000.pt
RESULTS=babilong_results/perdoc_chunk512_p8_nullsink_step1000
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_perdoc_chunk512_p8_nullsink_step1000
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(2 3 6)
i=0
for L in "${LENGTHS[@]}"; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name perdoc_chunk512_p8_nullsink_step1000_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
  i=$((i+1))
done
echo "launched ${#LENGTHS[@]} eval workers on GPU 0,1,2,3,5"
wait
echo "[$(date)] all eval workers done"
