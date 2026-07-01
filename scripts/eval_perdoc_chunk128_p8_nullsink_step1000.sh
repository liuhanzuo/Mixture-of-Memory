#!/usr/bin/env bash
# Offline BABILong eval for per-doc chunk128 P8-NULLSINK arm, checkpoint step1000.
# Uses GPU 5-7 in TWO WAVES so each GPU holds <=2 model copies (3+ -> OOM).
# Goal: confirm within-arm overtraining knee vs step500 (researcher: step500 good -> step1000 degrades).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_perdoc_chunk128_p8_nullsink
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step001000.pt
RESULTS=babilong_results/perdoc_chunk128_p8_nullsink_step1000
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_perdoc_chunk128_p8_nullsink_step1000
mkdir -p "$RESULTS" "$LOGDIR"

run_one() {
  local G=$1 L=$2
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name perdoc_chunk128_p8_nullsink_step1000_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 128 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
}

# Wave 1: short lengths (4 over 3 GPUs -> GPU5 doubles 0k+4k, fine)
run_one 5 0k
run_one 6 1k
run_one 7 2k
run_one 5 4k
wait
echo "[$(date)] wave1 (short) done"

# Wave 2: long lengths (3 over 3 GPUs, 1 each)
run_one 5 8k
run_one 6 16k
run_one 7 32k
wait
echo "[$(date)] wave2 (long) done"
echo "[$(date)] all eval workers done"
