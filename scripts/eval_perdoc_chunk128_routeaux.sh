#!/usr/bin/env bash
# Offline BABILong eval for per-doc chunk128 route_aux=1.0 arm (R3-2 gate).
# Mirror of eval_perdoc_chunk128_local.sh but for the routeaux checkpoint.
# Compares routing-supervision (route_aux) vs no-route_aux chunk128 adapter.
# One length per GPU across 8 H20. PYBIN/PROJECT_ROOT overridable for remote node.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_perdoc_chunk128_routeaux_remote
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
RESULTS=babilong_results/perdoc_chunk128_routeaux
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_perdoc_chunk128_routeaux
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPU=0
for L in "${LENGTHS[@]}"; do
  echo "[$(date)] GPU $GPU -> length $L"
  CUDA_VISIBLE_DEVICES=$GPU setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name perdoc_chunk128_routeaux_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 128 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
  GPU=$((GPU+1))
done
echo "launched ${#LENGTHS[@]} eval workers (GPU 0..$((GPU-1)))"
wait
echo "[$(date)] all eval workers done"
