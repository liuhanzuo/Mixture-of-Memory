#!/usr/bin/env bash
# Offline BABILong eval for P2 decoupled-read checkpoint (dolmino_p2_decoupled_local).
# Gate (see launch_dolmino_p2_decoupled_local.sh): >=4k accuracy must cross the
# ~1-2% noise floor that BOTH P1 arms hit. One length per GPU across 8 local H20.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/dolmino_p2_decoupled_local
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
RESULTS=babilong_results/p2_decoupled_local
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_p2_decoupled_local
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPU=0
for L in "${LENGTHS[@]}"; do
  echo "[$(date)] GPU $GPU -> length $L"
  CUDA_VISIBLE_DEVICES=$GPU setsid bash -c "$PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name p2_decoupled_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 1024 \
    --dtype bfloat16 --attn_impl sdpa" \
    </dev/null >"$LOGDIR/${L}.log" 2>&1 &
  GPU=$((GPU+1))
done
echo "launched ${#LENGTHS[@]} eval workers (GPU 0..$((GPU-1)))"
wait
echo "[$(date)] all eval workers done"
