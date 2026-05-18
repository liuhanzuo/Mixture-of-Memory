#!/usr/bin/env bash
# Launch full BABILong eval for one checkpoint from phase1b_v2_llama32_1b_base_5k.
#
# Usage:
#   bash scripts/launch_phase1b_v2_base_ckpt_eval.sh 500
#   bash scripts/launch_phase1b_v2_base_ckpt_eval.sh 1000
#   bash scripts/launch_phase1b_v2_base_ckpt_eval.sh 3000
#   bash scripts/launch_phase1b_v2_base_ckpt_eval.sh 5000
#   bash scripts/launch_phase1b_v2_base_ckpt_eval.sh final
#
# Notes:
#   * Uses one GPU per length (0k..32k on GPUs 0..6).
#   * Evaluates qa1..qa10 with NO chat template because the backbone is raw
#     Llama-3.2-1B, not Instruct.

set -euo pipefail

STEP_LABEL="${1:-final}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/Llama-3.2-1B}"
CKPT_DIR="${CKPT_DIR:-outputs/babilong_sft_phase1b_v2_llama32_1b_base_5k}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-$CKPT_DIR/adapter_config.json}"
TASKS=(qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10)
LENGTH_GPU=(0k:0 1k:1 2k:2 4k:3 8k:4 16k:5 32k:6)
TS="$(date +%Y%m%d_%H%M)"

if [[ "$STEP_LABEL" == "final" ]]; then
  LABEL="final"
  CHECKPOINT="$CKPT_DIR/mem_space_adapter.pt"
else
  STEP_NUM="$STEP_LABEL"
  PADDED="$(printf '%06d' "$STEP_NUM")"
  LABEL="step${STEP_NUM}"
  CHECKPOINT="$CKPT_DIR/mem_space_adapter_step${PADDED}.pt"
fi

RESULTS_FOLDER="${RESULTS_FOLDER:-outputs/eval_phase1b_v2_llama32_1b_base_${LABEL}}"
mkdir -p "$RESULTS_FOLDER" logs

for item in "${LENGTH_GPU[@]}"; do
  length="${item%%:*}"
  gpu="${item##*:}"
  logfile="logs/eval_phase1b_v2_llama32_1b_base_${LABEL}_${length}_${TS}.log"
  outname="p1bv2base_${LABEL}_${length}"
  echo "[launch] ${LABEL} ${length} gpu=${gpu} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup setsid "$PYTHON_BIN" scripts/run_babilong_mem_space.py \
    --model_path "$MODEL_PATH" \
    --checkpoint "$CHECKPOINT" \
    --adapter_config "$ADAPTER_CONFIG" \
    --results_folder "$RESULTS_FOLDER" \
    --output_name "$outname" \
    --tasks "${TASKS[@]}" \
    --lengths "$length" \
    --limit 100 \
    --device cuda:0 \
    </dev/null > "$logfile" 2>&1 &
  echo "  pid=$!"
done

echo "[launch] checkpoint=$CHECKPOINT"
echo "[launch] results_folder=$RESULTS_FOLDER"
