#!/usr/bin/env bash
# Offline BABILong eval for P8b chunk512 top_k8 ablation, checkpoint step500.
# EXACT same protocol as scripts/eval_perdoc_chunk512_p8_nullsink_step500.sh
# (the chunk512 top_k16 baseline eval): same entrypoint run_babilong_mem_space.py,
# same tasks qa1/qa2/qa5, same lengths 0k-32k, same --limit 100, --chunk_size 512,
# same dtype/attn. ONLY the ckpt dir + results/output names + GPU list differ.
# Runs on disk-B (share_304376610), reads the step500 ckpt written by the .76
# training run (shared FS across .76/.249).
#
# GPU SELECTION: pass GPUS env (space-separated) to avoid OOM contention with the
# still-running training. Default = single GPU "7" run SEQUENTIALLY over lengths
# (eval needs only 1 GPU; sequential keeps footprint to one 8B model at a time).
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# diskB nodes have NO internet; load_dataset must read the pre-warmed local Arrow
# cache (0k-32k all warmed 2026-06-06) instead of hanging on huggingface.co HEAD.
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p8b_chunk512_topk8_diskB
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
RESULTS=babilong_results/p8b_chunk512_topk8_step500
TASKS="qa1 qa2 qa5"
LOGDIR=logs/eval_p8b_chunk512_topk8_step500
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
# Space-separated GPU ids; lengths are round-robined across them. Default: 1 GPU.
read -r -a GPUS <<< "${GPUS:-7}"
i=0
for L in "${LENGTHS[@]}"; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "[$(date)] GPU $G -> length $L"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RESULTS --output_name p8b_chunk512_topk8_step500_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa \
    </dev/null >"$LOGDIR/${L}.log" 2>&1
  i=$((i+1))
done
echo "[$(date)] all eval lengths done"
