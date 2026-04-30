#!/usr/bin/env bash
# Resume driver for calib_size_disambig sweep: recovers the final 2 runs
# (r4_c256 and r8_c256) that crashed due to NCCL barrier timeout at 10 min
# (calibration on rank 0 alone exceeds the default 600s while ranks 1-7 wait).
#
# Strategy: pre-compute filters.pt on 1 GPU (no distributed barrier), then run
# the 8-GPU eval with --filters_cache pointing to the cached file, which
# bypasses the long single-rank calibration in the distributed process.
set -euo pipefail
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --kv_budget 512 --recent_window 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters"
ROOT_LOG=logs/llama3_calib_size_disambig_resume_${TS}.log
mkdir -p logs
echo "=== Resume c=256 runs starting $(date) ===" | tee "$ROOT_LOG"

resume_one() {
  local rank="$1"; local calib="$2"
  local tag="qf_r${rank}_c${calib}"
  local out="outputs/calib_size_disambig_llama3/${tag}"
  local cache="${out}/filters.pt"
  local calib_log="logs/llama3_calib_disambig_${tag}_calib_${TS}.log"
  local eval_log="logs/llama3_calib_disambig_${tag}_eval_${TS}.log"
  mkdir -p "$out"

  # Phase 1: single-GPU calibration (skip eval loop by using skip_chunks huge)
  if [ ! -f "$cache" ]; then
    echo "=== [${tag}] phase1 calib (1 GPU) at $(date) ===" | tee -a "$ROOT_LOG"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_qfilters.py \
      --model "$MODEL" --data "$DATA" $COMMON \
      --single_gpu \
      --filter_rank "$rank" --calibration_chunks "$calib" \
      --skip_chunks 99999 --max_chunks 1 \
      --output_dir "$out" >> "$calib_log" 2>&1 || true
    local t1=$(date +%s)
    echo "[${tag}] calib wall=$((t1-t0))s -> $(ls -la $cache 2>&1 | awk '{print $5}') bytes" | tee -a "$ROOT_LOG"
  else
    echo "[${tag}] filters.pt already present - skip calib" | tee -a "$ROOT_LOG"
  fi

  # Phase 2: 8-GPU eval with cached filters
  echo "=== [${tag}] phase2 eval (8 GPU, cached filters) at $(date) ===" | tee -a "$ROOT_LOG"
  local t2=$(date +%s)
  torchrun --nproc_per_node=8 --master_port=29553 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --filter_rank "$rank" --calibration_chunks "$calib" \
    --skip_chunks 200 --max_chunks 200 \
    --filters_cache "$cache" \
    --output_dir "$out" >> "$eval_log" 2>&1
  local t3=$(date +%s)
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] rank=${rank} calib=${calib} -> PPL=${ppl}  eval_wall=$((t3-t2))s" | tee -a "$ROOT_LOG"
}

resume_one 4 256
resume_one 8 256

echo "=== Resume DONE $(date) ===" | tee -a "$ROOT_LOG"
