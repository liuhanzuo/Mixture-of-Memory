#!/usr/bin/env bash
# Post-Patch-A monotonicity sanity (2026-04-25 23:00).
#
# If Patch A is correct, keeping more recent tokens should give lower PPL:
#   sw_b64   (64 recent tokens)   → highest PPL (most info lost)
#   sw_b256  (256 recent tokens)  → middle
#   sw_b1024 (1024 recent tokens) → lowest PPL (most info preserved)
#
# Pre-Patch-A, we saw sw_b64=161 < sw_b256=213 (monotonicity VIOLATED — smoking gun).
# Post-Patch-A we expect monotonicity restored.
#
# 10 chunks @ 4096, same Llama-3-8B + wikitext data.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/wikitext_chunks_llama3_4096.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 64 --max_chunks 10 --filter_rank 2 \
--calibration_chunks 64 --bf16 --attn_impl sdpa --recent_window 64 \
--mode sliding_window --sub_window_len 1024"
OUTROOT=outputs/qf_mono_${TS}
ROOT_LOG=logs/qf_mono_${TS}.log
mkdir -p "$OUTROOT"
echo "=== QF monotonicity starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"
  local out="${OUTROOT}/${tag}"
  local log="logs/qf_mono_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] kv=${kv} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29539 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] kv=${kv} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

run_one sw_b64     64
run_one sw_b256   256
run_one sw_b1024 1024

echo "=== QF mono DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in ${OUTROOT}/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):10s} kv={r['kv_budget']:5d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
