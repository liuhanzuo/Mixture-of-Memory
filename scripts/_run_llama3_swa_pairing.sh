#!/usr/bin/env bash
# Llama-3.0-8B pure-SWA control points paired with existing Q-Filters kv_curve.
# Fills in the missing SWA @ kv ∈ {64, 128, 512}, recent=64, to complete the
# "SWA vs SWA+Q-Filters @ same budget" table. SWA @ kv=256 r=64 already done
# (sw_b256_r64 = 133.62 PPL) — not rerun.
#
# mode=sliding_window ⇒ filters dict empty, compress_layer falls back to last-kv
# keys (pure SWA). No calibration required.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/pg19_chunks_llama3_noeos.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode sliding_window --recent_window 64"
ROOT_LOG=logs/llama3_swa_pairing_${TS}.log
echo "=== Llama-3 SWA pairing starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"
  local out="outputs/postfix_llama3_swa_pair/${tag}"
  local log="logs/llama3_swa_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] kv=${kv} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29536 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] kv=${kv} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

run_one sw_b64_r64    64
run_one sw_b128_r64  128
run_one sw_b512_r64  512

echo "=== Llama-3 SWA pairing DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in outputs/postfix_llama3_swa_pair/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
