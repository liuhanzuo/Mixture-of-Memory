#!/usr/bin/env bash
# Full Llama-2-7B Q-Filters sweep on the POST-FIX (double-label-shift bug
# fixed 2026-04-25) baseline. Reuses outputs/qfilters_baseline/filters.pt
# (rank-2, 64 calibration chunks — calibration is label-shift-invariant).
#
# Runs sequentially on 8× L20A (b200-1). Red line: no parallel 8-GPU.
#
# Sweep geometry (13 runs; 256/64 shared between kv_budget & recent sweeps):
#   Dense baseline        : mode=sliding_window, kv_budget=4096 (short-circuits)
#   kv_budget (recent=64) : {64, 128, 256, 512}
#   Filter-OFF control    : kv_budget=64, recent=64 (keep_old=0)
#   High-compression tune : kv_budget=64, recent=16
#   SW control            : mode=sliding_window, kv_budget=512, recent=64
#   recent_window @ 256   : {16, 32, 48, 96, 128}   # 64 already covered above
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
BASEFILTERS=outputs/qfilters_baseline/filters.pt
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 2 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa"
ROOT_LOG=logs/llama2_sweep_postfix_${TS}.log
echo "=== Llama-2 Q-Filters POST-FIX sweep starting $(date) ===" | tee "$ROOT_LOG"

run_one() {
  local tag="$1"; local mode="$2"; local kv="$3"; local rw="$4"
  local out="outputs/postfix_llama2/${tag}"
  local log="logs/llama2_postfix_${tag}_${TS}.log"
  mkdir -p "$out"
  # Reuse the existing calibration via --filters_cache (rank-0 reads it
  # verbatim; all ranks broadcast). SW/dense paths ignore filters but we pass
  # the flag for uniformity.
  echo "=== [${tag}] mode=${mode} kv_budget=${kv} recent=${rw} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29532 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --recent_window "$rw" --mode "$mode" \
    --filters_cache "$BASEFILTERS" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] mode=${mode} kv=${kv} recent=${rw} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

# --- Dense baseline (through harness) ---
run_one dense_4096          sliding_window 4096 64

# --- kv_budget sweep @ recent=64 ---
run_one qf_b64_r64           qfilters        64  64   # keep_old=0 → filter-OFF control
run_one qf_b128_r64          qfilters       128  64
run_one qf_b256_r64          qfilters       256  64   # headline point
run_one qf_b512_r64          qfilters       512  64

# --- Aggressive-compression re-tuned recent ---
run_one qf_b64_r16           qfilters        64  16

# --- Sliding-window control (no filter scoring) ---
run_one sw_b512_r64          sliding_window 512  64

# --- recent_window sweep @ kv_budget=256 ---
run_one qf_b256_r16          qfilters       256  16
run_one qf_b256_r32          qfilters       256  32
run_one qf_b256_r48          qfilters       256  48
run_one qf_b256_r96          qfilters       256  96
run_one qf_b256_r128         qfilters       256 128

echo "=== Llama-2 POST-FIX sweep DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Result summary ===" | tee -a "$ROOT_LOG"
for d in outputs/postfix_llama2/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} mode={r['mode']:15s} kv={r['kv_budget']:4d} recent={r['recent_window']:3d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
