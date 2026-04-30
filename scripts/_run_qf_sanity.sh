#!/usr/bin/env bash
# Q-Filters harness sanity check (2026-04-25, post-WikiText-sweep anomaly).
#
# WikiText sweep reported sw_b64=161.05 < sw_b256=213.82 (monotonicity violated)
# and qf_b64 ≡ sw_b64 ≡ 161.05 (filter vs no-filter path identical) while
# dense_4096 = 6.80 stayed healthy. Hypothesis: sub-window carryover in
# evaluate_ppl() does not re-rotate preserved K's RoPE after compression.
#
# This script isolates the three variables (carryover, compression,
# single-forward) so we can tell exactly which dial breaks PPL:
#
#   S1: dense control        — kv=4096, sw_len=4096 (no split, no compress)
#   S2: carryover only       — kv=4096, sw_len=1024 (split, but compress no-op
#                              because kv == seq_len)
#   S3: compress only        — kv=64,   sw_len=4096 (single forward per chunk,
#                              compression fires but cache is discarded per chunk)
#   S4: compress + carryover — kv=64,   sw_len=1024 (the suspect path)
#
# Expectation if the hypothesis is right:
#   S1, S2, S3 ≈ 6.8        (dense-equivalent)
#   S4 ≈ 161                (reproduces the suspect sweep number)
#
# If S2 breaks too → carryover itself is broken (not just RoPE).
# If S3 breaks → compression path is broken independent of carryover.
# If S4 is fine → our bug hypothesis is wrong; reopen investigation.
#
# 10 chunks each is enough — we just need directionality, not a proper PPL.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b
DATA=data/wikitext_chunks_llama3_4096.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 64 --max_chunks 10 --filter_rank 2 \
--calibration_chunks 64 --bf16 --attn_impl sdpa --recent_window 64 \
--mode sliding_window"
OUTROOT=outputs/qf_sanity_${TS}
ROOT_LOG=logs/qf_sanity_${TS}.log
mkdir -p "$OUTROOT"
echo "=== QF harness sanity starting $(date) ===" | tee "$ROOT_LOG"
echo "MODEL=$MODEL" | tee -a "$ROOT_LOG"
echo "DATA=$DATA" | tee -a "$ROOT_LOG"

run_one() {
  local tag="$1"; local kv="$2"; local swlen="$3"
  local out="${OUTROOT}/${tag}"
  local log="logs/qf_sanity_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "=== [${tag}] kv=${kv} sub_window_len=${swlen} at $(date) ===" | tee -a "$ROOT_LOG"
  torchrun --nproc_per_node=8 --master_port=29538 scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --sub_window_len "$swlen" \
    --output_dir "$out" >> "$log" 2>&1
  local ppl
  ppl=$(python -c "import json;print(json.load(open('${out}/eval_results.json'))['ppl'])" 2>/dev/null || echo "NA")
  echo "[${tag}] kv=${kv} sw_len=${swlen} -> PPL=${ppl}" | tee -a "$ROOT_LOG"
}

run_one s1_dense_sw4096       4096 4096
run_one s2_carryover_sw1024   4096 1024
run_one s3_compress_sw4096      64 4096
run_one s4_compress_sw1024      64 1024

echo "=== QF sanity DONE $(date) ===" | tee -a "$ROOT_LOG"
echo "=== Summary ===" | tee -a "$ROOT_LOG"
for d in ${OUTROOT}/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):28s} kv={r['kv_budget']:4d} sw_len={r['sub_window_len']:4d} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$ROOT_LOG"
