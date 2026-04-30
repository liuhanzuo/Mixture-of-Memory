#!/usr/bin/env bash
# Thread B H_phase experiment — rank1_kv256_recent_sweep_llama2.
#
# /researcher hypothesis (2026-04-26 18:05): kv>=192 PPL outlier band at
# rank=1 is driven by keep_old crossing the recent_window threshold. At
# kv=96/128: keep_old=32/64 <= recent_window=64. At kv=192/256:
# keep_old=128/192 > recent_window=64 → a filter-scored regime engages.
#
# Experiment: fix kv=256, filter_rank=1, calibration_chunks=64, seed=0, and
# sweep recent_window in {64, 128, 192, 256}. All other hyperparameters
# IDENTICAL to _run_issue110_multiseed_llama2.sh. Decision rule (per the
# /researcher report): recent_window=192 PPL<=200 confirms H_phase; >400
# promotes H_per_head offline audit.
#
# 4 single-GPU runs launched in parallel on b200-2 GPUs 0..3. No code
# change — reuses the same eval_qfilters.py path; filters.pt per-run so
# calibration is independent.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 1 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters --single_gpu --kv_budget 256"
# NB: --seed removed — eval_qfilters.py has no --seed flag. Prior /researcher verdict
# (2026-04-26) established Q-Filters pipeline is bit-identical across seeds post SVD fix
# (std=0.0 at kv=192/256), so determinism invariant holds without the flag.
OUTROOT=outputs/rank1_kv256_recent_sweep
mkdir -p "$OUTROOT" logs
SUMMARY=logs/rank1_kv256_recent_sweep_${TS}.summary.log
echo "=== rank1_kv256_recent_sweep START $(date) ===" | tee "$SUMMARY"

declare -a PIDS=()
declare -a TAGS=()

launch_one() {
  local gpu="$1"; local recent="$2"
  local tag="recent${recent}"
  local out="${OUTROOT}/${tag}"
  local log="logs/rank1_kv256_recent_sweep_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "[launch] GPU=${gpu} recent_window=${recent} out=${out} log=${log}" | tee -a "$SUMMARY"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --recent_window "$recent" \
    --output_dir "$out" >> "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  TAGS+=("$tag")
  echo "[launched] pid=${pid} tag=${tag}" | tee -a "$SUMMARY"
}

# 4 runs on GPUs 0..3 — recent_window in {64,128,192,256}
launch_one 0 64
launch_one 1 128
launch_one 2 192
launch_one 3 256

echo "=== All 4 runs backgrounded PIDs=${PIDS[*]} ===" | tee -a "$SUMMARY"

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  tag="${TAGS[$i]}"
  if wait "$pid"; then
    echo "[done] tag=${tag} pid=${pid} EXIT=0" | tee -a "$SUMMARY"
  else
    ec=$?
    echo "[done] tag=${tag} pid=${pid} EXIT=${ec}  FAILURE" | tee -a "$SUMMARY"
  fi
done

echo "=== rank1_kv256_recent_sweep END $(date) ===" | tee -a "$SUMMARY"
echo "=== Result summary ===" | tee -a "$SUMMARY"
for d in "${OUTROOT}"/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):18s} kv={r['kv_budget']:4d} recent={r.get('recent_window')} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$SUMMARY"
