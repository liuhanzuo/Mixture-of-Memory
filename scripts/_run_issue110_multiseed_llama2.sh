#!/usr/bin/env bash
# Issue #110 multi-seed characterization sweep.
#
# Post-fix (exact SVD applied 2026-04-26) Llama-2-7B Q-Filters rank=1 still
# lands in the contamination band at kv={192,256} (PPL 479 / 611) while
# adjacent kv points (96/128) are healthy (107 / 150). This sweep varies
# --seed across {0,1,2} at kv in {192,256} to decide whether the failure is
# stochastic (volatile across seeds → fix: higher rank / averaging) or
# reproducible (stable → different fix needed, probable attention-sink or
# calibration mismatch).
#
# 6 single-GPU runs launched in parallel on b200-3 GPUs 0..5 (2 idle for
# safety margin). All other hyperparameters IDENTICAL to
# _run_llama2_rank1_verify_sweep.sh at kv=192/256 (rank=1, recent_window=64,
# calibration_chunks=64, sub_window_len=1024, skip_chunks=200,
# max_chunks=200, seq_length=4096, bf16, sdpa).
#
# Each run writes outputs/issue110_multiseed/kv{kv}_seed{seed}/eval_results.json
# and logs/issue110_multiseed_kv{kv}_seed{seed}_${TS}.log.
set -euo pipefail
cd /root/Mixture-of-Memory
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch-base 2>/dev/null || true

MODEL=models/Llama--Llama2-7b
DATA=data/pg19_chunks.npy
TS=$(date +%Y%m%d_%H%M%S)
COMMON="--seq_length 4096 --skip_chunks 200 --max_chunks 200 --filter_rank 1 --recent_window 64 --calibration_chunks 64 --sub_window_len 1024 --bf16 --attn_impl sdpa --mode qfilters --single_gpu"
OUTROOT=outputs/issue110_multiseed
mkdir -p "$OUTROOT" logs
SUMMARY=logs/issue110_multiseed_${TS}.summary.log
echo "=== Issue #110 multi-seed sweep START $(date) ===" | tee "$SUMMARY"

declare -a PIDS=()
declare -a TAGS=()

launch_one() {
  local gpu="$1"; local kv="$2"; local seed="$3"
  local tag="kv${kv}_seed${seed}"
  local out="${OUTROOT}/${tag}"
  local log="logs/issue110_multiseed_${tag}_${TS}.log"
  mkdir -p "$out"
  echo "[launch] GPU=${gpu} kv=${kv} seed=${seed} out=${out} log=${log}" | tee -a "$SUMMARY"
  # Per-run filters.pt stays under the run's output_dir so each seed calibrates
  # independently (this is intentional — seed-dependence lives in the
  # calibration pipeline, not in eval). No --filters_cache override.
  CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/eval_qfilters.py \
    --model "$MODEL" --data "$DATA" $COMMON \
    --kv_budget "$kv" --seed "$seed" \
    --output_dir "$out" >> "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  TAGS+=("$tag")
  echo "[launched] pid=${pid} tag=${tag}" | tee -a "$SUMMARY"
}

# 6 runs on GPUs 0..5
launch_one 0 192 0
launch_one 1 192 1
launch_one 2 192 2
launch_one 3 256 0
launch_one 4 256 1
launch_one 5 256 2

echo "=== All 6 runs backgrounded PIDs=${PIDS[*]} ===" | tee -a "$SUMMARY"

# Wait and record per-run exit code
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

echo "=== Issue #110 multi-seed sweep END $(date) ===" | tee -a "$SUMMARY"
echo "=== Result summary ===" | tee -a "$SUMMARY"
for d in "${OUTROOT}"/*/eval_results.json; do
  python -c "import json,os; r=json.load(open('$d')); print(f\"{os.path.basename(os.path.dirname('$d')):20s} kv={r['kv_budget']:4d} seed={r.get('seed')} PPL={r['ppl']:.4f}\")" 2>/dev/null
done | tee -a "$SUMMARY"
