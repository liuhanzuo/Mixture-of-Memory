#!/usr/bin/env bash
# P0.12 ACCEPTANCE launcher — fans 2 arms x 3 reps (timing) + 1 consistency proc
# across the 8 free GPUs of .82 (28.82.250.82, diskB), then waits for all.
#
# Timing:  armA (resume_j=0) rep{1,2,3} -> GPU 0,1,2
#          armB (resume_j=12) rep{1,2,3} -> GPU 3,4,5
# (7):     consistency (both arms, one process)   -> GPU 6
#
# Each proc independently rebuilds + sha-verifies the identical top-12 pack and
# aborts (exit 2) on any sha mismatch, so no口径-inconsistent result is ever
# written. Outputs -> bench_results/p0_12_acceptance/.
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUT_DIR="${OUT_DIR:-bench_results/p0_12_acceptance}"
LENGTH="${LENGTH:-16k}"
CHUNK="${CHUNK:-512}"
TOPK="${TOPK:-12}"
NREP="${NREP:-20}"
WARMUP="${WARMUP:-3}"
NDECODE="${NDECODE:-16}"

cd "$PROJECT_ROOT" || { echo "cd $PROJECT_ROOT failed"; exit 1; }
mkdir -p "$OUT_DIR" logs
SCRIPT=scripts/bench_p0_12_acceptance.py

echo "=== P0.12 acceptance launch $(date) ==="
echo "PROJECT_ROOT=$PROJECT_ROOT PYTHON_BIN=$PYTHON_BIN"
echo "MODEL_PATH=$MODEL_PATH LORA=$LORA OUT_DIR=$OUT_DIR"

common() {
  echo --model_path "$MODEL_PATH" --lora_adapter "$LORA" \
       --length "$LENGTH" --chunk_size "$CHUNK" --topk "$TOPK" \
       --dtype bfloat16 --attn_impl sdpa --seed 0
}

pids=()

# ---- timing: armA (resume_j=0) rep 1..3 on GPU 0..2 ----
for rep in 1 2 3; do
  gpu=$((rep - 1))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$SCRIPT" --mode timing \
    --resume_j 0 --arm_name armA --rep_id "$rep" \
    --n_repeat "$NREP" --warmup "$WARMUP" $(common) \
    --output "$OUT_DIR/armA_rep${rep}.json" \
    >"logs/p012acc_armA_rep${rep}.out" 2>&1 &
  pids+=($!)
done

# ---- timing: armB (resume_j=12) rep 1..3 on GPU 3..5 ----
for rep in 1 2 3; do
  gpu=$((rep + 2))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$SCRIPT" --mode timing \
    --resume_j 12 --arm_name armB --rep_id "$rep" \
    --n_repeat "$NREP" --warmup "$WARMUP" $(common) \
    --output "$OUT_DIR/armB_rep${rep}.json" \
    >"logs/p012acc_armB_rep${rep}.out" 2>&1 &
  pids+=($!)
done

# ---- (7) output-consistency: both arms, one process, on GPU 6 ----
CUDA_VISIBLE_DEVICES=6 "$PYTHON_BIN" "$SCRIPT" --mode consistency \
  --n_decode "$NDECODE" $(common) \
  --output "$OUT_DIR/consistency.json" \
  >"logs/p012acc_consistency.out" 2>&1 &
pids+=($!)

echo "launched ${#pids[@]} procs: ${pids[*]}"
rc=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "proc $p exited non-zero"; rc=1; }
done
echo "=== ALL_DONE $(date) rc=$rc ==="
exit $rc
