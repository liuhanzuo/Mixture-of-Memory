#!/usr/bin/env bash
# Sliding-window long-context PPL sweep: 3 datasets x 2 models on one 8-GPU node.
# base (standard sliding window=8192/stride=4096) + mem_space (chunk1024 streaming).
# Each (dataset,model) on its own GPU -> 6 GPUs used in parallel.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"

MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk1024_deltarule_normreadout
ADAPTER=$CKPT_DIR/adapter_config.json
CKPT=$CKPT_DIR/mem_space_adapter.pt
SEQ=${SEQ:-32768}
WINDOW=${WINDOW:-8192}
STRIDE=${STRIDE:-4096}
CHUNK=${CHUNK:-1024}
MAXTOK=${MAXTOK:-1000000}
SKIP=${SKIP:-0}
OUT=results/sliding_ppl
LOG=logs/sliding_ppl
mkdir -p "$OUT" "$LOG"

declare -A DATAPATH=(
  [pg19]=data/pg19_chunks_llama3_noeos.npy
  [proofpile]=data/proofpile_llama3_noeos.npy
  [codeparrot]=data/codeparrot_llama3_noeos.npy
)

run_base() {
  local DS=$1 GPU=$2
  CUDA_VISIBLE_DEVICES=$GPU $PYBIN scripts/eval_sliding_ppl.py \
    --data $DS --data_path ${DATAPATH[$DS]} --model_path $MODEL \
    --seq_length $SEQ --window $WINDOW --stride $STRIDE \
    --skip_tokens $SKIP --max_tokens $MAXTOK --gpu 0 \
    --output_json $OUT/base_${DS}.json \
    >"$LOG/base_${DS}.log" 2>&1
}
run_mem() {
  local DS=$1 GPU=$2
  CUDA_VISIBLE_DEVICES=$GPU $PYBIN scripts/eval_sliding_ppl.py \
    --data $DS --data_path ${DATAPATH[$DS]} --model_path $MODEL \
    --adapter_config $ADAPTER --checkpoint $CKPT \
    --seq_length $SEQ --chunk_size $CHUNK \
    --skip_tokens $SKIP --max_tokens $MAXTOK --gpu 0 \
    --output_json $OUT/mem_${DS}.json \
    >"$LOG/mem_${DS}.log" 2>&1
}

# GPU assignment: base on 0/1/2, mem on 3/4/5
run_base pg19       0 &
run_base proofpile  1 &
run_base codeparrot 2 &
run_mem  pg19       3 &
run_mem  proofpile  4 &
run_mem  codeparrot 5 &
wait
echo "[$(date)] all 6 sliding-PPL jobs done"
for f in $OUT/*.json; do echo "=== $f ==="; cat "$f"; echo; done
