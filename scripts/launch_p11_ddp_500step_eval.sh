#!/usr/bin/env bash
# Eval the Exp A — P11 DDP 500-step diagnostic checkpoint.
# 6-way parallel BABILong eval (3 tasks × {short, long}).

set -u

TS="${TS:-$(date +%Y%m%d_%H%M)}"
RUN_DIR="outputs/babilong_sft_phase11_ddp_500step_validate"
CKPT="${RUN_DIR}/mem_space_adapter.pt"
ACFG="${RUN_DIR}/adapter_config.json"
MODEL="models/Meta-Llama-3-8B-Instruct"
RESULTS="outputs/eval_p11_ddp_500step"
LOGDIR="logs"

mkdir -p "$RESULTS" "$LOGDIR"

run_cell () {
  local task="$1"
  local split="$2"
  local lengths="$3"
  local gpu="$4"
  local out="${task}_${split}"
  local log="${LOGDIR}/eval_p11_ddp_500step_${task}_${split}_${TS}.log"

  CUDA_VISIBLE_DEVICES="$gpu" \
  python scripts/run_babilong_mem_space.py \
      --model_path        "$MODEL" \
      --checkpoint        "$CKPT" \
      --adapter_config    "$ACFG" \
      --results_folder    "$RESULTS" \
      --output_name       "p11_ddp_500step_${out}" \
      --tasks             "$task" \
      --lengths           $lengths \
      --chunk_size        4096 \
      --max_new_tokens    20 \
      --limit             100 \
      --device            cuda:0 \
      --use_chat_template \
      > "$log" 2>&1 &

  echo "[launcher] task=${task} split=${split} gpu=${gpu} pid=$! log=${log}"
}

run_cell qa1 short "0k 1k 2k 4k" 0
run_cell qa2 short "0k 1k 2k 4k" 1
run_cell qa5 short "0k 1k 2k 4k" 2
run_cell qa1 long  "8k 16k 32k"  3
run_cell qa2 long  "8k 16k 32k"  4
run_cell qa5 long  "8k 16k 32k"  5

wait
echo "[launcher] all 6 workers exited"
